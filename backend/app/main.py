"""
FastAPI backend for the Turbofan RUL dashboard.

Refactor of the original backend/main.py:
  - Data pipeline, loss, and MC Dropout logic now come from the shared
    `turbofan_rul` package instead of being duplicated inline.
  - Domain-adaptation artifacts (KMeans + scalers) are persisted via joblib and
    loaded on startup instead of being refit from raw data every time.
  - Config comes from environment variables (see app/config.py) instead of
    hardcoded constants.
  - Proper HTTP status codes (404) instead of 200-with-error-body.
  - Structured logging instead of print().
  - New: maintenance recommendation + risk level on every engine, plus a
    fleet-wide summary endpoint — pure post-hoc business logic on top of the
    existing model output, no changes to the model itself.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from turbofan_rul import CONFIG, CUSTOM_OBJECTS  # noqa: E402
from turbofan_rul.data import (  # noqa: E402
    load_all_datasets, apply_domain_adaptation, fit_domain_adaptation, make_test_sequences,
)
from turbofan_rul.artifacts import save_artifacts, load_artifacts, artifacts_exist  # noqa: E402
from turbofan_rul.inference import mc_predict, health_zone, maintenance_recommendation  # noqa: E402

from .config import SETTINGS

logging.basicConfig(level=SETTINGS.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("turbofan_rul.backend")

cfg = CONFIG


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #
class EngineSummary(BaseModel):
    id: str
    original_unit: int
    cycle: int
    rul_predicted: int
    rul_std: float
    status: str                    # health zone (RUL-bucket only)
    risk_level: str                # RUL + uncertainty combined
    maintenance_category: str      # Healthy / Inspection Recommended / Maintenance Required / Critical
    recommended_action: str
    uncertainty_elevated: bool


class FleetSummary(BaseModel):
    total_engines: int
    by_status: Dict[str, int]
    by_risk_level: Dict[str, int]
    by_maintenance_category: Dict[str, int]
    mean_uncertainty: float


# --------------------------------------------------------------------------- #
# Application state (populated at startup)
# --------------------------------------------------------------------------- #
class AppState:
    model = None
    kmeans = None
    scalers = None
    test_df = None
    test_last = None
    feature_cols: List[str] = []


state = AppState()


def _resolve_path(path: str) -> str:
    """Fall back to the path relative to CWD if the configured one doesn't exist
    (mirrors the original repo's behavior of being runnable from either the repo
    root or the backend/ directory)."""
    if os.path.exists(path):
        return path
    stripped = path.lstrip("./")
    if os.path.exists(stripped):
        return stripped
    return path


def _load_data_and_artifacts() -> None:
    import tensorflow as tf

    data_path = _resolve_path(SETTINGS.data_path)
    logger.info("Loading C-MAPSS data from %s", data_path)
    train_df, test_df, test_last = load_all_datasets(data_path, cfg)
    test_last["RUL"] = test_last["RUL"].clip(upper=cfg.max_rul)

    artifacts_path = _resolve_path(SETTINGS.artifacts_path)
    if artifacts_exist(artifacts_path):
        logger.info("Loading persisted domain-adaptation artifacts from %s", artifacts_path)
        kmeans, scalers = load_artifacts(artifacts_path)
    else:
        logger.warning(
            "No persisted artifacts found at %s — fitting from training data "
            "(this is slow; run training/train_v4.py once to persist them)",
            artifacts_path,
        )
        kmeans, scalers = fit_domain_adaptation(train_df, cfg)
        save_artifacts(kmeans, scalers, artifacts_path)

    test_df = apply_domain_adaptation(test_df, kmeans, scalers, cfg)
    test_last["condition"] = test_df.groupby("unit")["condition"].last().values

    state.kmeans = kmeans
    state.scalers = scalers
    state.test_df = test_df
    state.test_last = test_last
    state.feature_cols = cfg.feature_cols

    model_path = _resolve_path(SETTINGS.model_path)
    logger.info("Loading model from %s", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train it with training/train_v4.py "
            f"or place a compatible .keras file at this path (see MODELS.md)."
        )
    state.model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)

    os.makedirs(SETTINGS.cache_dir, exist_ok=True)
    logger.info("Startup complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_data_and_artifacts()
    yield


app = FastAPI(title="Aerospace RUL Monitor API", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _engine_summary(unit: int, cycle: int, rul_mean: float, rul_std: float) -> EngineSummary:
    zone = health_zone(rul_mean, cfg)
    rec = maintenance_recommendation(rul_mean, rul_std, cfg)
    return EngineSummary(
        id=f"ENG-{unit}",
        original_unit=int(unit),
        cycle=int(cycle),
        rul_predicted=max(0, round(rul_mean)),
        rul_std=round(float(rul_std), 1),
        status=zone,
        risk_level=rec.risk_level,
        maintenance_category=rec.maintenance_category,
        recommended_action=rec.recommended_action,
        uncertainty_elevated=rec.uncertainty_elevated,
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_loaded": state.model is not None}


@app.get("/api/engines", response_model=List[EngineSummary])
def get_engines():
    selected_units: List[int] = []
    for ds in cfg.datasets:
        units = state.test_last[state.test_last["dataset"] == ds]["unit"].head(
            SETTINGS.fleet_engines_per_dataset
        ).tolist()
        selected_units.extend(units)

    subset_last = state.test_last[state.test_last["unit"].isin(selected_units)].copy()
    subset_df = state.test_df[state.test_df["unit"].isin(selected_units)].copy()

    X_s, X_c = make_test_sequences(subset_df, cfg.sequence_len, state.feature_cols)
    y_mean, y_std = mc_predict(state.model, [X_s, X_c], n_samples=SETTINGS.mc_samples_fleet)

    results = []
    for i, unit in enumerate(selected_units):
        cycle = int(subset_last[subset_last["unit"] == unit]["cycle"].values[0])
        results.append(_engine_summary(unit, cycle, float(y_mean[i]), float(y_std[i])))

    logger.info("Served /api/engines — %d engines", len(results))
    return results


@app.get("/api/fleet/summary", response_model=FleetSummary)
def get_fleet_summary():
    engines = get_engines()
    by_status: Dict[str, int] = {}
    by_risk: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    for e in engines:
        by_status[e.status] = by_status.get(e.status, 0) + 1
        by_risk[e.risk_level] = by_risk.get(e.risk_level, 0) + 1
        by_category[e.maintenance_category] = by_category.get(e.maintenance_category, 0) + 1

    mean_uncertainty = float(np.mean([e.rul_std for e in engines])) if engines else 0.0
    return FleetSummary(
        total_engines=len(engines),
        by_status=by_status,
        by_risk_level=by_risk,
        by_maintenance_category=by_category,
        mean_uncertainty=round(mean_uncertainty, 2),
    )


@app.get("/api/engines/{unit_id}/telemetry")
def get_engine_telemetry(unit_id: int):
    engine_data = state.test_df[state.test_df["unit"] == unit_id].copy()
    if engine_data.empty:
        raise HTTPException(status_code=404, detail=f"Engine {unit_id} not found")

    cycles = engine_data["cycle"].values
    feature_cols = state.feature_cols

    t24_idx = feature_cols.index("s_2") if "s_2" in feature_cols else 0
    p30_idx = feature_cols.index("s_3") if "s_3" in feature_cols else 1  # NOTE: s_3 is T30, see MODELS.md

    telemetry = []
    for i, cycle in enumerate(cycles):
        row = engine_data.iloc[i]
        telemetry.append({
            "cycle": int(cycle),
            "t24": float(row[feature_cols[t24_idx]]),
            "t30": float(row[feature_cols[p30_idx]]),
        })

    degradation = []
    start_idx = max(0, len(cycles) - 20)
    batch_X_s, batch_X_c = [], []
    for i in range(start_idx, len(cycles)):
        seq_data = engine_data.iloc[: i + 1]
        X_s, X_c = make_test_sequences(seq_data, cfg.sequence_len, feature_cols)
        batch_X_s.append(X_s[0])
        batch_X_c.append(X_c[0])

    latest_rec: Optional[dict] = None
    if batch_X_s:
        batch_X_s = np.array(batch_X_s, dtype=np.float32)
        batch_X_c = np.array(batch_X_c, dtype=np.int32)
        n_actual = len(batch_X_s)

        # Pad to exactly 20 to avoid TF recurrent_dropout static batch size caching issues
        if n_actual < 20:
            pad_s = np.zeros((20 - n_actual, cfg.sequence_len, len(feature_cols)), dtype=np.float32)
            pad_c = np.zeros((20 - n_actual,), dtype=np.int32)
            batch_X_s = np.concatenate([batch_X_s, pad_s], axis=0)
            batch_X_c = np.concatenate([batch_X_c, pad_c], axis=0)

        mean_rul, std_rul = mc_predict(state.model, [batch_X_s, batch_X_c], n_samples=SETTINGS.mc_samples_detail)

        for idx in range(n_actual):
            m_val = float(mean_rul[idx])
            s_val = float(std_rul[idx])
            degradation.append({
                "cycle": int(cycles[start_idx + idx]),
                "predicted": m_val,
                "upper": m_val + (2 * s_val),
                "lower": m_val - (2 * s_val),
            })

        latest_mean, latest_std = float(mean_rul[n_actual - 1]), float(std_rul[n_actual - 1])
        rec = maintenance_recommendation(latest_mean, latest_std, cfg)
        latest_rec = {
            "rul_predicted": max(0, round(latest_mean)),
            "rul_std": round(latest_std, 1),
            "status": health_zone(latest_mean, cfg),
            "risk_level": rec.risk_level,
            "maintenance_category": rec.maintenance_category,
            "recommended_action": rec.recommended_action,
            "uncertainty_elevated": rec.uncertainty_elevated,
        }

    logger.info("Served telemetry for engine %d", unit_id)
    return {
        "telemetry": telemetry,
        "degradation": degradation,
        "recommendation": latest_rec,
        "details": {
            "sensor_legend": {"t24": cfg.sensor_names.get("s_2"), "t30": cfg.sensor_names.get("s_3")},
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
