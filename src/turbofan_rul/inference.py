"""
Inference-time utilities.

`mc_predict` is the Monte Carlo Dropout implementation, unchanged from main3.py.
`health_zone`, `risk_level`, and `maintenance_recommendation` are NEW — pure
post-hoc business logic layered on top of the model's (mean, std) output. None
of them touch the model, retrain anything, or change a single prediction; they
just turn "RUL=42, std=18" into something a maintenance planner can act on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import PipelineConfig


def mc_predict(model, inputs, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout: keep dropout ACTIVE at inference time (training=True) and
    average `n_samples` stochastic forward passes. Returns (mean, std) — the mean
    is the point RUL estimate, the std is the model's epistemic uncertainty.
    """
    preds = np.stack(
        [model(inputs, training=True).numpy().flatten() for _ in range(n_samples)],
        axis=0,
    )
    return preds.mean(axis=0), preds.std(axis=0)


def health_zone(rul: float, cfg: PipelineConfig) -> str:
    """
    Pure RUL-bucket classification (matches the health-zone confusion matrix used
    during training/evaluation). This is the model's raw output, categorized.
    """
    if rul <= 30:
        return "Critical"
    if rul <= 60:
        return "Warning"
    if rul <= 90:
        return "Moderate"
    return "Healthy"


@dataclass(frozen=True)
class RiskAssessment:
    risk_level: str          # Low / Medium / High / Severe
    maintenance_category: str  # Healthy / Inspection Recommended / Maintenance Required / Critical
    recommended_action: str
    uncertainty_elevated: bool  # True if uncertainty alone pushed the recommendation up


def compute_risk_level(rul: float, std: float, cfg: PipelineConfig) -> str:
    """
    Combines the RUL point estimate with MC Dropout uncertainty into a single
    Low / Medium / High / Severe risk level. A confident low-RUL prediction and an
    uncertain moderate-RUL prediction can both be genuinely risky — this rule
    captures that instead of looking at RUL alone.

    Thresholds are intentionally simple and documented here (not learned) so they
    stay easy to tune for a real fleet's risk tolerance.
    """
    high_uncertainty = std >= 20.0
    moderate_uncertainty = std >= cfg.uncertainty_threshold

    if rul <= 30 or (rul <= 60 and high_uncertainty):
        return "Severe"
    if rul <= 60 or (rul <= 90 and high_uncertainty):
        return "High"
    if rul <= 90 or moderate_uncertainty:
        return "Medium"
    return "Low"


_MAINTENANCE_RULES = {
    "Healthy": ("Healthy", "No action required — continue standard monitoring."),
    "Moderate": ("Inspection Recommended", "Schedule a visual inspection within the next maintenance window."),
    "Warning": ("Maintenance Required", "Plan a maintenance shop visit within the next {rul} cycles."),
    "Critical": ("Critical", "Immediate maintenance required — ground engine and schedule a shop visit."),
}

_CATEGORY_ORDER = ["Healthy", "Inspection Recommended", "Maintenance Required", "Critical"]


def maintenance_recommendation(rul: float, std: float, cfg: PipelineConfig) -> RiskAssessment:
    """
    Turns (predicted RUL, uncertainty) into an actionable maintenance category.

    Base category comes from the RUL health zone. If the combined risk level
    (which also factors in uncertainty) is more severe than the RUL-only zone
    would suggest, the recommendation is bumped up one level — e.g. a
    moderate RUL prediction with very high uncertainty gets treated more
    conservatively than the point estimate alone would imply, which is the
    whole practical point of having MC Dropout uncertainty in the first place.
    """
    zone = health_zone(rul, cfg)
    risk = compute_risk_level(rul, std, cfg)
    base_category, base_action = _MAINTENANCE_RULES[zone]

    bump_map = {"Low": 0, "Medium": 0, "High": 1, "Severe": 2}
    base_idx = _CATEGORY_ORDER.index(base_category)
    bumped_idx = min(base_idx + bump_map.get(risk, 0), len(_CATEGORY_ORDER) - 1)
    final_category = _CATEGORY_ORDER[bumped_idx]
    elevated = final_category != base_category

    if elevated:
        _, action = _MAINTENANCE_RULES[
            [k for k, v in _MAINTENANCE_RULES.items() if v[0] == final_category][0]
        ]
    else:
        action = base_action

    action = action.format(rul=max(1, round(rul)))

    return RiskAssessment(
        risk_level=risk,
        maintenance_category=final_category,
        recommended_action=action,
        uncertainty_elevated=elevated,
    )
