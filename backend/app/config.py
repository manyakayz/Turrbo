"""Backend configuration, loaded from environment variables with sensible
defaults for local development. See README.md for the full list of variables."""
import os
from dataclasses import dataclass, field
from typing import List


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    data_path: str = os.environ.get("TURBOFAN_DATA_PATH", "../data/")
    model_path: str = os.environ.get("TURBOFAN_MODEL_PATH", "../turbofan_rul_v4.keras")
    artifacts_path: str = os.environ.get("TURBOFAN_ARTIFACTS_PATH", "../artifacts/domain_adaptation.joblib")
    cache_dir: str = os.environ.get("TURBOFAN_CACHE_DIR", "cache")

    mc_samples_fleet: int = int(os.environ.get("TURBOFAN_MC_SAMPLES_FLEET", "10"))
    mc_samples_detail: int = int(os.environ.get("TURBOFAN_MC_SAMPLES_DETAIL", "50"))

    fleet_engines_per_dataset: int = int(os.environ.get("TURBOFAN_FLEET_PER_DATASET", "5"))

    cors_allow_origins: List[str] = field(
        default_factory=lambda: _env_list("TURBOFAN_CORS_ORIGINS", ["http://localhost:5173"])
    )

    log_level: str = os.environ.get("TURBOFAN_LOG_LEVEL", "INFO")


SETTINGS = Settings()
