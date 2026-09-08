"""
Central configuration for the Turbofan RUL pipeline.

Every constant here is imported by both the training script (training/train_v4.py)
and the serving backend (backend/app/main.py) so the two can never silently drift
apart. If you need to change a hyperparameter or threshold, change it here once.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PipelineConfig:
    # --- Data ---
    datasets: List[str] = field(default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"])
    sequence_len: int = 30
    max_rul: int = 150

    # Raw column layout of the C-MAPSS .txt files (26 columns per row)
    columns: List[str] = field(default_factory=lambda: (
        ["unit", "cycle"]
        + [f"os_{i}" for i in range(1, 4)]
        + [f"s_{i}" for i in range(1, 22)]
    ))
    os_cols: List[str] = field(default_factory=lambda: ["os_1", "os_2", "os_3"])
    # Near-zero-variance sensors carrying no degradation signal (dropped as features)
    drop_sensors: List[str] = field(default_factory=lambda: [
        "s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"
    ])

    # --- Domain adaptation ---
    n_conditions: int = 6  # KMeans clusters over operating settings

    # --- Training ---
    batch_size: int = 128
    epochs: int = 150
    val_fraction: float = 0.20

    # --- Monte Carlo Dropout ---
    mc_samples: int = 50
    uncertainty_threshold: float = 10.0  # cycles - flag engines above this

    # --- Health / risk zones (pure post-hoc business logic, not part of the model) ---
    zone_bins: List[int] = field(default_factory=lambda: [0, 30, 60, 90, 150])
    zone_labels: List[str] = field(default_factory=lambda: [
        "Critical", "Warning", "Moderate", "Healthy"
    ])

    # Real sensor names, for UI / documentation purposes only (does not affect the model).
    # Source: standard NASA C-MAPSS sensor definitions.
    sensor_names: dict = field(default_factory=lambda: {
        "s_1": "T2 (Fan inlet temp)", "s_2": "T24 (LPC outlet temp)",
        "s_3": "T30 (HPC outlet temp)", "s_4": "T50 (LPT outlet temp)",
        "s_5": "P2 (Fan inlet pressure)", "s_6": "P15 (Bypass-duct pressure)",
        "s_7": "P30 (HPC outlet pressure)", "s_8": "Nf (Fan speed)",
        "s_9": "Nc (Core speed)", "s_10": "epr (Engine pressure ratio)",
        "s_11": "Ps30 (HPC outlet static pressure)", "s_12": "phi (Fuel flow ratio)",
        "s_13": "NRf (Corrected fan speed)", "s_14": "NRc (Corrected core speed)",
        "s_15": "BPR (Bypass ratio)", "s_16": "farB (Burner fuel-air ratio)",
        "s_17": "htBleed (Bleed enthalpy)", "s_18": "Nf_dmd (Demanded fan speed)",
        "s_19": "PCNfR_dmd (Demanded corrected fan speed)",
        "s_20": "W31 (LPT coolant bleed)", "s_21": "W32 (HPT coolant bleed)",
    })

    @property
    def feature_cols(self) -> List[str]:
        """Sensor feature columns used by the v4 model (excludes unit/cycle/dropped sensors/os_*)."""
        exclude = set(["unit", "cycle"] + self.drop_sensors + self.os_cols)
        return [c for c in self.columns if c not in exclude]


CONFIG = PipelineConfig()
