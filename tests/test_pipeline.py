"""
Unit tests for src/turbofan_rul. These use small synthetic DataFrames instead of
the real C-MAPSS files (which are gitignored / not present in CI) so they run
fast and everywhere. See backend/tests/test_api.py for integration tests that
exercise the real data + model.
"""
import numpy as np
import pandas as pd
import pytest

from turbofan_rul.config import CONFIG
from turbofan_rul.data import add_rul, make_test_sequences, make_train_sequences
from turbofan_rul.inference import (
    compute_risk_level, health_zone, maintenance_recommendation,
)
from turbofan_rul.losses import asymmetric_huber_loss


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_feature_cols_excludes_dropped_and_meta_columns():
    cols = CONFIG.feature_cols
    assert "unit" not in cols
    assert "cycle" not in cols
    assert "os_1" not in cols
    for s in CONFIG.drop_sensors:
        assert s not in cols
    assert len(cols) == 14  # 21 sensors - 7 dropped


# --------------------------------------------------------------------------- #
# data.add_rul
# --------------------------------------------------------------------------- #
def test_add_rul_counts_down_to_failure_and_clips():
    df = pd.DataFrame({
        "unit": [1, 1, 1, 1, 1],
        "cycle": [1, 2, 3, 4, 5],
    })
    out = add_rul(df, max_rul=3)
    # last cycle (5) is failure -> RUL 0; earlier cycles count down, clipped at 3
    assert out["RUL"].tolist() == [3, 3, 2, 1, 0]


def test_add_rul_handles_multiple_engines_independently():
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2, 2],
        "cycle": [1, 2, 1, 2, 3],
    })
    out = add_rul(df, max_rul=150)
    engine1 = out[out["unit"] == 1]["RUL"].tolist()
    engine2 = out[out["unit"] == 2]["RUL"].tolist()
    assert engine1 == [1, 0]
    assert engine2 == [2, 1, 0]


# --------------------------------------------------------------------------- #
# data.make_test_sequences — front-padding for short engines
# --------------------------------------------------------------------------- #
def test_make_test_sequences_pads_short_engines():
    df = pd.DataFrame({
        "unit": [1, 1],
        "condition": [0, 0],
        "f1": [10.0, 20.0],
    })
    X_s, X_c = make_test_sequences(df, seq_len=5, feature_cols=["f1"])
    assert X_s.shape == (1, 5, 1)
    # first 3 rows are zero-padding, last 2 are the real (oldest-first) values
    assert np.allclose(X_s[0, :3, 0], 0.0)
    assert np.allclose(X_s[0, 3:, 0], [10.0, 20.0])
    assert X_c[0] == 0


def test_make_test_sequences_takes_last_n_when_long_enough():
    df = pd.DataFrame({
        "unit": [1] * 7,
        "condition": [0] * 7,
        "f1": list(range(7)),
    })
    X_s, _ = make_test_sequences(df, seq_len=3, feature_cols=["f1"])
    assert X_s.shape == (1, 3, 1)
    assert np.allclose(X_s[0, :, 0], [4, 5, 6])


# --------------------------------------------------------------------------- #
# data.make_train_sequences — sliding window
# --------------------------------------------------------------------------- #
def test_make_train_sequences_sliding_window_count():
    df = pd.DataFrame({
        "unit": [1] * 10,
        "condition": [0] * 10,
        "RUL": list(range(10)),
        "f1": list(range(10)),
    })
    X_s, X_c, y = make_train_sequences(df, seq_len=4, feature_cols=["f1"])
    # 10 cycles, window 4 -> 7 windows
    assert X_s.shape == (7, 4, 1)
    assert y.shape == (7,)


# --------------------------------------------------------------------------- #
# losses.asymmetric_huber_loss
# --------------------------------------------------------------------------- #
def test_asymmetric_huber_loss_zero_at_perfect_prediction():
    import tensorflow as tf
    y = tf.constant([10.0, 50.0, 100.0])
    loss = asymmetric_huber_loss(y, y)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_asymmetric_huber_loss_penalizes_late_predictions_more():
    import tensorflow as tf
    y_true = tf.constant([50.0])
    # symmetric error magnitude (5 cycles) in each direction, within the delta=10 quadratic region
    late_pred = tf.constant([55.0])   # prediction too HIGH (late / over-optimistic)
    early_pred = tf.constant([45.0])  # prediction too LOW (early / conservative)
    late_loss = float(asymmetric_huber_loss(y_true, late_pred))
    early_loss = float(asymmetric_huber_loss(y_true, early_pred))
    assert late_loss > early_loss
    assert late_loss == pytest.approx(2.0 * early_loss, rel=1e-4)


# --------------------------------------------------------------------------- #
# inference.health_zone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rul,expected", [
    (0, "Critical"), (30, "Critical"),
    (31, "Warning"), (60, "Warning"),
    (61, "Moderate"), (90, "Moderate"),
    (91, "Healthy"), (150, "Healthy"),
])
def test_health_zone_boundaries(rul, expected):
    assert health_zone(rul, CONFIG) == expected


# --------------------------------------------------------------------------- #
# inference.compute_risk_level
# --------------------------------------------------------------------------- #
def test_risk_level_low_rul_is_always_severe():
    assert compute_risk_level(rul=10, std=1.0, cfg=CONFIG) == "Severe"


def test_risk_level_high_uncertainty_elevates_moderate_rul():
    # RUL=75 alone would be "Medium" (61-90 band), but high uncertainty pushes it to High
    calm = compute_risk_level(rul=75, std=2.0, cfg=CONFIG)
    uncertain = compute_risk_level(rul=75, std=25.0, cfg=CONFIG)
    assert calm == "Medium"
    assert uncertain == "High"


def test_risk_level_confident_healthy_engine_is_low():
    assert compute_risk_level(rul=140, std=3.0, cfg=CONFIG) == "Low"


# --------------------------------------------------------------------------- #
# inference.maintenance_recommendation
# --------------------------------------------------------------------------- #
def test_maintenance_recommendation_healthy_engine():
    rec = maintenance_recommendation(rul=140, std=3.0, cfg=CONFIG)
    assert rec.maintenance_category == "Healthy"
    assert rec.uncertainty_elevated is False


def test_maintenance_recommendation_critical_rul_is_always_critical():
    rec = maintenance_recommendation(rul=5, std=1.0, cfg=CONFIG)
    assert rec.maintenance_category == "Critical"


def test_maintenance_recommendation_bumped_by_uncertainty():
    # RUL=45 alone -> Warning zone -> "Maintenance Required" base category.
    # High uncertainty should bump this up to "Critical".
    rec = maintenance_recommendation(rul=45, std=25.0, cfg=CONFIG)
    assert rec.maintenance_category == "Critical"
    assert rec.uncertainty_elevated is True


def test_maintenance_recommendation_never_exceeds_critical():
    rec = maintenance_recommendation(rul=1, std=100.0, cfg=CONFIG)
    assert rec.maintenance_category == "Critical"
