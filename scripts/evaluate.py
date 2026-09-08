"""
Evaluate a trained v4 model against the test set: regression metrics (RMSE, MAE,
R^2, NASA Score), health-zone classification metrics, and MC Dropout uncertainty
calibration. Loads weights only — does not retrain.

Consolidates what used to be two ad-hoc scratch scripts (temp_eval.py,
temp_eval2.py) into one documented entry point.

Usage:
    python scripts/evaluate.py --model best_turbofan_v4.keras --data data/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from turbofan_rul import CONFIG, CUSTOM_OBJECTS  # noqa: E402
from turbofan_rul.data import (  # noqa: E402
    load_all_datasets, add_rul, fit_domain_adaptation, apply_domain_adaptation, make_test_sequences,
)
from turbofan_rul.artifacts import artifacts_exist, load_artifacts  # noqa: E402
from turbofan_rul.inference import mc_predict, health_zone  # noqa: E402

cfg = CONFIG


def nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    return float(np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a .keras model file")
    parser.add_argument("--data", default="data/", help="Path to the C-MAPSS data directory")
    parser.add_argument("--artifacts", default="artifacts/domain_adaptation.joblib",
                         help="Path to persisted KMeans/scaler artifacts (fit fresh if absent)")
    parser.add_argument("--mc-samples", type=int, default=cfg.mc_samples)
    args = parser.parse_args()

    import tensorflow as tf  # deferred: keep --help fast

    print("Loading data...")
    train_df, test_df, test_last = load_all_datasets(args.data, cfg)
    train_df = add_rul(train_df, cfg.max_rul)
    test_last["RUL"] = test_last["RUL"].clip(upper=cfg.max_rul)

    if artifacts_exist(args.artifacts):
        print(f"Loading persisted domain-adaptation artifacts from {args.artifacts}")
        kmeans, scalers = load_artifacts(args.artifacts)
    else:
        print("No persisted artifacts found — fitting from training data...")
        kmeans, scalers = fit_domain_adaptation(train_df, cfg)

    test_df = apply_domain_adaptation(test_df, kmeans, scalers, cfg)
    X_test_s, X_test_c = make_test_sequences(test_df, cfg.sequence_len, cfg.feature_cols)
    y_test = test_last["RUL"].values.astype(np.float32)

    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model, custom_objects=CUSTOM_OBJECTS, compile=False)

    print(f"Running MC Dropout inference ({args.mc_samples} samples)...")
    y_pred_mean, y_pred_std = mc_predict(model, [X_test_s, X_test_c], n_samples=args.mc_samples)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    mae = mean_absolute_error(y_test, y_pred_mean)
    r2 = r2_score(y_test, y_pred_mean)
    score = nasa_score(y_test, y_pred_mean)

    print("\n" + "=" * 60)
    print("REGRESSION METRICS")
    print("=" * 60)
    print(f"RMSE            : {rmse:.4f} cycles")
    print(f"MAE             : {mae:.4f} cycles")
    print(f"R2 Score        : {r2:.4f}")
    print(f"NASA Score      : {score:.2f}  (lower is better)")
    print(f"Mean Uncertainty: {y_pred_std.mean():.2f} +/- {y_pred_std.std():.2f} cycles")

    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION: Is Engine Failing Soon? (RUL <= 30)")
    print("=" * 60)
    y_test_bin = (y_test <= 30).astype(int)
    y_pred_bin = (y_pred_mean <= 30).astype(int)
    print(f"Accuracy : {accuracy_score(y_test_bin, y_pred_bin) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test_bin, y_pred_bin) * 100:.2f}%")
    print(f"Recall   : {recall_score(y_test_bin, y_pred_bin) * 100:.2f}%")
    print(f"F1-Score : {f1_score(y_test_bin, y_pred_bin) * 100:.2f}%")

    print("\n" + "=" * 60)
    print("MULTI-CLASS: Health Zones")
    print("=" * 60)
    zone_labels = ["Critical", "Warning", "Moderate", "Healthy"]
    y_test_zones = pd.Series([health_zone(r, cfg) for r in y_test])
    y_pred_zones = pd.Series([health_zone(r, cfg) for r in y_pred_mean])
    print(classification_report(y_test_zones, y_pred_zones, labels=zone_labels))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test_zones, y_pred_zones, labels=zone_labels),
        index=zone_labels, columns=zone_labels,
    ))


if __name__ == "__main__":
    main()
