# =============================================================================
# Turbofan RUL Prediction — V4 training script
# BiLSTM + MC Dropout Uncertainty + Domain Adaptation + Anti-Overfitting
# NASA CMAPSS FD001+FD002+FD003+FD004
#
# This is a refactor of the original main3.py: identical pipeline, architecture,
# loss, and training procedure. The only *behavioral* change is the callback fix
# described below — everything else was moved into src/turbofan_rul so this
# script and the serving backend share one implementation.
#
# BUGFIX (vs original main3.py):
#   EarlyStopping(monitor='val_mae', min_delta=0.1, restore_best_weights=True) and
#   ModelCheckpoint(monitor='val_mae', save_best_only=True) both watched val_mae,
#   but disagreed on what counts as "an improvement": ModelCheckpoint saves on ANY
#   improvement, while EarlyStopping's min_delta=0.1 required at least a 0.1
#   improvement to update its internal "best" tracker. That mismatch is exactly
#   why, in practice, the checkpoint file ended up with better weights than the
#   final `model.save()` after EarlyStopping restored its (less strict) idea of
#   "best". Removing min_delta makes both callbacks agree on the same best epoch,
#   so the checkpoint and the final saved model are now guaranteed identical.
# =============================================================================
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, classification_report,
)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from turbofan_rul import CONFIG, asymmetric_huber_loss  # noqa: E402
from turbofan_rul.data import (  # noqa: E402
    load_all_datasets, add_rul, fit_domain_adaptation, apply_domain_adaptation,
    make_train_sequences, make_test_sequences,
)
from turbofan_rul.model import build_model  # noqa: E402
from turbofan_rul.inference import mc_predict  # noqa: E402
from turbofan_rul.artifacts import save_artifacts  # noqa: E402
from plots import plot_training_curves, plot_regression_analysis, plot_uncertainty  # noqa: E402

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = os.environ.get("TURBOFAN_DATA_PATH", "data/")
OUTPUT_DIR = os.environ.get("TURBOFAN_OUTPUT_DIR", ".")
cfg = CONFIG

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("=" * 60)
print("STEP 1: Loading All 4 Datasets")
print("=" * 60)

train_df, test_df, test_last = load_all_datasets(DATA_PATH, cfg)
print(f"Combined train : {train_df.shape[0]:,} rows, {train_df['unit'].nunique()} engines")
print(f"Combined test  : {test_last.shape[0]} engines")

# =============================================================================
# STEP 2: RUL LABELS
# =============================================================================
train_df = add_rul(train_df, cfg.max_rul)
test_last["RUL"] = test_last["RUL"].clip(upper=cfg.max_rul)

# =============================================================================
# STEP 3: DOMAIN ADAPTATION — Operating Condition Clustering
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Domain Adaptation — Operating Condition Clustering")
print("=" * 60)

kmeans, scalers = fit_domain_adaptation(train_df, cfg)
train_df = apply_domain_adaptation(train_df, kmeans, scalers, cfg)
test_df = apply_domain_adaptation(test_df, kmeans, scalers, cfg)
test_last["condition"] = test_df.groupby("unit")["condition"].last().values

# Persist for the backend so it doesn't need to refit on every startup
artifacts_path = os.path.join(OUTPUT_DIR, "artifacts", "domain_adaptation.joblib")
save_artifacts(kmeans, scalers, artifacts_path)
print(f"Saved domain-adaptation artifacts to {artifacts_path}")

feature_cols = cfg.feature_cols
print(f"Using {len(feature_cols)} sensor features + 1 condition ID (embedded)")

# =============================================================================
# STEP 4: SEQUENCE GENERATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Creating Sequences")
print("=" * 60)

n_units = train_df["unit"].nunique()
val_ids = train_df["unit"].unique()[-int(n_units * cfg.val_fraction):]
val_mask = train_df["unit"].isin(val_ids)

X_train_s, X_train_c, y_train = make_train_sequences(train_df[~val_mask], cfg.sequence_len, feature_cols)
X_val_s, X_val_c, y_val = make_train_sequences(train_df[val_mask], cfg.sequence_len, feature_cols)
X_test_s, X_test_c = make_test_sequences(test_df, cfg.sequence_len, feature_cols)
y_test = test_last["RUL"].values.astype(np.float32)

print(f"Train : {X_train_s.shape}, cond: {X_train_c.shape}")
print(f"Val   : {X_val_s.shape},   cond: {X_val_c.shape}")
print(f"Test  : {X_test_s.shape},  cond: {X_test_c.shape}")

# =============================================================================
# STEP 5: BUILD MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Building Model")
print("=" * 60)

model = build_model(cfg.sequence_len, len(feature_cols), cfg.n_conditions)
model.summary()

model.compile(optimizer=Adam(learning_rate=3e-4), loss=asymmetric_huber_loss, metrics=["mae"])

# =============================================================================
# STEP 6: COSINE LR + TRAIN
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Training")
print("=" * 60)


def cosine_lr(epoch, initial_lr=3e-4, min_lr=1e-6, total_epochs=cfg.epochs):
    decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    return float(min_lr + (initial_lr - min_lr) * decay)


checkpoint_path = os.path.join(OUTPUT_DIR, "best_turbofan_v4.keras")
final_path = os.path.join(OUTPUT_DIR, "turbofan_rul_v4.keras")

callbacks = [
    # NOTE: min_delta intentionally removed (see module docstring) so this
    # agrees with ModelCheckpoint on exactly which epoch is "best".
    EarlyStopping(monitor="val_mae", patience=15, restore_best_weights=True, verbose=1),
    LearningRateScheduler(cosine_lr, verbose=0),
    ModelCheckpoint(filepath=checkpoint_path, monitor="val_mae", save_best_only=True, verbose=1),
]

history = model.fit(
    [X_train_s, X_train_c], y_train,
    validation_data=([X_val_s, X_val_c], y_val),
    epochs=cfg.epochs,
    batch_size=cfg.batch_size,
    callbacks=callbacks,
    verbose=1,
)

# =============================================================================
# STEP 7: MONTE CARLO DROPOUT — Uncertainty Estimation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: MC Dropout Uncertainty Estimation")
print("=" * 60)

y_pred_mean, y_pred_std = mc_predict(model, [X_test_s, X_test_c], n_samples=cfg.mc_samples)

flagged_idx = np.where(y_pred_std > cfg.uncertainty_threshold)[0]
print(f"\nEngines flagged (std > {cfg.uncertainty_threshold} cycles): {len(flagged_idx)} / {len(y_test)}")

# =============================================================================
# STEP 8: EVALUATE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Evaluation")
print("=" * 60)

y_pred = y_pred_mean

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))


def nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    return float(np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)))


score = nasa_score(y_test, y_pred)

print(f"RMSE           : {rmse:.4f} cycles")
print(f"MAE            : {mae:.4f} cycles")
print(f"R2 Score       : {r2:.4f}")
print(f"NASA Score     : {score:.2f}  (lower is better)")
print(f"Mean Uncertainty: {y_pred_std.mean():.2f} +/- {y_pred_std.std():.2f} cycles")

# =============================================================================
# STEP 9: SAVE
#
# Because the callback fix above makes EarlyStopping and ModelCheckpoint agree,
# checkpoint_path and final_path will now contain identical weights (unlike the
# original run, where they differed). Both are still written for compatibility
# with existing tooling/scripts that expect either filename.
# =============================================================================
model.save(final_path)
print(f"\nModel saved as: {final_path}")
print(f"Checkpoint saved as: {checkpoint_path}")

# =============================================================================
# STEP 10: PLOTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: Generating Plots")
print("=" * 60)

suptitle = f"V4 | RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}  NASA Score={score:.0f}"
plot_training_curves(history, OUTPUT_DIR)
plot_regression_analysis(y_test, y_pred, cfg.max_rul, suptitle, OUTPUT_DIR)
plot_uncertainty(y_test, y_pred_mean, y_pred_std, test_last, cfg.datasets,
                  cfg.uncertainty_threshold, cfg.mc_samples, OUTPUT_DIR)
print("Plots saved.")
print("Done!")
