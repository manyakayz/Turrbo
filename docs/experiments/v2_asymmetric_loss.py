

# =============================================================================
# Turbofan Engine Remaining Useful Life (RUL) Prediction using LSTM
# Dataset : NASA CMAPSS — ALL 4 subsets combined (FD001 + FD002 + FD003 + FD004)
# Problem : Rolls-Royce Aerospace — Predictive Maintenance
# Task    : Regression — predict how many cycles remain before engine failure
#
# FIXES applied vs v1:
#   - Asymmetric loss (penalizes late predictions more → lowers NASA Score)
#   - Increased Dropout + L2 regularization (fixes overfitting)
#   - Lowered initial LR to 3e-4 (fixes early val loss spikes)
#   - Raised MAX_RUL to 150 (fixes ceiling clustering at 125)
#   - Patience increased to 20 for better convergence
#
# NEW PLOTS:
#   - Figure 1: Training curves (MAE + Loss)
#   - Figure 2: Predicted vs Actual, Residuals, Error vs Actual RUL
#   - Figure 3: Confusion Matrix (health zones), MAE per zone, MAE per dataset
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASETS     = ['FD001', 'FD002', 'FD003', 'FD004']
DATA_PATH    = 'CMaps/'
SEQUENCE_LEN = 30
MAX_RUL      = 150       # FIX: raised from 125 to reduce ceiling clustering
BATCH_SIZE   = 64
EPOCHS       = 150

COLUMNS = (
    ['unit', 'cycle'] +
    [f'os_{i}' for i in range(1, 4)] +
    [f's_{i}'  for i in range(1, 22)]
)
DROP_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

# Health zone bins for confusion matrix
ZONE_BINS   = [0, 30, 60, 90, 150]
ZONE_LABELS = ['Critical\n(0-30)', 'Warning\n(31-60)',
               'Moderate\n(61-90)', 'Healthy\n(91+)']

# =============================================================================
# 1. LOAD RAW DATA
# =============================================================================
print("=" * 60)
print("STEP 1: Loading All 4 Datasets (FD001-FD004)")
print("=" * 60)

def load_txt(path):
    return pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)

train_parts, test_parts, test_rul_parts = [], [], []

for i, ds in enumerate(DATASETS):
    tr = load_txt(f'{DATA_PATH}train_{ds}.txt')
    te = load_txt(f'{DATA_PATH}test_{ds}.txt')
    rl = pd.read_csv(f'{DATA_PATH}RUL_{ds}.txt', header=None, names=['RUL'])

    unit_offset   = i * 10000
    tr['unit']   += unit_offset
    te['unit']   += unit_offset
    tr['dataset'] = ds
    te['dataset'] = ds

    test_last_ds        = te.groupby('unit').last().reset_index()
    test_last_ds['RUL'] = rl['RUL'].values
    test_last_ds['dataset'] = ds

    train_parts.append(tr)
    test_parts.append(te)
    test_rul_parts.append(test_last_ds)

    print(f"  {ds} -> train: {tr.shape[0]:>6} rows, "
          f"test: {te.shape[0]:>6} rows, engines: {tr['unit'].nunique()}")

train_df  = pd.concat(train_parts,    ignore_index=True)
test_df   = pd.concat(test_parts,     ignore_index=True)
test_last = pd.concat(test_rul_parts, ignore_index=True)

print(f"\nCombined train : {train_df.shape[0]:,} rows, "
      f"{train_df['unit'].nunique()} engines")
print(f"Combined test  : {test_last.shape[0]} engines")

# =============================================================================
# 2. COMPUTE RUL LABELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Computing RUL Labels")
print("=" * 60)

def add_rul(df, max_rul=MAX_RUL):
    life = df.groupby('unit')['cycle'].max().reset_index()
    life.columns = ['unit', 'max_cycle']
    df = df.merge(life, on='unit')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=max_rul)
    df.drop(columns=['max_cycle'], inplace=True)
    return df

train_df         = add_rul(train_df)
test_last['RUL'] = test_last['RUL'].clip(upper=MAX_RUL)

print(f"Train RUL - min: {train_df['RUL'].min()}, max: {train_df['RUL'].max()}")
print(f"Test  RUL - min: {test_last['RUL'].min()}, max: {test_last['RUL'].max()}")

# =============================================================================
# 3. FEATURE ENGINEERING & NORMALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering & Normalization")
print("=" * 60)

feature_cols = [c for c in COLUMNS
                if c not in ['unit', 'cycle'] + DROP_SENSORS]
print(f"Using {len(feature_cols)} features: {feature_cols}")

scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

# =============================================================================
# 4. SEQUENCE GENERATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Creating Sequences")
print("=" * 60)

def make_sequences(df, seq_len, feature_cols, rul_col='RUL'):
    X_list, y_list = [], []
    for _, group in df.groupby('unit'):
        data   = group[feature_cols].values
        labels = group[rul_col].values
        for i in range(len(data) - seq_len + 1):
            X_list.append(data[i : i + seq_len])
            y_list.append(labels[i + seq_len - 1])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def make_test_sequences(df, seq_len, feature_cols):
    X_list = []
    for _, group in df.groupby('unit'):
        data = group[feature_cols].values
        if len(data) >= seq_len:
            X_list.append(data[-seq_len:])
        else:
            pad = np.zeros((seq_len - len(data), len(feature_cols)))
            X_list.append(np.vstack([pad, data]))
    return np.array(X_list, dtype=np.float32)

n_units  = train_df['unit'].nunique()
val_ids  = train_df['unit'].unique()[-int(n_units * 0.20):]
val_mask = train_df['unit'].isin(val_ids)

X_train, y_train = make_sequences(train_df[~val_mask], SEQUENCE_LEN, feature_cols)
X_val,   y_val   = make_sequences(train_df[val_mask],  SEQUENCE_LEN, feature_cols)
X_test           = make_test_sequences(test_df, SEQUENCE_LEN, feature_cols)
y_test           = test_last['RUL'].values.astype(np.float32)

print(f"Train sequences : {X_train.shape}")
print(f"Val   sequences : {X_val.shape}")
print(f"Test  sequences : {X_test.shape}")

# =============================================================================
# 5. ASYMMETRIC HUBER LOSS  (FIX: replaces plain Huber)
# =============================================================================
def asymmetric_huber_loss(y_true, y_pred, delta=10.0, late_penalty=2.0):
    """
    Standard Huber loss with asymmetric penalty:
      - Predicting RUL too HIGH (late prediction) -> 2x penalty
      - Predicting RUL too LOW  (early prediction) -> normal penalty
    This directly optimises toward a lower NASA Score.
    """
    error     = y_pred - y_true
    abs_error = K.abs(error)
    huber     = tf.where(abs_error <= delta,
                         0.5 * K.square(error),
                         delta * (abs_error - 0.5 * delta))
    penalty   = tf.where(error > 0, late_penalty * huber, huber)
    return K.mean(penalty)

# =============================================================================
# 6. MODEL — BiLSTM with L2 regularization  (FIX: adds regularizer + dropout)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Building Model")
print("=" * 60)

def build_lstm_model(seq_len, n_features):
    l2 = regularizers.l2(1e-4)
    inputs = Input(shape=(seq_len, n_features), name='sensor_input')

    x = Bidirectional(
        LSTM(128, return_sequences=True,
             kernel_regularizer=l2, recurrent_regularizer=l2),
        name='bilstm1'
    )(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.4, name='drop1')(x)

    x = Bidirectional(
        LSTM(64, return_sequences=False,
             kernel_regularizer=l2, recurrent_regularizer=l2),
        name='bilstm2'
    )(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.4, name='drop2')(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2, name='dense1')(x)
    x = Dropout(0.3, name='drop3')(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2, name='dense2')(x)
    output = Dense(1, activation='relu', name='rul_output')(x)

    return Model(inputs, output, name='BiLSTM_RUL_v2')

model = build_lstm_model(SEQUENCE_LEN, len(feature_cols))
model.summary()

model.compile(
    optimizer=Adam(learning_rate=3e-4),   # FIX: 1e-3 -> 3e-4
    loss=asymmetric_huber_loss,
    metrics=['mae']
)

# =============================================================================
# 7. TRAIN
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Training")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath='best_turbofan_model.keras',
                    monitor='val_mae', save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# 8. EVALUATE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Evaluation on Test Set")
print("=" * 60)

y_pred = model.predict(X_test).flatten()

rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)

def nasa_score(y_true, y_pred):
    diff  = y_pred - y_true
    score = np.where(diff < 0,
                     np.exp(-diff / 13) - 1,
                     np.exp( diff / 10) - 1)
    return np.sum(score)

score = nasa_score(y_test, y_pred)

print(f"RMSE       : {rmse:.4f} cycles")
print(f"MAE        : {mae:.4f} cycles")
print(f"R2 Score   : {r2:.4f}")
print(f"NASA Score : {score:.2f}  (lower is better)")

# Health zones for confusion matrix
def to_zones(rul_arr):
    return pd.cut(
        np.clip(rul_arr, 0, MAX_RUL),
        bins=ZONE_BINS, labels=ZONE_LABELS, right=True
    ).astype(str)

y_test_zones = to_zones(y_test)
y_pred_zones = to_zones(y_pred)

print("\nHealth Zone Classification Report:")
print(classification_report(y_test_zones, y_pred_zones, labels=ZONE_LABELS))

# =============================================================================
# 9. PLOTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Generating Plots")
print("=" * 60)

SUPTITLE = (f'NASA CMAPSS All Datasets  |  '
            f'RMSE={rmse:.2f}  MAE={mae:.2f}  '
            f'R2={r2:.4f}  NASA Score={score:.0f}')

# ── FIGURE 1: Training curves ─────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle('Training History', fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(history.history['mae'],     label='Train MAE', color='royalblue')
ax.plot(history.history['val_mae'], label='Val MAE',   color='darkorange')
ax.set_title('MAE over Epochs')
ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (cycles)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(history.history['loss'],     label='Train Loss', color='royalblue')
ax.plot(history.history['val_loss'], label='Val Loss',   color='darkorange')
ax.set_title('Asymmetric Huber Loss over Epochs')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot1_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot1_training_curves.png")

# ── FIGURE 2: Regression analysis ────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(21, 6))
fig2.suptitle(f'Regression Analysis  |  {SUPTITLE}', fontsize=11, fontweight='bold')

# Predicted vs Actual
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=18)
ax.plot([0, MAX_RUL], [0, MAX_RUL], 'r--', label='Perfect prediction')
ax.set_title('Predicted vs Actual RUL')
ax.set_xlabel('Actual RUL (cycles)')
ax.set_ylabel('Predicted RUL (cycles)')
ax.legend(); ax.grid(True, alpha=0.3)

# Residuals histogram
ax = axes[1]
residuals = y_pred - y_test
ax.hist(residuals, bins=35, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(0,                color='red',   linestyle='--', label='Zero error')
ax.axvline(residuals.mean(), color='green', linestyle='-',
           label=f'Mean = {residuals.mean():.1f}')
ax.set_title('Residual Distribution (Predicted - Actual)')
ax.set_xlabel('Residual (cycles)')
ax.set_ylabel('Count')
ax.legend(); ax.grid(True, alpha=0.3)

# Absolute error vs actual RUL
ax = axes[2]
sc = ax.scatter(y_test, np.abs(residuals), alpha=0.4,
                c=np.abs(residuals), cmap='RdYlGn_r', s=18)
plt.colorbar(sc, ax=ax, label='Absolute Error (cycles)')
ax.set_title('Absolute Error vs Actual RUL')
ax.set_xlabel('Actual RUL (cycles)')
ax.set_ylabel('|Predicted - Actual| (cycles)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot2_regression_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot2_regression_analysis.png")

# ── FIGURE 3: Confusion matrix + zone MAE + dataset MAE ──────────────────────
fig3, axes = plt.subplots(1, 3, figsize=(21, 6))
fig3.suptitle('Health Zone & Dataset Analysis', fontsize=13, fontweight='bold')

# Confusion matrix
ax = axes[0]
cm = confusion_matrix(y_test_zones, y_pred_zones, labels=ZONE_LABELS)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ZONE_LABELS, yticklabels=ZONE_LABELS,
            ax=ax, cbar=False, linewidths=0.5)
ax.set_title('Health Zone Confusion Matrix\n(True vs Predicted)')
ax.set_xlabel('Predicted Zone')
ax.set_ylabel('True Zone')

# MAE per health zone
ax = axes[1]
zone_df  = pd.DataFrame({'actual': y_test, 'pred': y_pred,
                          'zone': y_test_zones})
zone_mae = zone_df.groupby('zone').apply(
    lambda g: mean_absolute_error(g['actual'], g['pred'])
).reindex(ZONE_LABELS)
colors   = ['#d73027', '#fc8d59', '#fee090', '#91cf60']
bars     = ax.bar(ZONE_LABELS, zone_mae.values, color=colors, edgecolor='white',
                  width=0.5)
ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=10)
ax.set_title('MAE per Health Zone')
ax.set_xlabel('Health Zone')
ax.set_ylabel('MAE (cycles)')
ax.grid(True, alpha=0.3, axis='y')

# MAE per dataset
ax = axes[2]
ds_sizes = [len(p) for p in test_rul_parts]
ptr, ds_maes = 0, []
for sz in ds_sizes:
    yt = y_test[ptr:ptr+sz]
    yp = y_pred[ptr:ptr+sz]
    ds_maes.append(mean_absolute_error(yt, yp))
    ptr += sz

bar_colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
bars2 = ax.bar(DATASETS, ds_maes, color=bar_colors, edgecolor='white', width=0.5)
ax.bar_label(bars2, fmt='%.2f', padding=4, fontsize=10)
ax.set_title('MAE per Dataset')
ax.set_xlabel('Dataset')
ax.set_ylabel('MAE (cycles)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plot3_zone_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot3_zone_analysis.png")

# =============================================================================
# 10. SAVE MODEL
# =============================================================================
model.save('turbofan_rul_lstm_v2.keras')
print("\nModel saved as: turbofan_rul_lstm_v2.keras")
print("\nDone!")