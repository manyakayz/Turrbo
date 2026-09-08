# =============================================================================
# Turbofan Engine Remaining Useful Life (RUL) Prediction using LSTM
# Dataset : NASA CMAPSS — ALL 4 subsets combined (FD001 + FD002 + FD003 + FD004)
# Problem : Rolls-Royce Aerospace — Predictive Maintenance
# Task    : Regression — predict how many cycles remain before engine failure
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASETS     = ['FD001', 'FD002', 'FD003', 'FD004']   # all 4 combined
DATA_PATH    = 'data/'            # folder containing the txt files
SEQUENCE_LEN = 30                  # number of past cycles per input window
MAX_RUL      = 125                 # clip RUL ceiling (piece-wise linear target)
BATCH_SIZE   = 64
EPOCHS       = 100

# Column names (26 columns per row)
COLUMNS = (
    ['unit', 'cycle'] +
    [f'os_{i}' for i in range(1, 4)] +
    [f's_{i}'  for i in range(1, 22)]
)

# Sensors to DROP — near-zero variance, carry no degradation signal
DROP_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

# =============================================================================
# 1. LOAD RAW DATA — combine all 4 subsets
# =============================================================================
print("=" * 60)
print("STEP 1: Loading All 4 Datasets (FD001–FD004)")
print("=" * 60)

def load_txt(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    return df

train_parts = []
test_parts  = []
test_rul_parts = []

for i, ds in enumerate(DATASETS):
    tr = load_txt(f'{DATA_PATH}train_{ds}.txt')
    te = load_txt(f'{DATA_PATH}test_{ds}.txt')
    rl = pd.read_csv(f'{DATA_PATH}RUL_{ds}.txt', header=None, names=['RUL'])

    # Make unit IDs globally unique across datasets (offset by dataset index)
    unit_offset = i * 10000
    tr['unit'] += unit_offset
    te['unit'] += unit_offset

    # Tag which dataset each row came from (useful for analysis)
    tr['dataset'] = ds
    te['dataset'] = ds

    # Attach true RUL to last cycle of each test engine
    test_last = te.groupby('unit').last().reset_index()
    test_last['RUL'] = rl['RUL'].values

    train_parts.append(tr)
    test_parts.append(te)
    test_rul_parts.append(test_last)

    print(f"  {ds} → train: {tr.shape[0]:>6} rows, "
          f"test: {te.shape[0]:>6} rows, engines: {tr['unit'].nunique()}")

train_df  = pd.concat(train_parts,  ignore_index=True)
test_df   = pd.concat(test_parts,   ignore_index=True)
test_last = pd.concat(test_rul_parts, ignore_index=True)

print(f"\nCombined train : {train_df.shape[0]:,} rows, "
      f"{train_df['unit'].nunique()} engines")
print(f"Combined test  : {test_df.shape[0]:,} rows, "
      f"{test_last.shape[0]} engines")

# =============================================================================
# 2. COMPUTE RUL LABELS (piece-wise linear — plateau at MAX_RUL)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Computing RUL Labels")
print("=" * 60)

def add_rul(df, max_rul=MAX_RUL):
    # For each engine, max cycle = failure cycle
    life = df.groupby('unit')['cycle'].max().reset_index()
    life.columns = ['unit', 'max_cycle']
    df = df.merge(life, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=max_rul)   # piece-wise linear cap
    df.drop(columns=['max_cycle'], inplace=True)
    return df

train_df = add_rul(train_df)

# test_last already built during loading with true RUL values
test_last['RUL'] = test_last['RUL'].clip(upper=MAX_RUL)

print(f"Train RUL — min: {train_df['RUL'].min()}, max: {train_df['RUL'].max()}")
print(f"Test  RUL — min: {test_last['RUL'].min()}, max: {test_last['RUL'].max()}")

# =============================================================================
# 3. FEATURE ENGINEERING & NORMALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering & Normalization")
print("=" * 60)

# Drop low-variance sensors and metadata columns
feature_cols = [c for c in COLUMNS
                if c not in ['unit', 'cycle'] + DROP_SENSORS]
print(f"Using {len(feature_cols)} features: {feature_cols}")

# Fit scaler on training data only
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
    """
    Slide a window of `seq_len` cycles over each engine's time series.
    Returns X: (N, seq_len, n_features), y: (N,)
    """
    X_list, y_list = [], []
    for unit_id, group in df.groupby('unit'):
        data   = group[feature_cols].values
        labels = group[rul_col].values
        for i in range(len(data) - seq_len + 1):
            X_list.append(data[i : i + seq_len])
            y_list.append(labels[i + seq_len - 1])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def make_test_sequences(df, seq_len, feature_cols):
    """
    For test set: take the LAST seq_len cycles of each engine.
    Returns X: (n_engines, seq_len, n_features)
    """
    X_list = []
    for _, group in df.groupby('unit'):
        data = group[feature_cols].values
        if len(data) >= seq_len:
            X_list.append(data[-seq_len:])
        else:
            # Pad with zeros at the front if engine has fewer cycles
            pad  = np.zeros((seq_len - len(data), len(feature_cols)))
            X_list.append(np.vstack([pad, data]))
    return np.array(X_list, dtype=np.float32)

X_all, y_all = make_sequences(train_df, SEQUENCE_LEN, feature_cols)

# Split last 20% of engines as validation
n_units    = train_df['unit'].nunique()
val_units  = int(n_units * 0.20)
val_ids    = train_df['unit'].unique()[-val_units:]
val_mask   = train_df['unit'].isin(val_ids)

X_val_raw, y_val_raw = make_sequences(
    train_df[val_mask], SEQUENCE_LEN, feature_cols
)
X_train_raw, y_train_raw = make_sequences(
    train_df[~val_mask], SEQUENCE_LEN, feature_cols
)

X_test  = make_test_sequences(test_df, SEQUENCE_LEN, feature_cols)
y_test  = test_last['RUL'].values.astype(np.float32)

print(f"Train sequences : {X_train_raw.shape}, labels: {y_train_raw.shape}")
print(f"Val   sequences : {X_val_raw.shape},   labels: {y_val_raw.shape}")
print(f"Test  sequences : {X_test.shape},       labels: {y_test.shape}")

# =============================================================================
# 5. MODEL — Bidirectional LSTM
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Building Model")
print("=" * 60)

def build_lstm_model(seq_len, n_features):
    """
    Stacked Bidirectional LSTM for RUL regression.
    Output: single neuron (predicted RUL cycles)
    """
    inputs = Input(shape=(seq_len, n_features), name='sensor_input')

    # BiLSTM layer 1 — capture long-range degradation trends
    x = Bidirectional(
        LSTM(128, return_sequences=True), name='bilstm1'
    )(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3, name='drop1')(x)

    # BiLSTM layer 2 — refine temporal patterns
    x = Bidirectional(
        LSTM(64, return_sequences=False), name='bilstm2'
    )(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.3, name='drop2')(x)

    # Dense regression head
    x = Dense(64, activation='relu', name='dense1')(x)
    x = Dropout(0.2, name='drop3')(x)
    x = Dense(32, activation='relu', name='dense2')(x)
    output = Dense(1, activation='relu', name='rul_output')(x)  # ReLU → RUL ≥ 0

    model = Model(inputs, output, name='BiLSTM_RUL_Predictor')
    return model


model = build_lstm_model(SEQUENCE_LEN, len(feature_cols))
model.summary()

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='huber',            # robust to outliers vs plain MSE
    metrics=['mae']
)

# =============================================================================
# 6. TRAIN
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Training")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        filepath='best_turbofan_model.keras',
        monitor='val_mae', save_best_only=True, verbose=1
    )
]

history = model.fit(
    X_train_raw, y_train_raw,
    validation_data=(X_val_raw, y_val_raw),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# 7. EVALUATE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Evaluation on Test Set")
print("=" * 60)

y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

# NASA scoring function — penalizes late predictions more than early ones
def nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    score = np.where(
        diff < 0,
        np.exp(-diff / 13) - 1,
        np.exp( diff / 10) - 1
    )
    return np.sum(score)

score = nasa_score(y_test, y_pred)

print(f"RMSE       : {rmse:.4f} cycles")
print(f"MAE        : {mae:.4f} cycles")
print(f"R² Score   : {r2:.4f}")
print(f"NASA Score : {score:.2f}  (lower is better)")

# =============================================================================
# 8. PLOTS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Generating Plots")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f'Turbofan RUL Prediction — NASA CMAPSS (FD001+FD002+FD003+FD004)\n'
    f'RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  NASA Score={score:.1f}',
    fontsize=13, fontweight='bold'
)

# --- Training curves ---
ax = axes[0, 0]
ax.plot(history.history['mae'],     label='Train MAE', color='royalblue')
ax.plot(history.history['val_mae'], label='Val MAE',   color='orange')
ax.set_title('MAE over Epochs')
ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (cycles)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(history.history['loss'],     label='Train Loss', color='royalblue')
ax.plot(history.history['val_loss'], label='Val Loss',   color='orange')
ax.set_title('Loss (Huber) over Epochs')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# --- Predicted vs Actual ---
ax = axes[1, 0]
ax.scatter(y_test, y_pred, alpha=0.5, color='steelblue', s=20)
lims = [0, MAX_RUL]
ax.plot(lims, lims, 'r--', label='Perfect prediction')
ax.set_title('Predicted vs Actual RUL')
ax.set_xlabel('Actual RUL (cycles)')
ax.set_ylabel('Predicted RUL (cycles)')
ax.legend(); ax.grid(True, alpha=0.3)

# --- Residuals ---
ax = axes[1, 1]
residuals = y_pred - y_test
ax.hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', label='Zero error')
ax.set_title('Residual Distribution (Predicted − Actual)')
ax.set_xlabel('Residual (cycles)')
ax.set_ylabel('Count')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('turbofan_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as turbofan_results.png")

# =============================================================================
# 9. SAVE MODEL
# =============================================================================
model.save('turbofan_rul_lstm.keras')
print("\nModel saved as: turbofan_rul_lstm.keras")
print("\nDone!")