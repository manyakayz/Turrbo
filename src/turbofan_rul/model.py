"""
The v4 model architecture: dual-input BiLSTM with a learned operating-condition
embedding. Unchanged from the original main3.py — extracted here only so the
training script doesn't need to redefine it inline.
"""
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Concatenate, Dense, Dropout,
    Embedding, Flatten, Input, LSTM,
)
from tensorflow.keras.models import Model


def build_model(seq_len: int, n_features: int, n_conditions: int) -> Model:
    """Dual-input BiLSTM + domain-condition embedding for RUL regression."""
    l2 = regularizers.l2(3e-4)

    # --- Input A: sensor time series ---
    sensor_input = Input(shape=(seq_len, n_features), name="sensor_input")

    x = Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=l2,
             recurrent_regularizer=l2, recurrent_dropout=0.2),
        name="bilstm1",
    )(sensor_input)
    x = BatchNormalization(name="bn1")(x)
    x = Dropout(0.4, name="drop1")(x)

    x = Bidirectional(
        LSTM(32, return_sequences=False, kernel_regularizer=l2,
             recurrent_regularizer=l2, recurrent_dropout=0.2),
        name="bilstm2",
    )(x)
    x = BatchNormalization(name="bn2")(x)
    x = Dropout(0.4, name="drop2")(x)

    # --- Input B: operating condition ID -> learned embedding ---
    cond_input = Input(shape=(1,), dtype="int32", name="condition_input")
    cond_embed = Embedding(input_dim=n_conditions, output_dim=8,
                            name="condition_embedding")(cond_input)
    cond_flat = Flatten(name="cond_flatten")(cond_embed)

    # --- Merge + regression head ---
    merged = Concatenate(name="merge")([x, cond_flat])
    x = Dense(32, activation="relu", kernel_regularizer=l2, name="dense1")(merged)
    x = Dropout(0.3, name="drop3")(x)
    output = Dense(1, activation="relu", name="rul_output")(x)

    return Model(inputs=[sensor_input, cond_input], outputs=output,
                 name="BiLSTM_RUL_v4_DomainAdapt")
