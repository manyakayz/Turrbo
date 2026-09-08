"""
Data pipeline for the Turbofan RUL project.

This is a direct extraction of the pipeline from the v4 training script
(training/train_v4.py) — behavior is unchanged, it's just no longer copy-pasted
into three other files. Both training and the FastAPI backend import from here,
so preprocessing can never silently drift between train and serve time.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .config import PipelineConfig


def load_txt(path: str, columns: List[str]) -> pd.DataFrame:
    """Load a whitespace-delimited C-MAPSS .txt file."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns)


def load_all_datasets(data_path: str, cfg: PipelineConfig):
    """
    Load and concatenate all C-MAPSS subsets (FD001-FD004), with globally unique
    unit IDs (offset by 10000 * dataset index) so subsets can be merged safely.

    Returns (train_df, test_df, test_last_df) where test_last_df has one row per
    test engine (its final observed cycle) with the ground-truth RUL attached.
    """
    train_parts, test_parts, test_rul_parts = [], [], []

    for i, ds in enumerate(cfg.datasets):
        tr = load_txt(os.path.join(data_path, f"train_{ds}.txt"), cfg.columns)
        te = load_txt(os.path.join(data_path, f"test_{ds}.txt"), cfg.columns)
        rl = pd.read_csv(os.path.join(data_path, f"RUL_{ds}.txt"), header=None, names=["RUL"])

        unit_offset = i * 10000
        tr["unit"] += unit_offset
        te["unit"] += unit_offset
        tr["dataset"] = ds
        te["dataset"] = ds

        test_last_ds = te.groupby("unit").last().reset_index()
        test_last_ds["RUL"] = rl["RUL"].values
        test_last_ds["dataset"] = ds

        train_parts.append(tr)
        test_parts.append(te)
        test_rul_parts.append(test_last_ds)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    test_last = pd.concat(test_rul_parts, ignore_index=True)
    return train_df, test_df, test_last


def add_rul(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """Piecewise-linear RUL label: cycles-to-failure, clipped at max_rul."""
    life = df.groupby("unit")["cycle"].max().reset_index()
    life.columns = ["unit", "max_cycle"]
    df = df.merge(life, on="unit")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=max_rul)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def fit_domain_adaptation(train_df: pd.DataFrame, cfg: PipelineConfig
                           ) -> Tuple[KMeans, Dict[int, MinMaxScaler]]:
    """
    Fit the KMeans operating-condition clusterer and one MinMaxScaler per condition,
    on training data only. This is the "domain adaptation" step: FD002/FD004 engines
    fly under 6 distinct operating regimes, and sensor readings need to be normalized
    within each regime rather than globally.
    """
    kmeans = KMeans(n_clusters=cfg.n_conditions, random_state=42, n_init=20)
    kmeans.fit(train_df[cfg.os_cols])

    train_df = train_df.copy()
    train_df["condition"] = kmeans.predict(train_df[cfg.os_cols])

    scalers: Dict[int, MinMaxScaler] = {}
    feature_cols = cfg.feature_cols
    for cond in range(cfg.n_conditions):
        mask = train_df["condition"] == cond
        if mask.sum() > 0:
            sc = MinMaxScaler()
            sc.fit(train_df.loc[mask, feature_cols])
            scalers[cond] = sc

    return kmeans, scalers


def apply_domain_adaptation(df: pd.DataFrame, kmeans: KMeans,
                             scalers: Dict[int, MinMaxScaler], cfg: PipelineConfig
                             ) -> pd.DataFrame:
    """Assign operating-condition cluster and apply the matching per-condition scaler."""
    df = df.copy()
    df["condition"] = kmeans.predict(df[cfg.os_cols])

    feature_cols = cfg.feature_cols
    df[feature_cols] = df[feature_cols].astype(float)
    for cond, sc in scalers.items():
        mask = df["condition"] == cond
        if mask.sum() > 0:
            df.loc[mask, feature_cols] = sc.transform(df.loc[mask, feature_cols])
    return df


def make_train_sequences(df: pd.DataFrame, seq_len: int, feature_cols: List[str]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window sequences for training. Returns (X_sensor, X_condition, y)."""
    X_s, X_c, y_list = [], [], []
    for _, group in df.groupby("unit"):
        data = group[feature_cols].values
        conds = group["condition"].values
        labels = group["RUL"].values
        for i in range(len(data) - seq_len + 1):
            X_s.append(data[i: i + seq_len])
            X_c.append(conds[i + seq_len - 1])
            y_list.append(labels[i + seq_len - 1])
    return (
        np.array(X_s, dtype=np.float32),
        np.array(X_c, dtype=np.int32),
        np.array(y_list, dtype=np.float32),
    )


def make_test_sequences(df: pd.DataFrame, seq_len: int, feature_cols: List[str]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Last-`seq_len`-cycles window per engine (front-padded with zeros if an engine
    has fewer than seq_len observed cycles). Returns (X_sensor, X_condition).
    """
    X_s, X_c = [], []
    for _, group in df.groupby("unit"):
        data = group[feature_cols].values
        conds = group["condition"].values
        if len(data) >= seq_len:
            X_s.append(data[-seq_len:])
        else:
            pad = np.zeros((seq_len - len(data), len(feature_cols)))
            X_s.append(np.vstack([pad, data]))
        X_c.append(int(conds[-1]))
    return np.array(X_s, dtype=np.float32), np.array(X_c, dtype=np.int32)
