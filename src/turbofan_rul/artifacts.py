"""
Persist the fitted domain-adaptation artifacts (KMeans + per-condition scalers)
to disk so the backend doesn't refit them from raw training data on every single
startup. This changes nothing about the model or its outputs — `fit_domain_adaptation`
is deterministic (fixed random_state, fit only on train data that doesn't change) —
it just avoids repeating a ~160k-row KMeans fit every time the server boots.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def save_artifacts(kmeans: KMeans, scalers: Dict[int, MinMaxScaler], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"kmeans": kmeans, "scalers": scalers}, path)


def load_artifacts(path: str) -> Tuple[KMeans, Dict[int, MinMaxScaler]]:
    bundle = joblib.load(path)
    return bundle["kmeans"], bundle["scalers"]


def artifacts_exist(path: str) -> bool:
    return os.path.exists(path)
