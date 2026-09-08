# Experiment history

These are the earlier iterations of the training script, kept for reference —
they show the actual debugging/improvement process behind the v4 model that's
now in production (`training/train_v4.py`).

## v1 — `v1_baseline.py` (originally `main.py`)

Single-input BiLSTM (128 -> 64 units), plain Huber loss, `MAX_RUL` clipped at
125, global MinMax scaling across all 4 C-MAPSS subsets combined.

Measured on the real held-out test set: **RMSE ≈ 29.4, MAE ≈ 19.6, R² ≈ 0.61**.

## v2 — `v2_asymmetric_loss.py` (originally `main2.py`)

Same single-input architecture, three targeted fixes after diagnosing v1:

- **Asymmetric Huber loss** — a late (over-optimistic) RUL prediction is
  penalized 2x more than an early one, since predicting too late is the
  operationally dangerous failure mode. This targets the NASA scoring function
  directly rather than a generic regression loss.
- **L2 regularization + heavier dropout** — v1 was overfitting.
- **Lower initial learning rate (1e-3 -> 3e-4)** — v1 showed early validation
  loss spikes.
- **`MAX_RUL` raised 125 -> 150** — reduces prediction clustering at the RUL
  ceiling.

v2 was not run to a final saved checkpoint in this history — see v4 below for
the version that superseded it.

## v4 — `training/train_v4.py` (originally `main3.py`) — **current production model**

The two big architectural changes:

1. **Domain adaptation.** FD002/FD004 operate under 6 distinct operating
   conditions; FD001/FD003 effectively run under 1. Global normalization (as in
   v1/v2) blurs sensor baselines across regimes. v4 clusters the 3 operating
   settings with KMeans(k=6), normalizes sensor features *within* each cluster,
   and feeds the cluster ID into the network through a learned embedding layer.
2. **Monte Carlo Dropout** for uncertainty estimation — see `MODELS.md` and
   `src/turbofan_rul/inference.py` for the implementation.

Measured on the real held-out test set (the deployed checkpoint,
see `MODELS.md`): **RMSE ≈ 21.2, MAE ≈ 14.8, R² ≈ 0.80**.

Going from v1 to v4, RMSE dropped ~28% and R² improved from 0.61 to 0.80 —
almost entirely attributable to the domain adaptation step, since v1/v2's
biggest blind spot was treating all 4 operating-condition regimes identically.
