# Model provenance

This project went through 3 training iterations (see `docs/experiments/README.md`
for the full v1 -> v2 -> v4 story). This document is specifically about the v4
model files, since two different saves of the "same" model exist and they are
**not identical**.

## Where the weights live

`.keras` model files and the `data/` directory are intentionally **gitignored**
(see `.gitignore`) — they're large binaries that don't belong in git history.
To run this project you need to supply:

- `data/` — the NASA C-MAPSS dataset (`train_FD00{1..4}.txt`, `test_FD00{1..4}.txt`,
  `RUL_FD00{1..4}.txt`)
- `turbofan_rul_v4.keras` at the repo root — the trained v4 model (see below for
  how to get one)

Run `training/train_v4.py` to produce both from scratch, or place existing
`.keras` files as described below.

## The checkpoint vs. "final" discrepancy

`training/train_v4.py` writes two files during training:

- `best_turbofan_v4.keras` — written by `ModelCheckpoint(save_best_only=True)`,
  updated every time `val_mae` improves at all
- `turbofan_rul_v4.keras` — written by `model.save()` after training finishes,
  using the weights `EarlyStopping(restore_best_weights=True)` decided were best

In the original training run behind this project, **these two files ended up
with different weights**, and the checkpoint measurably outperformed the
"final" save on every metric:

| | RMSE | MAE | R² | NASA Score |
|---|---|---|---|---|
| `best_turbofan_v4.keras` (checkpoint) | ~21.2 | ~14.8 | ~0.80 | ~17k–30k |
| `turbofan_rul_v4.keras` (final, original) | ~21.8 | ~14.9 | ~0.78 | ~26k–38k |

**Root cause:** both callbacks watch `val_mae`, but they don't agree on what
counts as "an improvement." `ModelCheckpoint(save_best_only=True)` saves on
*any* improvement, however small. `EarlyStopping(..., min_delta=0.1)` only
updates its internal "best" tracker — the one `restore_best_weights` restores
from — if `val_mae` improves by at least 0.1. If validation MAE wobbled by
less than 0.1 after the checkpoint's best epoch, `ModelCheckpoint` would still
save that later (slightly better) epoch, while `EarlyStopping` wouldn't count
it and would restore an earlier, marginally worse epoch instead.

**Fix applied:** `training/train_v4.py` no longer sets `min_delta` on
`EarlyStopping`, so both callbacks now agree on the same "best" epoch. Future
training runs will produce identical `best_*.keras` and final `.keras` files.
This is a training-script correctness fix — it does not require retraining
the currently-deployed model and does not change any existing predictions.

## What's deployed right now

**`turbofan_rul_v4.keras` in this repo is the former `best_turbofan_v4.keras`
checkpoint** — the better-performing file — placed at the filename the backend
expects. If you train a fresh model with the fixed callback, `best_*.keras`
and the final save will match, and this distinction goes away.

## Verification

Both v4 candidate files were confirmed to:
- Load successfully with `custom_objects={'asymmetric_huber_loss': ...}`
- Accept the dual-input `[sensor_input, condition_input]` call signature the
  backend uses
- Produce numerically different (not just noisier) predictions — max absolute
  weight difference between the two files was ~0.11, confirmed via direct
  weight-array comparison, not just output variance

Two other candidate files (`best_turbofan_model.keras`, `turbofan_rul_lstm.keras`)
were also inspected — these are **v1 architecture** (single input, 17 features)
and are **not compatible** with the current backend, which expects the v4
dual-input, 14-feature architecture. Attempting to load a v1 model into the
current backend raises:

```
ValueError: Layer "BiLSTM_RUL_Predictor" expects 1 named input(s) with keys
['sensor_input'], but it received 2 input tensors.
```

They're kept in history for reference (see `docs/experiments/`) but aren't
usable as drop-in replacements without reverting the backend to the v1
single-input, globally-scaled feature pipeline.
