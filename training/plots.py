"""Plotting utilities for the v4 training run — extracted from the original
inline plotting code in main3.py so train_v4.py isn't 600 lines long."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error


def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(history.history["mae"], label="Train MAE", color="royalblue")
    ax.plot(history.history["val_mae"], label="Val MAE", color="darkorange")
    ax.set_title("MAE over Epochs")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (cycles)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history.history["loss"], label="Train Loss", color="royalblue")
    ax.plot(history.history["val_loss"], label="Val Loss", color="darkorange")
    ax.set_title("Asymmetric Huber Loss over Epochs")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    gap = np.array(history.history["val_mae"]) - np.array(history.history["mae"])
    ax.plot(gap, color="crimson", label="Val - Train MAE")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.fill_between(range(len(gap)), 0, gap, where=(gap > 0), alpha=0.3, color="crimson", label="Overfitting")
    ax.fill_between(range(len(gap)), 0, gap, where=(gap < 0), alpha=0.3, color="green", label="Underfitting")
    ax.set_title("Generalisation Gap\n(ideal: near 0)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap (cycles)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot1_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regression_analysis(y_test, y_pred, max_rul, suptitle, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"Regression Analysis  |  {suptitle}", fontsize=10, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=18)
    ax.plot([0, max_rul], [0, max_rul], "r--", label="Perfect prediction")
    ax.set_title("Predicted vs Actual RUL")
    ax.set_xlabel("Actual RUL (cycles)"); ax.set_ylabel("Predicted RUL (cycles)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    residuals = y_pred - y_test
    ax.hist(residuals, bins=35, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", label="Zero error")
    ax.axvline(residuals.mean(), color="green", linestyle="-", label=f"Mean={residuals.mean():.1f}")
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual (cycles)"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    sc = ax.scatter(y_test, np.abs(residuals), alpha=0.4, c=np.abs(residuals), cmap="RdYlGn_r", s=18)
    plt.colorbar(sc, ax=ax, label="Absolute Error (cycles)")
    ax.set_title("Absolute Error vs Actual RUL")
    ax.set_xlabel("Actual RUL (cycles)"); ax.set_ylabel("|Predicted - Actual| (cycles)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot2_regression_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(y_test, y_pred_mean, y_pred_std, test_last, datasets, uncertainty_threshold, mc_samples, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"MC Dropout Uncertainty ({mc_samples} samples)", fontsize=13, fontweight="bold")

    ax = axes[0]
    sorted_idx = np.argsort(y_test)
    x_range = range(len(y_test))
    ax.plot(y_test[sorted_idx], color="black", linewidth=1.2, label="Actual RUL")
    ax.plot(y_pred_mean[sorted_idx], color="steelblue", linewidth=1, label="Predicted RUL (MC mean)")
    ax.fill_between(x_range, (y_pred_mean - 2 * y_pred_std)[sorted_idx],
                     (y_pred_mean + 2 * y_pred_std)[sorted_idx], alpha=0.3, color="steelblue", label="+-2sigma uncertainty")
    ax.set_title("Predictions with Uncertainty Bands\n(engines sorted by actual RUL)")
    ax.set_xlabel("Engine index (sorted)"); ax.set_ylabel("RUL (cycles)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    abs_err = np.abs(y_pred_mean - y_test)
    ax.scatter(y_pred_std, abs_err, alpha=0.4, color="steelblue", s=18)
    ax.axvline(uncertainty_threshold, color="red", linestyle="--", label=f"Flag threshold ({uncertainty_threshold} cycles)")
    ax.set_title("Uncertainty vs Absolute Error\n(ideal: higher sigma -> higher error)")
    ax.set_xlabel("Prediction Std Dev (sigma, cycles)"); ax.set_ylabel("Absolute Error (cycles)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ptr = 0
    ds_sizes = [(test_last["dataset"] == ds).sum() for ds in datasets]
    for ds, sz in zip(datasets, ds_sizes):
        ax.hist(y_pred_std[ptr:ptr + sz], bins=20, alpha=0.6, label=ds, density=True)
        ptr += sz
    ax.axvline(uncertainty_threshold, color="red", linestyle="--", label=f"Flag threshold ({uncertainty_threshold})")
    ax.set_title("Uncertainty Distribution by Dataset")
    ax.set_xlabel("Prediction Std Dev (cycles)"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot4_uncertainty.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
