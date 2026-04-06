#!/usr/bin/env python3
"""
Plot training and validation loss curves from the Lightning CSVLogger metrics.csv.

Produces two PNGs:
  - loss_plot.png: combined train/val loss
  - component_loss_plot.png: two panels — (1) Lt vs α·Lr on one scale; (2) raw Lr on its own scale

Usage:
  python scripts/plot_train_losses.py CSV_DIR [-o OUTPUT_DIR] [--alpha 40]

  CSV_DIR: directory containing metrics.csv (e.g. logs/train/runs/.../csv/version_0)
  OUTPUT_DIR: where to save the PNGs (default: CSV_DIR/../../, i.e. the run root)
  --alpha: RPMG angle_weight α; if omitted, read model.criterion.angle_weight from run_root/.hydra/config.yaml
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_angle_weight(run_root: Path, explicit_alpha: float | None) -> float:
    """Return α for L = Lt + α·Lr. Prefer CLI; else parse Hydra config."""
    if explicit_alpha is not None:
        return float(explicit_alpha)
    cfg_path = run_root / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return 40.0
    text = cfg_path.read_text()
    # Prefer YAML parse if available
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        aw = data.get("model", {}).get("criterion", {}).get("angle_weight")
        if aw is not None:
            return float(aw)
    except Exception:
        pass
    m = re.search(r"^\s*angle_weight:\s*([\d.]+)\s*$", text, re.MULTILINE)
    if m:
        return float(m.group(1))
    return 40.0


def main():
    parser = argparse.ArgumentParser(description="Plot train/val losses from CSVLogger metrics.csv")
    parser.add_argument("csv_dir", type=Path, help="Directory containing metrics.csv")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output directory (default: run root)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="angle_weight α in L = Lt + α·Lr (default: read from run .hydra/config.yaml, else 40)",
    )
    args = parser.parse_args()

    metrics_path = args.csv_dir.resolve() / "metrics.csv"
    if not metrics_path.is_file():
        print(f"Error: {metrics_path} not found.")
        return

    train_epochs, train_vals = [], []
    val_epochs, val_vals = [], []
    train_lr_epochs, train_lr_vals = [], []
    train_lt_epochs, train_lt_vals = [], []
    val_lr_epochs, val_lr_vals = [], []
    val_lt_epochs, val_lt_vals = [], []

    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row.get("epoch", 0))

            if row.get("train/loss_epoch"):
                train_epochs.append(epoch)
                train_vals.append(float(row["train/loss_epoch"]))
            if row.get("val/loss"):
                val_epochs.append(epoch)
                val_vals.append(float(row["val/loss"]))
            if row.get("train/Lr_epoch"):
                train_lr_epochs.append(epoch)
                train_lr_vals.append(float(row["train/Lr_epoch"]))
            if row.get("train/Lt_epoch"):
                train_lt_epochs.append(epoch)
                train_lt_vals.append(float(row["train/Lt_epoch"]))
            if row.get("val/Lr"):
                val_lr_epochs.append(epoch)
                val_lr_vals.append(float(row["val/Lr"]))
            if row.get("val/Lt"):
                val_lt_epochs.append(epoch)
                val_lt_vals.append(float(row["val/Lt"]))

    out_dir = args.output_dir.resolve() if args.output_dir else args.csv_dir.resolve().parent.parent
    alpha = load_angle_weight(out_dir, args.alpha)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(train_epochs, train_vals, label="Train Loss", linewidth=1.5)
    ax.plot(val_epochs, val_vals, label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    loss_path = out_dir / "loss_plot.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {loss_path}")

    if train_lr_epochs:
        train_scaled = [alpha * v for v in train_lr_vals]
        val_scaled = [alpha * v for v in val_lr_vals] if val_lr_vals else []

        # Raw Lr is ~α× smaller than α·Lr in "visual weight" but numerically Lr << Lt on one axis
        # squashes the main curves. Use two panels: (1) Lt vs α·Lr (2) raw Lr vs raw Lt.
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(11, 9),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.1, 1]},
        )

        ax_top.plot(train_lt_epochs, train_lt_vals, label="Train Lt", linewidth=1.8, color="C0")
        ax_top.plot(train_lr_epochs, train_scaled, label=f"Train α·Lr (α={alpha:g})", linewidth=1.8, color="C1")
        if val_lt_vals:
            ax_top.plot(val_lt_epochs, val_lt_vals, label="Val Lt", linewidth=1.8, color="C0", linestyle="--")
        if val_lr_vals:
            ax_top.plot(val_lr_epochs, val_scaled, label=f"Val α·Lr", linewidth=1.8, color="C1", linestyle="--")
        ax_top.set_ylabel("Loss (objective terms)")
        ax_top.set_title(
            f"L = Lt + α·Lr (α={alpha:g}) — y-axis clipped so LR spikes do not hide typical values"
        )
        ax_top.legend(loc="best", fontsize=9)
        ax_top.grid(True, alpha=0.3)
        ax_top.margins(y=0.05)
        # Epoch metrics can spike (e.g. cosine restart); autoscale 0…~1.2 makes Lt≈α·Lr≈0.006 look ~0.
        combined_main = np.array(
            list(train_lt_vals) + list(train_scaled) + list(val_lt_vals or []) + list(val_scaled or []),
            dtype=float,
        )
        if combined_main.size:
            # 95th pct drops rare LR-restart spikes (~1+) while keeping early training (Lt≈0.09).
            y_hi = float(np.percentile(combined_main, 95))
            y_hi = max(y_hi * 1.08, 1e-9)
            ax_top.set_ylim(0, y_hi)

        ax_bot.plot(train_lr_epochs, train_lr_vals, label="Train Lr (raw rotation)", linewidth=1.8, color="C2")
        if val_lr_vals:
            ax_bot.plot(val_lr_epochs, val_lr_vals, label="Val Lr (raw)", linewidth=1.8, color="C3", linestyle="--")
        ax_bot.set_xlabel("Epoch")
        ax_bot.set_ylabel("Lr (unweighted)")
        ax_bot.set_title("Raw rotation loss Lr only (own y-scale; not comparable to Lt magnitude)")
        ax_bot.legend(loc="best", fontsize=9)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.margins(y=0.08)

        comp_path = out_dir / "component_loss_plot.png"
        plt.savefig(comp_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved: {comp_path}")


if __name__ == "__main__":
    main()
