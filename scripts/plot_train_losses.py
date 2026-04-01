#!/usr/bin/env python3
"""
Plot training and validation loss curves from the Lightning CSVLogger metrics.csv.

Produces two PNGs:
  - loss_plot.png: combined train/val loss
  - component_loss_plot.png: Lr (rotation) vs Lt (translation) breakdown

Usage:
  python scripts/plot_train_losses.py CSV_DIR [-o OUTPUT_DIR]

  CSV_DIR: directory containing metrics.csv (e.g. logs/train/runs/.../csv/version_0)
  OUTPUT_DIR: where to save the PNGs (default: CSV_DIR/../../, i.e. the run root)
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot train/val losses from CSVLogger metrics.csv")
    parser.add_argument("csv_dir", type=Path, help="Directory containing metrics.csv")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output directory (default: run root)")
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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_epochs, train_vals, label="Train Loss", linewidth=1.5)
    ax.plot(val_epochs, val_vals, label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = out_dir / "loss_plot.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {loss_path}")

    if train_lr_epochs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_lr_epochs, train_lr_vals, label="Train Lr (rotation)", linewidth=1.5)
        ax.plot(train_lt_epochs, train_lt_vals, label="Train Lt (translation)", linewidth=1.5)
        if val_lr_vals:
            ax.plot(val_lr_epochs, val_lr_vals, label="Val Lr", linewidth=1.5)
        if val_lt_vals:
            ax.plot(val_lt_epochs, val_lt_vals, label="Val Lt", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Rotation (Lr) vs Translation (Lt) Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        comp_path = out_dir / "component_loss_plot.png"
        plt.savefig(comp_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {comp_path}")


if __name__ == "__main__":
    main()
