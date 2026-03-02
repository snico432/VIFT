#!/usr/bin/env python3
"""
Plot trajectory and Y-over-time figures from eval output (paper-style 2x3 grid).

Usage (from VIFT directory):
  python scripts/plot_eval_trajectories.py RUN_DIR [--output path.png] [--method-name "label"]

  RUN_DIR: eval run directory (e.g. logs/eval/runs/2026-02-27_06-28-01). Poses are read from
           RUN_DIR/tensorboard/version_0/ by default, or from --poses-dir if set.
  --poses-dir: directory containing *_gt_poses.npy and *_estimated_poses.npy (default: RUN_DIR/tensorboard/version_0).
  --output: output filename or path (default: trajectories.png in RUN_DIR). If relative, under RUN_DIR.
  --method-name: legend label for the estimated trajectory. If not set, inferred from RUN_DIR/.hydra/config.yaml
                 (eval_dropout_mode, eval_dropout_prob, eval_dropout_style).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

# Add project root for src imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.kitti_utils import path_accu

SEQUENCES = ["05", "07", "10"]
FRAME_RATE_HZ = 10  # KITTI is 10 Hz


def load_global_poses(save_dir: Path, seq: str):
    """Load relative poses from .npy and convert to global (list of 4x4)."""
    gt_rel = np.load(save_dir / f"{seq}_gt_poses.npy")
    est_rel = np.load(save_dir / f"{seq}_estimated_poses.npy")
    gt_global = path_accu(gt_rel)
    est_global = path_accu(est_rel)
    return gt_global, est_global


def extract_xyz(pose_list):
    """From list of 4x4 matrices, extract (x, y, z) as arrays."""
    x = np.array([p[0, 3] for p in pose_list])
    y = np.array([p[1, 3] for p in pose_list])
    z = np.array([p[2, 3] for p in pose_list])
    return x, y, z


def method_name_from_config(run_dir: Path) -> str | None:
    """Build legend label from run_dir/.hydra/config.yaml (eval_dropout_*). Returns None if not found."""
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        return None
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None
    model_cfg = (cfg or {}).get("model") or {}
    tester = model_cfg.get("tester") if isinstance(model_cfg, dict) else {}
    if not isinstance(tester, dict):
        tester = {}
    mode = tester.get("eval_dropout_mode")
    if mode is None or (isinstance(mode, str) and mode.lower() in ("null", "none", "")):
        return "VIFT"
    else:
        if mode == "visual":
            mode = "Visual"
        elif mode == "imu":
            mode = "IMU"
        else:
            mode = "VIFT"
    prob = tester.get("eval_dropout_prob")
    prob_str = f"{prob:g}" if prob is not None else "?"
    style = (tester.get("eval_dropout_style") or "zero")
    style_lower = str(style).lower()
    parts = [f"{mode} dropout {prob_str}"]
    if style_lower not in ("zero", "null", "none", ""):
        if style_lower == "scale":
            scale_val = tester.get("eval_dropout_scale")
            parts.append(f"scale {scale_val:g}" if scale_val is not None else "scale")
        elif style_lower == "noise":
            noise_std = tester.get("eval_dropout_noise_std")
            parts.append(f"noise \u03c3={noise_std:g}" if noise_std is not None else "noise")
        else:
            parts.append(str(style))
    return "VIFT (" + ", ".join(parts) + ")"


def main():
    parser = argparse.ArgumentParser(description="Plot eval trajectories (paper-style).")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Eval run directory (e.g. logs/eval/runs/2026-02-27_06-28-01). Poses from run_dir/tensorboard/version_0/ unless --poses-dir is set.",
    )
    parser.add_argument(
        "--poses-dir", "-p",
        type=Path,
        default=None,
        help="Directory with *_gt_poses.npy and *_estimated_poses.npy (default: run_dir/tensorboard/version_0).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output filename or path (default: run_dir/trajectories.png). If relative, under run_dir.",
    )
    parser.add_argument(
        "--method-name", "-m",
        type=str,
        default=None,
        help="Legend label for the estimated trajectory. If not set, inferred from run_dir/.hydra/config.yaml",
    )
    args = parser.parse_args()
    raw = args.run_dir or ROOT / "logs/eval/runs/2026-02-27_04-14-01"
    run_dir = Path(raw).resolve()
    if not run_dir.is_dir():
        # Fallbacks: project-relative path with leading slash (e.g. /logs/eval/runs/XXX), or relative to ROOT/ROOT.parent
        candidates = []
        if raw.is_absolute() and len(raw.parts) > 1 and raw.parts[1] == "logs":
            candidates.append((ROOT / Path(*raw.parts[1:])).resolve())
        candidates.extend((base / raw).resolve() for base in (ROOT.parent, ROOT))
        for candidate in candidates:
            if candidate.is_dir():
                run_dir = candidate
                break
        else:
            print(f"Error: {run_dir} is not a directory.", file=sys.stderr)
            sys.exit(1)

    if args.poses_dir is not None:
        poses_dir = Path(args.poses_dir).resolve()
    else:
        # Prefer run_dir/version_0 (csv_eval), then run_dir/tensorboard/version_0 (legacy)
        for cand in (run_dir / "version_0", run_dir / "tensorboard" / "version_0"):
            if cand.is_dir() and (cand / "05_gt_poses.npy").is_file():
                poses_dir = cand
                break
        else:
            poses_dir = run_dir / "tensorboard" / "version_0"
    if not poses_dir.is_dir():
        print(f"Error: {poses_dir} not found (expected *_gt_poses.npy and *_estimated_poses.npy here).", file=sys.stderr)
        sys.exit(1)

    if args.output is not None:
        out_path = (run_dir / args.output) if not Path(args.output).is_absolute() else Path(args.output).resolve()
    else:
        out_path = run_dir / "trajectories.png"
    method_label = args.method_name or method_name_from_config(run_dir) or "VIFT"

    # Load data for all sequences
    data = {}
    for seq in SEQUENCES:
        gt_global, est_global = load_global_poses(poses_dir, seq)
        data[seq] = {
            "gt": extract_xyz(gt_global),
            "est": extract_xyz(est_global),
        }

    # Paper-style 2x3: top = X-Z path (same convention as kitti_eval.plotPath_2D), bottom = Y vs time
    # Match kitti_eval.plotPath_2D: x/z from pose[0,3], pose[2,3]; GT red, est blue; labels "Ground Truth" / "Ours"
    fontsize_ = 10
    style_gt = "r-"
    style_est = "b-"

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for col, seq in enumerate(SEQUENCES):
        x_gt, y_gt, z_gt = data[seq]["gt"]
        x_est, y_est, z_est = data[seq]["est"]
        n = len(x_gt)
        t = np.arange(n) / FRAME_RATE_HZ  # time in seconds

        # Top row: X-Z trajectory (same as plotPath_2D: x vs z, same colors/labels)
        ax_path = axes[0, col]
        ax_path.plot(x_gt, z_gt, style_gt, label="Ground Truth", linewidth=1.2)
        ax_path.plot(x_est, z_est, style_est, label=method_label, linewidth=1.2)
        ax_path.plot(0, 0, "ko", label="Start Point", markersize=4)
        ax_path.set_xlabel("x (m)", fontsize=fontsize_)
        ax_path.set_ylabel("z (m)", fontsize=fontsize_)
        ax_path.set_title(f"Sequence {seq}")
        ax_path.legend(loc="upper right", fontsize=8)
        ax_path.set_aspect("equal")
        ax_path.grid(True, alpha=0.3)

        # Bottom row: Y vs time
        ax_y = axes[1, col]
        ax_y.plot(t, y_gt, style_gt, label="Ground Truth", linewidth=1.2)
        ax_y.plot(t, y_est, style_est, label=method_label, linewidth=1.2)
        ax_y.set_xlabel("Time (s)", fontsize=fontsize_)
        ax_y.set_ylabel("y (m)", fontsize=fontsize_)
        ax_y.set_title(f"Sequence {seq}")
        ax_y.legend(loc="upper right", fontsize=8)
        ax_y.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
