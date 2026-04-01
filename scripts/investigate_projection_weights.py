#!/usr/bin/env python3
"""
Investigate the linear projection (fc1) weights of the pose transformer:
split by visual (first v_f_len dims) vs IMU (remaining i_f_len), report norms and stats.

Usage (from VIFT directory):
  python scripts/investigate_projection_weights.py --ckpt PATH_TO.ckpt [--output report.txt] [--plot] [--latents path/to/latents.npy]

  --ckpt: path to Lightning checkpoint (required).
  --output: optional path to write a text report (default: print only).
  --plot: if set, save a simple matplotlib figure comparing per-output norms (visual vs IMU).
  --latents: optional path to .npy of latents (shape (..., 768)). If omitted, uses the full eval/test dataloader. Used to report mean(||Wv xv||) vs mean(||Wi xi||).
"""

import argparse
import functools
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # so src/* "from utils..." resolves (utils = src.utils)
os.environ["PROJECT_ROOT"] = str(ROOT)

import torch
# Allow Lightning checkpoint to load with weights_only=True (PyTorch 2.6+)
torch.serialization.add_safe_globals([
    functools.partial,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
])

# Same as eval.py: allow model/criterion/tester etc. from checkpoint
from src.metrics.weighted_loss import RPMGPoseLoss
from src.metrics.kitti_metrics_calculator import KITTIMetricsCalculator
from src.data.components.latent_kitti_dataset import LatentVectorDataset
from src.models.components.pose_transformer import PoseTransformer, IMUToVisualCrossAttnPoseTransformer, VisualContextCrossAttnPoseTransformer
from src.models.weighted_vio_module import WeightedVIOLitModule
from src.testers.kitti_latent_tester import KITTILatentTester
torch.serialization.add_safe_globals([
    RPMGPoseLoss,
    KITTIMetricsCalculator,
    LatentVectorDataset,
    PoseTransformer,
    IMUToVisualCrossAttnPoseTransformer,
    VisualContextCrossAttnPoseTransformer,
    WeightedVIOLitModule,
    KITTILatentTester,
])


def get_net_from_checkpoint(ckpt_path: str):
    """Load Lightning model from checkpoint and return the underlying PoseTransformer (net)."""
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="eval.yaml", overrides=[f"ckpt_path={ckpt_path}"])
    # Instantiate the same model structure as eval
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.net, cfg


def analyze_weights(net, v_f_len: int = 512, i_f_len: int = 256):
    """Extract fc1 weight, split visual/IMU, compute and return stats."""
    linear = net.fc1[0]
    W = linear.weight.detach()
    # (out_features, in_features) = (embedding_dim, input_dim)
    emb_dim, in_dim = W.shape
    assert in_dim == v_f_len + i_f_len, f"Expected input_dim {v_f_len + i_f_len}, got {in_dim}"

    W_visual = W[:, :v_f_len]
    W_imu = W[:, v_f_len:]

    # Per-output norms (how much each embedding dim uses each modality)
    norm_visual_per_out = W_visual.norm(dim=1)
    norm_imu_per_out = W_imu.norm(dim=1)

    # Total norms (raw and normalized by sqrt(n_weights) for fair visual vs IMU comparison)
    total_visual = W_visual.norm().item()
    total_imu = W_imu.norm().item()
    n_visual = emb_dim * v_f_len
    n_imu = emb_dim * i_f_len
    total_visual_norm = total_visual / (n_visual ** 0.5)
    total_imu_norm = total_imu / (n_imu ** 0.5)

    # Per-input norms (which latent dims are used most)
    norm_per_input = W.norm(dim=0)
    norm_visual_per_in = norm_per_input[:v_f_len]
    norm_imu_per_in = norm_per_input[v_f_len:]

    return {
        "W": W,
        "W_visual": W_visual,
        "W_imu": W_imu,
        "emb_dim": emb_dim,
        "in_dim": in_dim,
        "total_visual": total_visual,
        "total_imu": total_imu,
        "total_visual_norm": total_visual_norm,
        "total_imu_norm": total_imu_norm,
        "norm_visual_per_out": norm_visual_per_out,
        "norm_imu_per_out": norm_imu_per_out,
        "norm_visual_per_in": norm_visual_per_in,
        "norm_imu_per_in": norm_imu_per_in,
        "v_f_len": v_f_len,
        "i_f_len": i_f_len,
    }


def get_latents(cfg, latents_npy: Optional[Path], v_f_len: int, i_f_len: int):
    """Return (latents_tensor, None) from .npy file, or (None, test_dataloader) from cfg. (None, None) if unavailable."""
    if latents_npy is not None:
        if not latents_npy.is_file():
            return None, None
        arr = __import__("numpy").load(latents_npy)
        t = torch.from_numpy(arr).float()
        if t.shape[-1] != v_f_len + i_f_len:
            return None, None
        return t, None
    try:
        datamodule = __import__("hydra").utils.instantiate(cfg.data)
        datamodule.setup("test")
        dl = datamodule.test_dataloader()
        return None, dl
    except Exception:
        return None, None


def compute_contribution_stats(latents: torch.Tensor, stats: dict):
    """Compute mean(||Wv xv||) and mean(||Wi xi||) over the given tensor. Merges into stats."""
    latents = latents.cpu().float()
    v_f_len = stats["v_f_len"]
    i_f_len = stats["i_f_len"]
    W_visual = stats["W_visual"]
    W_imu = stats["W_imu"]
    x_visual = latents[..., :v_f_len]
    x_imu = latents[..., v_f_len:]
    contrib_visual = x_visual @ W_visual.T
    contrib_imu = x_imu @ W_imu.T
    n_positions = latents.shape[:-1].numel()
    visual_contrib = torch.norm(contrib_visual, dim=-1).mean().item()
    imu_contrib = torch.norm(contrib_imu, dim=-1).mean().item()
    ratio = visual_contrib / max(imu_contrib, 1e-8)
    stats["contrib_visual_mean"] = visual_contrib
    stats["contrib_imu_mean"] = imu_contrib
    stats["contrib_ratio_visual_imu"] = ratio
    stats["contrib_n_positions"] = n_positions
    stats["contrib_shape"] = tuple(latents.shape)


def compute_contribution_stats_over_dataloader(dataloader, stats: dict):
    """Compute mean(||Wv xv||) and mean(||Wi xi||) over the full dataloader. Merges into stats."""
    v_f_len = stats["v_f_len"]
    i_f_len = stats["i_f_len"]
    W_visual = stats["W_visual"]
    W_imu = stats["W_imu"]
    sum_visual = 0.0
    sum_imu = 0.0
    n_positions = 0
    for batch in dataloader:
        latents = batch[0][0].cpu().float()
        x_visual = latents[..., :v_f_len]
        x_imu = latents[..., v_f_len:]
        contrib_visual = x_visual @ W_visual.T
        contrib_imu = x_imu @ W_imu.T
        norms_visual = torch.norm(contrib_visual, dim=-1)
        norms_imu = torch.norm(contrib_imu, dim=-1)
        n = latents.shape[:-1].numel()
        sum_visual += norms_visual.sum().item()
        sum_imu += norms_imu.sum().item()
        n_positions += n
    if n_positions == 0:
        return
    mean_visual = sum_visual / n_positions
    mean_imu = sum_imu / n_positions
    ratio = mean_visual / max(mean_imu, 1e-8)
    stats["contrib_visual_mean"] = mean_visual
    stats["contrib_imu_mean"] = mean_imu
    stats["contrib_ratio_visual_imu"] = ratio
    stats["contrib_n_positions"] = n_positions
    stats["contrib_shape"] = f"(full test set, {n_positions} positions)"


def print_report(stats: dict, file=None):
    f = file or sys.stdout
    v_f_len = stats["v_f_len"]
    i_f_len = stats["i_f_len"]
    f.write("Linear projection fc1: weight shape (embedding_dim, input_dim) = ")
    f.write(f"({stats['emb_dim']}, {stats['in_dim']})\n")
    f.write(f"Visual slice: cols [0:{v_f_len}], IMU slice: cols [{v_f_len}:{v_f_len + i_f_len}]\n\n")

    f.write("Total Frobenius norms (raw):\n")
    f.write(f"  Visual block: {stats['total_visual']:.6f}\n")
    f.write(f"  IMU block:   {stats['total_imu']:.6f}\n")
    f.write(f"  Ratio (visual/IMU): {stats['total_visual'] / max(stats['total_imu'], 1e-8):.4f}\n\n")
    f.write("Total Frobenius norms (normalized by sqrt(n_weights), comparable across modalities):\n")
    f.write(f"  Visual block: {stats['total_visual_norm']:.6f}\n")
    f.write(f"  IMU block:   {stats['total_imu_norm']:.6f}\n")
    f.write(f"  Ratio (visual/IMU): {stats['total_visual_norm'] / max(stats['total_imu_norm'], 1e-8):.4f}\n\n")

    nvo = stats["norm_visual_per_out"]
    nio = stats["norm_imu_per_out"]
    f.write("Per-output norms (mean ± std):\n")
    f.write(f"  Visual: {nvo.mean().item():.6f} ± {nvo.std().item():.6f}\n")
    f.write(f"  IMU:    {nio.mean().item():.6f} ± {nio.std().item():.6f}\n\n")

    nvi = stats["norm_visual_per_in"]
    nii = stats["norm_imu_per_in"]
    f.write("Per-input norms (which latent dims are used most):\n")
    f.write(f"  Visual: mean {nvi.mean().item():.6f}, max {nvi.max().item():.6f}\n")
    f.write(f"  IMU:    mean {nii.mean().item():.6f}, max {nii.max().item():.6f}\n")

    if "contrib_visual_mean" in stats:
        n_pos = stats.get("contrib_n_positions", 0)
        shape_str = stats.get("contrib_shape", "?")
        f.write("\nContribution norms on data (mean ||Wv xv||, mean ||Wi xi||):\n")
        f.write(f"  Computed over {n_pos} positions ({shape_str})\n")
        f.write(f"  Visual: {stats['contrib_visual_mean']:.6f}\n")
        f.write(f"  IMU:    {stats['contrib_imu_mean']:.6f}\n")
        f.write(f"  Ratio (visual/IMU): {stats['contrib_ratio_visual_imu']:.4f}\n")


def save_plot(stats: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Per-output norms: visual vs IMU
    ax = axes[0]
    ax.bar(range(stats["emb_dim"]), stats["norm_visual_per_out"].numpy(), alpha=0.7, label="Visual")
    ax.bar(range(stats["emb_dim"]), stats["norm_imu_per_out"].numpy(), alpha=0.7, label="IMU")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Weight norm")
    ax.set_title("Per-output norm (visual vs IMU block)")
    ax.legend()

    # Per-input norms: first 512 (visual) and last 256 (IMU)
    ax = axes[1]
    ax.plot(stats["norm_visual_per_in"].numpy(), label="Visual (first 512)")
    ax.plot(
        range(stats["v_f_len"], stats["v_f_len"] + stats["i_f_len"]),
        stats["norm_imu_per_in"].numpy(),
        label="IMU (last 256)",
    )
    ax.set_xlabel("Input dimension index")
    ax.set_ylabel("Weight norm")
    ax.set_title("Per-input norm along latent dimensions")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Investigate pose transformer fc1 projection weights.")
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--output", "-o", default=None, help="Optional path to write text report")
    parser.add_argument("--plot", action="store_true", help="Save a matplotlib figure")
    parser.add_argument("--v-f-len", type=int, default=512, help="Visual latent length (default 512)")
    parser.add_argument("--i-f-len", type=int, default=256, help="IMU latent length (default 256)")
    parser.add_argument("--latents", default=None, help="Optional path to .npy of latents (shape (..., 768)). If not set, use one batch from eval datamodule when available.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading model from checkpoint...")
    net, cfg = get_net_from_checkpoint(str(ckpt_path.resolve()))
    v_f_len = getattr(cfg.model.tester, "v_f_len", None) or args.v_f_len
    i_f_len = getattr(cfg.model.tester, "i_f_len", None) or args.i_f_len

    print("Analyzing fc1 weights (visual vs IMU split)...")
    stats = analyze_weights(net, v_f_len=v_f_len, i_f_len=i_f_len)

    latents_npy = Path(args.latents) if args.latents else None
    latents_tensor, dataloader = get_latents(cfg, latents_npy, v_f_len, i_f_len)
    if latents_tensor is not None:
        print("Computing contribution norms on data (from .npy file)...")
        compute_contribution_stats(latents_tensor, stats)
    elif dataloader is not None:
        print("Computing contribution norms over full eval/test dataloader...")
        compute_contribution_stats_over_dataloader(dataloader, stats)
    else:
        print("Skipping contribution norms (no latents: set --latents path/to/latents.npy or ensure eval data is available).")

    if args.output:
        with open(args.output, "w") as f:
            print_report(stats, file=f)
        print(f"Report written to {args.output}")
    else:
        print_report(stats)

    if args.plot:
        out_path = Path(args.output).with_suffix(".png") if args.output else (ckpt_path.parent / "projection_weights.png")
        save_plot(stats, out_path)


if __name__ == "__main__":
    main()
