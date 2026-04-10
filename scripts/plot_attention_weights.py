#!/usr/bin/env python3
"""
Load a Lightning checkpoint (IMUToVisualCrossAttnPoseTransformer), run a chosen KITTI
sequence through the eval-time wrapper, average attention weights over the whole sequence,
and save cross- vs self-attention heatmaps per layer.

Loads RUN/.hydra/config.yaml and the checkpoint in RUN/checkpoints/ that is not last.ckpt
(Lightning keeps last.ckpt plus the best epoch file). Saves the figure under RUN (default
RUN/attention_weights_SEQ.png). Point --run-dir at the Hydra output directory.

  python scripts/plot_attention_weights.py --run-dir logs/train/runs/my_run --sequence 05
"""

from __future__ import annotations

import argparse
import functools
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ["PROJECT_ROOT"] = str(ROOT)

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

torch.serialization.add_safe_globals(
    [
        functools.partial,
        torch.optim.AdamW,
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    ]
)

from src.metrics.weighted_loss import RPMGPoseLoss
from src.metrics.kitti_metrics_calculator import KITTIMetricsCalculator
from src.data.components.latent_kitti_dataset import LatentVectorDataset
from src.models.components.pose_transformer import (
    CrossAttnPoseTransformer,
    IMUToVisualCrossAttnPoseTransformer,
    PoseTransformer,
    VisualContextCrossAttnPoseTransformer,
)
from src.models.weighted_vio_module import WeightedVIOLitModule
from src.testers.kitti_latent_tester import KITTILatentTester

torch.serialization.add_safe_globals(
    [
        RPMGPoseLoss,
        KITTIMetricsCalculator,
        LatentVectorDataset,
        PoseTransformer,
        CrossAttnPoseTransformer,
        IMUToVisualCrossAttnPoseTransformer,
        VisualContextCrossAttnPoseTransformer,
        WeightedVIOLitModule,
        KITTILatentTester,
    ]
)


def _rewrite_paths_for_local_use(cfg: OmegaConf, project_root: Path) -> None:
    """Saved run configs use ${hydra:runtime.*} and ${oc.env:PROJECT_ROOT}; replace paths.* for offline use."""
    os.environ.setdefault("PROJECT_ROOT", str(project_root))
    cfg.paths = OmegaConf.create(
        {
            "root_dir": str(project_root),
            "data_dir": str(project_root / "data"),
            "log_dir": str(project_root / "logs"),
            "output_dir": str(project_root / "logs" / "train" / "runs" / "_plot_attention_weights"),
            "work_dir": str(project_root),
        }
    )


def load_config_from_run_dir(run_dir: Path, project_root: Path) -> OmegaConf:
    hydra_yaml = run_dir / ".hydra" / "config.yaml"
    if not hydra_yaml.is_file():
        raise FileNotFoundError(f"Missing {hydra_yaml}")
    cfg = OmegaConf.load(hydra_yaml)
    _rewrite_paths_for_local_use(cfg, project_root)
    OmegaConf.resolve(cfg)
    return cfg


def find_checkpoint_in_run(run_dir: Path) -> Path:
    """Return the *.ckpt that is not last.ckpt (the best-epoch checkpoint)."""
    for p in (run_dir / "checkpoints").glob("*.ckpt"):
        if p.name != "last.ckpt":
            return p
    raise FileNotFoundError(f"No checkpoint other than last.ckpt in {run_dir / 'checkpoints'}")


def get_net_from_cfg_and_ckpt(cfg: OmegaConf, ckpt_path: Path | str):
    import hydra

    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.net, cfg


def get_sequence_tester(cfg: OmegaConf, sequence: str):
    import hydra

    tester_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.tester, resolve=True))
    tester_cfg.val_seqs = [sequence]
    return hydra.utils.instantiate(tester_cfg)


def maybe_apply_eval_dropout(latents: torch.Tensor, tester_impl) -> torch.Tensor:
    drop_prob = getattr(tester_impl, "_effective_dropout_prob", tester_impl.eval_dropout_prob)
    if random.random() >= drop_prob:
        return latents

    latents = latents.clone()
    v_f_len = tester_impl.args.v_f_len
    if tester_impl.eval_dropout_mode == "visual":
        slc = latents[..., :v_f_len]
    else:
        slc = latents[..., v_f_len:]

    if tester_impl.eval_dropout_style == "zero":
        slc.zero_()
    elif tester_impl.eval_dropout_style == "scale":
        slc.mul_(tester_impl.eval_dropout_scale)
    else:
        noise_std = (
            tester_impl.eval_dropout_noise_std * slc.std().item()
            if slc.numel() > 0
            else tester_impl.eval_dropout_noise_std
        )
        slc.add_(torch.randn_like(slc, device=slc.device, dtype=slc.dtype) * noise_std)
    return latents


def update_attention_sums(attn_sums, attn_dict):
    cross_list = [t[0].detach().float().cpu() for t in attn_dict["cross_attn"]]
    self_list = [t[0].detach().float().cpu() for t in attn_dict["self_attn"]]

    if attn_sums is None:
        return {
            "cross_attn": [t.clone() for t in cross_list],
            "self_attn": [t.clone() for t in self_list],
            "seq_len": cross_list[0].shape[0],
        }

    if cross_list[0].shape[0] != attn_sums["seq_len"]:
        return attn_sums

    for i, t in enumerate(cross_list):
        attn_sums["cross_attn"][i] += t
    for i, t in enumerate(self_list):
        attn_sums["self_attn"][i] += t
    return attn_sums


def plot_attention(
    cross_mats: list[torch.Tensor],
    self_mats: list[torch.Tensor],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    n_layers = len(cross_mats)
    assert len(self_mats) == n_layers
    fig, axes = plt.subplots(n_layers, 2, figsize=(10, 3.5 * n_layers), squeeze=False)

    for i in range(n_layers):
        cw = cross_mats[i].detach().float().cpu().numpy()
        sw = self_mats[i].detach().float().cpu().numpy()

        ax = axes[i][0]
        im0 = ax.imshow(cw, aspect="auto", vmin=0.0, vmax=cw.max() if cw.size else 1.0)
        ax.set_title(f"Layer {i}: cross-attn (IMU queries × visual keys)")
        ax.set_xlabel("key index (visual time)")
        ax.set_ylabel("query index (IMU time)")
        fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[i][1]
        im1 = ax.imshow(sw, aspect="auto", vmin=0.0, vmax=sw.max() if sw.size else 1.0)
        ax.set_title(f"Layer {i}: self-attn (IMU)")
        ax.set_xlabel("key index")
        ax.set_ylabel("query index")
        fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


def run_sequence_average_and_plot(
    net: IMUToVisualCrossAttnPoseTransformer,
    cfg: OmegaConf,
    sequence: str,
    device: torch.device,
    out_path: Path,
) -> None:
    net.return_attention_weights = True
    net.eval()
    net.to(device)

    tester = get_sequence_tester(cfg, sequence)
    tester_impl = tester.kitti_latent_tester
    df = tester_impl.dataloader[0]
    tester_impl.hist = None

    attn_sums = None
    n_attn_calls = 0
    skipped_windows = 0

    for image_seq, imu_seq, gt_seq in tqdm(df, total=len(df), smoothing=0.9, desc=f"Seq {sequence}"):
        x_in = image_seq.unsqueeze(0).to(device)
        i_in = imu_seq.unsqueeze(0).to(device)

        with torch.inference_mode():
            latents = tester_impl.wrapper_model(x_in, i_in)
            latents = maybe_apply_eval_dropout(latents, tester_impl)

            if (tester_impl.hist is not None) and tester_impl.use_history_in_eval:
                for idx in range(latents.shape[1]):
                    tester_impl.hist = torch.roll(tester_impl.hist, -1, 1)
                    tester_impl.hist[:, -1, :] = latents[:, idx, :]
                    result = net((tester_impl.hist, None, None), gt_seq)
                    if not isinstance(result, tuple) or len(result) != 2:
                        print("Expected (poses, attn_dict) from net.", file=sys.stderr)
                        sys.exit(1)
                    _pose, attn = result
                    seq_len_before = None if attn_sums is None else attn_sums["seq_len"]
                    seq_len_now = attn["cross_attn"][0][0].shape[0]
                    if seq_len_before is not None and seq_len_now != seq_len_before:
                        skipped_windows += 1
                        continue
                    attn_sums = update_attention_sums(attn_sums, attn)
                    n_attn_calls += 1
            else:
                tester_impl.hist = latents
                result = net((latents, None, None), gt_seq)
                if not isinstance(result, tuple) or len(result) != 2:
                    print("Expected (poses, attn_dict) from net.", file=sys.stderr)
                    sys.exit(1)
                _pose, attn = result
                seq_len_before = None if attn_sums is None else attn_sums["seq_len"]
                seq_len_now = attn["cross_attn"][0][0].shape[0]
                if seq_len_before is not None and seq_len_now != seq_len_before:
                    skipped_windows += 1
                    continue
                attn_sums = update_attention_sums(attn_sums, attn)
                n_attn_calls += 1

    if attn_sums is None or n_attn_calls == 0:
        print(f"No full-length attention windows collected for sequence {sequence}.", file=sys.stderr)
        sys.exit(1)

    cross_avg = [t / n_attn_calls for t in attn_sums["cross_attn"]]
    self_avg = [t / n_attn_calls for t in attn_sums["self_attn"]]
    seq_len = attn_sums["seq_len"]

    title = (
        f"Attention weights (sequence {sequence}, averaged over {n_attn_calls} calls, "
        f"seq_len={seq_len})"
    )
    if skipped_windows > 0:
        print(
            f"Skipped {skipped_windows} shorter trailing window(s) with mismatched sequence length.",
            file=sys.stderr,
        )
    plot_attention(cross_avg, self_avg, out_path, title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot IMUToVisualCrossAttnPoseTransformer attention maps from a checkpoint."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory with .hydra/config.yaml and checkpoints/",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repo root for resolving data paths (default: parent of scripts/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image (.png/.pdf). Default: RUN_DIR/attention_weights_SEQ.png. Relative paths are under RUN_DIR.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="KITTI sequence to average over, e.g. 05 or 07",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu (default: cuda if available)",
    )
    args = parser.parse_args()

    project_root = (args.project_root or ROOT).resolve()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    run_dir = args.run_dir.expanduser().resolve()
    try:
        cfg = load_config_from_run_dir(run_dir, project_root)
        ckpt_path = find_checkpoint_in_run(run_dir).resolve()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"Using checkpoint: {ckpt_path}")

    if args.out is None:
        out_path = run_dir / f"attention_weights_{args.sequence}.png"
    elif args.out.is_absolute():
        out_path = args.out
    else:
        out_path = run_dir / args.out

    net, cfg = get_net_from_cfg_and_ckpt(cfg, ckpt_path)

    if not isinstance(net, IMUToVisualCrossAttnPoseTransformer):
        print(
            "Error: net must be IMUToVisualCrossAttnPoseTransformer. "
            "The saved config's model.net._target_ must match this architecture.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_sequence_average_and_plot(net, cfg, args.sequence, device, out_path)


if __name__ == "__main__":
    main()
