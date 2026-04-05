import functools
import subprocess
import sys
from pathlib import Path

import torch
torch.set_float32_matmul_precision("high")
torch.serialization.add_safe_globals([
    functools.partial,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
])

from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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
torch.serialization.add_safe_globals([
    RPMGPoseLoss,
    KITTIMetricsCalculator,
    LatentVectorDataset,
    PoseTransformer,
    CrossAttnPoseTransformer,
    IMUToVisualCrossAttnPoseTransformer,
    VisualContextCrossAttnPoseTransformer,
    WeightedVIOLitModule,
    KITTILatentTester,
])
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # Auto-plot trajectories in the same directory where eval results were written
    logger = trainer.logger
    if isinstance(logger, list) and logger:
        logger = logger[0]
    if logger is not None and getattr(logger, "log_dir", None):
        log_dir = Path(logger.log_dir)
        # Poses and .npy files are written to log_dir (works with any logger: tensorboard, csv, etc.)
        # run_root = Hydra output dir, e.g. .../runs/2026-02-27_06-45-00
        # For csv_eval: log_dir = run_root/version_0  => run_root = log_dir.parent
        # For tensorboard: log_dir = run_root/tensorboard/version_0  => run_root = log_dir.parent.parent
        run_root = log_dir.parent.parent if log_dir.parent.name in ("tensorboard", "csv") else log_dir.parent
        if log_dir.is_dir():
            script_path = Path(__file__).resolve().parent.parent / "scripts" / "plot_eval_trajectories.py"
            if script_path.is_file():
                log.info("Plotting trajectories into %s", run_root)
                subprocess.run(
                    [
                        sys.executable, str(script_path), str(run_root),
                        "--poses-dir", str(log_dir),
                        "-o", str(run_root / "trajectories.png"),
                    ],
                    cwd=Path(__file__).resolve().parent.parent,
                    check=False,
                )
            else:
                log.debug("Plot script not found at %s", script_path)
        else:
            log.debug("Log dir not found at %s, skipping trajectory plot", log_dir)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
