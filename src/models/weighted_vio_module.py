import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from pathlib import Path
import subprocess
import sys
import hydra
import json

class WeightedVIOLitModule(LightningModule):
    def __init__(
            self,
            net,
            optimizer,
            scheduler,
            criterion,
            compile,
            tester,
            metrics_calculator,
            plot_label="VIFT",
        ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.criterion = criterion
        self.tester = tester
        self.metrics_calculator = metrics_calculator
        self.plot_label = plot_label


    def forward(self, x, target):
        return self.net(x, target)

    def training_step(self, batch, batch_idx):
        x, target = batch
        out = self.forward(x, target)
        weight = x[-1]
        loss = self.criterion(out, target, weight)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if hasattr(self.criterion, '_last_angle_loss'):
            self.log("train/Lr", self.criterion._last_angle_loss, on_step=True, on_epoch=True)
            self.log("train/Lt", self.criterion._last_translation_loss, on_step=True, on_epoch=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        out = self.forward(x, target)
        weight = x[-1]
        loss = self.criterion(out, target, weight, use_weighted_loss=False)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if hasattr(self.criterion, '_last_angle_loss'):
            self.log("val/Lr", self.criterion._last_angle_loss, on_step=False, on_epoch=True)
            self.log("val/Lt", self.criterion._last_translation_loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        # This method is not used for our custom testing
        pass

    def on_test_epoch_end(self):
        with torch.inference_mode():
            results = self.tester.test(self.net)
        metrics = self.metrics_calculator.calculate_metrics(results)
        metric_sum = 0
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
            metric_sum += value
        self.log("hp_metric", metric_sum)
        
        save_dir = self.trainer.logger.log_dir
        self.tester.save_results(results, save_dir)

        if save_dir:
            log_dir = Path(save_dir)
            run_root = log_dir.parent.parent if log_dir.parent.name in ("tensorboard", "csv") else log_dir.parent
            scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"

            traj_script = scripts_dir / "plot_eval_trajectories.py"
            if traj_script.is_file():
                subprocess.run(
                    [
                        sys.executable, str(traj_script), str(run_root),
                        "--poses-dir", str(log_dir),
                        "-o", str(run_root / "trajectories.png"),
                        "-m", self.plot_label,
                    ],
                    cwd=scripts_dir.parent,
                    check=False,
                )

            loss_script = scripts_dir / "plot_train_losses.py"
            csv_dir = run_root / "csv" / "version_0"
            if loss_script.is_file() and csv_dir.is_dir():
                subprocess.run(
                    [
                        sys.executable, str(loss_script), str(csv_dir),
                        "-o", str(run_root),
                    ],
                    check=False,
                )
            summary_metrics = {
                f"test/{name}": value
                for name, value in metrics.items()
                if name in {
                    "05_r_rel",
                    "05_t_rel",
                    "05_r_rmse",
                    "05_t_rmse",
                    "07_r_rel",
                    "07_t_rel",
                    "07_r_rmse",
                    "07_t_rmse",
                    "10_r_rel",
                    "10_t_rel",
                    "10_r_rmse",
                    "10_t_rmse",
                }
            }
            with open(log_dir / "summary_metrics.json", "w") as f:
                json.dump(summary_metrics, f)

    def setup(self, stage):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
