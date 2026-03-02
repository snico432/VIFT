## VIFT with dropout ablation

### VIFT with dropout on visual features (zero out)

```bash
python src/eval.py ckpt_path=logs/train/runs/2026-02-15_23-50-17/checkpoints/epoch_145.ckpt \
  model.tester.eval_dropout_mode=visual model.tester.eval_dropout_prob=0.2 \
  trainer=gpu logger=tensorboard
```

### VIFT with visual degradation: scale (softer than zero)

When dropout triggers, multiply visual latents by a factor instead of zeroing (e.g. 10% signal):

```bash
python src/eval.py ckpt_path=logs/train/runs/2026-02-15_23-50-17/checkpoints/epoch_145.ckpt \
  model.tester.eval_dropout_mode=visual model.tester.eval_dropout_prob=0.2 \
  model.tester.eval_dropout_style=scale model.tester.eval_dropout_scale=0.1 \
  trainer=gpu logger=tensorboard
```

### VIFT with visual degradation: noise (softer than zero)

When dropout triggers, add Gaussian noise to visual latents (std relative to latent std):

```bash
python src/eval.py ckpt_path=logs/train/runs/2026-02-15_23-50-17/checkpoints/epoch_145.ckpt \
  model.tester.eval_dropout_mode=visual model.tester.eval_dropout_prob=0.2 \
  model.tester.eval_dropout_style=noise model.tester.eval_dropout_noise_std=1.0 \
  trainer=gpu logger=tensorboard
```

### VIFT with dropout on IMU features

```bash
python src/eval.py ckpt_path=logs/train/runs/2026-02-15_23-50-17/checkpoints/epoch_145.ckpt \
  model.tester.eval_dropout_mode=imu model.tester.eval_dropout_prob=0.5 \
  trainer=gpu logger=tensorboard
```

### Generate plots

Method name is inferred from the run’s `.hydra/config.yaml`; override with `-m` if needed.

```bash
# Run dir (poses read from run_dir/tensorboard/version_0/)
python scripts/plot_eval_trajectories.py logs/eval/runs/RUN_ID
```

```bash
# Custom output path and/or label
python scripts/plot_eval_trajectories.py logs/eval/runs/RUN_ID \
  -o trajectories_visual_dropout.png -m "VIFT (no visual)"


  VIFT/logs/eval/runs/2026-03-02_04-05-38