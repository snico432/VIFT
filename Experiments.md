## VIFT with dropout ablation

### VIFT with dropout on visual features

```bash
python src/eval.py ckpt_path=logs/train/runs/2026-02-15_23-50-17/checkpoints/last.ckpt \
  model.tester.eval_dropout_mode=visual model.tester.eval_dropout_prob=0.2 \
  'data.train_loader.root_dir=${paths.data_dir}/kitti_latent_data/train_10' \
  'data.val_loader.root_dir=${paths.data_dir}/kitti_latent_data/val_10' \
  'data.test_loader.root_dir=${paths.data_dir}/kitti_latent_data/val_10' \
  trainer=gpu logger=tensorboard
```

### VIFT with dropout on IMU features

```bash
python src/eval.py ckpt_path=logs/train/runs/2026-02-15_23-50-17/checkpoints/last.ckpt \
  model.tester.eval_dropout_mode=imu model.tester.eval_dropout_prob=0.5 \
  'data.train_loader.root_dir=${paths.data_dir}/kitti_latent_data/train_10' \
  'data.val_loader.root_dir=${paths.data_dir}/kitti_latent_data/val_10' \
  'data.test_loader.root_dir=${paths.data_dir}/kitti_latent_data/val_10' \
  trainer=gpu logger=tensorboard
```

### Generate plots

# Baseline
```bash
python scripts/plot_eval_trajectories.py logs/eval/runs/RUN_ID/tensorboard/version_0
```

# Visual dropout 1.0
```bash
python scripts/plot_eval_trajectories.py logs/eval/runs/RUN_ID/tensorboard/version_0 \
  --method-name "VIFT (visual dropout 1.0)"
```
# IMU dropout 0.5
```bash
python scripts/plot_eval_trajectories.py logs/eval/runs/RUN_ID/tensorboard/version_0 \
  -m "VIFT (IMU dropout 0.5)"
```
# Custom output path and label
```bash
python scripts/plot_eval_trajectories.py logs/eval/runs/RUN_ID/tensorboard/version_0 \
  -o trajectories_visual_dropout.png -m "VIFT (no visual)"