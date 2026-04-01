1. python src/train.py experiment=visual_context_cross_attn trainer=gpu logger=many_loggers
2. python src/eval.py ckpt_path="path/to/checkpoint.ckpt" \
  model=visual_context_cross_attn \
  data=latent_kitti_vio \
  trainer=gpu
3. python scripts/plot_eval_trajectories.py /lambda/nfs/cis-4910/VIFT/logs/eval/runs/2026-03-26_00-10-00 -m "Cross-Attn FFN=1024"
4. cd ../plot_losses
5. python plot_losses/plot_losses.py <directory_of_loss_files>

python src/eval.py \
  ckpt_path=/lambda/nfs/cis-4910/VIFT/logs/train/runs/2026-04-01_22-33-30/checkpoints/epoch_072.ckpt \
  model=cross_attn_latent_vio_tf \
  model.net.embedding_dim=512 \
  model.net.nhead=8 \
  model.net.dim_feedforward=1024 \
  model.criterion.angle_weight=10 \
  data=latent_kitti_vio \
  trainer=gpu