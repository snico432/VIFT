cd ~/cis-4910/VIFT
python src/eval.py \
  ckpt_path="/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-03-25_22-59-16/checkpoints/epoch_123.ckpt" \
  model=cross_attn_latent_vio_tf \
  data=latent_kitti_vio \
  trainer=gpu