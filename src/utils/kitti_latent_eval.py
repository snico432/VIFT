# import dependencies
import os
import random
from tqdm import tqdm
import numpy as np

import glob
import scipy.io as sio
import torch
from src.utils.kitti_utils import read_pose_from_text, saveSequence
from src.utils.kitti_eval import plotPath_2D, kitti_eval, data_partition
from src.models.components.vsvio import Encoder

from natsort import natsorted


# wrap the model with an encoder for testing
class WrapperModel(torch.nn.Module):
    def __init__(self, params):
        super(WrapperModel, self).__init__()
        self.Feature_net = Encoder(params)
    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        memory = torch.cat((feat_v, feat_i), 2)
        return memory


# IMU is 100 Hz, camera 10 Hz → ~10x more IMU samples per window. So the same
# per-window dropout prob removes ~10x more IMU "content" than visual. Rate-equalized
# IMU dropout scales prob by 1/IMU_TO_VISUAL_RATE for comparable ablation.
IMU_TO_VISUAL_RATE = 10


class KITTI_tester_latent():
    def __init__(self, args, wrapper_weights_path, use_history_in_eval=False, eval_dropout_mode=None, eval_dropout_prob=0.0, eval_dropout_rate_equal=True):
        super(KITTI_tester_latent, self).__init__()
        
        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))
        self.args = args

        # Initialize and load pretrained weights for the wrapper model
        self.wrapper_model = WrapperModel(args)
        self.load_wrapper_weights(wrapper_weights_path)
        self.wrapper_model.eval()
        self.wrapper_model.to(self.args.device)
        self.use_history_in_eval = use_history_in_eval
        self.eval_dropout_mode = eval_dropout_mode.lower() if isinstance(eval_dropout_mode, str) else None
        self.eval_dropout_prob = eval_dropout_prob
        self.eval_dropout_rate_equal = eval_dropout_rate_equal
        if self.eval_dropout_mode is not None:
            assert self.eval_dropout_mode in ("visual", "imu")
            assert 0.0 <= self.eval_dropout_prob <= 1.0
            # For IMU, per-window dropout removes ~10x more raw measurements than visual; optionally scale prob for comparable ablation
            self._effective_dropout_prob = self.eval_dropout_prob
            if self.eval_dropout_mode == "imu" and self.eval_dropout_rate_equal:
                self._effective_dropout_prob = self.eval_dropout_prob / IMU_TO_VISUAL_RATE
                print(f"Eval dropout: mode={self.eval_dropout_mode}, prob={self.eval_dropout_prob} (rate-equalized → effective {self._effective_dropout_prob:.3f})")
            else:
                print(f"Eval dropout: mode={self.eval_dropout_mode}, prob={self.eval_dropout_prob}")

    def load_wrapper_weights(self, weights_path):
        if os.path.exists(weights_path):
            pretrained_w = torch.load(weights_path, map_location='cpu')
            
            model_dict = self.wrapper_model.state_dict()
            update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
            
            # Check if update dict is equal to model dict
            assert len(update_dict.keys()) == len(self.wrapper_model.Feature_net.state_dict().keys()), "Some weights are not loaded"
            
            self.wrapper_model.load_state_dict(update_dict)
            print(f"Loaded wrapper model weights from {weights_path}")
        else:
            print(f"Warning: Wrapper model weights not found at {weights_path}")

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        self.hist = None
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):  
            x_in = image_seq.unsqueeze(0).repeat(num_gpu,1,1,1,1).to(self.args.device)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu,1,1).to(self.args.device)
            
            with torch.inference_mode():
                # Generate latent representations
                latents = self.wrapper_model(x_in, i_in)
                # Optionally zero out visual or IMU for eval ablation (uses rate-equalized prob for IMU when enabled)
                drop_prob = getattr(self, "_effective_dropout_prob", self.eval_dropout_prob)
                if random.random() < drop_prob:
                    latents = latents.clone()
                    v_f_len, i_f_len = self.args.v_f_len, self.args.i_f_len
                    if self.eval_dropout_mode == "visual":
                        latents[..., :v_f_len] = 0.0
                    else:
                        latents[..., v_f_len:] = 0.0
                # accumulate poses by passing latents to the main model

                if (self.hist is not None) and self.use_history_in_eval:
                    results = torch.zeros(latents.shape[0], latents.shape[1], 6)
                    for idx in range(latents.shape[1]):
                        self.hist = torch.roll(self.hist, -1, 1) # shift so that index 0 becomes last one, shift in seq dim
                        self.hist[:,-1,:] = latents[:,idx,:]
                        x = (self.hist, None, None)
                        result = net(x, gt_seq) # batch_size, seq_len, 6
                        results[:,idx,:] = result[:,-1,:]
                    pose = results
                else:
                    self.hist = latents
                    pose = net((latents, None, None), gt_seq) 
            pose_list.append(pose[0,:,:].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def eval(self, net, selection=None, num_gpu=1, p=0.5):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)            
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[i].poses_rel)
            
            self.est.append({'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global, 'speed':speed})
            self.errors.append({'t_rel':t_rel, 'r_rel':r_rel, 't_rmse':t_rmse, 'r_rmse':r_rmse})
            
        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(seq, 
                        self.est[i]['pose_gt_global'], 
                        self.est[i]['pose_est_global'], 
                        save_dir, 
                        self.est[i]['speed'], 
                        window_size)
    
    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir/'{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))


class KITTI_tester_latent_tokenized():
    def __init__(self, args, wrapper_weights_path, use_history_in_eval=False):
        super().__init__()
        
        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))
        self.args = args

        # Initialize and load pretrained weights for the wrapper model
        self.wrapper_model = WrapperModel(args)
        self.load_wrapper_weights(wrapper_weights_path)
        self.wrapper_model.eval()
        self.wrapper_model.to(self.args.device)
        self.use_history_in_eval = use_history_in_eval

    def load_wrapper_weights(self, weights_path):
        if os.path.exists(weights_path):
            pretrained_w = torch.load(weights_path, map_location='cpu')
            
            model_dict = self.wrapper_model.state_dict()
            update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
            
            # Check if update dict is equal to model dict
            assert len(update_dict.keys()) == len(self.wrapper_model.Feature_net.state_dict().keys()), "Some weights are not loaded"
            
            self.wrapper_model.load_state_dict(update_dict)
            print(f"Loaded wrapper model weights from {weights_path}")
        else:
            print(f"Warning: Wrapper model weights not found at {weights_path}")

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        self.hist = None
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):  
            x_in = image_seq.unsqueeze(0).repeat(num_gpu,1,1,1,1).to(self.args.device)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu,1,1).to(self.args.device)
            
            with torch.inference_mode():
                # Generate latent representations
                latents = self.wrapper_model(x_in, i_in)
                # accumulate poses by passing latents to the main model

                if (self.hist is not None) and self.use_history_in_eval:
                    results = torch.zeros(latents.shape[0], latents.shape[1], 6)
                    for idx in range(latents.shape[1]):
                        self.hist = torch.roll(self.hist, -1, 1) # shift so that index 0 becomes last one, shift in seq dim
                        self.hist[:,-1,:] = latents[:,idx,:]
                        x = (self.hist, None, None)
                        result, _ = net(x, gt_seq) # batch_size, seq_len, 6
                        results[:,idx,:] = result[:,-1,:]
                    pose = results
                else:
                    self.hist = latents
                    pose, _ = net((latents, None, None), gt_seq) 
            pose_list.append(pose[0,:,:].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def eval(self, net, selection=None, num_gpu=1, p=0.5):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)            
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[i].poses_rel)
            
            self.est.append({'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global, 'speed':speed})
            self.errors.append({'t_rel':t_rel, 'r_rel':r_rel, 't_rmse':t_rmse, 'r_rmse':r_rmse})
            
        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(seq, 
                        self.est[i]['pose_gt_global'], 
                        self.est[i]['pose_est_global'], 
                        save_dir, 
                        self.est[i]['speed'], 
                        window_size)
    
    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir/'{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))
