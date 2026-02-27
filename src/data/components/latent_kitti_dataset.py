import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LatentVectorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.latent_files = os.listdir(root_dir)

    def __len__(self):
        return int(len(self.latent_files) / 4)

    def __getitem__(self, idx):
        latent_vector = np.load(os.path.join(self.root_dir, f"{idx}.npy"))
        gt = np.load(os.path.join(self.root_dir, f"{idx}_gt.npy"))
        rot = np.load(os.path.join(self.root_dir, f"{idx}_rot.npy"))
        w = np.load(os.path.join(self.root_dir, f"{idx}_w.npy"))
        return (torch.from_numpy(latent_vector).to(torch.float), torch.from_numpy(rot), torch.from_numpy(w)), torch.from_numpy(gt).to(torch.float).squeeze()


class LatentVectorDatasetWithDropout(LatentVectorDataset):
    """Latent dataset that randomly zeros out visual or IMU dimensions (per sample).

    Latent layout: first v_f_len dims = visual, remaining i_f_len dims = IMU (total 768).
    With probability dropout_prob, the chosen modality is zeroed for the whole sample.
    """

    def __init__(
        self,
        root_dir: str,
        dropout_mode: str,
        dropout_prob: float = 0.5,
        v_f_len: int = 512,
        i_f_len: int = 256,
    ):
        super().__init__(root_dir)
        self.dropout_mode = dropout_mode.lower()
        self.dropout_prob = dropout_prob
        self.v_f_len = v_f_len
        self.i_f_len = i_f_len
        assert self.dropout_mode in ("visual", "imu")
        assert 0.0 <= self.dropout_prob <= 1.0
        assert self.v_f_len + self.i_f_len == 768, "v_f_len + i_f_len must equal 768"

    def __getitem__(self, idx):
        (latent, rot, w), gt = super().__getitem__(idx)
        if random.random() < self.dropout_prob:
            latent = latent.clone()
            if self.dropout_mode == "visual":
                latent[..., : self.v_f_len] = 0.0
            else:
                latent[..., self.v_f_len :] = 0.0
        return (latent, rot, w), gt


