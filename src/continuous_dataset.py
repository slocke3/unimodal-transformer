"""Continuous-input / discrete-target dataset for the output-resolution sweep.

Each sample:
    context : raw x-values, float32, shape (context_len,)   -- NOT binned
    target  : bin index of x_{n+1} at resolution n_out, long
    r_value : float32

Mirrors DiscreteMapDataset's trajectory generation, but leaves the context in
continuous form and only bins the target (at the swept output resolution).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .maps import iterate_map


class ContinuousMapDataset(Dataset):
    def __init__(self, r_values, n_out=64, context_len=50, burn_in=0,
                 traj_len=150, seed=0):
        super().__init__()
        self.context_len = context_len
        self.n_out = n_out

        rng = np.random.default_rng(seed)
        rs = np.asarray(r_values, dtype=float)
        x0s = rng.uniform(0.05, 0.95, size=len(rs))
        window = context_len + 1

        ctx_list, tgt_list, lab_list = [], [], []
        for i in range(len(rs)):
            traj = iterate_map(x0s[i], rs[i], burn_in + traj_len)[burn_in:]
            for t in range(len(traj) - window):
                ctx_list.append(traj[t : t + context_len])
                tgt_list.append(traj[t + context_len])
                lab_list.append(rs[i])

        if ctx_list:
            ctx = np.asarray(ctx_list, dtype=np.float32)
            tgt_f = np.asarray(tgt_list, dtype=np.float64)
        else:
            ctx = np.empty((0, context_len), dtype=np.float32)
            tgt_f = np.empty((0,), dtype=np.float64)

        tgt_bin = np.clip(np.floor(tgt_f * n_out), 0, n_out - 1).astype(np.int64)
        self.contexts = torch.tensor(ctx, dtype=torch.float32)
        self.targets = torch.tensor(tgt_bin, dtype=torch.long)
        self.r_labels = torch.tensor(np.asarray(lab_list, dtype=np.float32),
                                      dtype=torch.float32)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx], self.r_labels[idx]
