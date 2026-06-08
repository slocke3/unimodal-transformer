import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .maps import iterate_map, tokenize_trajectory, compute_lyapunov_array


class DiscreteMapDataset(Dataset):
    """
    Dataset of tokenized quadratic map trajectories.

    Each sample: (context_tokens, target_token, r_value)
        context_tokens : LongTensor (context_len,)
        target_token   : scalar LongTensor
        r_value        : float32
    """
    def __init__(self, n_trajectories=10_000, r_range=(0.5, 4.0),
                 context_len=50, burn_in=0, traj_len=200,
                 n_bins=64, seed=0, r_values=None):
        super().__init__()
        self.context_len = context_len
        self.n_bins = n_bins

        rng = np.random.default_rng(seed)
        if r_values is not None:
            rs = np.asarray(r_values)
            n_trajectories = len(rs)
        else:
            rs = rng.uniform(r_range[0], r_range[1], size=n_trajectories)

        x0s = rng.uniform(0.05, 0.95, size=n_trajectories)
        window_size = context_len + 1

        contexts_list, targets_list, r_labels_list = [], [], []
        for i in range(n_trajectories):
            traj = iterate_map(x0s[i], rs[i], burn_in + traj_len)[burn_in:]
            tokens = tokenize_trajectory(traj, n_bins)
            for t in range(len(tokens) - window_size):
                contexts_list.append(tokens[t : t + context_len])
                targets_list.append(tokens[t + context_len])
                r_labels_list.append(rs[i])

        self.contexts = torch.tensor(np.array(contexts_list), dtype=torch.long)
        self.targets  = torch.tensor(np.array(targets_list), dtype=torch.long)
        self.r_labels = torch.tensor(np.array(r_labels_list), dtype=torch.float32)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx], self.r_labels[idx]


def make_splits(n_trajectories=10_000, r_range=(0.5, 4.0),
                context_len=50, burn_in=0, traj_len=200,
                n_bins=64, train_frac=0.8, val_frac=0.1,
                seed=0, batch_size=256, num_workers=2):
    """Build train / val / test DataLoaders."""
    rng = np.random.default_rng(seed)
    rs = rng.uniform(r_range[0], r_range[1], size=n_trajectories)

    n_train = int(train_frac * n_trajectories)
    n_val   = int(val_frac * n_trajectories)
    perm    = rng.permutation(n_trajectories)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    def make_loader(indices, shuffle):
        ds = DiscreteMapDataset(
            r_values=rs[indices], context_len=context_len,
            burn_in=burn_in, traj_len=traj_len,
            n_bins=n_bins, seed=seed,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        make_loader(train_idx, True),
        make_loader(val_idx, False),
        make_loader(test_idx, False),
    )


def make_eval_grid(n_points=1000, r_range=(0.5, 4.0),
                   n_lyapunov_steps=100_000, verbose=True):
    r_grid = np.linspace(r_range[0], r_range[1], n_points)
    print(f"Computing {n_points} Lyapunov exponents...")
    lyapunovs = compute_lyapunov_array(r_grid, n_steps=n_lyapunov_steps, verbose=verbose)
    return r_grid, lyapunovs