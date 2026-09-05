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
                 n_bins=64, seed=0, r_values=None,
                 input_mode="bins", n_bins_in=None, n_bins_out=None,
                 output_mode="bins", synonyms=1):
        """input_mode "continuous" hands the model raw x instead of bin indices;
        output_mode "scalar" makes the target the raw next x for a square loss.
        n_bins_in / n_bins_out default to n_bins, which reproduces the original
        behaviour exactly -- the tokeniser and window count are unchanged."""
        super().__init__()
        assert input_mode in ("bins", "continuous")
        assert output_mode in ("bins", "scalar")
        self.context_len = context_len
        self.n_bins = n_bins
        self.input_mode, self.output_mode = input_mode, output_mode
        n_bins_in = n_bins if n_bins_in is None else n_bins_in
        n_bins_out = n_bins if n_bins_out is None else n_bins_out
        self.n_bins_in, self.n_bins_out = n_bins_in, n_bins_out

        rng = np.random.default_rng(seed)
        if r_values is not None:
            rs = np.asarray(r_values)
            n_trajectories = len(rs)
        else:
            rs = rng.uniform(r_range[0], r_range[1], size=n_trajectories)

        x0s = rng.uniform(0.05, 0.95, size=n_trajectories)
        window_size = context_len + 1

        # Collect raw windows once, then derive whichever representation each
        # mode needs. Tokenising afterwards is identical to tokenising the whole
        # trajectory first, since tokenize_trajectory is elementwise.
        raw_ctx_list, raw_tgt_list, r_labels_list = [], [], []
        for i in range(n_trajectories):
            traj = iterate_map(x0s[i], rs[i], burn_in + traj_len)[burn_in:]
            for t in range(len(traj) - window_size):
                raw_ctx_list.append(traj[t : t + context_len])
                raw_tgt_list.append(traj[t + context_len])
                r_labels_list.append(rs[i])

        # Handle the empty case (e.g. an empty split): np.array([]) defaults to
        # float64 with the wrong shape, which breaks the reshape below.
        if raw_ctx_list:
            raw_ctx = np.asarray(raw_ctx_list, dtype=np.float64)
            raw_tgt = np.asarray(raw_tgt_list, dtype=np.float64)
        else:
            raw_ctx = np.empty((0, context_len), dtype=np.float64)
            raw_tgt = np.empty((0,), dtype=np.float64)

        contexts = tokenize_trajectory(raw_ctx, n_bins_in)
        targets = tokenize_trajectory(raw_tgt, n_bins_out)
        self.synonyms = synonyms
        if synonyms > 1:
            assert input_mode == "bins", "synonyms only apply to binned input"
            contexts = contexts * synonyms + rng.integers(
                0, synonyms, size=contexts.shape)
        # Histogram of the token exposures in one pass through this dataset.
        # Positions repeated across sliding contexts are intentionally counted
        # repeatedly because the model sees each occurrence. Continuous-input
        # runs keep it, binned at n_bins_in, purely as a coverage diagnostic.
        self.token_counts = (
            np.bincount(tokenize_trajectory(raw_ctx, n_bins_in).reshape(-1),
                        minlength=n_bins_in)[:n_bins_in]
            + np.bincount(tokenize_trajectory(raw_tgt, n_bins_in),
                          minlength=n_bins_in)[:n_bins_in]
        ).astype(np.int64)

        if input_mode == "continuous":
            self.contexts = torch.tensor(raw_ctx, dtype=torch.float32)
        else:
            self.contexts = torch.tensor(contexts, dtype=torch.long)
        if output_mode == "scalar":
            self.targets = torch.tensor(raw_tgt, dtype=torch.float32)
        else:
            self.targets = torch.tensor(targets, dtype=torch.long)
        self.raw_targets = torch.tensor(raw_tgt, dtype=torch.float32)
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

class AsymMapDataset(Dataset):
    """
    Tokenized trajectories of the asymmetric unimodal family.

    Unlike DiscreteMapDataset, a task here is a PAIR (R, alpha): R is the peak
    height (R = r/4 recovers the logistic map at alpha = 1) and alpha sets the
    peak position x_c = alpha/2.

    Each sample: (context_tokens, target_token, R_value)
    The R label is returned in the third slot so the collate signature matches
    DiscreteMapDataset and the shared Trainer needs no changes.
    """
    def __init__(self, params, context_len=50, burn_in=0, traj_len=150,
                 n_bins=64, seed=0, family="asym"):
        super().__init__()
        from .maps import iterate_family  # local import keeps module light

        self.context_len = context_len
        self.n_bins = n_bins
        params = np.asarray(params, dtype=float).reshape(-1, 2)

        rng = np.random.default_rng(seed)
        x0s = rng.uniform(0.05, 0.95, size=len(params))
        window_size = context_len + 1

        contexts_list, targets_list, label_list = [], [], []
        for i, (R, alpha) in enumerate(params):
            traj = iterate_family(x0s[i], R, alpha, burn_in + traj_len,
                                  family)[burn_in:]
            tokens = tokenize_trajectory(traj, n_bins)
            for t in range(len(tokens) - window_size):
                contexts_list.append(tokens[t : t + context_len])
                targets_list.append(tokens[t + context_len])
                label_list.append(R)

        if contexts_list:
            contexts = np.asarray(contexts_list, dtype=np.int64)
            targets = np.asarray(targets_list, dtype=np.int64)
        else:
            contexts = np.empty((0, context_len), dtype=np.int64)
            targets = np.empty((0,), dtype=np.int64)

        self.token_counts = (
            np.bincount(contexts.reshape(-1), minlength=n_bins)
            + np.bincount(targets, minlength=n_bins)
        ).astype(np.int64)

        self.contexts = torch.tensor(contexts, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.r_labels = torch.tensor(np.array(label_list), dtype=torch.float32)
        self.params = params

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx], self.r_labels[idx]
