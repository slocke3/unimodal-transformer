"""Evaluation for the continuous-input / n_out-bin models.

Two metrics, both designed to be comparable ACROSS output resolutions:

  * reference-grid CE -- every model is scored at a common fine resolution
    (ref_bins), so the trivial "raw CE grows with n_out" effect is removed.
    With a softmax over n_out bins and no sub-bin information, the fair way to
    place a bin's probability on the finer grid is to spread it uniformly over
    the ref_bins/n_out reference cells it contains. That makes the per-example
    reference CE exactly the native CE plus log(ref_bins / n_out), so we just
    apply that shift (requires ref_bins to be a multiple of n_out).

  * implied-map RMS -- read E[x_{n+1} | context] = sum_j p_j * center_j and
    compare it to the true one-step map r*x_n*(1-x_n). Resolution-free, in
    x-units, so it is directly comparable across every n_out and to the
    continuous limit.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .maps import iterate_map


@torch.no_grad()
def evaluate_continuous_per_r(model, r_grid, device, context_len, n_out,
                              ref_bins=256, burn_in=0, n_eval_per_r=30,
                              traj_len=150, seed=99):
    if ref_bins % n_out != 0:
        raise ValueError(f"ref_bins ({ref_bins}) must be a multiple of n_out ({n_out})")
    model.eval()
    rng = np.random.default_rng(seed)
    window = context_len + 1
    ref_shift = float(np.log(ref_bins / n_out))
    centers = torch.tensor((np.arange(n_out) + 0.5) / n_out,
                           dtype=torch.float32, device=device)

    ce_native = np.empty(len(r_grid))
    ce_ref = np.empty(len(r_grid))
    rms = np.empty(len(r_grid))

    for i, r in enumerate(r_grid):
        ctx_list, nxt_list = [], []
        for _ in range(n_eval_per_r):
            x0 = rng.uniform(0.05, 0.95)
            traj = iterate_map(x0, r, burn_in + traj_len)[burn_in:]
            for t in range(len(traj) - window):
                ctx_list.append(traj[t : t + context_len])
                nxt_list.append(traj[t + context_len])

        ctx = torch.tensor(np.asarray(ctx_list, dtype=np.float32),
                           dtype=torch.float32, device=device)
        nxt = torch.tensor(np.asarray(nxt_list, dtype=np.float32),
                           dtype=torch.float32, device=device)
        tgt_bin = torch.clamp(torch.floor(nxt * n_out), 0, n_out - 1).long()

        logits = model(ctx)
        logp = F.log_softmax(logits, dim=-1)
        native = -logp.gather(1, tgt_bin.view(-1, 1)).squeeze(1)   # (M,)
        ce_native[i] = native.mean().item()
        ce_ref[i] = ce_native[i] + ref_shift

        probs = F.softmax(logits, dim=-1)
        pred_mean = probs @ centers                                # E[x_{n+1} | context]
        x_n = ctx[:, -1]
        true_next = r * x_n * (1.0 - x_n)
        rms[i] = torch.sqrt(((pred_mean - true_next) ** 2).mean()).item()

        if (i + 1) % 100 == 0:
            print(f"  eval: {i+1}/{len(r_grid)}", flush=True)

    return ce_ref, ce_native, rms
