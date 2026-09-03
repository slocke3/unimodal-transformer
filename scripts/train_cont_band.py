"""Band sweep with CONTINUOUS input embeddings instead of bin tokens.

Every earlier map sweep fed the model bin indices, so identifying the map meant
reconstructing a function from a coarsely quantised and unevenly sampled view of
it -- a 50-token context visits only ~32 of 64 bins and covers ~60% of the
invariant measure. Feeding x directly removes that bottleneck: each context
position becomes an exact (x_n, x_{n+1}) pair.

Two output modes isolate the input change from the output one:

  bins    logits over n_bins, cross-entropy against bin(x_{n+1}). Identical
          output to the earlier runs, so ONLY the input representation differs.
  scalar  a single real number, squared error against x_{n+1}. Continuous end
          to end.

Both families are supported. Bands are one-sided [1-w, 1] for asym (alpha > 1
makes the origin superattracting) and symmetric [-w, +w] for tilted (both
endpoints stay regular, so the logistic base map sits in the interior). Held-out
probes sit at a fixed DISTANCE d past each band edge, so a widening band does
not bring its own test closer.

Training predicts at every position in the window rather than only the last:
the model runs all L positions through every layer regardless, so scoring one
of them wastes the rest.

Reported per (parameter, R) cell, for both modes:
  implied_rms  RMS distance from the model's implied E[x_{n+1}|x_n] to the true
               map. In scalar mode the prediction IS the implied map; in bins
               mode it is sum_j p_j * centre_j.
  floor        the best achievable value of that quantity. Zero in scalar mode
               (x_n is known exactly, so nothing forces an error); in bins mode
               it is the output-quantisation residual, RMS(centre(bin(f)) - f).
"""
import argparse
import json
import os
import time

import numpy as np


def build_windows(params, family, context_len, traj_len, n_per, rng):
    """Sliding windows of raw x values, one fresh x0 per trajectory."""
    from src.maps import iterate_family
    W = context_len + 1
    out = []
    for (R, p) in params:
        for _ in range(n_per):
            t = iterate_family(rng.uniform(0.05, 0.95), R, p, traj_len, family)
            for s in range(len(t) - W):
                out.append(t[s:s + W])
    return np.asarray(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["asym", "tilted"], required=True)
    ap.add_argument("--output_mode", choices=["bins", "scalar"], required=True)
    ap.add_argument("--band_width", type=float, default=None,
                    help="band half-width in PARAMETER units")
    ap.add_argument("--heldout_d", type=float, default=0.2,
                    help="probe offset in PARAMETER units")
    ap.add_argument("--band_rho", type=float, default=None,
                    help="band half-width in MAP-SPACE arclength (bin widths); "
                         "mutually exclusive with --band_width")
    ap.add_argument("--heldout_bins", type=float, default=4.0,
                    help="probe offset in MAP-SPACE arclength (bin widths)")
    ap.add_argument("--R_lo", type=float, default=0.125)
    ap.add_argument("--R_hi", type=float, default=1.0)
    ap.add_argument("--n_tasks", type=int, default=8000)
    ap.add_argument("--n_train_traj", type=int, default=32000)
    ap.add_argument("--context_len", type=int, default=50)
    ap.add_argument("--n_bins", type=int, default=64)
    ap.add_argument("--traj_len", type=int, default=150)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--log_every", type=int, default=2500)
    ap.add_argument("--n_p_eval", type=int, default=41)
    ap.add_argument("--n_R_eval", type=int, default=60)
    ap.add_argument("--n_eval_per_point", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    if (a.band_width is None) == (a.band_rho is None):
        raise SystemExit("give exactly one of --band_width or --band_rho")
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    w, nb, L = a.band_width, a.n_bins, a.context_len

    from src import mapmetric as mm

    if a.band_rho is not None:
        # Map-space design. The band is an interval of arclength around the
        # logistic base map -- one-sided for asym (the base sits at the alpha=1
        # endpoint) and two-sided for tilted (the base is interior) -- and the
        # probes sit heldout_bins further along the SAME arclength, so their
        # distance from the band edge is constant in map units at every rho.
        rho, hb = a.band_rho, a.heldout_bins
        g_lo, g_hi = mm.sigma_range(a.family)
        if a.family == "asym":
            band_sig = (max(-rho, g_lo), 0.0)
            probe_sig = [-(rho + hb)]
        else:
            band_sig = (max(-rho, g_lo), min(rho, g_hi))
            probe_sig = [-(rho + hb), rho + hb]
        band_lo = mm.param_of_sigma(a.family, band_sig[0])
        band_hi = mm.param_of_sigma(a.family, band_sig[1])
        probe_sig = [g for g in probe_sig if g_lo - 1e-9 <= g <= g_hi + 1e-9]
        probes = [mm.param_of_sigma(a.family, g) for g in probe_sig]
        p_lo, p_hi = mm.FAMILY_RANGE[a.family]
        p_eval = np.linspace(p_lo, p_hi, a.n_p_eval)
    elif a.family == "asym":
        band_lo, band_hi = 1.0 - w, 1.0
        p_eval = np.linspace(0.2, 1.0, a.n_p_eval)
        probes = [band_lo - a.heldout_d]
    else:
        band_lo, band_hi = -w, w
        p_eval = np.linspace(-1.2, 1.2, a.n_p_eval)
        probes = [band_lo - a.heldout_d, band_hi + a.heldout_d]
    probes = [p for p in probes if p_eval.min() - 1e-9 <= p <= p_eval.max() + 1e-9]
    # heldout_mask matches on exact grid values, and an arclength-derived probe
    # will not land on a linspace, so splice the probes into the grid.
    p_eval = np.unique(np.concatenate([p_eval, np.asarray(probes, dtype=float)]))

    import torch
    import torch.nn as nn
    from src.model import ContinuousTrajectoryTransformer
    from src.maps import family_map_vec

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)

    # -- training data: raw x windows --------------------------------------
    if a.band_rho is not None:
        # uniform in arclength, so tasks are spread evenly in MAP space
        ps = np.array([mm.param_of_sigma(a.family, g) for g in
                       rng.uniform(band_sig[0], band_sig[1], size=a.n_tasks)])
    elif w == 0:
        ps = np.full(a.n_tasks, band_hi)
    else:
        ps = rng.uniform(band_lo, band_hi, size=a.n_tasks)
    Rs = rng.uniform(a.R_lo, a.R_hi, size=a.n_tasks)
    n_per = max(1, a.n_train_traj // a.n_tasks)
    win = build_windows(list(zip(Rs, ps)), a.family, L, a.traj_len, n_per, rng)
    rng.shuffle(win)
    print(f"[cont] family={a.family} out={a.output_mode} "
          f"band=[{band_lo:.3f},{band_hi:.3f}] probes={[round(p,3) for p in probes]} "
          f"windows={len(win)} dev={dev}", flush=True)

    model = ContinuousTrajectoryTransformer(
        n_bins=nb, context_len=L, d_model=a.d_model, n_heads=a.n_heads,
        n_layers=a.n_layers, dropout=a.dropout,
        output_mode=a.output_mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_steps)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    data = torch.as_tensor(win, device=dev)

    model.train()
    hist = []
    for step in range(1, a.max_steps + 1):
        idx = torch.randint(0, len(data), (a.batch_size,), device=dev)
        b = data[idx]
        xb, yb = b[:, :-1], b[:, 1:]
        opt.zero_grad()
        out = model(xb, all_positions=True)
        if a.output_mode == "bins":
            tgt = torch.clamp((yb * nb).long(), 0, nb - 1)
            loss = ce(out.reshape(-1, nb), tgt.reshape(-1))
        else:
            loss = mse(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % a.log_every == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.item())})
            print(f"  step {step:6d}/{a.max_steps} loss {loss.item():.5f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    # -- evaluation over the (parameter, R) plane ---------------------------
    model.eval()
    R_eval = np.linspace(a.R_lo, a.R_hi, a.n_R_eval)
    centres = torch.as_tensor((np.arange(nb) + 0.5) / nb, dtype=torch.float32,
                              device=dev)
    shape = (len(p_eval), len(R_eval))
    implied = np.full(shape, np.nan)
    floor = np.full(shape, np.nan)
    loss_grid = np.full(shape, np.nan)
    erng = np.random.default_rng(a.seed + 7)
    for i, pv in enumerate(p_eval):
        for j, Rv in enumerate(R_eval):
            wins = build_windows([(Rv, pv)], a.family, L, a.traj_len,
                                 a.n_eval_per_point, erng)
            b = torch.as_tensor(wins, device=dev)
            xb, yb = b[:, :-1], b[:, -1]
            with torch.no_grad():
                out = model(xb)
                if a.output_mode == "bins":
                    pr = torch.softmax(out, dim=-1)
                    e_next = (pr @ centres).cpu().numpy()
                    tgt = torch.clamp((yb * nb).long(), 0, nb - 1)
                    loss_grid[i, j] = float(ce(out, tgt).item())
                else:
                    e_next = out.cpu().numpy()
                    loss_grid[i, j] = float(mse(out, yb).item())
            last_x = wins[:, -2].astype(float)
            truth = family_map_vec(last_x, Rv, pv, a.family)
            implied[i, j] = float(np.sqrt(np.mean((e_next - truth) ** 2)))
            if a.output_mode == "bins":
                q = (np.floor(truth * nb) + 0.5) / nb
                floor[i, j] = float(np.sqrt(np.mean((q - truth) ** 2)))
            else:
                floor[i, j] = 0.0
        if (i + 1) % 10 == 0:
            print(f"  eval {i+1}/{len(p_eval)}", flush=True)

    in_band = (p_eval >= band_lo - 1e-9) & (p_eval <= band_hi + 1e-9)
    held = np.array([any(abs(g - p) < 1e-6 for p in probes) for g in p_eval])
    keep = R_eval >= 0.25
    np.savez(os.path.join(a.out_dir, "eval_cont.npz"),
             p_grid=p_eval, R_grid=R_eval, implied_rms=implied,
             binning_floor=floor, loss_grid=loss_grid, in_band=in_band,
             heldout_mask=held, probes=np.array(probes), band_lo=band_lo,
             band_hi=band_hi, band_width=(np.nan if w is None else w),
             family=a.family, output_mode=a.output_mode,
             sigma_grid=np.array([mm.sigma_of(a.family, p) for p in p_eval]),
             band_rho=(np.nan if a.band_rho is None else a.band_rho),
             heldout_bins=(np.nan if a.band_rho is None else a.heldout_bins),
             band_sigma=np.array([mm.sigma_of(a.family, band_lo),
                                  mm.sigma_of(a.family, band_hi)]))
    with open(os.path.join(a.out_dir, "params.json"), "w") as f:
        json.dump({**vars(a), "band_lo": band_lo, "band_hi": band_hi,
                   "history": hist, "n_params": model.count_parameters(),
                   "wall_sec": round(time.time() - t0, 1)}, f, indent=2)
    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(a.out_dir, "model.pt"))
    print(f"[cont] DONE {time.time()-t0:.0f}s | in-band RMS "
          f"{np.nanmean(implied[in_band][:, keep]):.5f} | held-out RMS "
          f"{np.nanmean(implied[held][:, keep]) if held.any() else float('nan'):.5f}",
          flush=True)


if __name__ == "__main__":
    main()
