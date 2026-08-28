"""Train one model on variable-order Markov sources, and test on unseen orders.

This is the concept-shift experiment: the training arms differ in which ORDERS
of dependence appear, and the test includes orders that never appeared. Unlike
sweeping a map parameter, that changes the functional form of the rule rather
than its parameters.

The prediction being tested:

  fixed-order arm   every sequence has the same k, so order-k counting is the
                    optimal solution and nothing rewards a general routine.
                    Should fail on unseen orders.
  mixed-order arm   k varies per sequence, so no single order is right and the
                    in-distribution optimum IS "match the longest suffix you can
                    find". That routine has no built-in ceiling, so it may
                    extrapolate to orders never trained on.

A fresh conditional table is drawn for every sequence, so task diversity is
infinite by construction and the model cannot memorise tasks.

Everything is scored against EXACT references (Dirichlet-multinomial conjugacy):
full Bayes over all admissible orders, and Bayes restricted to the training
orders -- the latter being the concept-shift analogue of a mis-specified prior,
and the line the model must beat to have generalized.
"""
import argparse
import json
import os
import time

import numpy as np


def generate_batch(n, ks, beta, length, rng):
    """Sequences with a fresh order-k table each, vectorised over the batch."""
    B = len(ks)
    x = np.empty((B, length), dtype=np.int64)
    for k in np.unique(ks):
        idx = np.where(ks == k)[0]
        m = len(idx)
        chains = rng.dirichlet(np.full(n, beta), size=(m, n ** k))
        cdf = np.cumsum(chains, axis=2)
        seq = np.empty((m, length), dtype=np.int64)
        seq[:, :k] = rng.integers(0, n, size=(m, k))
        cur = np.zeros(m, dtype=np.int64)
        for i in range(k):
            cur = cur * n + seq[:, i]
        rows = np.arange(m)
        for t in range(k, length):
            u = rng.random(m)[:, None]
            seq[:, t] = (u > cdf[rows, cur]).sum(axis=1)
            cur = (cur * n + seq[:, t]) % (n ** k)
        x[idx] = seq
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_orders", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--test_orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--n_symbols", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--context_len", type=int, default=512)
    p.add_argument("--k_max_bayes", type=int, default=6,
                   help="orders the FULL Bayes reference marginalises over")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=40000)
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--n_eval", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    n, beta, L = a.n_symbols, a.beta, a.context_len

    import torch
    import torch.nn as nn
    from src.model import DiscreteTrajectoryTransformer
    from src.markov import bayes_mixture

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)

    model = DiscreteTrajectoryTransformer(
        n_bins=n, context_len=L, d_model=a.d_model, n_heads=a.n_heads,
        n_layers=a.n_layers, dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_steps)
    crit = nn.CrossEntropyLoss()

    print(f"[markov] train_orders={a.train_orders} n={n} beta={beta} L={L} "
          f"steps={a.max_steps} params={model.count_parameters():,} dev={dev}",
          flush=True)

    # ---- training: fresh tasks every batch --------------------------------
    model.train()
    hist = []
    for step in range(1, a.max_steps + 1):
        ks = rng.choice(a.train_orders, size=a.batch_size)
        seq = generate_batch(n, ks, beta, L + 1, rng)
        xb = torch.as_tensor(seq[:, :-1], dtype=torch.long, device=dev)
        yb = torch.as_tensor(seq[:, -1], dtype=torch.long, device=dev)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % a.log_every == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.item())})
            print(f"  step {step:6d}/{a.max_steps}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    # ---- evaluation: model vs exact Bayes references, per test order ------
    model.eval()
    results = {}
    print(f"\n%8s %10s %12s %14s %12s" % ("test k", "in train", "model NLL",
                                          "restricted", "full Bayes"), flush=True)
    for k in a.test_orders:
        ks = np.full(a.n_eval, k)
        seq = generate_batch(n, ks, beta, L + 1, np.random.default_rng(1000 + k))
        ctx, tgt = seq[:, :-1], seq[:, -1]
        with torch.no_grad():
            lp = torch.log_softmax(
                model(torch.as_tensor(ctx, dtype=torch.long, device=dev)),
                dim=-1).cpu().numpy()
        m_nll = float(-lp[np.arange(len(tgt)), tgt].mean())
        full, restr = [], []
        for i in range(len(ctx)):
            pf, _ = bayes_mixture(ctx[i], n, range(1, a.k_max_bayes + 1), beta)
            pr, _ = bayes_mixture(ctx[i], n, a.train_orders, beta)
            full.append(-np.log(max(pf[tgt[i]], 1e-300)))
            restr.append(-np.log(max(pr[tgt[i]], 1e-300)))
        results[k] = {"model": m_nll, "full": float(np.mean(full)),
                      "restricted": float(np.mean(restr)),
                      "in_train": bool(k in a.train_orders)}
        print("%8d %10s %12.4f %14.4f %12.4f%s"
              % (k, "yes" if k in a.train_orders else "NO", m_nll,
                 results[k]["restricted"], results[k]["full"],
                 "   <- beats restricted" if m_nll < results[k]["restricted"]
                 and k not in a.train_orders else ""), flush=True)

    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(a.out_dir, "model.pt"))
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(a), "history": hist, "results": results,
                   "wall_sec": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\n[markov] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
