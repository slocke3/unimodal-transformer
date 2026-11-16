# Rebuttal figures (ICLR 2027, Nov 2026)

Each figure has a `*_summary.txt` with the numbers behind it. Scripts are in
`scripts/rebuttal/`. All of them run without torch except `probe_matched_final.py`.

| Figure | Reviewer question | Script |
|---|---|---|
| `taskdiv_resolution_tradeoff` | pqmj: "finer tokenizations need more maps" mixes quantization error with task-diversity error | `taskdiv_resolution_tradeoff.py` |
| `baselines_taskdiv`, `baselines_taskdiv_all` | FWzs: parameter-estimation and transition-based baselines | `baselines_taskdiv.py`, then `plot_baselines_taskdiv.py` |
| `identifiability` | FWzs: why context length doesn't help; pqmj: connect results to the dynamics | `identifiability.py` |
| `context_short` | FWzs: L in {1, 2, 5} | `make_jobs.py ctxshort` → RunPod → `plot_context_short.py` |
| `window_transfer/transition_controls` | FWzs: state coverage versus predictive compatibility; transfer robustness | `window_transfer_controls.py`; details and replication status in [`window_transfer/README.md`](window_transfer/README.md) |

Existing figures that answer 68V6's design questions (not remade):
- `figures_taskdiv/noise_vs_bins.png`: why tokenize instead of using real-valued input. With uncorrupted continuous input the error is ~5e-7. Noise the size of a bin beats quantization at every resolution, because noise averages out across the context and quantization does not.
- `figures_taskdiv/allpos.png`: last-position vs every-position training. Training on every position removes most of the crossover but costs accuracy at fine resolution (256 bins: about 6e-5 vs 1.7e-6 new-task MSE).

## Findings so far

**Resolution vs task diversity (pqmj).** Both readings are true, and the curves reconcile them: **the best input resolution grows with the number of training tasks.**

| m | best N, CE | best N, MSE |
|---|---|---|
| 100 | 64 | 16 |
| 250–500 | 64 | 32 |
| 1000 | 64 | 64 |
| 2000 | 128 | 128 |
| 4000–8000 | 256 | 256 |

- Relative to each resolution's own plateau (the paper's m*), finer needs more tasks: CE m* rises from 184 (8 bins) to 677 (256 bins), MSE from 182 to 723.
- At a common absolute error, pqmj is right about the coarse arms: 256 bins reach the 8-bin model's final loss (1.6 nats) at m≈315. Among the fine resolutions, 64 bins still reach any shared target first (CE 0.1 nats: m≈830 at 64 bins, 940 at 128, 1700 at 256).

**Baselines (FWzs).**
- Counting predictors use the last 1, 2, 4, 6, or 10 input tokens, fitted to the
  same training examples. Unseen histories back off to the longest observed
  suffix. CE smoothing is chosen on validation data separately for each order;
  MSE predictions use the empirical mean continuous target. To extend saved
  results, run `python scripts/rebuttal/baselines_taskdiv.py --ks 10 --workers 3`,
  then regenerate the figures with `plot_baselines_taskdiv.py` (use
  `--ns 8 16 32 64 128 256 --out figures_rebuttal/baselines_taskdiv_all.pdf`
  for the full-resolution figure).
- Ten-token histories use exact two-word keys where needed to avoid integer
  overflow, and joint history/output counts use dense history IDs. The 5-gram
  results remain saved but are no longer included in the figures.
- The figures show the transformer and k-gram predictors. The least-squares
  parameter-estimation line is omitted; its results remain in the numerical
  summaries and saved baseline data.
- **Longer-history extension (four-seed medians, m=8000).** At 64 input bins,
  moving from 4 to 6 tokens lowers CE from 0.162 to 0.147 and MSE from
  2.22e-5 to 1.98e-5 (transformer: 0.0642 and 1.69e-5). At 256 bins, CE
  increases from 0.104 to 0.121, while MSE improves slightly from 3.32e-6 to
  3.27e-6 (transformer: 0.0302 and 1.67e-6). Longer histories therefore do
  not uniformly improve these finite-data counting predictors.
- **Ten-token extension (four-seed medians, m=8000).** At 8 and 16 bins, the
  10-gram's MSE is about 7--8% above the transformer's. At 64 bins, moving
  from 6 to 10 tokens changes MSE from 1.98e-5 to 2.00e-5 and increases CE
  from 0.147 to 0.198 (transformer: 1.69e-5 and 0.0642). At 256 bins, MSE
  remains about twice the transformer's (3.29e-6 versus 1.67e-6), while CE
  increases to 0.134 (transformer: 0.0302). Longer histories do not eliminate
  the gap, and ten tokens are often worse than six under this counting and
  validation-selected smoothing procedure.
- The 1-gram (last token only) stays far above the transformer at every m, so history matters.
- k-gram counters also show a task-diversity crossover (64 bins, 4-gram CE: 3.9 at m=100 → 0.22 at m=1000). The crossover is therefore partly a coverage effect, not something only in-context inference produces.
- The transformer reaches a given error with fewer tasks and plateaus lower: at m=8000 the 4-gram's CE is 2.5× the transformer's at 64 bins and 3.4× at 256. At coarse resolution (16 bins) the 4-gram is within about 10%.
- Least-squares r̂ (knows the family, no training tasks) is a real bar under square loss. The transformer beats it at 16 and 64 bins once past the crossover and matches it at 256 bins for m ≥ 4000. Before the crossover r̂ wins (64 bins, m=500: r̂ 3.9e-5 vs transformer 6.1e-4).

**Identifiability (FWzs, pqmj).** Exact set-membership computation of what the last K tokens allow, given the known family (1500 training-like contexts).
- The range of next states consistent with the context levels off after about 4–6 tokens at every resolution:

  | Input bins | K=1 | K=2 | K=4 | K=10 | r known |
  |---|---|---|---|---|---|
  | 16 | 0.83 | 0.12 | 0.065 | 0.063 | 0.040 |
  | 64 | 0.81 | 0.030 | 0.016 | 0.016 | 0.010 |
  | 256 | 0.79 | 0.0077 | 0.0040 | 0.0039 | 0.0026 |

- Only chaotic contexts keep narrowing r as K grows (64 bins: consistent r-set length 0.070 at K=4 → 0.015 at K=10). Fixed-point and periodic contexts level off by K≈4–6.
- Extra r information past K≈6 doesn't narrow the next state. The remaining bottleneck is where the current state sits within its bin, and expanding dynamics erase what past observations said about it.
- Prediction for the L = 1, 2, 5 runs: most of the gain by L≈5, flat beyond. This is consistent with the paper's flat L = 10–75 curve.
- Dropped: a per-context comparison of the r-set shrink rate with the Lyapunov exponent. The per-context rates didn't track λ (correlation −0.03), even though their median was 0.96 λ, so no λ claim is made.

**Hénon pilot (`henon_pilot`, `plot_henon_pilot.py`; all reviewers: "a single family").** Hénon map with b = 0.3, a in [0, 1.4], only x observed. Same recipe as the logistic sweep (CE, 64 output bins, seed 0), m in {100, 500, 2000, 8000}, 32 and 128 input bins (`runs_henon`, RunPod).
- The crossover replicates: new-task CE falls with m and meets training-task CE by m≈2000 at both resolutions (128 bins: 0.73 → 0.12 → 0.071 → 0.064; training tasks 0.06–0.07).
- Below the crossover the penalty is much milder than in the logistic family (128 bins, m=100: 0.73 vs 3.8 nats).
- The plateau at 128 bins sits above the logistic one (0.064 vs 0.042). This is a mix effect, not a harder task. The chaotic band plateaus by m≈500 at 0.25 nats, and the logistic chaotic band at 128 bins is about the same (0.33). But chaos covers 24% of the Hénon a-range against 12% of the logistic r-range. Fixed-point and period-doubling regimes keep improving to 0.004–0.006. The hidden y is not the bottleneck: x_{n+1} = 1 − a x_n² + 0.3 x_{n−1}, so two tokens and a fix the next state up to quantization.
- Resolution shift is weak here. At m=100, 128 bins beats 32 bins outright (0.73 vs 1.35), unlike the logistic family (3.8 vs 2.1). Relative to each resolution's own plateau, finer is still further away (11× vs 3.2×), but the m* estimates barely differ (≈300 vs ≈260). One seed and four m values, so this is a pilot.

**Matched final tokens (FWzs; `probe_matched_final.py` → `runs_rebuttal_probe/matched_final.csv`).** Six 64-input-bin models at L=50 (m in {100, 1000, 8000}, CE and MSE, seed 0, retrained on RunPod as `runs_probe`). 200k fresh new-task contexts each, run on CPU (MPS gives wrong outputs). Pairs: contexts with the same final token whose final states differ by under 1/4 of a bin, and whose true next states differ by more than one output bin (1/64); 58k pairs. A last-token-only predictor must give both contexts in a pair the same output (slope 0).

| model | slope, all | corr | slope, periodic | slope, chaotic | within-bin R², periodic / chaotic |
|---|---|---|---|---|---|
| MSE, m=100 | 0.88 | 0.95 | 0.95 | 1.00 | 0.70 / 0.87 |
| MSE, m=1000 | 1.00 | 1.00 | 1.00 | 1.01 | 0.96 / 0.92 |
| MSE, m=8000 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 / 0.93 |
| CE, m=100 | 0.84 | 0.91 | 1.60 (corr 0.72) | 1.05 | −35 / −0.03 |
| CE, m=1000 | 1.00 | 1.00 | 1.01 | 1.00 | 0.92 / 0.90 |
| CE, m=8000 | 1.00 | 1.00 | 1.01 | 1.00 | 0.93 / 0.93 |

- Past the crossover, the model's prediction moves one-for-one with the true next state between contexts that end in the same token, in every regime. Within a regime it removes 90–97% of the squared error left by the best possible last-token predictor. That predictor is an oracle fit to these same contexts; pooled over regimes the figure is 99.8%, but that pooled baseline is weaker.
- Fixed-point regime: a last-token predictor is already nearly exact there (next state ≈ current state once converged), so R² is uninformative. The pair slope is still 1.02–1.03.
- Below the crossover (m=100) history is still used: slope 0.84–0.88. The error is concentrated at r in [1, 2) (MSE model: 5.9e-2 at [1, 1.5) vs ≤1e-4 for r > 2). There the fixed point 1 − 1/r moves fastest with r, so 100 training tasks are several output bins apart, and convergence is slow, so contexts are mostly transients. The CE model at m=100 also struggles in 3 < r < 3.57 (overreacting, slope 1.6), where the square-loss model does not.

**Short contexts (`context_short`, runs_ctx_short): L = 2 and L = 5 are NOT usable as trained (2026-11-14).** With dropout 0.1, these models fit their training data (L=2 train loss 5.3e-5, at the 2-gram counter's level of 5e-5). But with dropout switched off for evaluation, their error is 18–20× higher (L=2: 1.1e-3 eval vs 6.2e-5 with dropout on; L=5: 4.2e-4 vs 2.2e-5). It's the same on their own training windows, so this is not overfitting: the network has adapted to dropout's noise, which a 1–5-token attention never averages out. L=10 is unaffected (1.78e-5 vs 1.88e-5), as are L=1 (matches the 1-gram) and the L=10 RunPod checks (match the original runs). Fix: retrain L = 2, 5 with dropout 0, plus L=10 dropout-0 controls. The appendix context sweep (`figsweep`, "only L=2 is deficient") may have the same problem; check its dropout before citing it.
