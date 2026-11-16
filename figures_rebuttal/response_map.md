# Rebuttal response map (ICLR 2027, submission 51913)

For each reviewer point: what we have, which figure carries it, and the status.
Figures are in this folder unless another path is given. The numbers behind them are in `README.md` and the `*_summary.txt` files.

Status key: **answered** = result in hand · **partial** = result in hand but incomplete · **running** = waiting on compute · **deferred** = planned, not started · **text** = needs writing only

## Shared concern: novelty (all three reviewers)

Decision (2026-11-14): no new analysis will fix this before the deadline. Answer it in text; the items below support soundness and depth, not novelty.

All three say the crossover is already known from Raventós et al. (2023) and the setting is too simple to add much. New results we can point to:

1. **Resolution and diversity reconciled.** The best input resolution grows with the number of training tasks (64 bins best at m ≤ 1000, 256 bins at m ≥ 4000). This answers pqmj's objection and replaces a loosely worded claim with a precise one. → `taskdiv_resolution_tradeoff`
2. **The crossover is partly a coverage effect.** Count-based k-gram predictors also show a crossover, so its existence comes from the data. What is specific to the transformer is that it reaches a given error with fewer tasks and plateaus 2.5–3.4× lower (64 and 256 bins). → `baselines_taskdiv`
3. **The flat context curve has a dynamical explanation.** An exact set-membership calculation shows that the last 4–6 tokens determine everything the context can say about the next state, at every resolution. More context keeps narrowing r only in the chaotic band, and there it no longer narrows the next state, because the expanding dynamics erase within-bin information. → `identifiability`
4. **The model infers the map from history.** In matched-final-token pairs, the prediction tracks the true next state one-for-one past the crossover. → probe table (no figure yet)
5. ~~A second family (Hénon)~~ dropped 2026-11-15: same period-doubling route to chaos, so replication is not evidence of generality. Concede "one system" in text.

Positioning sentence to develop: in standard ICL regression the input distribution is the same for every task. Here the dynamics set the input distribution, so tasks differ in which states they visit as well as in their rule. That is what produces the coverage effects (item 2, the asymmetric window transfer) and the identifiability limits (item 3).

## pqmj (weak reject; informativeness 1)

| Point | Response | Figure | Status |
|---|---|---|---|
| How is m* measured (line 330)? | Define it: half the log-scale drop from the m=100 error to the plateau (mean of m = 4000 and 8000), interpolated log-log | — | text |
| Synonym tokens: how is the ambiguity handled? | Input side only. Each occurrence of a bin draws one of its j synonyms uniformly at random, so the bin is token // j. Training contexts are drawn once when the dataset is built; evaluation contexts are drawn fresh. Targets and loss stay on the original 64 output bins | — | text |
| CE vs MSE reported inconsistently | Task diversity: both losses already shown. Window transfer: move the MSE heatmap into the main text and discuss the difference. Context length: MSE only (the new short-context runs are MSE too) | `figures_window/window_fixed_mse` | partial; a CE context arm is optional |
| Results not connected to the dynamics | Identifiability analysis by regime. Probe split by regime | `identifiability` | answered |
| Only one dynamical system | Concede in text; explain the choice (exact dynamical ground truth). Hénon dropped: same route to chaos | — | text |
| Findings incremental | Shared novelty response | — | text |
| Is the 64-bin output fixed? Is that standard? | Yes, fixed at 64 so CE is comparable across input resolutions. Chronos-style models also quantize into a fixed vocabulary. MSE: tokenized input, scalar output | — | text |
| r=3 transfer: why only windows starting at 3? Why not under MSE? | Our check: the overlap formula also predicts transfer for windows starting above 3, and the wider windows pass 1+√6, where period 2 is unstable. So narrow the claim. The repeated-token and same-scorecard checks would say more | — | deferred (item 5) |
| "Finer tokenizations need more maps" is misleading | Agree in part. At a common absolute error, finer bins win against the coarse arms. Relative to each resolution's own plateau, finer needs more tasks. The best resolution grows with m. Reword the claim | `taskdiv_resolution_tradeoff` | answered |

## FWzs (weak reject; soundness 2)

| Point | Response | Figure | Status |
|---|---|---|---|
| Limited novelty | Shared novelty response | — | text |
| Is the threshold governed by general properties of the dynamics? | No second-family evidence (Hénon dropped). At m=100 the error concentrates at r in [1, 2), where the fixed point moves fastest with r (a lead toward a dynamical measure of task distance; one seed) | `henon_pilot` | partial |
| Higher dimensions, multiple parameters, noisy observations? | Concede higher dimensions in text. Noise: existing noise-vs-bins runs | `figures_taskdiv/noise_vs_bins` | partial |
| Why doesn't context length help? | The identifiability analysis predicts saturation at 4–6 tokens. The L = 1, 2, 5 runs test this directly | `identifiability`, `context_short` | running |
| Has L ∈ {1, 2, 5} been tried? | 99 runs (11 m values, 3 seeds) plus 3 L=10 checks | `context_short` | running |
| How does state coverage help when transition rules differ? | It doesn't on its own, and our conjugate-map experiment shows this directly. Along R = 1, every map is topologically conjugate to the logistic map at r = 4. Out-of-band maps visit the same states (occupancy overlap ≥ 0.92) but share at most 0.31 of their token transitions (mostly ≤ 0.15). CE jumps from ≈ 1.2 nats in band to 2.7–9.7 nats out of band (uniform guessing is 4.16). State coverage is necessary (occupancy-overlap law for the windows, ρ ≈ −0.5); the transitions must be covered too | `paper_iclr2027/family_overlap.png` (top right), `figures_asym2/conjugacy_curve.png` (left) | answered; one seed; a combined two-panel figure would help |
| What algorithm does the model use: infers r, or local transitions? | It is not last-token or short-memory: it beats the 1-gram at every m and the 4-gram past the crossover. It beats least-squares r̂ at 16 and 64 bins past the crossover. Matched-pair slope 1.00 | `baselines_taskdiv` | answered; the Bayes comparison would add more (item 6, deferred) |
| Suggested baselines: parameter estimation, transition predictor | Least-squares r̂ and k-grams with k = 1, 2, 4 | `baselines_taskdiv` | answered |
| Suggested experiment: matched final states | Done; slope 0.84–0.88 at m=100, 1.00 at m ≥ 1000, both losses, all regimes | table in `README.md` | answered; figure optional |

## 68V6 (weak reject)

| Point | Response | Figure | Status |
|---|---|---|---|
| Novelty; setting too simple | Shared novelty response | — | text |
| No architecture or capacity ablation | Capacity sweep | — | deferred; check Chase's runs first |
| Why exclude r < 0.5? | For r < 0.5, x at least halves every step, so it falls into bin 0 (64 bins) within 6 steps. Over 90% of windows are then all zeros, followed by a zero. These "tasks" can't be told apart and add nothing to diversity. For 0.5 ≤ r < 1 the decay is slow enough that transients stay visible. The cutoff is a convention; offer a control run | — | text |
| Why tokenize instead of using real values? | With noise-free real values, two observations give r exactly (r = x′ / (x(1−x))), and the continuous-input model reaches about 5e-7 error. Tokenization models finite measurement precision, which is the object of study. Against the smoothing/regularization reading: noise of one bin width on continuous input does better than binning at every resolution | `figures_taskdiv/noise_vs_bins` | answered |
| Terminal one-step loss vs a sequential loss | Training on every position removes most of the crossover but costs accuracy at fine resolution (256 bins: 6e-5 vs 1.7e-6). In chaos, multi-step rollouts diverge at rate λ, so one-step error is the clean quantity | `figures_taskdiv/allpos` | answered (rollouts not evaluated) |
| Isn't this just ICL regression with y = f(x)? | (a) The dynamics set the input distribution per task, which is what drives the coverage effects and asymmetric transfer. (b) Quantization: the transformer beats least-squares r̂ on bin midpoints at 16 and 64 bins past the crossover. (c) Chaos limits identifiability | `baselines_taskdiv`, `identifiability` | text with figures |

## Figures to use, by priority

1. `taskdiv_resolution_tradeoff`: pqmj's main objection.
2. `baselines_taskdiv`: FWzs's baselines and algorithm question; 68V6 on regression.
3. `identifiability` + `context_short` (pending): FWzs on context length; pqmj on dynamics.
4. ~~`henon_pilot`~~ dropped.
5. Existing figures to cite: `figures_taskdiv/noise_vs_bins`, `figures_taskdiv/allpos`, `figures_window/window_fixed_mse`.
6. Matched-pair probe: a table may be enough.

## Gaps

- **Capacity ablation (68V6).** The only point with nothing in hand.
- **The r=3 mechanism (pqmj).** Narrow the claim now; checks deferred.
- **Only one system.** Hénon dropped (same route to chaos); concede in text.
