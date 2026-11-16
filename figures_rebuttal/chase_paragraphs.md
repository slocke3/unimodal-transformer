# Rebuttal paragraphs for discussion with Chase (2026-11-15)

Short versions agreed with Simon. Figures are in `figures_rebuttal/` unless another path is given.

## pqmj: "finer tokenizations need more maps" is misleading — `taskdiv_resolution_tradeoff`

pqmj says the claim mixes up quantization error with task-diversity error. They're partly right: on absolute error, a fine model can match a coarse model's final loss with fewer tasks. But measured against each resolution's own error floor, which is what our m* does, finer resolutions do need more tasks, at every threshold. The two views reconcile because the best resolution grows with the number of training tasks. Proposed fix: define m*, reword the claim, and add one small figure (or replace Fig. 2 with the own-floor panels c and f).

## FWzs: does the model infer r, or use simpler strategies? — `baselines_taskdiv`

FWzs asks whether the model infers r or relies on simpler strategies, and suggests comparing it against a parameter-estimation baseline and a transition-based predictor. We ran both, on the same data: k-gram transition counters, and a least-squares fit of r that is given the logistic formula. Past the crossover the transformer beats both. Its error is 2.5–3.4× lower than the best counter at fine resolution, and lower than the least-squares fit at 16 and 64 bins, since that fit ignores quantization. So it does more than count transitions or plug in an estimate of r. One caveat to state openly: the counters also show a crossover, so the crossover itself reflects task coverage. What is specific to the transformer is needing fewer tasks and reaching a lower error floor.

Supporting sentence (FWzs's matched-final-state experiment): the 1-gram is the best possible last-token predictor, and the transformer beats it at every task count, so it must use the history. We also ran the experiment as proposed: past the crossover, predictions for pairs ending in the same token differ exactly as the true next states do (slope 1.00; about 0.85 at m = 100).

## FWzs: why doesn't longer context help? — `identifiability`

FWzs asks why longer context doesn't help, and whether that means the model isn't inferring the system from context. We computed exactly how much the last K tokens can tell anyone about the next state. That information levels off after 4–6 tokens at every resolution, so a flat curve from L = 10 to 75 is what the dynamics predict. Only chaotic contexts keep narrowing r with more tokens, and even that stops helping, because chaotic stretching erases what older tokens said about the current state. The model does use the history (matched-pair slope ≈ 1), and our appendix context sweep already shows the knee at L = 4.

Open issue: the new L = 2 and 5 runs were trained with dropout 0.1 (the script default used for every paper run). At L ≤ 5 this distorts eval-mode error about 18×; L ≥ 10 is unaffected. They need retraining with dropout 0 (about $5–9). Check whether the appendix L = 2 point shares the problem.

## FWzs Q1: how does state coverage help when transition rules differ? — `paper_iclr2027/family_overlap.png` (top right), `figures_asym2/conjugacy_curve.png` (left)

We agree that state coverage alone isn't enough, and our conjugate-map experiment shows this directly. Along R = 1, every map is topologically conjugate to the logistic map at r = 4. Maps outside the training band visit the same states (occupancy overlap ≥ 0.92) but share at most 0.31 of their token transitions, and the model fails on them: CE rises from about 1.2 nats in band to 2.7–9.7 nats out of band, worse than uniform guessing for most bands. So state coverage is necessary, as the occupancy-overlap law shows for the windows, but the training data must also cover the transitions. One seed. This section is in main.tex but not in the submitted version.

## 68V6: why exclude r < 0.5?

68V6 asks why we exclude r < 0.5 if everything below r = 1 converges to 0. Below r = 1 the orbit shrinks by roughly a factor of r each step, so the smaller r is, the faster the trajectory hits zero and the less it says about the map. Below 0.5 a window is essentially all zeros (97% at 64 bins). The 0.5 cutoff drops the least informative of these maps, but it is a convention: maps just above it are nearly as uninformative (96% all-zero windows at r = 0.5). Maps below 1 are predicted almost perfectly, so they only dilute the averages. Recomputing every curve with r ≥ 1 raises all errors by the same ~16% and leaves m* unchanged.

## 68V6: terminal one-step loss vs a sequential loss — `figures_taskdiv/allpos`

No paragraph yet. Key result: training with the loss on every position removes most of the crossover but raises the fine-resolution error floor about 35×.

## All three: only one system

Drop Hénon from the rebuttal (Simon, 2026-11-15). At b = 0.3 it follows the same period-doubling route to chaos as the logistic family, so replication is expected and reviewers would not read it as generality. Concede the limitation in text, and explain the choice: the logistic family gives exact dynamical ground truth (λ, invariant measure, periodic windows). A system with a different route to chaos (e.g. the sine circle map) is for the ICML version.

## Novelty (all three)

Nothing we have, or can finish by the deadline, fixes this. Answer in text; the results above add soundness and depth, not novelty.

## To decide with Chase

- Capacity ablation (68V6): status of Chase's runs. The only point with nothing in hand.
- Chase's short-context runs: which L, and what dropout?
- Replace Fig. 2 with the own-floor panels?
- How to use λ and the invariant measure in the main text (pqmj: defined but only used in the appendix).
- Restore the conjugate-map section in the revision for FWzs Q1.
- State dropout 0.1 in the methods.
- Still to write (text only): CE vs MSE reporting (pqmj), synonym-token details, why the output is fixed at 64 bins.
