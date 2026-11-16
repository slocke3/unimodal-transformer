# Restricted-window transfer follow-up

Status: original fixed-step CE results checked; replication jobs prepared, **not
launched**. The saved Runpod SSH endpoint refused the connection. The connected
API returned no pods and no organization, so it did not provide access to the
team machines. No new resources were provisioned.

## What the current results establish

The analysis uses the existing 53 windows and 14,658 out-of-support pairs. The
original CSV's penalties were checked numerically against `runs_win_fix_ce/`;
these are the final 160k-step results, not the earlier early-stopping sweep.

| Association with transfer penalty | Manhattan support distance | Euclidean support distance |
|---|---:|---:|
| Pooled Spearman | 0.439 | 0.411 |
| Partial: parameter distance and categorical width | 0.562 | 0.551 |
| Also control context-state occupancy JS | 0.276 | 0.278 |
| Also control target identity | 0.162 | 0.163 |
| Control target and window identity, state JS, and parameter distance | 0.010 | 0.005 |

Partial correlations residualize ranks; categorical effects are dummy-variable
effects, not ranks of category labels. The final row removes every window's
average rank effect and every target's average rank effect, leaving their
interaction. It answers a stricter, different question from the pooled result.
It also removes between-window variation that could itself reflect coverage.

The window-mean support/penalty correlation is **0.585**. Within each window,
the median support/penalty Spearman correlation across targets is only **0.120**
(IQR -0.096 to 0.230; positive in 33/53 windows). Context-state JS has median
within-window correlation 0.433. These are descriptive summaries, not estimates
from 14,658 independent experimental units. Windows and targets are shared.

**Interpretation:** transition mismatch is associated with differences in broad
transfer ability across training intervals. It is a weak local predictor of
which particular targets a fixed model predicts well. The data do not establish
that broader transition support is the mechanism causing better transfer.
The additional controls are useful qualifications, not proof that transitions
are irrelevant. See `transition_controls.png`, `pair_controls.csv`,
`within_window.csv`, and `summary.txt` for the full results.

## Why state coverage is insufficient

For a current state x, the logistic map predicts r*x*(1-x). At x=1/2, for
example, r=3 predicts 3/4 and r=4 predicts 1. A shared observed state does not
specify its continuation; the history may provide evidence about r.

More generally, state occupancy p(i) is a marginal, whereas prediction requires
a conditional distribution P(j | history). Two symmetric binary Markov chains
can both have stationary occupancy (1/2,1/2), while one stays in its current
state with probability .9 and the other stays with probability .1. Their state
distributions and even their transition SUPPORTS are identical, but their
conditional predictions differ. Thus:

- Occupancy overlap measures availability of observed states.
- Transition-support overlap also checks which continuations are represented.
- Matching support still does not imply matching transition probabilities or
  the conditional law given a longer token history.

The measured support distance is the minimum over individual source-grid maps
of the target-mass-weighted distance to that map's transition support. It is
**not** distance to the union of all transitions in a training interval. This
is a finite-trajectory diagnostic, not an identified learned algorithm.

Suggested response wording:

> We agree that state coverage alone does not establish predictive
> compatibility: systems visiting the same states can require different
> continuations. Our transition analysis compares observed state-to-state
> transitions rather than state occupancy alone. Transition-support mismatch
> remains associated with transfer penalty after adjusting for occupancy
> mismatch, parameter distance, and window width (partial rank correlation
> 0.28). However, the association is primarily between training windows and
> becomes negligible after also removing window and target effects. We
> therefore interpret these statistics as descriptive evidence about broad
> transfer patterns, rather than a causal account of individual predictions.

## Stage 1: replicate the original intervals

`jobs_rb_window_replicate.txt`: **32 new CE runs**, seeds 1 and 2 for 16 original
intervals. Combined with the existing seed 0, each selected interval has three
seeds. All eight original positive-chaotic-coverage intervals are included;
the controls are eight preceding regular intervals matched by width. Selection
uses geometry and the existing Lyapunov classification, never transfer loss.

| Width | Positive-chaotic-coverage starts | Regular-control starts |
|---|---|---|
| 1 | 3 | 2.1667 |
| .5 | 3.5 | 3 |
| .25 | 3.5, 3.75 | 3, 3.25 |
| .125 | 3.5, 3.625, 3.75, 3.875 | 3, 3.125, 3.25, 3.375 |

Protocol matches the original experiment: 8,000 total trajectories (600
validation, 7,400 optimization), 64-bin input/output, L=50, trajectory length
150, no burn-in, dropout .1, final checkpoint at 160k steps. Training-parameter
counts remain 2286/1143/571/286 for widths 1/.5/.25/.125. The original DataLoader
is the default. A seed changes the parameter draw, initial conditions, model
initialization, minibatch stream and evaluation draws; this is a full-pipeline
replication, not an initialization-only ablation.

This stage tests reproducibility at existing supports. It does **not** increase
the number of distinct chaotic windows or isolate chaos from parameter location.
The width-1 regular control extends beyond r=3 and overlaps its chaotic
comparison's support; it is regular according to Lyapunov coverage, not a
fixed-point-only control. These are comparisons, not randomized interventions
on chaos holding everything else fixed.

The analysis preserves the original same-width/same-target reference, so new
seeds cannot change old penalties. It averages seeds within intervals before
computing correlations and saves individual run summaries and seed ranges.
`common_regular_targets.csv` additionally compares these 16 intervals on the
**same** targets r < 2.1667, all out of support for every selected interval,
avoiding the changing target mix of each window's full complement. This
restricted comparison addresses transfer to that common regular region only.

## Stage 2: sample additional intervals, if useful after replication

`jobs_rb_window_expand.txt`: **33 optional CE runs**, 11 new intervals with
seeds 0/1/2. Width .125 starts at 3.0625, 3.1875, ..., 3.8125; width .25 starts
at 3.125, 3.375, 3.625; width .5 starts at 3.25. These fill half-step offsets
across [3,4], including regular, mixed, and chaotic supports. They were selected
without model results. They are additional overlapping supports, not independent
dynamical families. This stage is prepared separately and has not been launched.

## Commands

From the repository root, with the project's Python environment:

```bash
python scripts/rebuttal/make_window_replication_jobs.py
python scripts/rebuttal/test_window_transfer_controls.py
python scripts/rebuttal/window_transfer_controls.py
```

On a reachable team GPU pod, sync code and benchmark **one replication job**
before choosing concurrency or quoting cost. The benchmark writes to `bench/`,
not the final run directories. Actual wall time/cost has not been measured for
this sweep on a current pod.

```bash
python scripts/runpod/bench_pod.py jobs_rb_window_replicate.txt --steps 2000 --conc 1 4
python scripts/runpod/run_jobs.py jobs_rb_window_replicate.txt --per_gpu 4
```

The `4` in the launch example is provisional; use the measured concurrency.
After pulling completed results, rerun:

```bash
python scripts/rebuttal/window_transfer_controls.py \
  --additional-runs runs_win_rebuttal_replicate_ce \
  --output-dir figures_rebuttal/window_transfer_replicated
```

If stage 2 is run, supply both directories after `--additional-runs` and use a
new output directory. Do not overwrite the original manuscript figures.
