# Separating a dynamical failure from the binning moving

When a model trained on one part of a map family loses accuracy on another part,
two very different things can be responsible:

1. **Dynamical failure** — the model cannot infer the new map's dynamics from
   context. This is the interesting result.
2. **Representation shift** — the dynamics may be perfectly inferable, but the
   orbit now sits somewhere else relative to the *fixed uniform partition* of
   `[0,1]`, so the token statistics are ones the model never saw. This is an
   artifact of the tokenization, not a fact about in-context learning.

A single train/test loss difference confounds the two, and simply swapping map
families does not fix it: a new family changes dynamics *and* density together.
What is needed is an intervention that moves one factor while pinning the other.

Three such controls are described below. The first is implemented; the second
and third are not yet built. Section 5 argues that the second and third are
diagnostics rather than fixes, and that the fix is an augmentation.

## Notation

The asymmetric family used by the `asymband` sweep is

```
g(x) = R (x/x_c)^alpha ((1-x)/(1-x_c))^beta,   beta = 2 - alpha,  x_c = alpha/2
```

`R` is the peak height (`max g = R`), so `R = r/4` and `alpha = 1` recover the
logistic map exactly. `alpha` moves the critical point to `x_c = alpha/2` while
keeping it quadratic, so the family stays in the logistic universality class.

Sweeps stay at `alpha <= 1`: near the origin `g ~ x^alpha`, so `g'(0) = 0` for
`alpha > 1` and the origin becomes superattracting. Above `alpha = 1` a growing
share of orbits die there, and a dead orbit is a constant token stream that is
trivially predictable — out-of-band loss would improve the more degenerate the
task became.

## 1. The R = 1 conjugacy curve (implemented)

At `R = 1` the critical orbit is `x_c -> 1 -> 0`, and the origin is repelling
for `alpha < 1`. Every map on the curve `{(R=1, alpha) : alpha <= 1}` therefore
has the same kneading sequence as the logistic map at `r = 4`, so they all share
their symbolic dynamics. Topological entropy is pinned at `ln 2` while the SRB
Lyapunov exponent slides (0.693 at `alpha=1` down to 0.622 at `alpha=0.5`) —
precisely the signature of the measure moving under a fixed topology.

Cross-entropy variation along this curve therefore **cannot** be a failure to
infer the dynamics. It is the binning moving. Since the curve is just the
`R = 1` column of the evaluation grid, this control is free.

Implemented in `scripts/analyze_conjugacy_curve.py`, run automatically by
`scripts/make_conjugacy_figure.slurm`. The model-free half of the story --
the densities, the transition matrices and the two overlap curves -- is
plotted by `scripts/plot_family_overlap.py` (`figures_asym/family_overlap.png`),
which needs no checkpoint and can be regenerated at any bin count.

One measurement from that script is worth recording up front, because it decides
which statistic to trust. Comparing each `alpha` against `alpha = 1` at `R = 1`:

| alpha | marginal overlap | transition overlap |
|-------|------------------|--------------------|
| 0.5   | 0.959            | **0.075**          |
| 0.7   | 0.978            | 0.106              |
| 0.9   | 0.992            | 0.220              |

(at the sweep's `n_bins = 64`; a coarser partition raises both numbers, so
always quote the bin count with them. Stable to three decimals from 50k to 1M
orbit steps and across seeds, with no orbit death.)

The *marginal* occupancy barely moves — both maps are full-chaos with density
piling up at the endpoints — but the *transition* distribution moves enormously.
At `alpha = 0.5` more than 92% of the transition mass lands in (bin, next bin)
cells that essentially never occur at `alpha = 1`, even though 96% of the
1-point occupancy is shared. For a next-token task the
transition statistic is the relevant one, so marginal histogram overlap badly
understates how far the representation has shifted. Any future diagnostic that
uses occupancy overlap as a proxy should use the order-1 joint, not the
marginal.

## 2. Peak-anchored binning (not built)

**Idea.** Replace the uniform partition with one that moves with the peak: map
`[0, x_c] -> [0, 1/2]` and `[x_c, 1] -> [1/2, 1]` piecewise-linearly, then bin
uniformly in those coordinates. The critical point then always falls on the same
token boundary, and the peak shift is absorbed by construction.

**What it tests.** Train on logistic and test across `alpha` under this
tokenization. If generalization recovers, the failure was the partition sliding
out from under the model. If it does not, the failure is dynamical.

**Cost.** A change to `tokenize_trajectory` alone — no invariant measure to
estimate, no new training machinery. This is the cheapest of the three and
should be tried before the decomposition in section 3.

**Caveat.** The rescaling uses `x_c = alpha/2`, i.e. knowledge of the test map.
That is fine for a diagnostic — it answers "if the representation were aligned,
would it generalize?" — but it is oracle preprocessing and must not be described
as the model solving the problem itself.

## 3. Density-matched surrogates (not built)

**Idea.** Applying a monotone homeomorphism `h` to the orbits of `f` produces
orbits of the conjugate map `h . f . h^-1`: identical dynamics (same kneading,
same entropy), different invariant density. This is exactly the mechanism
already used by `src/graded_transfer.py`, where `h_eps` slides the density from
arcsine to uniform while holding logistic dynamics fixed.

Generalize `h_eps` to `h_alpha = F_alpha^-1 . F_logistic`, where `F` denotes the
CDF of the relevant invariant measure. Then, at matched `R`, evaluate three sets
of sequences:

| sequences                          | dynamics | density  | CE |
|------------------------------------|----------|----------|----|
| logistic orbits                    | logistic | logistic | C  |
| `h_alpha`-warped logistic orbits   | logistic | alpha's  | B  |
| true `alpha`-map orbits            | alpha's  | alpha's  | A  |

`B - C` is the pure representation effect and `A - B` the pure dynamical
effect, and the two add up to the total that a naive train/test comparison
reports as a single number.

**Cost.** `warp_h`, `_ce_on_orbits` and `_windows_from_orbits` already exist in
`src/graded_transfer.py`. The new piece is estimating `F_alpha` by histogramming
a long orbit and inverting it.

**Measured, and it does not work.** Warping logistic `r=4` orbits by the
density-matching map and comparing the result's transition distribution against
the real `g_alpha` at `R=1` moves the overlap from 0.075 only to 0.110 — the
surrogate does *not* reproduce the target's dynamics.

The reason is worth stating, because it invalidates the naive version of this
control. The conjugacy to the tent map pushes forward the **measure of maximal
entropy**, not the physical (SRB) measure. Those coincide for logistic `r=4` —
which is precisely why `lambda = ln 2` there — but nowhere else on the curve:
`lambda_SRB` is 0.689 at `alpha=0.9` and 0.622 at `alpha=0.5`, all against
`h_top = ln 2`. So the CDF of the invariant density is *not* the conjugacy, and
matching densities leaves the dynamics unmatched.

**Use the topological conjugacy instead.** Build `h` from the itinerary: take
the Gray-coded binary expansion of the symbol sequence relative to the critical
point, which is the standard conjugacy to the full tent map. It satisfies
`h(x_c) = 0.5` exactly for every `alpha` (verified numerically), as any
conjugacy must, and it works:

| alpha | raw transition overlap | after conjugacy warp |
|-------|------------------------|----------------------|
| 0.9   | 0.220                  | **0.922**            |
| 0.7   | 0.106                  | **0.767**            |
| 0.5   | 0.076                  | **0.616**            |

The residual shortfall from 1.0 is finite grid, finite expansion depth and
float precision in chaotic orbits, not a failure of the construction. The
practical consequence: `B` must be produced with the itinerary conjugacy, and
the surrogate then really does hold the dynamics fixed.

## 4. The full 2x2, if a definitive answer is wanted

Sections 2 and 3 each move one factor. Combining them gives a factorial design
in which both main effects and their interaction are identifiable:

|                       | matched density            | shifted density            |
|-----------------------|----------------------------|----------------------------|
| **matched dynamics**  | logistic (baseline)        | `h_alpha`-warped logistic  |
| **shifted dynamics**  | density-matched alpha-map  | raw alpha-map              |

The bottom-left cell is the one that never arises by accident: conjugate the
`alpha`-map by `F_logistic^-1 . F_alpha` so its density matches logistic while
its kneading stays its own. Only with all four cells does "switching families
for train/test" become an identifying design rather than another confounded
comparison.

## 5. Why the diagnostics are not fixes, and what the fix would be

Sections 2 and 3 both hand the model information about the test map — the peak
position `x_c` in one case, the target measure in the other. They can answer
"would it generalize if the representation were aligned?" but never "can it
learn to align the representation itself?" For that the model has to be *taught
an invariance*, and the honest way to choose one is to pick a nuisance
transformation you would assert anyway, not one reverse-engineered from the
shift you are trying to beat.

The defensible assertion here is that **the coordinate on the interval is
arbitrary**. Nothing in the dynamics privileges uniform `x`; the uniform
partition is a modelling convenience. Any monotone reparameterization `h` turns
orbits of `f` into orbits of `h . f . h^-1`, which has identical symbolic
dynamics and an arbitrarily different appearance under a fixed grid.

The measurement in section 3 shows this is not merely a nice principle: the
`alpha` family at `R = 1` *is* reachable from logistic by such a warp (overlap
0.076 -> 0.616 under the itinerary conjugacy). So a model trained with random
monotone recoordinatizations is being trained on a distribution that already
contains the asymmetric family, without anyone having looked at it. That is the
difference between an augmentation and a patch.

Candidate augmentations, most to least principled:

1. **Random monotone warps.** Draw `h` per sequence from a generic family of
   monotone maps fixing 0 and 1 (Beta CDFs, random-knot monotone splines) and
   tokenize `h(x)`. `src/graded_transfer.py` already sweeps a one-parameter
   version of exactly this (`h_eps`); the augmentation randomizes it.
2. **Multi-family training.** Train on the existing `FAMILIES` registry (tent,
   sine, cubic) rather than logistic alone. Motivated by "we want a model of
   unimodal dynamics, not of one map", and requires no knowledge of `g_alpha`.
3. **Reflection.** Train on `1 - x` as well as `x`. Legitimate: if `x_n` is an
   orbit of `f` then `1 - x_n` is an orbit of `sigma . f . sigma`, a genuine
   conjugate. Cheap, and motivated by "left and right are arbitrary labels".
4. **Grid dither and time subsampling.** Random sub-bin offsets of the
   partition, and training on `f^k` for random small `k`. Motivated by "the
   discretization origin and the sampling rate are arbitrary".

Expected costs, worth stating before running: in-band loss should *rise*,
because the model can no longer memorize one token-transition table and must
infer the coordinate from context. That also makes context length a binding
constraint — this becomes the in-context system-identification problem that
`src/system_id.py` already studies, so the natural companion measurement is
performance against context length `L`.
