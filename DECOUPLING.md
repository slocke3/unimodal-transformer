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
and third are not yet built.

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
`scripts/make_conjugacy_figure.slurm`.

One measurement from that script is worth recording up front, because it decides
which statistic to trust. Comparing each `alpha` against `alpha = 1` at `R = 1`:

| alpha | marginal overlap | transition overlap |
|-------|------------------|--------------------|
| 0.5   | 0.958            | **0.169**          |
| 0.7   | 0.978            | 0.287              |
| 0.9   | 0.994            | 0.641              |

The *marginal* occupancy barely moves — both maps are full-chaos with density
piling up at the endpoints — but the *transition* distribution moves enormously.
At `alpha = 0.5` more than 80% of the transition mass lands in (bin, next bin)
cells that essentially never occur at `alpha = 1`. For a next-token task the
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

**Caveat.** This matches the one-point density, not the transition structure —
and section 1 shows those diverge sharply here. So `B - C` captures only the
marginal part of the representation effect, and some of the rest leaks into
`A - B`. Matching the order-1 joint instead would be stronger but is harder,
since no scalar coordinate change can generally align a two-dimensional
transition distribution.

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
