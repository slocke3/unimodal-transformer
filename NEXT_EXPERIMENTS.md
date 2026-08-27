# Next experiments: out-of-task-distribution generalization in the map family

Working notes, ordered by how much they would change our mind. Framed against
Goddard, Smith, Ngampruetikorn & Schwab, *When can in-context learning
generalize out of task distribution?* (ICML 2025, arXiv:2506.05574), whose
design this project is the dynamical-systems analogue of.

## The mapping between the two settings

| that paper | here |
|------------|------|
| task = weight vector `w` on `S^(d-1)` | task = a map, i.e. a pair `(R, alpha)` |
| pretraining tasks from a cap of half-angle `phi` | training band `alpha ~ U[1-w, 1]` |
| task diversity = `phi`, an angular *width* | band half-width `w` |
| number of pretraining tasks `N` | number of distinct `(R, alpha)` pairs |
| OOD = bands at angle `delta`, out to 175 deg | evaluation `alpha` below the band edge |
| generalized solution resembles OLS | "infer the return map from context, apply it" |

The essential structural difference: their generalized solution is a *closed-form,
task-independent algorithm* (least squares) that is expressible in the model's
native continuous representation. Ours would be nonparametric — estimate a
function on `[0,1]` from a binned context — so it is not obvious the analogous
solution is even reachable. That is the interesting question, not a defect.

## 1. We never reached the diversity threshold (highest priority)

Their transition sits at `phi ~ 120 deg` out of 180, i.e. the training cap must
span roughly **two thirds of the whole task space** before the model switches
from the specialized to the generalized solution.

Our `asymband` sweep never came close. The usable parameter range is
`alpha in [0.15, 1.0]` — above 1 the origin is superattracting and orbits die,
below ~0.15 collapse starts appearing — so the span is 0.85. The widest band we
ran, `w = 0.4`, covers `0.4 / 0.85 = 47%` of it. That is well inside the regime
where their paper predicts a specialized solution, and "every model fails just
past its own band edge" is exactly what a specialized solution looks like.

Our own numbers already hint at an approaching transition. The ratio of
out-of-band to in-band cross-entropy is flat and then falls hard:

| w | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 |
|---|---|------|-----|------|-----|------|-----|-----|
| ratio | 22.7 | 17.1 | 19.1 | 18.6 | 16.5 | 13.8 | 10.8 | **5.0** |

Extrapolating the collapse suggests parity somewhere near 60-70% coverage —
close to where their threshold sits.

**Experiment.** Extend the evaluation range down to `alpha = 0.2` and sweep band
widths spanning roughly 10% to 90% of the usable span:
`w in {0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72}`, i.e. band edges
from 0.92 down to 0.28. Everything else as in the current sweep.

**What would settle it.** A *transition*, not a trend: performance becoming
roughly uniform across the whole `alpha` range, including far below the band
edge. That is generalization genuinely outside the convex hull, since in a
one-dimensional task space the hull of the training band is just the band.
A smooth improvement with no transition would be the boring interpolation
story, and would say this system does not admit the generalized solution.

## 2. Port the phase diagram

Their central figure is two-dimensional: task diversity `phi` against the number
of pretraining tasks `N`, giving three phases — in-weights learning, in-task-
distribution generalization, and out-of-task-distribution generalization.

We have measured both axes here, but separately and in different systems:
`N` in the logistic task-diversity sweep (threshold `m ~ 1000`) and `w` in the
`asymband` sweep. Their interaction is unmeasured. The port is a grid over
(band width, number of distinct `(R, alpha)` tasks) with everything else fixed.
This is the natural "same analysis, new system" contribution, and it is what
would show whether the dynamics setting has the same three-phase structure.

## 3. The compositional split

Recorded as the fallback if no transition appears in experiment 1.

The task space is two-dimensional, `(R, alpha)`, and so far only *extrapolation*
in `alpha` has been tested. Instead train on a cross and test the interior:

- **Train:** all `R` at `alpha in [0.9, 1.0]`, plus all `alpha` at `R in [0.8, 0.9]`
- **Test:** `(alpha = 0.6, R = 1.0)` — a cell never seen, though both coordinate
  values were seen separately

This asks whether `R` and `alpha` are represented as separable factors that can
be recombined. It is not coverage: the test cell is genuinely absent from
training. There is precedent in the paper itself — its nonlinear experiments
compare sampling the full parameter vector on one sphere against sampling
`vec(W_1)` and `w_2` on separate spheres, and the transition moves substantially
(`phi ~ 135 deg` versus `~60 deg`). Factorizing the task distribution changes
when out-of-distribution generalization appears, which is exactly the
compositional handle.

## 4. Diagnose which solution is implemented

Their specialized/generalized distinction is diagnosed behaviourally: the
generalized solution performs uniformly across all test angles and resembles
OLS. Our analogue is already built — `src/system_id.py` reads the model's
implied return map `E[x_(n+1) | x_n]` out of its predictions.

Run it on the `w = 0` model at out-of-band `alpha`. If the implied map still
traces the *logistic* parabola, the model is applying a memorized transition
table and never attempts system identification. If it traces the true `g_alpha`
parabola while cross-entropy stays at 9-11 nats, the map is being identified and
the loss is coming from the readout — a completely different problem. This costs
no new training and should be run before any of the above.

## Ruled out

- **Random monotone warp augmentation.** Puts the target family in
  distribution; measures interpolation, not out-of-distribution generalization.
- **Multi-family training (tent / sine / cubic).** Tent is topologically
  conjugate to logistic `r=4`; what it adds is a different invariant density.
  But the density is the statistic that does *not* move along the `alpha` axis
  (marginal overlap 0.96 at `alpha=0.5` against 0.075 for transitions), so this
  varies the wrong axis.
- **Context-derived normalization.** Normalizing by the context window's own
  empirical CDF uses no oracle information, but that CDF is ~96% invariant
  across `alpha` — it would normalize by the one thing that does not change.
- **Reformulating as in-context regression on the return map.** Already done, in
  the paper above.
