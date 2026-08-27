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

## CORRECTION (after the extended sweep): no extrapolation was demonstrated

The extended sweep ran and the held-out loss did fall by 9x across the coverage
range. That was initially read as the transition. It does not survive the right
control.

**The null was wrong.** The return-map probe compared the model's implied map
against the map at `alpha = 1`. That is the correct null only for the `w = 0`
model, which trained on `alpha = 1` alone. Every other model trained on a whole
*band*, so the meaningful null is the nearest map it actually saw — its band
edge `alpha_lo` — not `alpha = 1`, which for a wide band is just a distant
member of its own training set.

Against the correct null, at held-out `alpha = 0.25`:

| run | band | ->true | ->alpha=1 | ->band edge |
|-----|------|--------|-----------|-------------|
| w=0.05 | [0.95, 1] | 0.465 | 0.089 | **0.089** |
| w=0.30 | [0.70, 1] | 0.381 | 0.158 | **0.124** |
| w=0.55 | [0.45, 1] | 0.286 | 0.286 | **0.175** |
| w=0.70 | [0.30, 1] | 0.114 | 0.430 | **0.089** |

Every model, at every held-out alpha, sits closer to its band edge than to the
truth. They clamp to the nearest map they were trained on. Plotted with the
envelope of the whole training band shaded, the implied maps stay *inside* the
envelope in every held-out panel — reproducible by some map already seen.

**The falling loss has a mundane explanation.** Holding the held-out region
fixed at `alpha in [0.20, 0.30]` fixes the *tasks* but not their *distance to
the band edge*, which shrinks from 0.65 at `w=0.05` to 0.05 at `w=0.70`.
Clamping to the edge therefore becomes a steadily better approximation, which
reproduces the loss curve with no generalized solution anywhere.

**The widest run cannot even discriminate.** At `w=0.70`, `alpha=0.25`, the true
map and the band-edge map differ by RMS 0.046, while the model's own error to
either is 0.09-0.11. The hypotheses are closer together than the measurement is
precise, so that panel is uninformative by construction.

### What is actually established

In-band, identification is real and excellent: the implied map sits at the
binning floor for any alpha inside the training band, at every band width. In
the phase language of arXiv:2506.05574 all 14 models reached
*in-task-distribution generalization* and none reached *out-of-task-distribution
generalization*.

### The design fix

Hold the held-out tasks at a fixed *distance* from the band edge rather than at
fixed *positions*: for each `w`, test at `alpha_lo - d` for a common `d`. That
separates "the model extrapolates" from "the test got easier", which the current
design confounds. The alpha axis is bounded below near 0.15, so a fixed-distance
design needs either a smaller `d` or a parameterisation with more room — worth
settling before spending another 14 GPU-days.

## The confound-free criterion: distance to the binning floor

The clamping problem has a clean way out that needs no change of geometry.
Rather than asking "is the implied map closer to the truth or to the band edge",
ask **how close is it to the binning floor** — the RMS error that bin
quantisation alone forces, which no model can beat. Clamping to the band edge
produces a specific, non-vanishing error; only genuine identification reaches
the floor. The criterion is absolute, so it does not care how far the held-out
task is from the band.

Applied to the existing probe (RMS / floor):

| band | interior of band | near band edge | held out |
|------|------------------|----------------|----------|
| [0.95, 1] | 1.2 | — | 9.4 - 27.4 |
| [0.70, 1] | 0.9 - 1.2 | — | 7.9 - 22.4 |
| [0.45, 1] | 1.1 - 1.9 | 3.5 | 9.7 - 16.8 |
| [0.30, 1] | 1.2 - 1.9 | 3.2 - 5.0 | 6.7 |

Identification is *at the floor* in the interior of every training band,
degrades toward the edges, and never approaches the floor on a held-out task —
best case 6.7x, for a task sitting immediately adjacent to the band. This
confirms the negative result independently of the bad null.

It also exposes the specialization cost sharply: the widest model is 5.0x the
floor at `alpha = 1`, a map it trained on, against 1.2x for the narrow-band
models. Wide bands buy coverage at the price of precision at their own edges.

## The design fix: hold the distance, not the position

For each band width `w`, evaluate at `alpha_lo - d` for a common `d`, instead of
at fixed absolute positions. Then widening the band does not bring the test
closer, and the loss curve against `w` is no longer contaminated by the test
becoming easier. Report RMS/floor alongside cross-entropy so the absolute
criterion is always visible.

## Families tested for a cleaner transition

Both candidates were checked numerically before proposing them, because the
`alpha` family's superattracting-origin degeneracy was missed the first time.

### A. Tilted logistic — RECOMMENDED

    f_s(x) = C x(1-x) (1 + s(x - 1/2)),   C normalising the peak to R

`s = 0` is the logistic map exactly. Both endpoint exponents stay at 1.00 for
every `s`, so neither endpoint ever becomes superattracting — the defect that
forced `alpha <= 1` is absent by construction. Verified usable over
`s in [-1.9, +1.4]`: zero orbit death throughout, `f'(0) > 1` everywhere in
range (it crosses 1 near `s = 1.45`, which is the true boundary), and the peak
sweeps `x_c` from 0.338 to 0.636.

The advantage over the `alpha` family is that the base map sits in the
*interior* of the usable range, so a band `|s| <= w` centred on logistic has
held-out regions on **both** sides. Combined with fixed-distance testing that
gives two independent held-out probes per model.

The cost is a smaller deformation range: the peak moves +/-0.14 about 0.5,
against 0.25-0.50 for the `alpha` family. Whether that is enough deformation to
make the task hard is worth checking before committing GPU time.

### B. Spherical deformation — REJECTED, does not stay unimodal

    f_u(x) = C x(1-x) exp(rho * sum_k u_k sin(k pi x)),   u on the unit sphere

The idea was to reproduce the paper's geometry exactly: a homogeneous sphere of
task *directions* at fixed deformation amplitude, logistic as the undeformed
centre, cap training and antipodal testing. Endpoints stay regular and orbits
survive at moderate `rho`.

But most directions do not give a unimodal map, and it worsens with dimension —
the fraction with exactly one interior maximum is 0.75 at `K=3, rho=0.4`, 0.45
at `K=4`, and **0.07 at `K=6`**. Rejecting the multimodal directions would
destroy the homogeneity that motivated the sphere in the first place. Any
sphere-like construction here needs a basis that preserves unimodality by
construction, which the sine basis does not.

### What the geometry question actually turns on

The bounded one-dimensional task space is not the real obstacle, because the
floor criterion above is absolute and does not degrade as the band widens. The
open question is whether the generalized solution is *reachable* at all: the
models already identify maps at the binning floor **inside** the band, so the
identification mechanism exists — it simply does not extend past the training
support. A family with fewer parameters and the base map interior to the range
(A) is the best next bet, and the floor criterion will detect a transition in
whatever geometry it happens.
