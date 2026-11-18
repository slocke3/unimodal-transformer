# Manuscript changes for the rebuttal (2026-11-16)

Line numbers refer to the manuscript Simon pasted on 2026-11-16, saved at
`/private/tmp/claude-501/-Users-simonlocke-Documents-GitHub-unimodal-transformer/14411c54-dce6-495c-b46b-0fa8dc70117f/scratchpad/manuscript_2026-11-16.tex`
(the final robustness table after line 405 is not included in that copy).

Status tags: **[text]** writing only · **[fig ready]** result and figure exist · **[needs run]** compute required · **[needs Chase]** waiting on Chase · **[decide]** decision needed · **[check]** quick analysis needed first.

Reviewer codes: **P** = pqmj, **F** = FWzs, **V** = 68V6.

---

## Part 1. Changes in manuscript order

### Abstract (L30)
1. **Keep the resolution claim** (Simon, 2026-11-16). The output is fixed at 64 bins for every input resolution, so all arms are scored on the same prediction problem. Make sure the paper defines m\* and says that "generalize" means new-task error closing on the model's own plateau (item 24). *(P)* **[text]**
2. **Sharpen the regression contrast** ("In contrast, time series prediction involves…"). The sentence should say that the task generates its own inputs, so the dynamics decide what evidence the context contains, and that observations resolve the state only to finite precision. *(V: regression; novelty)* **[text]**

### Introduction: how dynamics differ from regression (L39–41)
3. **Strengthen with the "isn't this just regression?" paragraph:**
   - the orbit, not the context length, limits the evidence: a fixed-point context is one repeated value, a period-2 context has two;
   - the invariant measure decides which inputs each task shows, so tasks differ in their inputs as well as their rule;
   - tokenization is not observation noise: it is deterministic, identical at every step, does not average out over the context, and sets a floor that more context cannot remove.

   Forward-reference the noise-vs-bins appendix (identifiability dropped 2026-11-17). *(V: "isn't this just ICL regression?"; novelty)* **[text]**

### Introduction: why the logistic family (L43–50)
4. **r < 0.5 (L48): DONE (2026-11-16).** Orbits shrink by about a factor r per step and soon stop carrying information about r; exclude r < 0.5 (over 96% all-zero windows at 64 bins); the cutoff is a convention, ideally followed by "restricting evaluation to r ≥ 1 leaves m\* unchanged". The rebuttal reply carries the numbers: ≈16% uniform rise; m\* identical on the CE arms and within 8 tasks on the square-loss arms. *(V: why exclude r < 0.5)*
5. **Add 1–2 sentences on why a single family.** The logistic family has exact dynamical ground truth (Lyapunov exponent, invariant measure, periodic windows, conjugacies), which is what lets us explain the model's behaviour. State the scope limitation plainly here and in the Limitations. *(P/F/V: only one system)* **[text]**
6. **Why tokenize (L50): DONE (2026-11-16).** With noise-free real values two observations give r exactly; we instead tokenize as Chronos does. Typo fixed. Suggested: the "evidence … depends on" sentence should say it depends on the dynamics as well as on context length, and the range should be r ∈ [0.5, 4]. *(V: why tokenize)*

### Introduction: summary of findings (L52)
7. **Reword the resolution sentence** "Finer tokenizations require more training maps…" to match item 1. *(P)* **[text]**
8. **Context sentence: DONE (2026-11-17).** "Context length, in contrast, lowers the error only until the context covers the last few tokens, which is all a count-based predictor needs; longer contexts reach the same high-diversity error floor but delay the crossover." Simon: say "last few tokens", not "ten tokens". *(F: context length)*
9. **Narrow the r = 3 sentence** ("models trained just above the first period-doubling bifurcation…"). The transfer is seen only for windows starting at r = 3, and state overlap is necessary but not sufficient (see item 30). *(P: r = 3)* **[text]**
10. **Add the new findings:**
    - the baselines and probe (the model infers the task from context; the crossover itself is partly a coverage effect);
    - transfer requires coverage of token transitions, not just states.

    *(F: algorithm, Q1)* **[text]**
11. **Optional: set up the two hypotheses here** (A: identifies the system from context, as in regression ICL and in-context operator estimation; B: looks up transitions pooled from training) and answer them in the Discussion (item 34). *(novelty: results were not foregone)* **[text]**

### Related work (L58–61)
12. **L58: the regression comparison.** Make it explicit that regression ICL draws inputs from one distribution shared by all tasks, whereas here the dynamics set each task's inputs. The existing accounts (wu2024how, lu2025asymptotic) assume independence. *(V)* **[text]**
13. **L58 and L61: tie the baselines to prior work.**
    - L61: our k-gram counters are the transition-statistics strategy of edelman2024evolution.
    - L58: our least-squares r̂ is the parameter-estimation strategy of garg2022what and akyurek2023what.
    - L61: contrast with liu2024llms ("accuracy improves as the context grows") and our saturation.

    *(F: algorithm)* **[text]**

### Contributions (L72–76)
14. **L72:** "model behavior can be explained by known properties of each map". Name them: orbit structure (identifiability), Lyapunov exponent (chaotic coverage), represented transitions. *(P: results not connected to the dynamics)* **[text]**
15. **L74:** reword the resolution claim (item 1). Context half DONE (2026-11-17): "and context length lowers the attainable error only until it covers the last few tokens, beyond which it delays the crossover." *(P, F)* **[text: resolution half]**
16. **L76:** narrow the r = 3 claim (item 30), and add that transfer follows represented transitions, not parameter distance (ρ = 0.07) or state occupancy alone (conjugate maps). *(P, F Q1)* **[text]**
17. **New item:** the transformer beats transition counting and parameter estimation past the crossover and tracks the true next state in matched-final-token pairs. The counters also show a crossover, so the crossover reflects task coverage. *(F: algorithm and baselines)* **[fig ready]**

### Methods: dynamical quantities (L83–118)
18. **The invariant measure and λ are defined here but only used in the appendix.**
    - Minimum fix: one forward reference per quantity saying where it is used:
      - λ: chaotic coverage, the identifiability analysis, and the error floor in chaos;
      - μ_r: occupancy and D_state.
    - Larger fix: bring the λ-based results into the main text (item 29).

    *(P: sections not connected)* **[decide with Chase]**

### Methods: tokenization (L129–142)
19. **L138, "We choose uniform bins…".** Expand with the why-tokenize point (item 6) if it isn't in the introduction. *(V)* **[text]**
20. **Synonym construction details: DONE (2026-11-16)** in the L191 paragraph. Synonyms are drawn independently for every occurrence in every window, fixed when the training set is built and drawn fresh for evaluation. They are many-to-one, so no information is added or lost, and they apply to inputs only (the loss stays on the 64 output bins). Seed-0 m\*: synonyms 254–312 for vocabularies 64–1024; at a vocabulary of 256, 256 bins gives 831 and j = 4 gives 312. Keep seed-0 numbers in this panel because the synonym arm has one seed. *(P: synonym ambiguity)*
21. **Why the output is fixed at 64 bins: DONE (2026-11-16).** One sentence added at L142: fixing the output (64 bins or a real number) means every model is scored on the same target regardless of input resolution, so errors are directly comparable. "Is it standard?" and "why 64?" go in the rebuttal reply only: it is a deliberate control (Chronos shares one vocabulary for inputs and outputs; verify before citing), and 64 is intermediate in the 8–256 range. *(P: is the 64-bin output standard?)*
22. **Which loss each experiment uses: DEFERRED (2026-11-16)**; placement undecided. Draft sentence: "We report task diversity under both objectives (Fig. 2), the context-length sweep under square loss and the synonym control under cross-entropy (Fig. 3), and restricted-support transfer under cross-entropy in the main text and square loss in App. window-mse." Also name cross-entropy in the Fig. 3b caption. Still open: no CE context sweep in the current manuscript. *(P: CE vs MSE)*

### Methods: architecture (L144–147)
23. **Point to the capacity ablation** (new appendix section). Chase's width sweep (32 to 256, 0.05M to 3.2M parameters, MSE, 8–256 bins, one seed) shows the crossover and the resolution reversal at every width. Width 256 matches the default width 128, so the default is past capacity saturation; only width 32 is limited, at 128–256 bins. Depth and training length were not varied. Dropout 0.1 is already stated here. *(V: no capacity ablation)* **[fig ready: Chase's width figure]**

### Methods: evaluation (L149)
24. **m\* definition: DONE (2026-11-16).** Placed in the Results opening paragraph, before Fig. 2 (not in the Methods). The Fig. 2 caption, L191 and the L200 caption now refer to it. Open: seed 0 (194 → 831) vs four-seed median (191 → 677) for Fig. 3b. *(P: missing m\* definition)*

### Methods: terminal-position objective (L151–167)
25. **Justify the one-step terminal loss.**
    - Each model is trained and evaluated at a single context length, which makes L a controlled variable; an every-position loss mixes contexts of length 1 to L.
    - Every-position training removes most of the crossover but raises the fine-resolution floor (256 bins: 6e-5 vs 1.7e-6 new-task MSE).
    - In chaos, multi-step rollouts diverge at rate λ, so one-step error is the clean quantity.

    *(V: one-step vs sequential loss)* **[fig ready: `figures_taskdiv/allpos`]**

### Results: task diversity (L169–184, Fig. 2 at L172–178)
26. **L180, "A transition is not guaranteed here…".** Expand this into the reason the result is not foregone, then add the baselines:
    - the k-gram counters also show a crossover, so its existence reflects task coverage;
    - the transformer needs fewer tasks and reaches a 2.5–3.4× lower floor;
    - it beats least-squares r̂ at 16 and 64 bins past the crossover;
    - in matched-final-token pairs the slope is 1.00 past the crossover and about 0.85 at m = 100.

    *(F: algorithm, baselines, matched pairs; novelty)* **[fig ready: `figures_rebuttal/baselines_taskdiv`; probe is a table]**
27. **L182–184, the tradeoff paragraph.** Replace it with the reconciled claim:
    - at a common absolute error, finer bins reach a coarse model's floor with fewer tasks, so pqmj is right on that measure;
    - relative to each resolution's own floor, finer resolutions need more tasks, at every threshold;
    - the best resolution grows with m (64 bins best at m ≤ 1000, 256 bins at m ≥ 4000).

    Fig. 2 caption (L176): reword "Finer input partitions move the transition to larger m". Add the own-floor panels to Fig. 2, or replace it with them. *(P)* **[fig ready: `figures_rebuttal/taskdiv_resolution_tradeoff` panels c and f; decide: replace or add]**

### Results: context length (L186–188, Fig. 3a)
28. **Context paragraph and Fig. 3a: DONE (2026-11-17).**
    - Fig. 3a now shows L ∈ {1, 3, 5, 10, 20, 50} from Chase's sweep (`scripts/plot_context_synonym_combined.py`, y-limit raised for the legend). No appendix with all eleven lengths (Simon).
    - L186–188 rewritten: L = 1 is close to the last-token counter; L = 3 and 5 show no crossover and plateau higher (no floor numbers, dropout 0.1); from L = 10 on all lengths share the floor, matching the 4–6 tokens after which the counters stop improving (cites `app:baselines`); longer contexts delay the crossover, m\* ≈ 130 at L = 10 to ≈ 530 at L = 50 (median curves; L = 20 ≈ 210).
    - Identifiability analysis dropped from the paper (Simon): the counters make the same point.
    - Not in the text: the delay comes almost entirely from fixed-point maps with 1 ≤ r < 2 (m = 250: 4e-5 at L = 10 vs 9e-3 at L = 50; periodic and chaotic flat across L ≥ 10). Mechanism unknown. `scratchpad/ctx_regime.py`.
    - Caveat for replies: L = 5 (≈ 6e-5) is worse than the 4-token counter (2.2e-5), so the counter match holds for where saturation happens, not for each short L.

    *(F: why context doesn't help, L ∈ {1, 2, 5}; P: dynamics)*

### Results: synonyms and wrap-up (L191–195, Fig. 3 caption L200)
- **L191:** add the synonym details (item 20) if they aren't in the Methods, and refer to the m\* definition (item 24). **[text]**
- **L193: DONE (2026-11-17).** "Beyond L = 10, longer context leaves the high-diversity error floor unchanged but delays the crossover. Finer observation resolution also delays the crossover, but ultimately enables substantially lower prediction error."
- **L200 caption: DONE (2026-11-17).** Title "Effects of context length and vocabulary size."; (a) lists the six lengths and says L ≤ 5 show no crossover, L ≥ 10 share the floor, longer reach it at larger m; (b) names cross-entropy and one seed. Remove Simon's "(I updated the figure…)" note before submission.

### Window transfer (L203–242, Fig. 4 at L216–223)
29. **After L225, bring the appendix's main result into the main text** (2–3 sentences):
    - out-of-support error tracks represented-process mismatch (occupancy ρ = 0.56, transition support ρ = 0.44) and not parameter distance (ρ = 0.07);
    - chaotic coverage, defined by λ > 0, lowers transition mismatch (ρ = −0.60).

    This is where λ and the invariant measure connect to the results. *(P: results not connected to the dynamics; F)* **[text; analysis exists at L333–392]**
30. **L227–240, the r = 3 paragraph: rewrite it.**
    - Clarify the notation: R (training parameter) vs r (target), the branch x₋(R), and what is matched.
    - Answer "why only windows starting at 3?". The overlap formula also predicts transfer for windows starting above 3, so state overlap is necessary but not sufficient. Candidate explanation: just above r = 3 the two period-2 branches fall in the same bin, so training sees repeated tokens, the same transition a fixed point produces. Windows starting above 3 resolve the branches into different bins and never show a repeated token.
    - Fig. 4 caption (L219–221): narrow the "range predicted by state overlap" claim to match.

    *(P: notation, r = 3)* **[check: count repeated-token (i→i) transitions per window; CPU only]**
31. **L242, the MSE paragraph: expand it into a short discussion in the main text.**
    - Low-loss regions are weaker under MSE.
    - Likely reason: 64-bin CE only needs the right bin, while MSE also penalizes within-bin and continuation errors, and a matched state with a different continuation produces exactly those.
    - The represented-process correlations hold under MSE (table after L405).
    - Soften "Thus, the phenomenon is not specific to cross-entropy".
    - Optionally move the MSE heatmap into the main text.

    *(P: CE vs MSE)* **[text; decide on moving the figure]**
32. **New paragraph (main text) plus appendix: conjugate maps.**
    - Out-of-band maps visit the same states (occupancy overlap ≥ 0.92) but share at most 0.31 of their token transitions.
    - CE goes from about 1.2 nats in band to 2.7–9.7 nats out of band.
    - So state coverage is necessary, but the transitions must be covered too.

    *(F Q1)* **[fig ready: `paper_iclr2027/family_overlap.png` top right + `figures_asym2/conjugacy_curve.png` left; combine into one figure; one seed]**

### Discussion (L246–255)
33. **L249.** Context half DONE (2026-11-17): "Context length, by contrast, lowers the error only until the context covers the last few tokens, the same history that count-based predictors need; longer contexts achieve the same minimum error." Still to do: reword the resolution sentence (item 1). *(F)* **[text: resolution half]**
34. **L251: answer the two hypotheses.**
    - Replace or extend "a mix of strategies (parroting vs operator estimation)".
    - Neither pure hypothesis fits. The model infers the task from context (baselines, probe), but only among token processes covered in training: it needs ~10³ tasks, fails just outside its training support, and fails on conjugate maps.
    - Fix the typo "should informed".

    *(novelty; F: algorithm; V: regression)* **[text]**
35. **L253–255, Limitations: concede and point.**
    - Concede: one system; one parameter; no higher dimensions; few seeds for some analyses.
    - Noise: point to the noise-vs-bins appendix.
    - Capacity: point to the ablation.
    - Future work: a system with a different route to chaos (e.g. the circle map), and coordinate changes (the conjugacy result).

    *(P/F/V scope)* **[text]**

### Appendix: new sections (insert after L272 or at the end)
36. m\* definition and the r ≥ 1 robustness check. *(P, V)* **[text + small table]**
37. Resolution tradeoff, own-floor vs absolute (if not in Fig. 2). *(P)* **[fig ready]**
38. Baselines (k-grams, least-squares r̂) and the matched-final-token probe. *(F)* **[fig ready + table]**
    - Appendix figure: `baselines_taskdiv_all_3col` (all six resolutions, 3 across; CE rows then MSE rows). If moved to the main text, show 16/64/256 only (`baselines_taskdiv`).
    - Candidate figure: `baselines_6gram_crossover` (transformer vs 6-gram per resolution, CE | MSE). Under CE the counter's m\* matches the transformer's at every resolution (177–706 vs 182–673), so crossover location reflects transition coverage; past it the transformer is 2–4× better at ≥ 64 bins.
    - Not doing (2026-11-17): counters matched to each context length (no CE context sweep; L = 3, 5 dropout confound), smoothed/synonym counters, the Bayes (training-task vs family posterior) comparison. Bayes comparison = future work / ICML.
39. ~~Identifiability analysis.~~ Dropped (2026-11-17): the counters carry the point.
40. ~~Short contexts appendix.~~ Not needed (2026-11-17): L = 1, 3, 5 are in the main Fig. 3a; no full-sweep appendix.
41. Every-position loss. *(V)* **[fig ready: `figures_taskdiv/allpos`]**
42. Noise vs bins: tokenization is not noise. *(V, F noise)* **[fig ready: `figures_taskdiv/noise_vs_bins`]**
43. Conjugate maps and transition coverage. *(F Q1)* **[figs exist; combine]**
44. Capacity ablation (Chase's width sweep). *(V)* **[fig ready]**
45. Synonym construction details, if too long for the Methods. *(P)* **[text]**

---

## Part 2. Reviewer point → where it is addressed (for writing the rebuttal)

| Reviewer point | Items | Status |
|---|---|---|
| **P** m\* not defined | 24, 36, L200 caption | text |
| **P** synonym ambiguity | 20 (L138 or L191), 45 | text |
| **P** CE vs MSE reported inconsistently | 22, 31, (28 if CE context arm) | text; decide |
| **P** results not connected to dynamics | 14, 18, 28, 29 | text; decide λ placement |
| **P** only one system / scope | 5, 35 | text |
| **P** findings incremental | 2, 10, 11, 17, 26, 34 | text |
| **P** 64-bin output standard? MSE setup? | 21 | text |
| **P** notation at end of §5.1 | 30 | text |
| **P** why only windows starting at 3? MSE? | 30, 31 | check, then text |
| **P** "finer needs more maps" misleading | 1, 7, 15, 27, 37 | fig ready |
| **F** limited novelty | as P incremental | text |
| **F** threshold governed by general dynamics? higher dimensions, more parameters, noise | 29, 35, 42 | text; partial |
| **F** why doesn't context help? | 8, 28, 33, 38 | done (needs the baselines appendix, 38) |
| **F** L ∈ {1, 2, 5}? | 28 | done: L = 1, 3, 5 in Fig. 3a |
| **F** what algorithm? | 13, 17, 26, 34, 38 | fig ready |
| **F** suggested baselines | 17, 26, 38 | fig ready |
| **F** matched final states | 26, 38 | table ready |
| **F** Q1 state coverage vs transitions | 16, 29, 32, 43 | figs exist |
| **V** novelty / setting too simple | as P incremental; 3, 11, 34 | text |
| **V** no capacity ablation | 23, 44 | fig ready (width only) |
| **V** why exclude r < 0.5? | 4, 36 | text + table |
| **V** why tokenize? | 6, 19, 42 | text + fig ready |
| **V** one-step vs sequential loss | 25, 41 | fig ready |
| **V** isn't this just ICL regression? | 3, 12, 26, 34 | text |

## Part 3. Blocking items

- **Chase:** nothing outstanding. Short contexts and capacity come from his runs (40, 44); his L = 3 and 5 runs used dropout 0.1, so don't read much into their exact floors.
- **Decisions:** Fig. 2 replace vs add (27); where λ and the invariant measure go (18, 29); MSE heatmap to the main text (31); CE context arm (22).
- **Checks before writing:**
  - repeated-token transitions for the r = 3 windows (30), CPU only.
- **New figure to build:** the conjugate-map panel combining occupancy/transition overlap with CE along R = 1 (32, 43). Uses saved data; no runs.
