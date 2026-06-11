# Thesis Outline — chapter map

**Working title:** *Riesgo de automatización laboral por inteligencia
artificial en Jalisco: exposición cognitiva y encarnada, incentivo económico
y proyección 2025–2030*

This document is the writing plan: chapter → argument → the exact artifacts
(tables, figures, scripts) each section draws from. The thesis text will be
written in Spanish; this map and all repo documentation stay in English.
Theory references resolve in `docs/THEORETICAL_FRAMEWORK.md` (TF); data and
methods provenance in `analysis/README.md` and `data/raw/README.md`.

---

## Cap. 1 — Introducción

Problem: AI automation discussed as one risk; it is two technologies with
different physics, adopters and clocks, landing on a labor market that is
55% informal and a nearshoring hub. RQ (two-axis), H1–H4 (as derived in
TF §8), contributions (DBOE, DEOE, the bipolar structure, the evidenced
IRA moderation, the Jalisco quantification), structure of the document.
*Draws on:* README research question; TF §7–8.

## Cap. 2 — Marco teórico

Expansion of TF into prose: task framework (§1) → race/profitability/so-so
technologies (§2) → Moravec and the two frontiers (§3) → measurement lineage
and the thesis's position (§4) → J-curve and the policy window (§5) →
developing-economy adaptations (§6). Close with the integrated causal chain
(§7) and the derived hypotheses (§8).
*Draws on:* `THEORETICAL_FRAMEWORK.md` (the skeleton IS this chapter).

## Cap. 3 — Datos y métodos

3.1 Sources (the table in README + DATA_INDEX; the SQL Server architecture).
3.2 **DBOE**: Felten reproduction (r=.94), c_j(t) construction, the
    benchmark-mapping defense, the monotonicity convention.
3.3 **DEOE**: five subdomains, alphas, PC1, Webb validation (+.76),
    discriminant validity; the Moravec/RL reclassification finding.
3.4 Crosswalk chain: official INEGI tables, broad-SOC key, coverage 88% and
    the bounds strategy.
3.5 Two-level architecture: hurdle Level-1 (zero-inflation rationale),
    CE panel for H4, scenario apparatus (c(t), r(t), 2024 rebasing, IRA
    moderation), threshold design for workers-at-risk.
3.6 Inference: permutation tests for small-N panels; robustness battery.
*Draws on:* builder docstrings; `exposure_cfa.py`, `level1_exposure_model.py`,
`h4_adoption_test.py`, `level2_sector_projection.py`, `permutation_inference.py`,
`robustness_batch.py`.

## Cap. 4 — Resultados I: La estructura de la exposición

4.1 The bipolar structure (H2): EFA F1 57%, CFA ΔAIC +2277, φ=−.67.
    *Figure:* loadings table. *Tables:* EFA/CFA outputs.
4.2 The occupational gradient (H1): SINCO divisions, monotone mirror.
    *Table:* `sinco_exposure_scores`.
4.3 Who holds the exposed jobs: education mirror (−1.00→+0.50 vs
    +0.47→−0.54), income, the gendered split, formality.
    *Figure:* `worker_exposure_profile.png`.
4.4 Jalisco in Mexico (H3 refined): bi-frontal, rank 8/32 vs 29/32; the
    32-state bipolar line. *Figure:* `state_exposure_map.png`.
4.5 The territory: municipal map, metro-cognitive vs corridor-embodied.
    *Figure:* `municipal_exposure_map.png`.
4.6 External test: the hurdle model vs observed usage (AUC .66→.85; the
    intensive margin belongs to DEOE); AEI Mexico validation (r=.97).

## Cap. 5 — Resultados II: ¿El mercado busca automatizar?

5.1 H4 — twenty years of revealed behavior: capital deepening (p_perm=.013),
    falling labor share (p_perm=.004), the so-so pattern (+0.6 vs +3.7 %/yr).
    Placebo and permutation. *Figure:* `h4_adoption_test.png`.
5.2 The wage incentive at worker level: the +23%/SD premium concentrated in
    the educated. *Table:* `wage_premium`.
5.3 The generative-AI shock so far: no significant employment response in
    either direction (permutation-corrected); the absorption test (the
    informality objection answered); nearshoring as upward confounder.
    *Figures:* `chatgpt_event_imss.png`, `absorption_informality.png`,
    `nearshoring_fdi.png`.
5.4 The other side of the race: reinstatement check (descriptive).
5.5 Perception: +12.5pp across the ChatGPT moment; the optimism gap.
    *Figure:* `perception_trend.png`.
5.6 Synthesis: capability shock ✓, perception shock ✓, labor shock ✗ —
    **the policy window** (TF §5).

## Cap. 6 — Resultados III: Proyección 2025–2030

6.1 The two technology curves: c(t) saturating, r(t) compounding — the
    asymmetry and the nearshoring justification of the accelerated scenario.
6.2 Sector pressure scenarios (IRA-moderated); top sectors by pole.
6.3 **Workers at risk**: 1.10M above today's bar → 1.51/1.88/1.98M by 2030;
    the pole flip under acceleration; threshold sensitivity.
    *Figure:* `workers_at_risk.png`.
6.4 Severity: 908k unprotected; the 348k triple-vulnerability cell.
6.5 Scope statement: scenario analysis anchored in observed curves and an
    evidenced mechanism — not a forecast (TF §10.4).

## Cap. 7 — Discusión

7.1 Findings vs literature: where Jalisco confirms Webb/Acemoglu-Restrepo,
    where it diverges (bi-frontal position; informality as severity, not
    absorber — so far).
7.2 Maintained assumptions and their tested status (TF §10): the O*NET
    transplant (AEI r=.97; Lewandowski caveat), adoption-based r(t),
    AEI-target scope.
7.3 Limitations and the data frontier (what does not exist: occupation
    panel, Mexican adoption survey, robot benchmarks) — as research agenda.
7.4 What would change the conclusions (falsifiability: the signatures to
    monitor in IMSS/ENOE/ENDUTIH as the window advances).

## Cap. 8 — Conclusiones y política pública

8.1 Answers to the RQ and H1–H4, one paragraph each.
8.2 Differentiated policy by pole: cognitive (educated/formal/metro —
    augmentation, task transition) vs embodied (less-educated/informal/
    corridor — reconversion, safety net; the 348k cell first).
8.3 The policy window as the organizing concept; monitoring dashboard
    proposal (the indicators this thesis built are reusable).
8.4 Research agenda (= the documented data frontier).

## Anexos

A. Index construction detail + validation tables. B. Robustness/permutation
appendix. C. Crosswalk and coverage bounds. D. Reproducibility (the
`analysis/README.md` map + `sql/` rebuild). E. Figure inventory.

---

## Drafting order (recommended)

2 (skeleton exists) → 3 (docstrings nearly write it) → 4 → 5 → 6 → 1
(intro last-but-one, written against finished results) → 7 → 8.
One chapter per session; each draft pulls numbers live from the server
tables, never retyped.
