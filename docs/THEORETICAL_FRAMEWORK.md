# Theoretical Framework

**Thesis:** Labor Automation Risk from AI in Jalisco, Mexico
**Role of this document:** the intellectual base of the thesis — the theories
the empirical work operationalizes, the hypotheses they generate, and the
explicit map from each theoretical claim to the artifact that tests it.
This is the skeleton of the thesis's theoretical chapter; every section ends
with *where the thesis cashes the theory out*.

---

## 1. The task-based framework: technology substitutes tasks, not jobs

The foundation of all modern automation economics is the task model of
Autor, Levy & Murnane (2003), generalized by Acemoglu & Autor (2011): an
occupation is not an indivisible unit but a **bundle of tasks**, and
technology competes with workers task by task. Two consequences organize
this thesis:

1. Exposure must be measured at the level of what occupations *do* — task
   and ability content — not at the level of job titles. This licenses the
   use of O*NET descriptors as the measurement substrate, and the
   construction of exposure indices as weighted aggregations of
   task/ability-level technology capabilities.
2. "Automation risk" is not binary job destruction. An occupation can have
   a fraction of its bundle automated, with effects on wages, task
   composition and hiring before any effect on headcounts — which
   constrains what early evidence can look like (see §5).

**Cashed out in:** the O*NET-based construction of both indices
(`build_dynamic_aioe.py`, `build_embodied_exposure.py`); the occupation
grain of the Level-1 model; the interpretation rules of the ChatGPT test.

## 2. Displacement, reinstatement, and the profitability condition

Acemoglu & Restrepo (2018, 2019) formalize automation as a **race**: the
displacement effect (machines take tasks) against the reinstatement effect
(new tasks are created for labor). Three implications drive the design:

- **The net employment effect is an empirical question**, not an assumption.
  A credible thesis must look for both sides.
- **Feasibility is not adoption.** A task is automated when it is *cheaper*
  to automate than to keep paying the worker — adoption depends on the
  wage/capital-cost ratio, not on technical capability alone. This is the
  theoretical definition of the thesis's IRA (Índice de Rentabilidad de la
  Automatización: labor cost / capital cost proxies) and the reason H4
  exists.
- **"So-so technologies"** (Acemoglu & Restrepo 2019): automation profitable
  enough to displace but not productive enough to expand output produces
  the least favorable outcome — not mass layoffs but *missing job growth*.
  This is precisely the realized pattern found in Jalisco's twenty-year
  census panel (+0.6%/yr employment growth in exposed-and-profitable
  sectors vs +3.7%/yr elsewhere).

**Cashed out in:** `build_ira.py` (the incentive); `h4_adoption_test.py`
(the market responds: capital deepening p_perm = .013, labor share
p_perm = .004); `reinstatement_emerging_tasks.py` (the other side of the
race); the IRA moderation inside `level2_sector_projection.py` and
`workers_at_risk.py`.

## 3. Two technological frontiers: Moravec's paradox as the theory of the two axes

Moravec's paradox — high-level reasoning is computationally cheap;
sensorimotor skill is computationally expensive — implies that "AI
automation" is not one technology but **two frontiers with different
physics, different cost curves, and different adopters**:

| | Cognitive frontier (LLMs) | Embodied frontier (robots) |
|---|---|---|
| Capability driver | benchmark performance of language/reasoning models | manipulation, navigation, cost-per-robot |
| Capital requirement | near zero (a phone, a subscription) | formal fixed capital (a firm buys the machine) |
| Diffusion channel | person-by-person, instant, crosses informality | investment-by-investment, slow, formal sector only |
| Measured curve in this thesis | c_j(t), Epoch benchmark frontier | r(t), IFR world operational stock |

Two non-obvious predictions follow. First, occupational exposure to the two
frontiers should be **negatively related** — the abilities cheap for
software are dear for robots and vice versa — which is Webb's (2020)
empirical observation and this thesis's H2 (confirmed: EFA bipolar factor;
CFA two correlated axes ≫ one factor, φ = −0.67; the 32 states of Mexico
fall on a single bipolar line). Second, **exposure indices must declare
their frontier**. The thesis's audit finding that two indices marketed as
physical (Moravec auto_w, RL feasibility) are empirically cognitive
(r = −0.66 with physical work content) shows what happens when they don't.

**Cashed out in:** the two-index design (DBOE/DEOE);
`exposure_factor_structure.py`, `exposure_cfa.py`,
`state_exposure_comparison.py`; the reclassification of Moravec/RL; the
asymmetric formal-bias argument in `informality_severity.py`.

## 4. The measurement lineage, and what this thesis adds

The exposure-measurement literature is a progression of answers to "how do
we know which occupations the technology touches":

1. **Frey & Osborne (2017)** — expert workshop labels on 70 occupations,
   extrapolated by a Gaussian-process classifier through nine "engineering
   bottlenecks". First mover; criticized for occupation-level (not task-
   level) treatment and subjective anchors (Arntz, Gregory & Zierahn 2016).
2. **Webb (2020)** — overlap between patent text and task text; separate
   indices for software, robots and AI; first clear evidence the frontiers
   point at different occupations.
3. **Felten, Raj & Seamans (2021)** — the AIOE: 52 O*NET abilities × 10 AI
   application families, weighted by occupational ability importance.
   Systematic and reproducible, but **static** — capability enters as a
   constant.
4. **Eloundou et al. (2023)** — GPT-4 rates task-level LLM exposure; high
   predictive power, but model-subjective and a snapshot.
5. **Anthropic Economic Index (2025–26)** — *observed* AI usage by
   occupation: behavior, not opinion. The first outcome-flavored measure,
   and this thesis's non-circular Level-1 target.

**This thesis's position in the lineage:**

- **DBOE** makes Felten's framework *dynamic*: the application-capability
  vector becomes c_j(t), measured yearly from frontier benchmark scores
  (Epoch AI), so exposure carries a clock. Validated by reproducing the
  published AIOE at r = 0.94.
- **DEOE** builds the *embodied mirror* that the lineage lacks: five O*NET
  physical subdomains aggregated by the same weighting logic, validated
  against Webb's robot-patent index (r = +0.76) with clean discriminant
  validity against the cognitive cluster (r ≈ 0).
- The pair, plus the bipolar structure they reveal, is the methodological
  contribution: a two-frontier, time-indexed exposure system anchored in
  measured capabilities, applied to a developing-economy labor market.

**Cashed out in:** the index builders and their validation blocks;
`level1_exposure_model.py` (hurdle model against observed usage — AUC
0.66 → 0.85 over Frey-Osborne); `robustness_batch.py` (leave-one-benchmark-
out); `aei_mexico_validation.py` (the US-anchor transplant tested:
Mexico–US occupational usage correlation 0.97).

## 5. Adoption in time: the productivity J-curve and the policy window

General-purpose technologies show long lags between capability and
measurable economic effects (Brynjolfsson, Rock & Syverson's productivity
J-curve): adoption requires complementary investment, reorganization and
diffusion. For a developing economy adopting at 0.44× the expected rate
(AEI), theory predicts exactly what the thesis finds in 2023–24: **a
capability shock without a labor-market shock**. Early effects, when they
arrive, appear first in hiring flows, entry-level postings, and within-job
task composition — margins invisible to sector-level employment stocks.

The corollary is the thesis's central policy concept: the gap between the
capability shock (arrived), the perception shock (arrived: +12.5pp in
personal displacement expectation, 2020→2023), and the realized labor shock
(not yet) is **the policy window** — measurable, finite, and the actionable
object of the final chapter.

**Cashed out in:** `chatgpt_event_imss.py` and `absorption_informality.py`
(no significant employment response in either direction; permutation-
corrected), `perception_latinobarometro.py`, and the interpretation
framework written into both scripts.

## 6. Developing-economy labor markets: informality, transplanted task content, and nearshoring

Three strands adapt the framework to Mexico:

- **Informality as the adjustment margin** (Maloney 2004; Levy 2008).
  In Latin America displaced formal workers reappear as informal workers,
  not as unemployed — so formal-sector nulls can mask displacement. The
  thesis treats this as a testable hypothesis, not a caveat: the absorption
  test finds no rise in the informal share of exposed sectors. Informality
  instead enters as a **severity multiplier**: 48% of the at-risk
  population (908k workers) faces the same exposure with no safety net.
- **Task content varies with development** (Lewandowski et al. 2022): the
  same occupation title is more routine-intensive in poorer countries, the
  known caveat of transplanting O*NET. The thesis bounds this risk with the
  AEI validation (usage profile r = 0.97 with the US) and reports it as the
  framework's main maintained assumption.
- **Nearshoring and global value chains**: relocation of manufacturing
  to Mexico raises FDI (+27% in Jalisco, 2021–23) that arrives at the
  global automation frontier — capital inflow is simultaneously job
  creation today and imported robot adoption tomorrow, the concrete
  mechanism behind the accelerated r(t) scenario.

**Cashed out in:** `informality_severity.py`, `absorption_informality.py`,
`nearshoring_channel.py`, `crosswalk_coverage_bounds.py`.

## 7. The integrated framework

The causal chain the thesis estimates, theory block by theory block:

```
technology capability (global)          §3, §4:  c_j(t), r(t)
        ×
occupational task content               §1:      O*NET → DBOE, DEOE
        ×
local employment composition            §6:      ENOE → who is exposed (H1, H3)
        ×
economic incentive                      §2:      IRA — feasibility ≠ adoption (H4)
        ↓
realized substitution                   §2, §5:  capital deepening, labor share,
                                                 missing job growth; lags (policy window)
        ↓
distributional incidence                §6:      education / gender / formality /
                                                 territory; severity multipliers
```

## 8. Hypotheses as derivations

Each hypothesis is a prediction of a specific block, not an ad-hoc claim:

- **H1 (cognitive gradient)** ← §1 + §3: LLM capability concentrates on
  high-ability-content occupations, inverting Frey-Osborne's safe zone.
- **H2 (bipolar structure)** ← §3: the two frontiers price abilities
  oppositely, so exposures anti-correlate.
- **H3 (Jalisco's position, bi-frontal)** ← §3 + §6: employment composition
  places Jalisco below the cognitive frontier's center of mass and above
  the embodied one — and, within Mexico, nearer the cognitive pole than
  most states (rank 8/32).
- **H4 (economic moderation)** ← §2: exposure translates into pressure only
  where the wage/capital ratio makes automation profitable.

## 9. Theory → evidence map

| Theory block | Hypothesis / claim | Artifact | Headline result |
|---|---|---|---|
| Task framework (§1) | exposure measurable from task content | `build_dynamic_aioe.py`, `build_embodied_exposure.py` | DBOE r=.94 vs AIOE; DEOE r=+.76 vs Webb robot |
| Two frontiers (§3) | H2 bipolar | `exposure_factor_structure.py`, `exposure_cfa.py` | F1 bipolar 57% var; 2-factor ΔAIC +2277, φ=−.67 |
| Two frontiers (§3) | indices must declare frontier | DEOE validation | Moravec/RL reclassified cognitive (r=−.66 w/ physical) |
| Lineage (§4) | DBOE/DEOE add power over baseline | `level1_exposure_model.py` | AUC .66→.85 over Frey-Osborne (target: observed usage) |
| Lineage (§4) | US anchor transplants | `aei_mexico_validation.py` | MEX–USA usage profile r=.97 |
| Task framework (§1+§6) | H1, H3 gradients | `build_sinco_exposure.py`, `worker_profile.py`, `state_exposure_comparison.py` | monotone division gradient; education mirror; Jalisco bi-frontal 8/32–29/32 |
| Race + profitability (§2) | H4 market responds | `h4_adoption_test.py`, `permutation_inference.py` | K/L +0.143 (p_perm=.013); labor share −0.040 (p_perm=.004) |
| So-so technologies (§2) | missing growth, not layoffs | H4 quadrants | +0.6 vs +3.7 %/yr over 20 years |
| Reinstatement (§2) | exposed occupations also gain tasks | `reinstatement_emerging_tasks.py` | renewal tilts cognitive (descriptive, n.s.) |
| J-curve (§5) | capability shock ≠ labor shock | `chatgpt_event_imss.py`, `absorption_informality.py` | no significant response either direction (p_perm=.26); no absorption signature |
| Perception (§5) | expectations lead outcomes | `perception_latinobarometro.py` | +12.5pp personal expectation 2020→2023 |
| Informality (§6) | severity multiplier, not separate risk | `informality_severity.py` | 908k unprotected at-risk; 348k triple-vulnerable |
| Nearshoring (§6) | capital imports the robot frontier | `nearshoring_channel.py` | FDI +27%, share 5.5→6.8% |
| Synthesis (§7) | scenario projection, not forecast | `level2_sector_projection.py`, `workers_at_risk.py` | 1.10M → 1.88M above today's bar by 2030 (baseline) |

## 10. Maintained assumptions and their status

1. **O*NET task content transports to Mexico** — standard in the literature
   (ILO, IDB, OECD); tested here (AEI r=.97); caveat cited (Lewandowski).
2. **Capability benchmarks proxy the cognitive frontier** — leave-one-out
   robust for the cross-section; curve level is where choice matters
   (documented).
3. **Robot adoption proxies the embodied frontier** — no capability
   benchmark series exists anywhere; adoption (IFR stock) is the literature
   standard (Acemoglu-Restrepo 2020; Graetz & Michaels 2018).
4. **Scenario projection, not estimation** — no occupation-level employment
   panel exists for Mexico; the projection is anchored in observed curves
   and an evidenced mechanism, and is labeled accordingly.

## References

Acemoglu, D., & Autor, D. (2011). Skills, tasks and technologies. *Handbook of Labor Economics*, 4B, 1043–1171.

Acemoglu, D., & Restrepo, P. (2018). The race between man and machine. *American Economic Review*, 108(6), 1488–1542.

Acemoglu, D., & Restrepo, P. (2019). Automation and new tasks: How technology displaces and reinstates labor. *Journal of Economic Perspectives*, 33(2), 3–30.

Acemoglu, D., & Restrepo, P. (2020). Robots and jobs: Evidence from US labor markets. *Journal of Political Economy*, 128(6), 2188–2244.

Arntz, M., Gregory, T., & Zierahn, U. (2016). The risk of automation for jobs in OECD countries. *OECD Working Paper* 189.

Autor, D., Levy, F., & Murnane, R. (2003). The skill content of recent technological change. *Quarterly Journal of Economics*, 118(4), 1279–1333.

Brynjolfsson, E., Rock, D., & Syverson, C. (2021). The productivity J-curve. *American Economic Journal: Macroeconomics*, 13(1), 333–372.

Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). GPTs are GPTs. *arXiv:2303.10130*.

Felten, E., Raj, M., & Seamans, R. (2021). Occupational, industry, and geographic exposure to artificial intelligence. *Strategic Management Journal*, 42(12), 2195–2217.

Frey, C. B., & Osborne, M. A. (2017). The future of employment. *Technological Forecasting and Social Change*, 114, 254–280.

Graetz, G., & Michaels, G. (2018). Robots at work. *Review of Economics and Statistics*, 100(5), 753–768.

Gmyrek, P., Berg, J., & Bescond, D. (2023). *Generative AI and jobs*. ILO Working Paper 96.

Handa, K., et al. (2025–2026). *The Anthropic Economic Index*. Anthropic.

Levy, S. (2008). *Good intentions, bad outcomes: Social policy, informality, and economic growth in Mexico*. Brookings.

Lewandowski, P., Park, A., Hardy, W., Du, Y., & Wu, S. (2022). Technology, skills, and globalization: Explaining international differences in routine and nonroutine work. *World Bank Economic Review*, 36(3), 670–686.

Maloney, W. (2004). Informality revisited. *World Development*, 32(7), 1159–1178.

Moravec, H. (1988). *Mind children*. Harvard University Press.

Webb, M. (2020). *The impact of artificial intelligence on the labor market*. Stanford working paper.
