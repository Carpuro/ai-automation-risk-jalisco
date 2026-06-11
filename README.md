# Labor Automation Risk from AI in Jalisco, Mexico

**Thesis â€” MaestrÃ­a en Ciencias de los Datos**
Carlos Pulido Rosas Â· CUCEA, Universidad de Guadalajara Â· 2025â€“2026

---

## Research Question

> *How is AI automation risk distributed in Jalisco's labor market between the
> cognitive frontier (large language models) and the embodied frontier (robots,
> drones, autonomous machines) â€” and what determines which of the two dominates
> for each occupation and sector?*

The study covers **all of AI, on two axes of equal weight**: cognitive AI
(LLMs and perceptual AI) and embodied AI (robots, drones, autonomous vehicles,
physical manipulation). It moves beyond Frey-Osborne (2013) in three ways:
(1) it measures LLM-specific exposure with a dynamic index anchored in real
benchmark capability curves; (2) it measures embodied exposure with a parallel
index validated against real robot-patent exposure; and (3) it adds the
economic incentive (Acemoglu & Restrepo, 2018) that determines whether
technically feasible substitution is also profitable.

---

## Hypotheses

1. **H1 â€” Cognitive gradient.** LLM exposure follows the occupational
   hierarchy: highest for managerial/professional occupations, lowest for
   elementary/manual ones â€” inverting Frey-Osborne's assumption that
   non-routine cognitive work is safe.
2. **H2 â€” Bipolar structure.** Cognitive and embodied exposure are not two
   independent risks but opposite poles of one dominant dimension: the
   occupations most exposed to LLMs are the least exposed to robots, and vice
   versa (Webb, 2020, confirmed with own indices).
3. **H3 â€” Jalisco leans physical.** Jalisco's employment-weighted exposure
   profile sits on the embodied pole (manufacturing, agriculture, transport):
   below-average LLM exposure, above-average robot exposure.
4. **H4 â€” Economic moderation.** The Automation Profitability Index
   (IRA = annual wage / capital-cost proxy) moderates whether technical
   exposure translates into real adoption pressure by sector.

---

## Own Indices (methodological contribution)

| Index | Axis | Construction | Validation |
|---|---|---|---|
| **DBOE** â€” Dynamic LLM Occupational Exposure | Cognitive | Extends Felten's AIOE with yearly frontier-capability curves c_j(t) from Epoch AI benchmarks (3 LLM applications) | Reproduces published AIOE r = 0.94; tracks LM-AIOE r = 0.93 |
| **DEOE** â€” Embodied Occupational Exposure | Physical | 5 O*NET physical subdomains (importance Ã— level, mirror of DBOE weights), summary = PC1 (63% variance) | Webb robot-patent exposure r = +0.76; discriminant vs cognitive indices r â‰ˆ 0 |

Builders: `data/raw/build_dynamic_aioe.py`, `data/raw/build_embodied_exposure.py`.

A finding along the way: the indices previously treated as physical
(Moravec auto_w, RL feasibility) are empirically **cognitive** (r = âˆ’0.66 with
real physical work, r â‰ˆ +0.75 with DBOE/AIOE) â€” without the DEOE the battery
had no physical axis at all.

---

## Key Results So Far

- **Bipolar factor structure (H2 confirmed).** EFA over 8 de-duplicated
  indices (N = 747 SOC): one dominant factor (57% variance) with cognitive
  indices loading positive (DBOE +0.87) and embodied indices negative
  (DEOE âˆ’0.87, Webb robot âˆ’0.82). `analysis/exposure_factor_structure.py`.
- **Cognitive gradient (H1 confirmed).** DBOE by SINCO division descends
  monotonically from Funcionarios/directores (+0.89) to occupations in the
  elementary divisions (âˆ’1.2); DEOE mirrors it, peaking at machine operators
  (+1.07) and agriculture (+0.72). Official INEGI SINCOâ†”SOC crosswalk,
  `data/raw/build_sinco_exposure.py`.
- **Jalisco leans physical (H3 confirmed).** Employment-weighted (ENOE
  fac_tri, 88% coverage): DBOE âˆ’0.38 vs DEOE +0.09 â€” Jalisco's labor force is
  below-average in LLM exposure and above-average in robot exposure.
- **Level-1 model (non-circular).** Observed AI usage
  (`anthropic_observed_exposure`, Anthropic Economic Index) is zero-inflated
  (52% zeros), so a two-part hurdle model is used. Extensive margin: adding
  DBOE lifts AUC 0.66 â†’ 0.85 over Frey-Osborne (LR p â‰ˆ 1e-48); DEOE adds
  significantly with opposite sign. Intensive margin is dominated by DEOE
  (Î² = âˆ’0.76). `analysis/level1_exposure_model.py`.

Both index magnitudes are kept in the risk model (decision: each occupation
carries an AI risk *and* a robot risk); the bipolar axis is reported as a
structural finding.

---

## Methodology â€” Two-Level Architecture

**Level 1 â€” Occupation exposure model (N â‰ˆ 678 SOC).** Hierarchical two-part
(hurdle) model of observed AI usage: Frey-Osborne baseline â†’ +DBOE â†’ +DEOE,
with likelihood-ratio / incremental F tests. The target is *observed* usage,
not a constructed score, so the test is external and non-circular.

**Level 2 â€” Jalisco sector projection (2025â€“2030).** Exposure aggregated to
SINCO occupations (official INEGI crosswalk) and SCIAN sectors, interacted
with the IRA economic incentive, anchored in IMSS formal-employment
trajectories 2000â€“2024. Scenario projection driven by the capability curves
(c_j(t) for LLMs; robotics capability curve pending) â€” **scenario analysis
anchored in real data, not a trained forecaster** (no observed automation
panel exists for Jalisco).

Statistical validation: EFA/CFA for construct structure, Cronbach's alpha per
DEOE subdomain, convergent/discriminant correlations, VIF, sensitivity checks
(PC1 without weak subdomains, benchmark-mapping robustness).

> **Note on the two dynamic layers:** DBOE's dynamics come from benchmark
> capability curves c_j(t) (Epoch AI). DEOE's dynamics come from the robotics
> adoption curve r(t) â€” world operational stock of industrial robots (IFR via
> OWID, 2012â€“2024) indexed to 2022, projected 2025â€“2030 under three scenarios,
> with Mexico's own IFR installations series as the local anchor
> (`data/raw/build_robot_capability.py`). No public benchmark series measures
> robot *capability* the way Epoch measures LLMs, so the embodied curve is
> adoption-based â€” cost-decline evidence (Graetz & Michaels, 2018) is cited as
> supporting context.

---

## Data Sources

All sources below are downloaded and (except where noted) loaded into SQL
Server. See [`docs/SQL_SERVER_SCHEMA.md`](docs/SQL_SERVER_SCHEMA.md) for the
DB inventory and [`data/DATA_INDEX.md`](data/DATA_INDEX.md) for details.

| Source | Content | Status |
|---|---|---|
| ENOE Q3 2024, ent=14 | Jalisco worker microdata (SDEMT 13,839; 6,147 occupied with SINCO-4d) | Loaded |
| O*NET 28.3 | Occupation descriptors (abilities, activities, context, tech skills) | Loaded |
| INEGI SINCOâ†”SOC/CIUO comparative tables | Official 4-digit crosswalk (via `occupationcross`) | Loaded |
| ESCO ISCO â†” SOC crosswalk | Secondary bridge | Loaded |
| Felten et al. (2021) AIOE | Cognitive exposure + Appendix D matrix (DBOE input) | Loaded |
| Epoch AI Capabilities | LLM benchmark scores (DBOE c_j(t) input) | Loaded |
| Webb (2020) / Comparison of Indices | Robot/software/AI patent exposure + Frey-Osborne, SML, Eloundou | Loaded |
| Anthropic Economic Index | Observed AI usage by occupation (Level-1 target) | Loaded |
| ILO WP140, Moravec, RL feasibility | Additional cognitive indices (battery) | Loaded |
| INEGI Censos EconÃ³micos 2003â€“2023 | Capital/labor by SCIAN (IRA, longitudinal) | Loaded |
| INEGI PIBE 2003â€“2024 | Sectoral GDP Jalisco | Downloaded |
| IMSS (IIEG) 2000â€“2024 | Formal employment by sector, monthly | Loaded |
| LatinobarÃ³metro 2017â€“2023 | AI/robot job-displacement perception (Mexico) | Loaded |

---

## Project Structure

```
ai-automation-risk-jalisco/
â”œâ”€â”€ README.md
â”œâ”€â”€ analysis/                  â€” statistical models (EFA, Level-1 hurdle)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                   â€” sources + index builders (build_*.py, load_*.py)
â”‚   â”œâ”€â”€ processed/             â€” built indices (CSV)
â”‚   â””â”€â”€ DATA_INDEX.md
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ SQL_SERVER_SCHEMA.md   â€” live DB inventory
â”‚   â””â”€â”€ DATA_SOURCES.md
â””â”€â”€ legacy/                    â€” superseded coursework demo (see legacy/README.md)
```

---

## Current Status

_Last updated: 2026-06-10._

- [x] Data collection complete (ENOE, O*NET, INEGI, IMSS, LatinobarÃ³metro, Epoch, index battery)
- [x] **DBOE** built and validated (r = 0.94 vs published AIOE)
- [x] **DEOE** built and validated (r = +0.76 vs Webb robot exposure)
- [x] Moravec/RL reclassified as cognitive (empirical finding)
- [x] EFA: bipolar cognitiveâ†”physical structure (H2)
- [x] Official SINCOâ†”SOC crosswalk; 88â€“89% of Jalisco workers with exposure attached
- [x] SINCO division gradient with corrected official labels (H1)
- [x] Jalisco employment-weighted exposure profile (H3)
- [x] Level-1 hurdle model vs Frey-Osborne and published rivals
- [x] DEOE dynamic layer: r(t) world robot-stock curve (IFR/OWID) + Mexico installations, 3 scenarios 2025â€“2030
- [x] Level-2 sector projection 2025â€“2030 (exposure Ã— IRA moderation Ã— technology curves, IMSS anchor; 3 scenarios)
- [x] Municipal automation-pressure map (125 municipios, CE2023 sector mix Ã— exposure; `figures/`)
- [ ] Perception chapter (LatinobarÃ³metro, scale harmonization)
- [ ] Port core-table DDL from `mcd_cucea` for full reproducibility

---

## Key References

Acemoglu, D., & Restrepo, P. (2018). The race between man and machine. *American Economic Review*, 108(6), 1488â€“1542.

Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). GPTs are GPTs: An early look at the labor market impact potential of large language models. *arXiv:2303.10130*.

Felten, E., Raj, M., & Seamans, R. (2021). Occupational, industry, and geographic exposure to artificial intelligence. *Strategic Management Journal*, 42(12), 2195â€“2217.

Frey, C. B., & Osborne, M. A. (2017). The future of employment. *Technological Forecasting and Social Change*, 114, 254â€“280.

Gmyrek, P., Berg, J., & Bescond, D. (2023). *Generative AI and jobs*. ILO Working Paper 96.

Handa, K., et al. (2025). *The Anthropic Economic Index*. Anthropic.

Webb, M. (2020). *The impact of artificial intelligence on the labor market*. Stanford University working paper.

---

## Contact

Carlos Pulido Rosas Â· carlos.pulido.rosas@gmail.com
CUCEA â€” Universidad de Guadalajara
GitHub: [github.com/carpuro](https://github.com/carpuro)
