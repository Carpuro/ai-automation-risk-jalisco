# Labor Automation Risk from AI in Jalisco, Mexico

**Thesis — Maestría en Ciencias de los Datos**  
Carlos Pulido Rosas · CUCEA, Universidad de Guadalajara · 2025–2026

---

## Research Question

> *To what extent does specific exposure to large language models (LLMs) modify occupational automation risk beyond what the Frey-Osborne model predicts, and what economic incentive determines whether that substitution actually occurs in Jalisco's labor market?*

This study moves beyond technical feasibility (Frey-Osborne, 2013) in two directions: (1) it incorporates LLM-specific exposure measures that invert the original model's assumptions about non-routine cognitive tasks, and (2) it introduces an economic incentive variable that determines whether automation is not just possible but profitable for firms — following the task-based framework of Acemoglu & Restrepo (2018).

---

## Hypotheses

1. LLM exposure significantly predicts automation risk for non-routine cognitive occupations, independent of the Frey-Osborne score (H1).
2. The Automation Profitability Index (IRA: annual wage / automation cost proxy) moderates the relationship between technical risk and actual adoption (H2).
3. Education level remains the strongest protective factor, but its effect is weaker for language-intensive occupations with high LLM exposure (H3).
4. Agriculture and manufacturing retain the highest risk; white-collar clerical occupations show higher risk than Frey-Osborne predicted (H4).

---

## Variable Structure

### Block 1 — Worker profile (ENOE)
`education`, `age`, `income (INGOCUP)`, `sector (SCIAN)`, `formality`, `firm size`, `urban/rural`

### Block 2 — Task profile (O*NET, Phase 1)
`routine_task_intensity (RTI)`, `frey_osborne_score`, `cognitive_demand`, `social_interaction`, `creativity`

### Block 3 — LLM exposure (Phase 2)
`dboe` — **Dynamic LLM Occupational Exposure** (own contribution): extends Felten's AIOE with real Epoch AI benchmark scores per year. Validated against the published AIOE (r = 0.94). See `data/raw/build_dynamic_aioe.py`.  
`gpt_exposure_score` — ILO WP140 task-level exposure (2023 + predicted 2025), crosswalked ISCO → SINCO  
`moravec_index` — Arora et al. (2025), model-agnostic robustness check  
`anthropic_observed_exposure` — Anthropic Economic Index, observed vs. theoretical  
`rl_feasibility` — RL learnability index (2030 horizon)  
`aioe` — AI Occupational Exposure Index, Felten et al. (2021), used as control  

### Block 4 — Economic incentive
`ira` — Automation Profitability Index: `annual_wage / capital_intensity_proxy`  
Source: INGOCUP (ENOE) + fixed assets per worker (INEGI Censos Económicos 2019)

---

## Methodology

**Phase 1 (complete):** Frey-Osborne baseline with ENOE Jalisco data. Random Forest R² ≈ 0.75.  
Key finding: agriculture at highest risk; education is the dominant protective factor (77–81% feature importance).

**Phase 2 (in progress):** Add Blocks 3 and 4. Model specification:

```
automation_risk = f(
    Block 1: ENOE worker profile,
    Block 2: O*NET task profile (Frey-Osborne baseline),
    Block 3: LLM exposure (GPT score + LTII + AIOE),
    Block 4: IRA economic incentive
)
```

**Statistical validation:**
- Pearson vs. Spearman correlation — detect non-linearity before model selection
- Ramsey RESET — test OLS functional form
- VIF — multicollinearity between education, income, sector
- Generalized Additive Models (GAM) — non-linear baseline for comparison
- SHAP values — variable importance interpretation for Random Forest
- Confirmatory Factor Analysis (CFA) — validate LLM exposure construct

---

## Data Sources

All sources below are downloaded and (except where noted) loaded into SQL Server.
See [`docs/SQL_SERVER_SCHEMA.md`](docs/SQL_SERVER_SCHEMA.md) for the full DB inventory
and [`data/DATA_INDEX.md`](data/DATA_INDEX.md) for source details.

| Source | Content | Status |
|---|---|---|
| ENOE Q3 2024, ent=14 | Jalisco worker microdata (SDEMT 13,839 / COE1) | Loaded |
| O*NET 28.3 | Occupation task/ability/context descriptors (detail) | Loaded |
| ESCO ISCO ↔ SOC crosswalk | bridge SINCO → ISCO → SOC | Loaded |
| ILO WP140 (2025) | task-level GenAI exposure (2023 + predicted 2025) | Loaded |
| Felten et al. (2021) | AIOE by occupation | Loaded |
| Arora et al. (2025) Moravec | model-agnostic exposure | Loaded |
| Anthropic Economic Index | observed exposure by occupation | Loaded |
| Epoch AI Capabilities | LLM benchmark scores (DBOE input) | Downloaded |
| INEGI Censos Económicos 2024 | capital/labor by SCIAN + municipio (IRA) | Loaded |
| INEGI PIBE 2003–2022 | sectoral GDP Jalisco | Downloaded |
| IMSS (IIEG) 2000–2024 | formal employment by sector, monthly | Loaded |
| Latinobarómetro 2017–2023 | AI/robot job-displacement perception (Mexico) | Loaded |

---

## Project Structure

```
ai-automation-risk-jalisco/
├── README.md
├── environment.yml
├── requirements.txt
├── data/
│   ├── raw/            — ENOE, O*NET source files
│   ├── mappings/       — SOC-SINCO crosswalk
│   ├── processed/      — cleaned, joined datasets
│   └── sample/         — sample for testing
├── notebooks/
│   └── automation_risk_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── automation_analyzer.py
│   ├── statistical_inference.py
│   ├── visualizations.py
│   └── main.py
├── outputs/
│   ├── models/
│   ├── visualizations/
│   └── reports/
└── docs/
    ├── METHODOLOGY.md
    ├── DATA_SOURCES.md
    └── ANALYSIS_GUIDE.md
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate ai_automation_thesis
python verify_setup.py
jupyter notebook notebooks/automation_risk_analysis.ipynb
```

---

## Current Status

_Last updated: 2026-05-30. DB inventory: [`docs/SQL_SERVER_SCHEMA.md`](docs/SQL_SERVER_SCHEMA.md)._

- [x] Phase 1: Frey-Osborne baseline (Random Forest R² = 0.75)
- [x] ENOE Jalisco data processed and loaded (SDEMT 13,839 / COE1 11,352)
- [x] O*NET descriptors integrated (task ratings, work context, skills, abilities)
- [x] SINCO → ISCO → SOC crosswalk (ESCO) built and loaded
- [x] Data collection complete (ENOE, O*NET, INEGI CE2024/PIBE, IMSS, Latinobarómetro, Epoch)
- [x] Block 3 indices loaded (AIOE, ILO WP140, Moravec, Anthropic, RL feasibility)
- [x] **DBOE** dynamic LLM-exposure index built and validated (r = 0.94 vs published AIOE)
- [x] DBOE / IMSS / Latinobarómetro loaded to SQL Server
- [ ] Feed real DBOE into `ocupaciones_onet.gpt_exposure_score` (currently placeholder)
- [ ] IRA economic incentive (Block 4) — recompute from CE2024 full file (Q000C/Q400A)
- [ ] Phase 2 model M1→M4 (hierarchical, incremental F-test)
- [ ] ENOE COE1 `digital_access` to be sourced from O*NET (p5f not collected in Q3 2024)
- [ ] Statistical validation (RESET, VIF, GAM, SHAP, CFA)
- [ ] Port core-table DDL from `mcd_cucea` into this repo for full reproducibility

---

## Key References

Acemoglu, D., & Restrepo, P. (2018). The race between man and machine. *American Economic Review*, 108(6), 1488–1542.

Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). GPTs are GPTs: An early look at the labor market impact potential of large language models. *arXiv:2303.10130*.

Felten, E., Raj, M., & Seamans, R. (2023). How will language models use tool use, planning, and reasoning? *SSRN Working Paper*.

Frey, C. B., & Osborne, M. A. (2017). The future of employment. *Technological Forecasting and Social Change*, 114, 254–280.

Gmyrek, P., Berg, J., & Bescond, D. (2023). *Generative AI and jobs: A global analysis of potential effects on job quantity and quality*. ILO Working Paper 96.

Nedelkoska, L., & Quintini, G. (2018). *Automation, skills use and training*. OECD Social, Employment and Migration Working Papers, No. 202.

---

## Contact

Carlos Pulido Rosas · carlos.pulido.rosas@gmail.com  
CUCEA — Universidad de Guadalajara  
GitHub: [github.com/carpuro](https://github.com/carpuro)
