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
`gpt_exposure_score` — Eloundou et al. (2023), crosswalked SOC → SINCO  
`ltii` — LLM Task Intensity Index, constructed from O*NET items  
`aioe` — AI Occupational Exposure Index, Felten et al. (2023), used as control  

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

| Source | Content | Status |
|---|---|---|
| ENOE Q3 2024, ent=14 | Jalisco worker microdata | Available |
| O*NET 28.3 | Occupation task descriptors | Available |
| SOC → SINCO crosswalk | INEGI equivalence table | Pending |
| Eloundou et al. (2023) | GPT Exposure Scores by SOC | Available (paper) |
| Felten et al. (2023) | AIOE by occupation | Available (paper) |
| Censos Económicos INEGI 2019 | Capital intensity by SCIAN sector | Available |
| ENAPROCE | Technology adoption rate by state | Available |

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

- [x] Phase 1: Frey-Osborne baseline (Random Forest R² = 0.75)
- [x] ENOE Jalisco data processed
- [x] O*NET task descriptors integrated
- [ ] SOC-SINCO crosswalk — pending acquisition
- [ ] LLM exposure scores (Block 3) — pending crosswalk
- [ ] IRA economic incentive variable (Block 4) — pending Censos Económicos integration
- [ ] Phase 2 model — pending Blocks 3 and 4
- [ ] Statistical validation (RESET, VIF, GAM, SHAP, CFA)

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
