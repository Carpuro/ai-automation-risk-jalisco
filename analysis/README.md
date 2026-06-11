# Analysis scripts — map and run order

Every statistical result in the thesis is produced by one script in this
folder, reading from SQL Server tables built by `data/raw/*.py` and writing
results back as tables (+ CSV in `data/processed/`, figures in `figures/`).
This file is the reproducibility map: **what each script answers, what it
needs, what it produces.**

All scripts assume the local SQL Server (`localhost,1433`,
`ai_automation_risk_jalisco`, trusted connection) and the Anaconda Python
at `C:\Users\carlo\anaconda3\python.exe`.

## Construct structure

| Script | Question | Key inputs | Outputs |
|---|---|---|---|
| `exposure_factor_structure.py` | EFA: one factor or two? | `model_exposure_soc`, `external_index_comparison` | printed (bipolar F1, 57% var) |
| `exposure_cfa.py` | CFA: 2 correlated axes vs 1 factor | same | printed (ΔAIC +2277, φ=−0.67) |

## Level-1: occupation model

| Script | Question | Key inputs | Outputs |
|---|---|---|---|
| `level1_exposure_model.py` | Do DBOE/DEOE explain observed AI usage beyond Frey-Osborne? (hurdle: logit + positive-part OLS) | `model_exposure_soc`, `external_index_comparison` | printed (AUC .66→.85; DEOE β=−.76 intensive) |
| `aei_mexico_validation.py` | Does Mexican usage match the US-derived target? | `data/raw/aei/*.csv` | `aei_mexico_soc` (r=.97) |

## Demand side (does the market act on it?)

| Script | Question | Key inputs | Outputs |
|---|---|---|---|
| `h4_adoption_test.py` | Did high-IRA sectors actually substitute? (CE panel 2003–2023) | `external_ira_by_sector_full`, `sector_exposure_profile` | `h4_adoption_test` + figure |
| `chatgpt_event_imss.py` | Formal-employment response post-ChatGPT | `imss_empleo_sector`, `sector_exposure_profile` | `chatgpt_event_imss` + figure |
| `absorption_informality.py` | Same on TOTAL employment + informality absorption | `enoe_jalisco_quarterly`, `sector_exposure_profile` | `absorption_test` + figure |
| `wage_premium.py` | Is exposed work already paid more? (Mincer) | `enoe_jalisco_workers`, `sinco4_exposure_scores` | `wage_premium` |
| `reinstatement_emerging_tasks.py` | Do exposed occupations also gain tasks? | `onet_emerging_tasks`, `model_exposure_soc` | `reinstatement_emerging` |
| `nearshoring_channel.py` | FDI as the robot-frontier import channel | `data/raw/nearshoring/ied_entidad_tipo.csv`, IMSS, robot curve | `nearshoring_fdi` + figure |

## Level-2: projection and incidence

| Script | Question | Key inputs | Outputs |
|---|---|---|---|
| `level2_sector_projection.py` | Sector pressure scenarios 2025–2030 (exposure × curves × IRA) | workers, `sinco4_exposure_scores`, IRA, IMSS, `robot_capability_curve`, Epoch zip | `sector_exposure_profile`, `sector_pressure_projection` |
| `workers_at_risk.py` | Headline counts: workers above today's high-pressure bar by 2030 | Level-2 prerequisites | `workers_at_risk`, `workers_at_risk_profile` + figure |
| `worker_profile.py` | Who holds the exposed jobs (educ/income/sex/formality) | `enoe_jalisco_workers`, `sinco4_exposure_scores` | `worker_exposure_profile` + figure |
| `informality_severity.py` | Unprotected at-risk population | workers-at-risk chain | `informality_severity` |
| `state_exposure_comparison.py` | Jalisco vs the 32 states | cached ENOE 2024T3 zip, `sinco4_exposure_scores` | `state_exposure_comparison` + figure |
| `perception_latinobarometro.py` | Do workers expect displacement? (4 harmonized waves) | `latinobarometro_mx` | `perception_trend`, `perception_breakdown_2023` + figure |

## Hardening (inference and robustness)

| Script | Question | Outputs |
|---|---|---|
| `permutation_inference.py` | Exact p-values for the small-N sector tests (B=1000) | `permutation_inference` — H4 p=.013/.004; ChatGPT +3.3% → p=.255 (overclaim corrected) |
| `crosswalk_coverage_bounds.py` | Can the 12% uncovered workers flip H3? | `crosswalk_coverage_bounds` — no (bounds + H3-conservative direction) |
| `robustness_batch.py` | Threshold sensitivity, DBOE leave-one-benchmark-out, H4 reverse-timing placebo | `robustness_threshold`, `robustness_dboe_loo`, `robustness_h4_placebo` |

## Run order (full rebuild)

Builders first (`data/raw/`): `build_dynamic_aioe` → `load_new_tables` /
`load_model_tables` → `build_embodied_exposure` → `build_sinco_crosswalk` →
`build_sinco_exposure` → `build_ira` → `build_robot_capability` →
`build_enoe_quarterly` → `build_municipal_exposure`. Then this folder:
structure → Level-1 → demand side → Level-2 → hardening (scripts state
their own prerequisites in the docstring).

Theory behind each test: `docs/THEORETICAL_FRAMEWORK.md` (§9 maps theory →
hypothesis → script → result). DB inventory: `docs/SQL_SERVER_SCHEMA.md`.
