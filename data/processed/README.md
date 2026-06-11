# data/processed — generated outputs (CSV mirrors of SQL Server tables)

Everything here is **generated** — never edit by hand; rerun the producing
script instead (see `analysis/README.md` and `data/raw/README.md` for the
script map). The authoritative store is SQL Server
(`ai_automation_risk_jalisco`, inventory in `docs/SQL_SERVER_SCHEMA.md`);
these CSVs are convenience mirrors of selected result tables. Most are
gitignored.

## Index construction

| File | Producer |
|---|---|
| `dynamic_aioe_scores.csv` (DBOE per SOC, 2022–2026 + z) | `build_dynamic_aioe.py` |
| `sinco_dboe_scores.csv` (superseded — crude bridge, wrong labels) | `build_dynamic_aioe.py` |
| `sinco_exposure_scores.csv` (DBOE+DEOE per SINCO division, official) | `build_sinco_exposure.py` |
| `robot_capability_curve.csv` (r(t), 3 scenarios) | `build_robot_capability.py` |
| `onet_scores.csv`, `isco4_onet_scores.csv`, `sinco_group_scores.csv`, `enoe_jalisco.csv` | legacy-era processing (`process_*.py`) |

## Results (analysis outputs)

| File | Producer |
|---|---|
| `sector_exposure_profile.csv`, `sector_pressure_projection.csv` | `level2_sector_projection.py` |
| `workers_at_risk.csv` | `workers_at_risk.py` |
| `worker_exposure_profile.csv` | `worker_profile.py` |
| `informality_severity.csv` | `informality_severity.py` |
| `municipal_exposure.csv` | `build_municipal_exposure.py` |
| `state_exposure_comparison.csv` | `state_exposure_comparison.py` |
| `enoe_jalisco_quarterly.csv` | `build_enoe_quarterly.py` |
| `absorption_test.csv` | `absorption_informality.py` |
| `h4_adoption_test.csv` | `h4_adoption_test.py` |
| `chatgpt_event_imss.csv` | `chatgpt_event_imss.py` |
| `wage_premium.csv` | `wage_premium.py` |
| `reinstatement_emerging.csv` | `reinstatement_emerging_tasks.py` |
| `perception_trend.csv` | `perception_latinobarometro.py` |
| `aei_mexico_soc.csv` | `aei_mexico_validation.py` |
| `nearshoring_fdi.csv` | `nearshoring_channel.py` |
| `crosswalk_coverage_bounds.csv`, `permutation_inference.csv`, `robustness_*.csv` | hardening scripts |
