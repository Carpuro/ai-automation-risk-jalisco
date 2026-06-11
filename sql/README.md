# sql — core-table DDL (ported from mcd_cucea)

DDL for the tables that this repo's loaders assume to exist but did not
create. Ported 2026-06-11 from
`mcd_cucea/2do-semestre/programacion-ii/proyectos/ProyectoFinal/` for full
reproducibility of the SQL Server database from scratch.

| File | Creates | Status in the thesis |
|---|---|---|
| `01_core_schema.sql` | `ocupaciones_onet`, `trabajadores` | Demo-era core (Programación II). `trabajadores` (n=2,000 sample) and the Block-3 placeholder columns of `ocupaciones_onet` are superseded by `enoe_jalisco_workers` and the real indices; `ocupaciones_onet` remains in use as the SINCO-group table carrying the non-destructive `dboe_2026_z` / `digital_intensity_z` additions. |
| `02_views.sql` | `vista_riesgo_jalisco` | **Demo-era — do not use as a modeling target.** Its `automation_risk` is a fixed-weight CASE formula (the circularity documented in `legacy/README.md`). Kept for provenance. |
| `03_detail_tables.sql` | `enoe_sdemt_full`, `enoe_coe1_jalisco`, `onet_task_ratings`, `onet_technology_skills`, `onet_task_statements`, `onet_emerging_tasks`, `onet_skills`, `onet_work_context`, `onet_education_training`, `onet_work_activities_detail` | **Current.** These are the detail tables the real builders read (DEOE reads `onet_work_activities_detail` + `onet_work_context`; digital intensity reads `onet_technology_skills`; reinstatement reads `onet_emerging_tasks`). |

Everything else in the database (~35 result and index tables) is created by
the Python builders via `to_sql` — no separate DDL needed; see
`data/raw/README.md` and `analysis/README.md` for the producer map.

## Rebuild from scratch

1. Create the database `ai_automation_risk_jalisco`.
2. Run `01_core_schema.sql`, then `03_detail_tables.sql` (skip `02_views.sql`
   unless reproducing the legacy demo).
3. Populate detail tables with `data/raw/process_onet.py` /
   `process_enoe.py` / `process_external.py`.
4. Follow the builder run order in `data/raw/README.md`, then the analysis
   run order in `analysis/README.md`.
