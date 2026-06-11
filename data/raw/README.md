# data/raw — sources and builders

Source files (downloaded, some gitignored for size) and the `build_*.py` /
`load_*.py` / `process_*.py` scripts that turn them into SQL Server tables.
Full source documentation: `data/DATA_INDEX.md`; DB inventory:
`docs/SQL_SERVER_SCHEMA.md`.

## Builders (run these; each docstring states its prerequisites)

| Script | Builds | From |
|---|---|---|
| `build_dynamic_aioe.py` | DBOE index (LLM exposure, c_j(t) capability curve) | O*NET Abilities, AIOE appendix, `epoch_ai_capabilities.zip` |
| `build_embodied_exposure.py` | DEOE index (embodied exposure, PC1 of 5 O*NET physical subdomains) | `onet_work_activities_detail`, `onet_work_context` (DB) |
| `build_sinco_crosswalk.py` | `crosswalk_sinco_ciuo`, `crosswalk_sinco_soc` | `crosswalks/mexico_sinco_tablas_comparativas.xlsx` (official INEGI) |
| `build_sinco_exposure.py` | `sinco4_exposure_scores`, `sinco_exposure_scores` (official labels) | DB tables |
| `build_ira.py` | `external_ira_by_sector_full` (IRA, 5 censuses 2003–2023) | `INEGI_CE2024_jalisco_completo.csv` |
| `build_robot_capability.py` | `robot_capability_curve` (r(t), 3 scenarios) | `robotics/owid_robots_global_stock.csv` + IFR PDFs (see `robotics/README.md`) |
| `build_enoe_quarterly.py` | `enoe_jalisco_quarterly` (total employment incl. informal, 2022T1–2024T3) | ENOE microdata zips, auto-downloaded to `enoe/quarters/` (gitignored, ~350 MB) |
| `build_municipal_exposure.py` | `municipal_exposure` + choropleth | CE2024 raw (municipal grain) + `INEGI_Jalisco_municipios.geojson` |
| `build_digital_intensity.py` | `sinco_digital_intensity` | O*NET Technology Skills |
| `load_new_tables.py` / `load_model_tables.py` / `process_external.py` / `process_enoe.py` / `process_onet.py` | core loads (workers, indices battery, IMSS, Latinobarómetro, `model_exposure_soc`, …) | various raw files |

Deprecated (kept for provenance): `build_crosswalk.py` and the SINCO
aggregation inside `build_dynamic_aioe.py` (crude first-digit bridge,
shifted labels) — superseded by the official INEGI crosswalk chain.

## Source folders

- `onet/` — O*NET 28.3 text database
- `enoe/` — ENOE Q3 2024 microdata; `enoe/quarters/` quarterly zips (gitignored)
- `crosswalks/` — official INEGI SINCO comparative tables + manuals
- `robotics/` — IFR executive-summary PDFs + OWID series (sources documented in its README)
- `moravec_index/` — Moravec files incl. `Comparison of Indices.csv` (Webb, Frey-Osborne, SML, Eloundou, routine — the external benchmark battery)
- `aei/` — Anthropic Economic Index country release (25 MB CSV, gitignored)
- `nearshoring/` — Secretaría de Economía FDI by state and type
- Loose files: AIOE appendix, Epoch zip, CE2024 full CSV, Latinobarómetro zips, ILO/RL/Anthropic index files, PIBE, geojson

## Gotchas (hard-won)

- O*NET txt are tab-separated; INEGI/IIEG files come in latin-1/cp1252.
- ENOE SDEMT lacks the 4-digit occupation — it comes from COE1 `p3` via the
  person-key join (`ent,con,upm,v_sel,n_hog,h_mud,n_ren,per`).
- ENOE zip naming: `enoe_n_2022_trimQ_csv.zip` (2022) vs `enoe_YYYY_trimQ_csv.zip` (2023+).
- The `felten` column in `Comparison of Indices.csv` is NOT the published
  AIOE (r=0.26 with it) — use the validated `aioe_score`.
- CE2024 server table is state-level only; the municipal grain lives in the
  raw CSV (skiprows=4, municipio = col 2).
- repodatos.atdt.gob.mx blocks scripted downloads; use the
  datos.gob.mx `/download/` resource paths.
