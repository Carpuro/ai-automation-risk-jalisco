# SQL Server — `ai_automation_risk_jalisco` (katana)

Inventario y procedencia de la base de datos de la tesis.
Levantado por inspección directa el **2026-05-30**.

## Conexión
- **Servidor:** SQL Server 2025 (RTM 17.0) en katana (Windows), `localhost,1433`
- **Base:** `ai_automation_risk_jalisco`
- **Auth:** `Trusted_Connection=yes` (Windows auth local). Driver ODBC: `ODBC Driver 17/18 for SQL Server` o `SQL Server`.
- **SQLAlchemy:** `mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco?trusted_connection=yes`
- **Python:** usar `C:\Users\carlo\anaconda3\python.exe` (el binario de `envs\MCD` falla al invocarse directo en PowerShell con exit 127).

> ⚠️ El bloque "Estado actual de la base de datos" de `data/DATA_INDEX.md` quedó
> obsoleto: declaraba la mayoría de tablas como pendientes. **Este documento es la
> fuente de verdad** del contenido real del servidor.

---

## Tablas (25) — inventario verificado

### Núcleo de modelado
| Tabla | Filas | Contenido | Creada por |
|---|---|---|---|
| `trabajadores` | 2,000 | Muestra ML (sinco_code, escoacum, ingocup, scian_sector, formalidad, edad, tamanio_empresa, ira, urban_rural) | Programación II `02_insercion.py` (mcd_cucea) — **a confirmar** |
| `ocupaciones_onet` | 8 | Scores por grupo SINCO (1,2,3,4,6,7,8,9 — faltan 5 y 0). Columnas Block 3 placeholder sintéticas + reales `dboe_2026_z` y `digital_intensity_z` (ver notas) | ídem + `load_new_tables.py` §5 + `build_digital_intensity.py` |
| `vista_riesgo_jalisco` (vista) | — | JOIN trabajadores + ocupaciones_onet | `03_joins_views.sql` — **a confirmar** |

### ENOE Q3 2024 (microdatos)
| Tabla | Filas | Contenido | Creada por |
|---|---|---|---|
| `enoe_sdemt_full` | 13,839 | Sociodemográfico Jalisco completo (31 cols) | loader DB — **a confirmar** (CSV de `process_enoe.py`) |
| `enoe_coe1_jalisco` | 11,352 | COE1 Jalisco (32 cols: tipo_local, tipo_contrato, horas, llaves) | ídem |

> Nota de exploración: la batería `p5f` de condiciones físicas NO se recolectó en
> esta oleada; `enoe_coe1_jalisco` no contiene condiciones físicas usables.

### O*NET 28.3 (detalle)
| Tabla | Filas | Creada por |
|---|---|---|
| `onet_work_context` | 289,173 | loader DB desde `data/raw/onet/db_28_3_text/` — **a confirmar** |
| `onet_task_ratings` | 161,847 | ídem |
| `onet_work_activities_detail` | 71,586 | ídem |
| `onet_skills` | 61,110 | ídem |
| `onet_education_training` | 35,881 | ídem |
| `onet_technology_skills` | 32,470 | ídem (example, commodity, hot_technology, in_demand) |
| `onet_task_statements` | 19,281 | ídem |
| `onet_emerging_tasks` | 218 | ídem |

### Índices de exposición a IA / LLM — creadas por `data/raw/process_external.py`
| Tabla | Filas | Grano | Score |
|---|---|---|---|
| `external_aioe` | 774 | SOC | aioe_score, lm_aioe_score (Felten) |
| `external_anthropic_job` | 756 | SOC | anthropic_observed_exposure |
| `external_task_penetration` | 17,998 | tarea O*NET | claude_penetration |
| `external_ilo_wp140` | 3,265 | tarea ISCO | score_2023, score_2025 |
| `external_ilo_wp140_occ` | 427 | ISCO-08 | ilo_exposure_2023/2025 |
| `external_moravec_occ` | 923 | SOC | auto_w + componentes Moravec |
| `external_moravec_comparison` | 923 | SOC | Moravec + Eloundou (alpha/beta/gamma) |
| `external_rl_feasibility` | 17,951 | tarea | rl_index |
| `external_rl_feasibility_occ` | 894 | SOC | rl_index_mean, rl_index_max |
| `external_esco_crosswalk` | 8,627 | ISCO↔SOC | puente crosswalk |
| `sinco_aioe_scores` | 9 | SINCO mayor | aioe_mean, lm_aioe_mean, anthropic_obs_mean |

### Económico — creadas por `data/raw/process_external.py`
| Tabla | Filas | Contenido |
|---|---|---|
| `external_inegi_ce2024` | 95 | CE2024 agregado por sector (unidades, personal_rem, remuneraciones, activo_fijo) |
| `external_ira_by_sector` | 19 | IRA por sector (archivo CE2024 **viejo**, solo 2023, sin Q000B/Q400A). **Superado** por `external_ira_by_sector_full` (ver datos nuevos) |

---

## Orden de reproducción (scripts en este repo)
1. `data/raw/process_onet.py` → CSVs O*NET en `data/processed/`
2. `data/raw/process_enoe.py` → `enoe_jalisco.csv`
3. `data/raw/build_crosswalk.py` → `sinco_group_scores.csv`, `isco4_onet_scores.csv`
4. `data/raw/process_external.py` → carga todas las tablas `external_*` + `sinco_aioe_scores` a SQL Server
5. `data/raw/build_dynamic_aioe.py` → `dynamic_aioe_scores.csv`, `sinco_dboe_scores.csv` (DBOE)
6. `data/raw/load_new_tables.py` → carga DBOE + IMSS + Latinobarómetro a SQL Server; integra DBOE en `ocupaciones_onet`
7. `data/raw/build_ira.py` → `external_ira_by_sector_full` (IRA longitudinal 2003–2023 del CE2024 completo)
8. `data/raw/build_digital_intensity.py` → `sinco_digital_intensity` + columna `ocupaciones_onet.digital_intensity_z`

> **Gap de procedencia:** el DDL/carga de las tablas núcleo (`trabajadores`,
> `ocupaciones_onet`), ENOE full y O*NET detalle no está en este repo. Probable
> origen: scripts SQL de Programación II (`01_esquema.sql`, `02_insercion.py`,
> `03_joins_views.sql`) en el monorepo `mcd_cucea`. Confirmar y portar a este repo.

---

## Datos nuevos cargados (2026-05-30) — creadas por `data/raw/load_new_tables.py`
| Tabla | Filas | Contenido |
|---|---|---|
| `dynamic_aioe_scores` | 759 | DBOE por SOC (dboe_2022..2026 + dboe_2026_z) — índice propio |
| `sinco_dboe_scores` | 10 | DBOE agregado a grupo SINCO mayor |
| `imss_empleo_sector` | 2,682 | Empleo formal IMSS Jalisco, tidy long (sector, anio, mes, fecha, trabajadores); 9 sectores × ene-2000 a oct-2024. Total Jalisco 1.03M (2000) → 2.05M (2024), con caída COVID en 2020 |
| `latinobarometro_mx` | 4,800 | Subset México (idenpa=484) × 4 oleadas (2017/18/20/23); cols: year, weight, age, sex, robot_jobs_perception, robot_var, internet_home |
| `external_ira_by_sector_full` | 95 | IRA recalculado del CE2024 completo — **serie longitudinal sector × año** (19 sectores SCIAN × 5 censos 2003/08/13/18/23). Variantes `ira_base`, `ira_real` (depreciación Q000B), `ira_tech` (cómputo Q400A). Creada por `data/raw/build_ira.py` |
| `sinco_digital_intensity` | 10 | Intensidad digital real por grupo SINCO desde O\*NET Technology Skills (tech_skill_count, hot_tech_count, in_demand_count + z). Reemplaza el `digital_access` sintético. Gradiente: Directivos/Profesionistas (z≈+1.6/+1.8) → Operadores/Elementales (z≈−1.0). Creada por `data/raw/build_digital_intensity.py` |

> ⚠️ **Caveat Latinobarómetro:** el ítem de percepción robots/IA-desplazan-empleo
> fue verificado manualmente por oleada (las etiquetas Stata se truncan a 80 chars
> y rompen la detección por keyword):
> - 2017 `P56N_A` — "IA y robótica harán desaparecer la mayoría de empleos" (marco **social/general**)
> - 2018 `P61N` — "robots van a quitar tu empleo" (**personal**)
> - 2020 `p29n_e` — "en 10 años los robots habrán quitado mi puesto" (**personal**)
> - 2023 `P30STIN_A` — "robots quitarán mi empleo en 10 años" (**personal**)
>
> 2017 es de marco social y las escalas difieren entre oleadas → la comparación
> longitudinal directa requiere armonización (documentar en el capítulo de percepción).

## Tablas de modelado cargadas (2026-05-31) — creadas por `data/raw/load_model_tables.py`
Consolidación: dejar el server como fuente única antes de buscar el crosswalk SINCO↔ISCO.
| Tabla | Filas | Grano | Contenido |
|---|---|---|---|
| `enoe_jalisco_workers` | 6,147 | trabajador ocupado | **Ocupación SINCO-4 díg granular** (`sinco4`, **340 ocupaciones** distintas) recuperada de COE1 `p3` vía join SDEMT+COE1 por llave de persona. Recupera el grano fino que el `enoe_coe1_jalisco` viejo había perdido (solo traía 11 grupos `c_ocu11c`). Incluye fac_tri, sex, eda, anios_esc, scian, seg_soc, emple7c, ingocup, hrsocup |
| `crosswalk_isco4_onet_scores` | 436 | ISCO-08 4 díg | features Frey-Osborne derivados de O*NET (cognitive_demand, social, creativity, manual_dexterity, rti, frey_osborne_score); col `sinco_major` VACÍA (gap del puente SINCO↔ISCO) |
| `crosswalk_onet_scores` | 1,016 | O*NET SOC | scores O*NET por ocupación |
| `crosswalk_sinco_group_scores` | 10 | SINCO mayor | scores agregados a grupo |
| `model_exposure_soc` | **667** | SOC 6 díg | **Tabla Nivel-1 consolidada.** Cognitivo: dboe_2026/dboe_2026_z, aioe_score, lm_aioe_score, anthropic_observed_exposure, moravec_auto_w, rl_index_mean (estos 2 son cognitivos, ver corrección abajo). Físico (DEOE, 2026-05-31): phys_manual, phys_machine, phys_vehicle_field, phys_routine, phys_dexterity_bottleneck, embodied_exposure, embodied_exposure_z. Ahora 20 cols. Base del modelo de exposición de 2 ejes |
| `embodied_exposure_soc` | 759 | SOC 6 díg | **DEOE — Dynamic Embodied Occupational Exposure (núcleo estático).** Índice físico propio desde O*NET (importancia×nivel z, espejo del W_ok del DBOE). 5 subdominios (phys_manual, phys_machine, phys_vehicle_field, phys_routine, phys_dexterity; Cronbach α 0.57–0.95) + resumen `embodied_exposure` = **PC1 (63% varianza)**. **Validado convergente: r=+0.76 vs Webb 2020 robot-patent exposure**, ~0 vs AI/AIOE. Creada por `data/raw/build_embodied_exposure.py` |

> **⚠️ CORRECCIÓN DE CORRELACIONES (2026-05-31, deja obsoleta la nota previa):**
> Al construir el DEOE se midió que **`moravec_auto_w` y `rl_index_mean` NO son
> físicos — son cognitivos** (paradoja de Moravec): correlacionan −0.66/−0.72 con
> trabajo físico real (`phys_manual`) y +0.74/+0.78 con DBOE/AIOE. El viejo
> "1 factor dominante / físico-cognitivo 0.70-0.78" era artefacto: los 5 índices
> eran TODOS cognitivos. El **DEOE (`embodied_exposure_z`) es el primer eje físico
> real**, ortogonal a los cognitivos (r −0.10 a −0.20). DBOE↔AIOE=0.96 sigue válido
> (colineales). Anthropic_observed sigue siendo el mejor target externo no circular.

> **Pendiente que NO es de datos locales:** puente SINCO↔ISCO-08 (correspondencia
> oficial INEGI) para conectar `enoe_jalisco_workers.sinco4` (340 occ) con
> `crosswalk_isco4_onet_scores` (436 ISCO) y `model_exposure_soc` (667 SOC).
> Único hueco que requiere descargar un dato externo.

### Integración DBOE → `ocupaciones_onet` (2026-05-31, `load_new_tables.py` §5)
Se agregó la columna `dboe_2026_z` (exposición LLM real, validada r=0.94) a
`ocupaciones_onet`, poblada por join SINCO. **No destructivo:** las columnas Block 3
sintéticas previas se conservan.

> ⚠️ Las columnas Block 3 sintéticas (`gpt_exposure_score`, `moravec_index`,
> `dual_factor_score`, `iceberg_score`) tenían valores **invertidos** — p.ej.
> agropecuario salía como la ocupación más expuesta a GPT (0.76) y directivo como
> la menos (0.20). El `dboe_2026_z` corrige el orden (Profesionista +0.88 →
> Operador −1.02). Para el modelo Phase 2 usar `dboe_2026_z`, no las sintéticas.
>
> `vista_riesgo_jalisco` enumera columnas explícitamente y NO recoge `dboe_2026_z`
> automáticamente; exponerla en la vista queda pendiente para el modelo Phase 2.
