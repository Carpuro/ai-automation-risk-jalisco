# Data Index — ai_automation_risk_jalisco

Base de datos SQL Server en homelab (localhost:1433 / Tailscale 100.64.0.11:1433).
Última actualización: 2026-05-30.

---

## Estado actual de la base de datos

> ⚠️ **DESACTUALIZADO.** El estado real de SQL Server (25 tablas, ENOE/O\*NET/índices
> ya cargados) está documentado en [`docs/SQL_SERVER_SCHEMA.md`](../docs/SQL_SERVER_SCHEMA.md)
> (verificado 2026-05-30). La tabla siguiente refleja el estado del proyecto de
> Programación II y se conserva solo como referencia histórica.

| Tabla / Vista | Filas | Estado | Fuente |
|---|---|---|---|
| `trabajadores` | 2,000 | ✅ Cargada | ENOE SDEMT Q3 2024 — muestra Jalisco |
| `ocupaciones_onet` | 8 | ✅ Cargada | O\*NET 28.3 — promediado a grupos SINCO |
| `vista_riesgo_jalisco` | 2,000 | ✅ Vista activa | JOIN trabajadores + ocupaciones_onet |
| `enoe_sdemt_full` | — | ⏳ Pendiente | ENOE SDEMT Q3 2024 — todos Jalisco (13,839) |
| `enoe_coe1_jalisco` | — | ⏳ Pendiente | ENOE COE1 Q3 2024 — condiciones de trabajo Jalisco |
| `onet_task_ratings` | — | ⏳ Pendiente | O\*NET 28.3 Task Ratings |
| `onet_technology_skills` | — | ⏳ Pendiente | O\*NET 28.3 Technology Skills |
| `onet_task_statements` | — | ⏳ Pendiente | O\*NET 28.3 Task Statements |
| `onet_emerging_tasks` | — | ⏳ Pendiente | O\*NET 28.3 Emerging Tasks |
| `onet_skills` | — | ⏳ Pendiente | O\*NET 28.3 Skills |
| `onet_work_context` | — | ⏳ Pendiente | O\*NET 28.3 Work Context |
| `onet_education_training` | — | ⏳ Pendiente | O\*NET 28.3 Education, Training & Experience |
| `onet_work_activities_detail` | — | ⏳ Pendiente | O\*NET 28.3 Work Activities (detalle) |

---

## Fuentes de datos — detalle

### 1. ENOE Q3 2024 (INEGI)
**Ruta local:** `data/raw/enoe/`
**URL:** https://www.inegi.org.mx/programas/enoe/15ymas/

Encuesta Nacional de Ocupación y Empleo, tercer trimestre 2024.
5 archivos relacionados por clave compuesta: `ent + con + upm + d_sem + n_pro_viv + v_sel + n_hog + h_mud + n_ent + per + n_ren`.

#### ENOE_SDEMT324.csv — Sociodemográfico
- **Filas totales:** 423,118 | **Jalisco (ent=14):** 13,839 | **Columnas:** 114
- **En DB:** 2,000 (muestra aleatoria estratificada por sector)
- **Variables clave usadas:**
  | Variable ENOE | Variable DB | Descripción |
  |---|---|---|
  | sex | sexo | 1=Hombre, 2=Mujer |
  | eda | edad | Edad en años |
  | cs_p17 | nivel_educativo | Escolaridad 1-6 (sin→posgrado) |
  | seg_soc | formalidad | Acceso a seguridad social |
  | ingocup | ingreso_mensual | Ingreso mensual MXN |
  | emple7c | empleo_tipo | Tipo de empleo (7 categorías) |
  | scian | scian_sector | Sector SCIAN |
  | c_ocu11c | c_ocu11c | Código SINCO 11 categorías |
  | t_loc_tri | urban_rural | 1=urbano, 0=rural |
  | anios_esc | — | Años de escolaridad |
  | rama_est2 | rama_est2 | Rama de actividad (2 dígitos) |

#### ENOE_COE1T324.csv — Cuestionario de Ocupación y Empleo Parte 1
- **Filas totales:** 347,258 | **Jalisco:** 11,352 | **Columnas:** 172
- **En DB:** ❌ No cargado
- **Encoding:** latin-1 (no UTF-8).

> ⚠️ **CORRECCIÓN (exploración 2026-05-30):** la batería `p5f` de condiciones
> físicas NO se recolectó en la oleada Q3 2024 — todas las columnas p5f1–p5f13
> están ~0% llenas en Jalisco (p5f7 computadora = 0.0%; solo p5f14 = 49%). El
> supuesto previo de que `p5f7` daría `digital_access` y las p5f darían el RTI
> empírico **queda descartado**. `digital_access` debe salir de O\*NET, no de ENOE.

- **Variables COE1 realmente usables (fill rate Jalisco):**
  | Variable COE1 | Nombre propuesto | Fill | Descripción |
  |---|---|---|---|
  | p1 | tipo_local | alto | Tipo de local de trabajo |
  | p5b_thrs | horas_semana | 54% | Total de horas trabajadas por semana |
  | p5b_tdia | dias_semana | 54% | Días trabajados por semana |
  | p5c | — | 54% | Horas relacionadas |
  | (llaves) | — | 100% | ent, con, upm, ... n_ren (join a SDEMT) |

  > **Conclusión:** COE1 aporta horas/días trabajados y las llaves de join.
  > No aporta el módulo de condiciones físicas esperado.

#### ENOE_COE2T324.csv — COE Parte 2
- **Filas:** 347,258 | **Jalisco:** 11,352 | **Columnas:** 82
- Segundo empleo, actividad por cuenta propia. Menor prioridad para la tesis.

#### ENOE_HOGT324.csv — Hogar
- **Filas:** 151,676 | **Jalisco:** 4,870 | **Columnas:** 37
- Datos del hogar (no del trabajador individual). Menor prioridad.

#### ENOE_VIVT324.csv — Vivienda
- **Filas:** nacional | **Columnas:** 25. Datos de la vivienda. Baja prioridad.

---

### 2. O*NET 28.3 (U.S. Department of Labor)
**Ruta local:** `data/raw/onet/db_28_3_text/`
**URL:** https://www.onetcenter.org/database.html

Base de datos de descriptores ocupacionales para 923 ocupaciones SOC.
Clave primaria: `O*NET-SOC Code` (formato XX-XXXX.XX).
Crosswalk a SINCO: vía IDB México 2024 o ISCO-08 (ver sección Crosswalk).

#### Work Activities.txt — ✅ Parcialmente usado
- Usado para calcular RTI, cognitive_demand, social_interaction, creativity, manual_dexterity.
- Solo el promedio por grupo SINCO está en `ocupaciones_onet`.
- El detalle por SOC code NO está cargado.

#### Task Ratings.txt — ⏳ Pendiente
- **Tamaño:** 11.4 MB
- Ratings de importancia, frecuencia y relevancia para miles de tareas por ocupación.
- Columnas: O*NET-SOC Code, Task ID, Task Type, Scale ID (IM/FT/RL), Data Value, N, SE.
- **Relevancia:** permite análisis a nivel de tarea individual, más granular que promedios de ocupación.

#### Technology Skills.txt — ⏳ Pendiente
- **Tamaño:** 2.5 MB
- Tecnologías específicas requeridas por ocupación (software, hardware, plataformas).
- Hot Technology flag = tecnología actualmente en demanda.
- **Relevancia:** reemplaza `digital_access` binario generado con un indicador real.

#### Task Statements.txt — ⏳ Pendiente
- **Tamaño:** 2.7 MB
- Descripción textual de cada tarea por ocupación.
- **Relevancia:** input para LLM-based scoring manual o automático (Fase 2 tesis).

#### Emerging Tasks.txt — ⏳ Pendiente
- **Tamaño:** 26 KB
- Tareas nuevas emergentes que están apareciendo en cada ocupación.
- **Relevancia:** identifica qué ocupaciones están adoptando tareas AI-relacionadas.

#### Skills.txt — ⏳ Pendiente
- **Tamaño:** 5.3 MB
- Habilidades requeridas (35 habilidades: comprensión lectora, matemáticas, programación, etc.).
- Escalas: Importance (IM) y Level (LV).

#### Work Context.txt — ⏳ Pendiente
- **Tamaño:** 32.9 MB — el más grande
- Condiciones físicas y sociales del entorno de trabajo.
- 57 elementos: nivel de ruido, trabajo al aire libre, contacto con el público, etc.
- **Relevancia:** complementa las variables p5f de COE1 con referencia O*NET.

#### Education, Training, and Experience.txt — ⏳ Pendiente
- **Tamaño:** 3.3 MB
- Nivel educativo típico, años de experiencia, tipo de formación requeridos por ocupación.
- **Relevancia:** valida la variable `nivel_educativo` de ENOE contra el requerimiento de la ocupación.

#### Abilities.txt — ✅ Parcialmente usado
- Usado para parte del cálculo de scores O*NET. No cargado el detalle.

---

### 3. INEGI Censos Económicos 2024 — Jalisco (SAIC)
**Ruta local:** `data/raw/INEGI_CE2024_jalisco_completo.csv`
**Origen:** SAIC INEGI, consulta 2026-05-24 — Solo entidad 14 Jalisco, 125 municipios, todos los sectores SCIAN.
**Filas:** 258,616 | **Columnas:** 105 | **Tamaño:** ~129 MB

Archivo definitivo — reemplaza `INEGI_CE2024_activo_fijo_scian_OLD.csv`.

#### Variables clave para la tesis

| Variable | Descripción | Uso |
|---|---|---|
| `H001A` | Personal ocupado total | Denominador IRA — todos los trabajadores |
| `H010A` | Personal remunerado total | Denominador IRA alternativo |
| `H001B/C` | Personal ocupado hombres/mujeres | Análisis de género |
| `H101A` | Personal de producción, ventas y servicios | Perfil laboral por sector |
| `H203A` | Personal administrativo y de dirección | Perfil laboral por sector |
| `J000A` | Total de remuneraciones | Numerador IRA (costo laboral base) |
| `J300A` | Contribuciones patronales a seguridad social | Costo laboral completo |
| `J400A` | Otras prestaciones sociales | Costo laboral completo |
| `Q000A` | Acervo total de activos fijos | IRA denominador — capital total |
| `Q000B` | **Depreciación total de activos fijos** | IRA mejorado — amortización real |
| `Q000C` | Compra y adquisición de activos fijos | Inversión del año — flujo |
| `Q010A` | Acervo maquinaria y equipo de producción | Capital industrial |
| `Q400A` | **Acervo equipo de cómputo y periféricos** | Capital tecnológico — indicador de automatización activa |
| `A131A` | **Valor agregado censal bruto** | IRA robusto — productividad real |
| `A111A` | Producción bruta total | Contexto económico por sector |

#### IRA mejorado con este archivo
```
IRA_base    = J000A / (Q000A / 5)                          # versión original
IRA_real    = (J000A + J300A + J400A) / Q000B              # costo laboral total / depreciación real
IRA_tech    = (J000A + J300A + J400A) / Q400A              # costo laboral / capital tecnológico
```

---

### 4. INEGI PIBE — PIB por Entidad Federativa Jalisco (2003–2024)
**Ruta local:** `data/raw/INEGI_PIBE_jalisco_2003_2024.xlsx`
**Origen:** INEGI Sistema de Cuentas Nacionales de México, PIBE base 2018. Descargado 2026-05-24.
**Filas:** ~510 conceptos | **Años:** 2003–2024 (2024 preliminar) | **Tamaño:** ~0.1 MB

Serie anual del PIB de Jalisco por actividad económica a precios constantes de 2018.
Hoja principal: `Tabulado` (datos) | `MetaInfo` (metadatos)

#### Variables clave

| Concepto | Descripción | Uso |
|---|---|---|
| PIB total Jalisco | `B.1bP` | Contexto macroeconómico |
| Valor Agregado Bruto | `B.1bV` | Base para participación sectorial |
| Actividades primarias (11) | Agricultura, ganadería, pesca | Sector de alto IRA, bajo riesgo percibido |
| Actividades secundarias (21–33) | Minería, manufactura, construcción | Automatización industrial |
| Actividades terciarias (43–93) | Comercio, servicios, gobierno | Clusters de riesgo medio-alto por LLMs |

#### Uso en la tesis

- **Análisis longitudinal 2003–2024:** evolución del peso sectorial en el PIB vs. riesgo de automatización
- **Orientación económica de Jalisco:** qué sectores dominan y hacia dónde crece la economía
- **Contexto H2 (IRA):** sectores con alto PIB + alto IRA = presión real de sustitución

---

### 5. IMSS — Empleo formal histórico Jalisco (vía IIEG)
**Ruta local:** `data/raw/IMSS_empleo_*.xlsx` y `.zip`
**Origen:** IIEG Jalisco (DIEEF-IIEG), serie del IMSS. Descargado 2026-05-30.
**Nota:** el portal `datos.imss.gob.mx` (datos crudos mensuales 1997–2024) estaba caído (HTTP 503); el IIEG republica las series consolidadas en GitHub. Cobertura máxima del IIEG: **2000–2024**. No hay 1997–1999 consolidado.

| Archivo | Cobertura | Descripción |
|---|---|---|
| `IMSS_empleo_jalisco_por_sector.xlsx` | ene 2000 – oct 2024 (mensual) | 9 sectores económicos IMSS de Jalisco — serie principal para análisis longitudinal capital/empleo |
| `IMSS_empleo_jalisco_sector_actividad.xlsx` | dic 2023 – oct 2024 | Por división/grupo/fracción económica (corto plazo, granular) |
| `IMSS_empleo_jalisco_2015_2023.zip` | 2015–2023 | CSV granular completo (~246 MB descomprimido) |
| `IMSS_empleo_nacional_por_entidad.xlsx` | 2000–2024 (mensual) | 32 entidades federativas — contexto nacional |

**Uso en la tesis:** análisis longitudinal de sustitución capital/trabajo por sector SCIAN (conecta con H2/IRA y los 5 censos económicos INEGI 2003–2023).

---

### 6. Latinobarómetro — Percepciones IA y empleo LATAM
**Ruta local:** `data/raw/Latinobarometro_<año>_stata.zip`
**Origen:** latinobarometro.org (descarga libre, sin login). Formato Stata `.dta`. Descargado 2026-05-30.
**Oleadas disponibles en rango:** 2017, 2018, 2020, 2023 (no hubo oleada 2019, 2021 ni 2022).

**Uso en la tesis:** contexto de percepciones sobre IA y riesgo laboral en LATAM/México (capítulo introductorio / política pública). Complementa arXiv:2505.08841.

---

### 7. Epoch AI — Capabilities dataset (benchmarks LLM)
**Ruta local:** `data/raw/epoch_ai_capabilities.zip` + `data/raw/epoch_eci_repo.zip`
**Origen:** epoch.ai/data (Creative Commons BY). Descargado 2026-05-30.
**Contenido:** 49 benchmarks (MMLU, GPQA Diamond, MATH Level 5, FrontierMath, AIME, SWE-bench, ARC-AGI, etc.) con score, fecha de release, organización; más `epoch_capabilities_index.csv` (643 modelos, 2021–2026).

**Uso en la tesis:**
- Insumo del índice **DBOE** (ver sección 8) — capacidad frontier por aplicación y año.
- Curva de capacidad `c_j(t)` como evidencia del horizonte 2025–2030 (razonamiento matemático 0.00→0.79 en 2022–2026).

---

### 8. DBOE — Dynamic LLM Occupational Exposure (índice propio)
**Script:** `data/raw/build_dynamic_aioe.py`
**Salidas:** `data/processed/dynamic_aioe_scores.csv`, `data/processed/sinco_dboe_scores.csv`
**Construido:** 2026-05-30. Contribución metodológica original de la tesis.

Extensión temporal del AIOE de Felten, Raj & Seamans (2021) usando scores reales de benchmarks LLM (Epoch AI). Reemplaza el `gpt_exposure_score` sintético del Block 3.

**Método (validado):**
```
A_k(t)    = Σ_j rel[k,j] · c_j(t)          # exposición de habilidad k en año t
W_ok      = z_occ(importancia_ok) + z_occ(nivel_ok)
DBOE_o(t) = Σ_k z_abil(A_k(t)) · W_ok
```
- `rel[k,j]` = matriz AIOE Appendix D (52 habilidades × aplicaciones IA).
- Restringido a 3 aplicaciones LLM: Language Modeling, Reading Comprehension, Abstract Strategy Games (razonamiento/matemáticas).
- `c_j(t)` = capacidad frontier (máx score entre modelos liberados hasta el año t) promediada por aplicación.

**Validación:**
- Reproduce el AIOE publicado (Appendix A): **r = 0.942**.
- Sigue al LM AIOE de Felten: **r = 0.925**.

**Hallazgo clave (H1):** gradiente por SINCO mayor — Profesionistas (z=+0.88) y Directivos (+0.78) más expuestos; Operadores (−1.02) y Elementales (−1.21) al fondo. La exposición LLM se concentra en ocupaciones cognitivas.

**Limitación temporal (reportar abiertamente):** el re-ranking ocupacional 2022→2026 es mínimo (Spearman 0.998). La señal temporal la lleva la curva `c_j(t)`, no el reordenamiento ocupacional. La exposición es estructural y estable, no volátil.

| Archivo | Descripción | Merge |
|---|---|---|
| `dynamic_aioe_scores.csv` | 759 SOC × {dboe_2022..2026, dboe_2026_z} | nivel SOC |
| `sinco_dboe_scores.csv` | 10 grupos SINCO mayores | ✅ vía `c_ocu11c` a ENOE |

---

### 9. Archivos procesados (O*NET base)
**Ruta local:** `data/processed/`

| Archivo | Descripción | En DB |
|---|---|---|
| `enoe_jalisco.csv` | 2,000 trabajadores Jalisco — muestra ML | ✅ Como `trabajadores` |
| `onet_scores.csv` | Scores O\*NET por SOC code (Work Activities + Abilities) | ✅ Procesado → `ocupaciones_onet` |
| `sinco_group_scores.csv` | Scores O\*NET promediados por grupo SINCO (1-9) | ✅ Como `ocupaciones_onet` |
| `isco4_onet_scores.csv` | Scores a nivel ISCO-4 (más granular que grupos SINCO) | ❌ No cargado |

---

## Crosswalk SOC → SINCO

El puente entre O\*NET (SOC) y ENOE (SINCO) es el elemento más crítico de la metodología.

### Implementación actual (simplificada)
- SINCO 2011 primer dígito → ISCO-08 grupo mayor → SOC major group → promedio O\*NET
- Resultado: 8 grupos SINCO mapeados a promedios de docenas de ocupaciones SOC
- **Limitación:** pierde toda la variación dentro de cada grupo SINCO

### Implementación pendiente (IDB México 2024)
- IDB construyó crosswalk SOC→SINCO-2011 siguiendo metodología Eloundou et al.
- Disponible en: https://publications.iadb.org/en/ai-and-increase-productivity-and-labor-inequality-latin-america
- Permitiría mapear a nivel de SINCO 4 dígitos (mucho más granular)

---

## Diagrama de relaciones

```
ENOE_SDEMT ──(join key)──> enoe_sdemt_full
                               |
                          [ent+con+...+n_ren]
                               |
ENOE_COE1 ──(join key)──> enoe_coe1_jalisco

enoe_sdemt_full ──(c_ocu11c → sinco_code)──> ocupaciones_onet
                                                    |
                                              (onetsoc_code → SOC)
                                                    |
O*NET files ──────────────────────────────> onet_task_ratings
                                            onet_technology_skills
                                            onet_task_statements
                                            onet_emerging_tasks
                                            onet_skills
                                            onet_work_context
                                            onet_education_training

trabajadores (2,000 sample) ──(id_trabajador)──> vista_riesgo_jalisco
```

---

## Variables derivadas pendientes

Una vez cargados los datos, se calcularán:

> ⚠️ Las variables `*_empirico` y `digital_access_real` basadas en p5f de COE1
> NO son factibles (p5f no se recolectó — ver corrección en sección ENOE COE1).
> Se sustituyen por las fuentes O\*NET, que sí tienen estos descriptores.

| Variable | Cómo calcular | Reemplaza / fuente |
|---|---|---|
| `digital_access` | Technology Skills O\*NET por SOC → SINCO | O\*NET (no ENOE) |
| `rti` / `manual` / `social` | Work Activities + Work Context O\*NET | O\*NET (no ENOE) |
| `horas_semana` | p5b_thrs de COE1 (54% fill) | ENOE COE1 (sí factible) |
| `tech_skill_count` | COUNT de Technology Skills por SOC | variable nueva |
| `hot_tech_count` | COUNT donde hot_technology=1 | variable nueva |
| `n_tareas` | COUNT de Task Ratings por SOC | granularidad tarea |
| `pct_tareas_alto_IM` | % tareas con importance >= 4.0 | variable nueva |
| `nivel_edu_requerido` | Education & Training data value | validación vs `nivel_educativo` |
