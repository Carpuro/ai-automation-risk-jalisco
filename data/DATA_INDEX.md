# Data Index — ai_automation_risk_jalisco

Base de datos SQL Server en homelab (localhost:1433 / Tailscale 100.64.0.11:1433).
Última actualización: 2026-05-20.

---

## Estado actual de la base de datos

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
- **Variables clave a extraer:**
  | Variable COE1 | Nombre propuesto | Descripción |
  |---|---|---|
  | p1 | tipo_local | Tipo de local: 1=casa, 2=negocio/ofic., 3=vía pública, 4=campo, 5=otro |
  | p2c | tipo_contrato | 1=indefinido, 2=temporal/proyecto, 3=sin contrato |
  | p3h | tam_establecimiento_n | Número de trabajadores (continuo, vs categorizado en SDEMT) |
  | p3i | contrato_escrito | 1=sí, 2=no |
  | p5b_thrs | horas_semana | Total de horas trabajadas por semana |
  | p5f1 | cond_exterior | Trabaja al aire libre expuesto al clima |
  | p5f2 | cond_de_pie | De pie la mayor parte del tiempo |
  | p5f3 | cond_peso | Levanta o mueve objetos pesados |
  | p5f4 | cond_repetitivo | Movimientos o acciones repetitivas |
  | p5f5 | cond_maquinaria | Opera maquinaria fija (tornos, fresadoras) |
  | p5f6 | cond_transporte | Opera equipo de transporte |
  | p5f7 | cond_computadora | Usa computadora, smartphone o tableta |
  | p5f8 | cond_clientes | Atiende o ayuda directamente a clientes |
  | p5f9 | cond_vigilancia | Protege o vigila personas o bienes |
  | p5f10 | cond_temperatura | Trabaja a altas temperaturas |
  | p5f11 | cond_peligroso | Contacto con sustancias peligrosas |
  | p5f12 | cond_altura | Trabaja en altura |
  | p5f13 | cond_herramientas | Usa herramientas manuales |
  | p5f14 | cond_elaborar | Elabora o transforma objetos/productos |
  | p5f15 | cond_reparacion | Reparación o mantenimiento |

  > **Relevancia para la tesis:** las variables p5f son medidas empíricas directas
  > de las condiciones físicas de trabajo — equivalente observado del RTI de O\*NET.
  > p5f7 (cond_computadora) reemplaza el `digital_access` generado sintéticamente.

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

### 3. Archivos procesados
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

| Variable | Cómo calcular | Reemplaza |
|---|---|---|
| `digital_access_real` | p5f7 de COE1 (usa computadora) | `digital_access` sintético |
| `rti_empirico` | promedio(p5f1,p5f2,p5f3,p5f4) de COE1 | `rti` de O\*NET |
| `manual_empirico` | promedio(p5f5,p5f12,p5f13,p5f14,p5f15) | `manual_dexterity` O\*NET |
| `social_empirico` | p5f8 de COE1 | `social_interaction` O\*NET |
| `tech_skill_count` | COUNT de Technology Skills por SOC | variable nueva |
| `hot_tech_count` | COUNT donde hot_technology=1 | variable nueva |
| `n_tareas` | COUNT de Task Ratings por SOC | granularidad tarea |
| `pct_tareas_alto_IM` | % tareas con importance >= 4.0 | variable nueva |
| `nivel_edu_requerido` | Education & Training data value | validación vs `nivel_educativo` |
