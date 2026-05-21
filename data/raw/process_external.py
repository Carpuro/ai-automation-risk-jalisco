"""
process_external.py
Procesa y carga a SQL Server los datos externos descargados:
  1. external_aioe         — AIOE + Language Modeling AIOE por SOC (Felten et al.)
  2. external_anthropic    — observed_exposure por SOC (Anthropic Economic Index)
  3. external_task_penet   — penetracion de Claude por tarea O*NET (Anthropic)
  4. Mapea a SINCO groups y actualiza ocupaciones_onet con scores reales
"""

import os, warnings
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
warnings.filterwarnings('ignore')

BASE   = os.path.dirname(__file__)
engine = create_engine(
    'mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco'
    '?driver=ODBC+Driver+17+for+SQL+Server'
    '&trusted_connection=yes&TrustServerCertificate=yes'
)

# ── Crosswalk SOC -> SINCO (lo que ya tenemos procesado) ──────────────────────
# isco4_onet_scores.csv tiene SOC codes mapeados a grupos SINCO
xwalk_path = os.path.join(os.path.dirname(BASE), 'processed', 'isco4_onet_scores.csv')
if os.path.exists(xwalk_path):
    xwalk = pd.read_csv(xwalk_path)
    print(f'Crosswalk cargado: {xwalk.shape} — cols: {list(xwalk.columns)}')
else:
    print('WARN: crosswalk isco4_onet_scores.csv no encontrado — mapping manual')
    xwalk = None

# ── 1. AIOE + Language Modeling AIOE ─────────────────────────────────────────
print('\n=== 1. AIOE scores (Felten et al.) ===')
aioe_occ = pd.read_excel(os.path.join(BASE, 'AIOE_DataAppendix.xlsx'),
                          sheet_name='Appendix A')
aioe_occ.columns = ['soc_code', 'occupation_title', 'aioe_score']

lm_aioe  = pd.read_excel(os.path.join(BASE, 'Language_Modeling_AIOE_AIIE.xlsx'),
                          sheet_name='LM AIOE')
lm_aioe.columns = ['soc_code', 'occupation_title', 'lm_aioe_score']

# Merge ambos
aioe_full = aioe_occ.merge(
    lm_aioe[['soc_code','lm_aioe_score']], on='soc_code', how='left'
)
print(f'  AIOE por SOC: {len(aioe_full):,} ocupaciones')
print(f'  Rango AIOE:     {aioe_full.aioe_score.min():.2f} – {aioe_full.aioe_score.max():.2f}')
print(f'  Rango LM_AIOE:  {aioe_full.lm_aioe_score.min():.2f} – {aioe_full.lm_aioe_score.max():.2f}')

aioe_full.to_sql('external_aioe', engine, if_exists='replace', index=False)
print('  [OK] Cargado -> external_aioe')

# ── 2. Anthropic observed exposure por ocupacion ──────────────────────────────
print('\n=== 2. Anthropic job exposure ===')
anthro_job = pd.read_csv(os.path.join(BASE, 'anthropic_job_exposure.csv'))
anthro_job.columns = ['soc_code', 'occupation_title', 'anthropic_observed_exposure']
print(f'  Ocupaciones: {len(anthro_job):,}')
print(f'  Rango exposure: {anthro_job.anthropic_observed_exposure.min():.3f} – '
      f'{anthro_job.anthropic_observed_exposure.max():.3f}')
print(f'  Media exposure: {anthro_job.anthropic_observed_exposure.mean():.3f}')

anthro_job.to_sql('external_anthropic_job', engine, if_exists='replace', index=False)
print('  [OK] Cargado -> external_anthropic_job')

# ── 3. Anthropic task penetration ─────────────────────────────────────────────
print('\n=== 3. Anthropic task penetration ===')
task_pen = pd.read_csv(os.path.join(BASE, 'anthropic_task_penetration.csv'))
task_pen.columns = ['task_description', 'claude_penetration']
print(f'  Tareas: {len(task_pen):,}')
print(f'  Tareas con penetracion > 0: {(task_pen.claude_penetration > 0).sum():,}')
print(f'  Max penetracion: {task_pen.claude_penetration.max():.3f}')
print(f'  Top 5 tareas:')
print(task_pen.nlargest(5, 'claude_penetration')[['task_description','claude_penetration']].to_string())

task_pen.to_sql('external_task_penetration', engine, if_exists='replace',
                index=False, chunksize=2000)
print('  [OK] Cargado -> external_task_penetration')

# ── 4. Mapear AIOE a grupos SINCO via crosswalk existente ─────────────────────
print('\n=== 4. Mapping SOC -> SINCO groups ===')

# Cargar el crosswalk desde la tabla onet_scores procesada
try:
    with engine.connect() as conn:
        onet_cur = pd.read_sql(text('SELECT * FROM dbo.ocupaciones_onet'), conn)
    print(f'  ocupaciones_onet actual: {len(onet_cur)} grupos')
    print(f'  Columnas: {list(onet_cur.columns)}')
except Exception as e:
    print(f'  ERROR leyendo ocupaciones_onet: {e}')
    onet_cur = None

# El crosswalk isco4_onet_scores tiene el puente SOC -> SINCO
# Calculamos promedios de AIOE real por grupo SINCO
if xwalk is not None:
    print(f'\n  Columnas crosswalk: {list(xwalk.columns)}')
    # Intentar unir AIOE a crosswalk via SOC code
    # Los SOC codes en AIOE son formato "11-1011"
    # Verificar formato en crosswalk
    soc_col = [c for c in xwalk.columns if 'soc' in c.lower() or 'code' in c.lower()]
    sinco_col = [c for c in xwalk.columns if 'sinco' in c.lower() or 'group' in c.lower()]
    print(f'  Posibles col SOC: {soc_col}')
    print(f'  Posibles col SINCO: {sinco_col}')

# ── 5. Construir tabla de scores reales por SINCO group ───────────────────────
print('\n=== 5. Scores reales por SINCO group ===')
# Mapeamos los grupos SINCO 1-9 a SOC major groups via ISCO-08
# Mapping aproximado (mismo que se usó para generar los datos originales):
SINCO_SOC_MAP = {
    1: ['11-','13-','15-','17-','19-'],  # Directivos/Profesionistas
    2: ['11-','13-'],                     # Profesionistas
    3: ['13-','15-','17-'],               # Tecnicos
    4: ['43-','11-3'],                    # Personal de apoyo administrativo
    5: ['41-','43-'],                     # Comerciantes/vendedores
    6: ['35-','37-','39-'],               # Servicios personales
    7: ['45-','47-','49-'],               # Actividades agropecuarias/construccion
    8: ['51-','53-'],                     # Operadores de maquinaria
    9: ['53-','47-'],                     # Ocupaciones elementales
}

SINCO_NAMES = {
    1: 'Directivos y funcionarios',
    2: 'Profesionistas',
    3: 'Tecnicos',
    4: 'Trabajadores de apoyo administrativo',
    5: 'Comerciantes y trabajadores en ventas',
    6: 'Trabajadores en servicios personales',
    7: 'Trabajadores agropecuarios y pesqueros',
    8: 'Operadores de maquinaria industrial',
    9: 'Trabajadores en actividades elementales',
}

records = []
for sinco_group, prefixes in SINCO_SOC_MAP.items():
    # Filtrar ocupaciones SOC que pertenecen a este grupo SINCO
    mask = aioe_full['soc_code'].apply(
        lambda s: any(str(s).startswith(p) for p in prefixes))
    subset = aioe_full[mask]
    anthro_mask = anthro_job['soc_code'].apply(
        lambda s: any(str(s).startswith(p) for p in prefixes))
    anthro_sub = anthro_job[anthro_mask]

    if len(subset) > 0:
        rec = {
            'sinco_code':          sinco_group,
            'sinco_name':          SINCO_NAMES[sinco_group],
            'n_soc_occupations':   len(subset),
            'aioe_mean':           round(subset['aioe_score'].mean(), 4),
            'aioe_std':            round(subset['aioe_score'].std(), 4),
            'lm_aioe_mean':        round(subset['lm_aioe_score'].mean(), 4),
            'lm_aioe_std':         round(subset['lm_aioe_score'].std(), 4),
            'anthropic_obs_mean':  round(anthro_sub['anthropic_observed_exposure'].mean(), 4)
                                   if len(anthro_sub) > 0 else None,
        }
        records.append(rec)

sinco_aioe = pd.DataFrame(records)
print(sinco_aioe.to_string(index=False))

sinco_aioe.to_sql('sinco_aioe_scores', engine, if_exists='replace', index=False)
print('\n[OK] Cargado -> sinco_aioe_scores')

# ── 6. ILO WP140 2025 — scores GenAI por tarea ISCO-08 ───────────────────────
print('\n=== 6. ILO WP140 2025 (ISCO-08) ===')
ilo = pd.read_excel(os.path.join(BASE, 'ILO_WP140_Final_Scores_ISCO08_2025.xlsx'))
keep = ['label4d', 'label1d', 'ISCO_08', 'Title', 'taskID', 'Task_ISCO',
        'score_2023', 'score_2025', 'mean_score_2023', 'mean_score_2025',
        'SD_2023', 'SD_2025', 'potential25']
ilo_clean = ilo[[c for c in keep if c in ilo.columns]].copy()
print(f'  Tareas: {len(ilo_clean):,} | Ocupaciones ISCO: {ilo_clean["ISCO_08"].nunique()}')
ilo_clean.to_sql('external_ilo_wp140', engine, if_exists='replace', index=False, chunksize=2000)
print('  [OK] -> external_ilo_wp140')

# Agregado por ocupacion ISCO-08
ilo_occ = ilo_clean.groupby(['ISCO_08', 'Title', 'label1d', 'label4d']).agg(
    n_tasks=('taskID', 'count'),
    ilo_exposure_2023=('score_2023', 'mean'),
    ilo_exposure_2025=('score_2025', 'mean'),
    ilo_sd_2025=('SD_2025', 'mean'),
).reset_index().round(4)
ilo_occ.to_sql('external_ilo_wp140_occ', engine, if_exists='replace', index=False)
print(f'  [OK] -> external_ilo_wp140_occ ({len(ilo_occ)} ocupaciones)')

# ── 7. Moravec Index (Schaal 2025) ────────────────────────────────────────────
print('\n=== 7. Moravec Index ===')
moravec_dir = os.path.join(BASE, 'moravec_index')

morav_occ = pd.read_csv(os.path.join(moravec_dir, 'occupation_agg.csv'))
print(f'  Ocupaciones: {len(morav_occ):,} | Cols: {list(morav_occ.columns)}')
morav_occ.to_sql('external_moravec_occ', engine, if_exists='replace', index=False)
print('  [OK] -> external_moravec_occ')

morav_cmp = pd.read_csv(os.path.join(moravec_dir, 'Comparison of Indices.csv'))
print(f'  Comparacion indices: {len(morav_cmp):,} ocupaciones')
morav_cmp.to_sql('external_moravec_comparison', engine, if_exists='replace', index=False)
print('  [OK] -> external_moravec_comparison')

# ── 8. RL Feasibility Index (arXiv:2605.02598) ───────────────────────────────
print('\n=== 8. RL Feasibility Index ===')
rl = pd.read_csv(os.path.join(BASE, 'RL_feasibility_index.csv'))
rl_cols = ['O*NET-SOC Code', 'Title', 'Task ID', 'Task',
           'gemini_rl_weighted_avg', 'gemini_rl_index']
rl_clean = rl[[c for c in rl_cols if c in rl.columns]].copy()
rl_clean.columns = ['soc_code', 'title', 'task_id', 'task',
                    'rl_weighted_avg', 'rl_index']
print(f'  Tareas: {len(rl_clean):,}')
rl_clean.to_sql('external_rl_feasibility', engine, if_exists='replace',
                index=False, chunksize=2000)
print('  [OK] -> external_rl_feasibility')

# Agregado por ocupacion
rl_occ = rl_clean.groupby(['soc_code', 'title']).agg(
    n_tasks=('task_id', 'count'),
    rl_index_mean=('rl_index', 'mean'),
    rl_index_max=('rl_index', 'max'),
).reset_index().round(4)
rl_occ.to_sql('external_rl_feasibility_occ', engine, if_exists='replace', index=False)
print(f'  [OK] -> external_rl_feasibility_occ ({len(rl_occ)} ocupaciones)')

# ── 9. ESCO crosswalk (ISCO-08 <-> O*NET-SOC) ────────────────────────────────
print('\n=== 9. ESCO crosswalk ===')
esco_raw = pd.read_excel(os.path.join(BASE, 'ESCO_to_ONET_SOC_crosswalk.xlsx'), header=2)
esco_raw.columns = ['isco_code', 'isco_title', 'soc_code', 'soc_title']
esco_raw = esco_raw.dropna(subset=['isco_code', 'soc_code'])
esco_raw = esco_raw[~esco_raw['isco_code'].astype(str).str.contains('ESCO|Code', na=True)]
esco_raw['isco_4'] = esco_raw['isco_code'].astype(str).str[:4].str.replace('.', '', regex=False)
print(f'  Registros: {len(esco_raw):,} | ISCO-4 grupos: {esco_raw["isco_4"].nunique()}')
esco_raw.to_sql('external_esco_crosswalk', engine, if_exists='replace',
                index=False, chunksize=2000)
print('  [OK] -> external_esco_crosswalk')

# ── 10. INEGI Censos Economicos 2024 — Jalisco sector 2 digitos ──────────────
print('\n=== 10. INEGI CE2024 — Jalisco ===')
inegi = pd.read_csv(os.path.join(BASE, 'INEGI_CE2024_activo_fijo_scian.csv'),
                    encoding='latin1', skiprows=3, header=0, low_memory=False)
inegi.columns = ['anio', 'entidad', 'municipio', 'sector',
                 'unidades', 'personal_rem', 'remuneraciones', 'activo_fijo', 'drop']

# Filtrar: Jalisco + nivel estatal (sin municipio) + sectores SCIAN 2 digitos
inegi_jal = inegi[
    inegi['entidad'].astype(str).str.contains('Jalisco', na=False) &
    inegi['municipio'].isna() &
    inegi['sector'].astype(str).str.match(r'^Sector \d')
].copy()

# Limpiar y convertir a numerico
for col in ['personal_rem', 'remuneraciones', 'activo_fijo', 'unidades']:
    inegi_jal[col] = pd.to_numeric(inegi_jal[col], errors='coerce')

inegi_jal['sector_code'] = inegi_jal['sector'].str.extract(r'Sector (\S+)')
inegi_jal = inegi_jal[['anio', 'sector_code', 'sector', 'unidades',
                         'personal_rem', 'remuneraciones', 'activo_fijo']].dropna(subset=['activo_fijo'])
print(f'  Filas Jalisco: {len(inegi_jal)} | Anos: {sorted(inegi_jal["anio"].unique())}')
inegi_jal.to_sql('external_inegi_ce2024', engine, if_exists='replace', index=False)
print('  [OK] -> external_inegi_ce2024')

# IRA por sector (anio 2023)
inegi_2023 = inegi_jal[inegi_jal['anio'] == '2023'].copy()
inegi_2023['salario_anual_prom'] = (inegi_2023['remuneraciones'] * 1e6) / inegi_2023['personal_rem']
inegi_2023['activo_fijo_por_trab'] = (inegi_2023['activo_fijo'] * 1e6) / inegi_2023['personal_rem']
inegi_2023['activo_fijo_amort_5y'] = inegi_2023['activo_fijo_por_trab'] / 5
inegi_2023['ira'] = (inegi_2023['salario_anual_prom'] / inegi_2023['activo_fijo_amort_5y']).round(4)
ira_out = inegi_2023[['sector_code', 'sector', 'personal_rem', 'remuneraciones',
                       'activo_fijo', 'salario_anual_prom', 'activo_fijo_amort_5y', 'ira']]
print('\n  IRA por sector SCIAN (2023):')
print(ira_out[['sector_code', 'sector', 'ira']].sort_values('ira').to_string(index=False))
ira_out.to_sql('external_ira_by_sector', engine, if_exists='replace', index=False)
print('\n  [OK] -> external_ira_by_sector')

# ── 11. Verificacion final ────────────────────────────────────────────────────
print('\n=== Tablas cargadas ===')
with engine.connect() as conn:
    for t in ['external_aioe', 'external_anthropic_job', 'external_task_penetration',
              'sinco_aioe_scores', 'external_ilo_wp140', 'external_ilo_wp140_occ',
              'external_moravec_occ', 'external_moravec_comparison',
              'external_rl_feasibility', 'external_rl_feasibility_occ',
              'external_esco_crosswalk', 'external_inegi_ce2024', 'external_ira_by_sector']:
        try:
            n = conn.execute(text(f'SELECT COUNT(*) FROM dbo.{t}')).scalar()
            print(f'  {t:<40} {n:>10,} filas')
        except Exception as e:
            print(f'  {t:<40} ERROR: {e}')
