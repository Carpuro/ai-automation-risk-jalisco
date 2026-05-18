"""
Procesa O*NET 28.3 real para calcular:
  - cognitive_demand    (Work Activities: analisis, procesamiento, decision)
  - social_interaction  (Work Activities: relaciones, comunicacion, asistencia)
  - creativity          (Work Activities: pensamiento creativo + Abilities: originalidad)
  - manual_dexterity    (Abilities: destreza manual y dactilar)
  - rti                 (Work Activities: tareas rutinarias manuales + cognitivas)
  - automation_risk     (Frey-Osborne proxy: inverso de cognitive+social+creative)

Salida: data/processed/onet_scores.csv
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.join(os.path.dirname(__file__), 'onet', 'db_28_3_text')
OUT  = os.path.join(os.path.dirname(__file__), '..', 'processed')
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------
# 1. Cargar archivos O*NET
# ------------------------------------------------------------------
occ = pd.read_csv(f'{BASE}/Occupation Data.txt', sep='\t', encoding='utf-8',
                  usecols=['O*NET-SOC Code', 'Title'])
occ.columns = ['soc_code', 'occupation_name']

wa  = pd.read_csv(f'{BASE}/Work Activities.txt', sep='\t', encoding='utf-8',
                  usecols=['O*NET-SOC Code', 'Element Name', 'Scale ID', 'Data Value'])
wa.columns = ['soc_code', 'element', 'scale', 'value']

ab  = pd.read_csv(f'{BASE}/Abilities.txt', sep='\t', encoding='utf-8',
                  usecols=['O*NET-SOC Code', 'Element Name', 'Scale ID', 'Data Value'])
ab.columns = ['soc_code', 'element', 'scale', 'value']

wc  = pd.read_csv(f'{BASE}/Work Context.txt', sep='\t', encoding='utf-8',
                  usecols=['O*NET-SOC Code', 'Element Name', 'Scale ID', 'Data Value'])
wc.columns = ['soc_code', 'element', 'scale', 'value']

print(f'Ocupaciones: {len(occ):,}')
print(f'Work Activities: {len(wa):,} registros')
print(f'Abilities: {len(ab):,} registros')

# Usar solo importancia (IM) para Work Activities y Abilities
wa_im = wa[wa['scale'] == 'IM'].copy()
ab_im = ab[ab['scale'] == 'IM'].copy()

# ------------------------------------------------------------------
# 2. Definir elementos por dimension
# ------------------------------------------------------------------

# Demanda cognitiva alta = difícil de automatizar
COGNITIVE = [
    'Analyzing Data or Information',
    'Processing Information',
    'Thinking Creatively',
    'Making Decisions and Solving Problems',
    'Developing Objectives and Strategies',
    'Updating and Using Relevant Knowledge',
    'Judging the Qualities of Objects, Services, or People',
]

# Interaccion social alta = difícil de automatizar
SOCIAL = [
    'Establishing and Maintaining Interpersonal Relationships',
    'Communicating with Supervisors, Peers, or Subordinates',
    'Communicating with People Outside the Organization',
    'Assisting and Caring for Others',
    'Coaching and Developing Others',
    'Resolving Conflicts and Negotiating with Others',
    'Selling or Influencing Others',
]

# Creatividad
CREATIVE = [
    'Thinking Creatively',
    'Developing Objectives and Strategies',
]

# Tareas rutinarias (positivamente correlacionadas con riesgo)
ROUTINE = [
    'Controlling Machines and Processes',
    'Operating Vehicles, Mechanized Devices, or Equipment',
    'Handling and Moving Objects',
    'Performing General Physical Activities',
    'Inspecting Equipment, Structures, or Materials',
]

# Habilidades manuales (abilities)
MANUAL_AB = [
    'Manual Dexterity',
    'Finger Dexterity',
    'Arm-Hand Steadiness',
]

# Originality (ability — parte de creatividad)
CREATIVE_AB = [
    'Originality',
    'Fluency of Ideas',
]


def score_dim(df, elements, col_name):
    """Promedio de importancia de los elementos seleccionados por ocupacion."""
    subset = df[df['element'].isin(elements)]
    pivot  = subset.groupby('soc_code')['value'].mean().reset_index()
    pivot.columns = ['soc_code', col_name]
    return pivot


# ------------------------------------------------------------------
# 3. Calcular dimensiones
# ------------------------------------------------------------------
cog  = score_dim(wa_im, COGNITIVE, 'cognitive_raw')
soc  = score_dim(wa_im, SOCIAL,    'social_raw')
cre  = score_dim(wa_im, CREATIVE,  'creativity_raw')
rout = score_dim(wa_im, ROUTINE,   'routine_raw')
man  = score_dim(ab_im, MANUAL_AB, 'manual_raw')
crb  = score_dim(ab_im, CREATIVE_AB, 'creative_ab_raw')

df = occ.copy()
for d in [cog, soc, cre, rout, man, crb]:
    df = df.merge(d, on='soc_code', how='left')

df = df.fillna(df.median(numeric_only=True))

# ------------------------------------------------------------------
# 4. Normalizar a 0-100
# ------------------------------------------------------------------
def norm100(col):
    mn, mx = col.min(), col.max()
    return ((col - mn) / (mx - mn) * 100).round(2)

df['cognitive_demand']   = norm100(df['cognitive_raw'])
df['social_interaction'] = norm100(df['social_raw'])
df['creativity']         = norm100((df['creativity_raw'] + df['creative_ab_raw']) / 2)
df['manual_dexterity']   = norm100(df['manual_raw'])
df['rti']                = norm100(df['routine_raw'])

# ------------------------------------------------------------------
# 5. Automation risk (proxy Frey-Osborne)
# Ocupaciones son difíciles de automatizar si tienen:
#   alta demanda cognitiva, alta interaccion social, alta creatividad
# Faciles de automatizar: alta rutina, alta destreza manual (baja cognitiva)
# ------------------------------------------------------------------
protective = (0.50 * df['cognitive_demand'] +
              0.30 * df['social_interaction'] +
              0.20 * df['creativity'])

risk_raw = (0.55 * df['rti'] +
            0.25 * df['manual_dexterity'] +
            0.20 * (100 - df['cognitive_demand']))

# Normalizar a 0-1
df['automation_risk']     = ((risk_raw - risk_raw.min()) /
                              (risk_raw.max() - risk_raw.min())).round(4)
df['frey_osborne_score']  = df['automation_risk']

# ------------------------------------------------------------------
# 6. Job Zone como proxy de education_level (1-5 → 1-6)
# ------------------------------------------------------------------
jz = pd.read_csv(f'{BASE}/Job Zones.txt', sep='\t', encoding='utf-8',
                 usecols=['O*NET-SOC Code', 'Job Zone'])
jz.columns = ['soc_code', 'job_zone']
df = df.merge(jz, on='soc_code', how='left')
df['education_level'] = df['job_zone'].fillna(3).clip(1, 5).astype(int)

# ------------------------------------------------------------------
# 7. Guardar
# ------------------------------------------------------------------
cols_out = ['soc_code', 'occupation_name', 'cognitive_demand', 'social_interaction',
            'creativity', 'manual_dexterity', 'rti',
            'automation_risk', 'frey_osborne_score', 'education_level']

df[cols_out].to_csv(f'{OUT}/onet_scores.csv', index=False)
print(f'\nGuardado: {OUT}/onet_scores.csv ({len(df):,} ocupaciones)')

print('\n=== ESTADISTICAS ===')
for col in ['cognitive_demand','social_interaction','creativity','rti','automation_risk']:
    print(f'  {col:<22}: mean={df[col].mean():.2f} | std={df[col].std():.2f}')

print('\n=== TOP 10 MAYOR RIESGO ===')
print(df.nlargest(10, 'automation_risk')[['occupation_name','automation_risk','rti','cognitive_demand']].to_string())

print('\n=== TOP 10 MENOR RIESGO ===')
print(df.nsmallest(10, 'automation_risk')[['occupation_name','automation_risk','rti','cognitive_demand']].to_string())
