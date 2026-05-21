"""
Construye crosswalk SINCO -> SOC -> O*NET usando el puente ISCO-08.

SINCO 2011 (usado en ENOE) esta basado en ISCO-08.
El primer digito de SINCO corresponde al primer digito de ISCO-08.
El crosswalk ESCO (ISCO-08 -> SOC) nos da el puente.

Salida: data/processed/sinco_onet_crosswalk.csv
"""

import pandas as pd
import numpy as np
import os

RAW  = os.path.dirname(__file__)
PROC = os.path.join(RAW, '..', 'processed')
os.makedirs(PROC, exist_ok=True)

# ------------------------------------------------------------------
# 1. Cargar crosswalk ESCO (ISCO-08 <-> SOC)
# ------------------------------------------------------------------
esco_path = os.path.join(RAW, 'ESCO_to_ONET_SOC_crosswalk.xlsx')
esco = pd.read_excel(esco_path, header=2)
esco.columns = ['isco_code', 'isco_title', 'soc_code', 'soc_title']
esco = esco.dropna(subset=['isco_code', 'soc_code'])
esco = esco[~esco['isco_code'].astype(str).str.contains('ESCO', na=True)]

# Extraer grupo ISCO de 4 digitos (primeros 4 chars antes del punto)
esco['isco_4'] = esco['isco_code'].astype(str).str[:4].str.replace('.', '', regex=False)
esco['isco_major'] = esco['isco_4'].str[0]  # primer digito = grupo mayor

print(f'ESCO crosswalk: {len(esco):,} registros')
print(f'Grupos ISCO mayores: {sorted(esco["isco_major"].unique())}')

# ------------------------------------------------------------------
# 2. Cargar O*NET scores (calculados de datos reales)
# ------------------------------------------------------------------
onet = pd.read_csv(f'{PROC}/onet_scores.csv')
print(f'O*NET scores: {len(onet):,} ocupaciones')

# ------------------------------------------------------------------
# 3. Join ESCO + O*NET para obtener scores por SOC
# ------------------------------------------------------------------
esco_onet = esco.merge(onet, left_on='soc_code', right_on='soc_code', how='inner')
print(f'ESCO con scores O*NET: {len(esco_onet):,} registros')

# ------------------------------------------------------------------
# 4. Mapeo SINCO mayor -> ISCO mayor (compatible 1:1 en 1er digito)
#    SINCO 2011 grupos principales:
#    1=Directivos  2=Profesionistas  3=Tecnicos  4=Administrativos
#    5=Comercio    6=Servicios       7=Agropecuarios
#    8=Artesanos   9=Operadores      0=Elementales
#
#    ISCO-08 grupos principales:
#    1=Directivos  2=Profesionales  3=Tecnicos  4=Apoyo oficina
#    5=Servicios   6=Agropecuarios  7=Artesanos 8=Operadores
#    9=Elementales 0=FFAA
#
#    Mapeo aproximado SINCO1 -> ISCO1:
# ------------------------------------------------------------------
SINCO_TO_ISCO = {
    '1': '1',  # Directivos -> Managers/Directors
    '2': '2',  # Profesionistas -> Professionals
    '3': '3',  # Tecnicos -> Technicians
    '4': '4',  # Administrativos -> Clerical support
    '5': '5',  # Comercio/Ventas -> Service/Sales
    '6': '5',  # Servicios personales -> Service workers
    '7': '6',  # Agropecuarios -> Skilled agri
    '8': '7',  # Artesanos -> Craft
    '9': '8',  # Operadores -> Plant operators
    '0': '9',  # Elementales -> Elementary
}

# ------------------------------------------------------------------
# 5. Scores promedio por SINCO grupo mayor
# ------------------------------------------------------------------
score_cols = ['cognitive_demand', 'social_interaction', 'creativity',
              'manual_dexterity', 'rti', 'automation_risk', 'frey_osborne_score']

group_scores = {}
for sinco_major, isco_major in SINCO_TO_ISCO.items():
    subset = esco_onet[esco_onet['isco_major'] == isco_major]
    if len(subset) > 0:
        means = subset[score_cols].mean()
        group_scores[sinco_major] = means
    else:
        # Fallback: media global
        group_scores[sinco_major] = esco_onet[score_cols].mean()

sinco_group_df = pd.DataFrame(group_scores).T
sinco_group_df.index.name = 'sinco_major'
sinco_group_df = sinco_group_df.reset_index()

print('\n=== SCORES PROMEDIO POR GRUPO SINCO MAYOR ===')
print(sinco_group_df[['sinco_major','automation_risk','rti','cognitive_demand']].to_string())

# ------------------------------------------------------------------
# 6. Mapa de nombres SINCO major -> etiqueta
# ------------------------------------------------------------------
SINCO_LABELS = {
    '1': 'Directivos y gerentes',
    '2': 'Profesionistas y técnicos de alto nivel',
    '3': 'Técnicos especializados',
    '4': 'Trabajadores de apoyo administrativo',
    '5': 'Comerciantes y vendedores',
    '6': 'Trabajadores en servicios personales',
    '7': 'Trabajadores agropecuarios',
    '8': 'Artesanos y trabajadores en manufactura',
    '9': 'Operadores de maquinaria y transporte',
    '0': 'Trabajadores en ocupaciones elementales',
}

sinco_group_df['sinco_label'] = sinco_group_df['sinco_major'].map(SINCO_LABELS)
sinco_group_df.to_csv(f'{PROC}/sinco_group_scores.csv', index=False)

# ------------------------------------------------------------------
# 7. Crosswalk detallado ISCO_4 -> SOC -> O*NET scores
#    (para usar cuando tengamos codigos SINCO de 4 digitos de ENOE)
# ------------------------------------------------------------------
isco4_scores = esco_onet.groupby('isco_4')[score_cols].mean().reset_index()
isco4_scores['sinco_major'] = isco4_scores['isco_4'].str[0].map(
    {v: k for k, v in SINCO_TO_ISCO.items()}
)
isco4_scores.to_csv(f'{PROC}/isco4_onet_scores.csv', index=False)

print(f'\nGuardados:')
print(f'  {PROC}/sinco_group_scores.csv  ({len(sinco_group_df)} grupos)')
print(f'  {PROC}/isco4_onet_scores.csv   ({len(isco4_scores)} grupos ISCO-4)')
print('\nListo para recibir ENOE.')
