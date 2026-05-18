"""
Procesa ENOE Q3 2024 para extraer trabajadores de Jalisco (entidad 14).
Listo para correr cuando el ZIP de ENOE este en data/raw/enoe/.

Variables que extrae del ENOE:
  - sinco_code  : codigo SINCO (clase2 o c_ocu11c)
  - escoacum    : escolaridad acumulada (anios_esc -> niveles 1-6)
  - ingocup     : ingreso mensual por ocupacion
  - scian_sector: sector SCIAN (rama_est2)
  - formalidad  : formal/informal (via pos_ocu o tipo_emp)
  - edad        : eda
  - tamanio_empresa: segempleado

Salida: data/processed/enoe_jalisco.csv
"""

import pandas as pd
import numpy as np
import os
import glob

RAW  = os.path.join(os.path.dirname(__file__), 'enoe')
PROC = os.path.join(os.path.dirname(__file__), '..', 'processed')
os.makedirs(PROC, exist_ok=True)

# ------------------------------------------------------------------
# 1. Encontrar archivo ENOE (busca CSV con SDEM o COE1)
# ------------------------------------------------------------------
csvs = glob.glob(f'{RAW}/**/*.csv', recursive=True) + \
       glob.glob(f'{RAW}/**/*.CSV', recursive=True)

print(f'Archivos CSV encontrados en enoe/: {len(csvs)}')
for f in csvs:
    size = os.path.getsize(f) / 1e6
    print(f'  {os.path.basename(f)}: {size:.1f} MB')

if not csvs:
    print('\nERROR: No hay archivos CSV en data/raw/enoe/')
    print('Descarga ENOE Q3 2024 de:')
    print('  https://www.inegi.org.mx/programas/enoe/15ymas/#Microdatos')
    print('  -> 2024 Trimestre 3 -> Datos -> CSV')
    print(f'  Extrae el ZIP en: {RAW}')
    exit(1)

# ------------------------------------------------------------------
# 2. Identificar el archivo SDEM (sociodemografico principal)
#    Tiene las variables de ocupacion e ingreso
# ------------------------------------------------------------------
sdem_files = [f for f in csvs if 'SDEM' in f.upper() or 'sdem' in f.lower()]
coe1_files = [f for f in csvs if 'COE1' in f.upper() or 'coe1' in f.lower()]

target_file = None
if sdem_files:
    target_file = sdem_files[0]
    print(f'\nUsando SDEM: {os.path.basename(target_file)}')
elif coe1_files:
    target_file = coe1_files[0]
    print(f'\nUsando COE1: {os.path.basename(target_file)}')
else:
    # Intentar con el CSV mas grande
    target_file = max(csvs, key=os.path.getsize)
    print(f'\nUsando archivo mas grande: {os.path.basename(target_file)}')

# ------------------------------------------------------------------
# 3. Cargar y filtrar Jalisco (ent = 14)
# ------------------------------------------------------------------
print(f'\nCargando {os.path.basename(target_file)}...')
try:
    df_raw = pd.read_csv(target_file, encoding='latin-1', low_memory=False)
except Exception:
    df_raw = pd.read_csv(target_file, encoding='utf-8', low_memory=False)

print(f'Total registros: {len(df_raw):,}')
print(f'Columnas disponibles: {list(df_raw.columns[:20])}...')

# Filtrar Jalisco
ent_col = next((c for c in df_raw.columns
                if c.lower() in ['ent', 'entidad', 'ent_pais']), None)
if ent_col:
    df = df_raw[df_raw[ent_col].astype(str).str.zfill(2) == '14'].copy()
    print(f'Jalisco (ent=14): {len(df):,} registros')
else:
    print('WARN: no se encontro columna de entidad, usando todo el dataset')
    df = df_raw.copy()

# ------------------------------------------------------------------
# 4. Mapear columnas ENOE a esquema del proyecto
# ------------------------------------------------------------------
col_map = {}

# SINCO / ocupacion — c_ocu11c es el codigo de ocupacion de 11 categorias SINCO
# clase2 es la clase de trabajador (1=privado, 2=publico, 3=empleador...) — NO usar
for cand in ['c_ocu11c', 'C_OCU11C', 'ocu11c', 'sinco']:
    if cand in df.columns:
        col_map['sinco_code'] = cand
        break

# Ingreso
for cand in ['ingocup', 'INGOCUP', 'ing_x_hrs', 'ingreso']:
    if cand in df.columns:
        col_map['ingocup'] = cand
        break

# Educacion
for cand in ['anios_esc', 'ANIOS_ESC', 'escoacum', 'nivel']:
    if cand in df.columns:
        col_map['escoacum_raw'] = cand
        break

# Edad
for cand in ['eda', 'EDA', 'edad']:
    if cand in df.columns:
        col_map['edad'] = cand
        break

# Sector SCIAN
for cand in ['rama_est2', 'RAMA_EST2', 'rama', 'scian', 'actividad']:
    if cand in df.columns:
        col_map['sector_raw'] = cand
        break

# Formalidad — seg_soc (seguridad social) es el indicador estandar ENOE
# seg_soc=1 tiene seg social (formal), seg_soc=2 no tiene (informal)
for cand in ['seg_soc', 'SEG_SOC', 'tip_con', 'TIP_CON', 'pos_ocu', 'POS_OCU']:
    if cand in df.columns:
        col_map['formalidad_raw'] = cand
        break

# Tamano empresa — emple7c: 1=micro, 2=pequena, 3-4=mediana, 5-7=grande
for cand in ['emple7c', 'EMPLE7C', 'segempleado', 'SEGEMPLEADO', 'npertra']:
    if cand in df.columns:
        col_map['tamanio_raw'] = cand
        break

print(f'\nColumnas mapeadas: {col_map}')

# ------------------------------------------------------------------
# 5. Transformar variables
# ------------------------------------------------------------------
out = pd.DataFrame()

# SINCO code (4 digitos)
if 'sinco_code' in col_map:
    out['sinco_code_raw'] = pd.to_numeric(df[col_map['sinco_code']], errors='coerce')
    # c_ocu11c tiene 11 categorias: mapear a grupo SINCO mayor
    C_OCU_TO_SINCO = {
        1: 2,   # Profesionistas -> SINCO 2
        2: 3,   # Tecnicos -> SINCO 3
        3: 3,   # Educacion -> SINCO 3
        4: 2,   # Arte/deporte -> SINCO 2
        5: 1,   # Directivos -> SINCO 1
        6: 4,   # Administrativos -> SINCO 4
        7: 7,   # Agropecuarios -> SINCO 7
        8: 8,   # Artesanos -> SINCO 8
        9: 9,   # Operadores -> SINCO 9
        10: 6,  # Servicios personales -> SINCO 6
        11: 5,  # Comerciantes -> SINCO 5
    }
    sinco_mapped = out['sinco_code_raw'].map(C_OCU_TO_SINCO)
    out['sinco_code_raw'] = sinco_mapped.fillna(6)  # default: servicios
    out['sinco_major'] = out['sinco_code_raw'].astype(int).astype(str)
else:
    out['sinco_code_raw'] = 1110
    out['sinco_major'] = '1'

# Ingreso mensual
if 'ingocup' in col_map:
    out['ingocup'] = pd.to_numeric(df[col_map['ingocup']], errors='coerce').fillna(0)
    # Filtrar valores razonables (entre 500 y 200,000 MXN/mes)
    out.loc[out['ingocup'] < 500, 'ingocup'] = np.nan
    out.loc[out['ingocup'] > 200_000, 'ingocup'] = np.nan
    out['ingocup'] = out['ingocup'].fillna(out['ingocup'].median())
else:
    out['ingocup'] = 10_000.0

# Escolaridad -> niveles 1-6
if 'escoacum_raw' in col_map:
    anios = pd.to_numeric(df[col_map['escoacum_raw']], errors='coerce').fillna(0)
    # Convertir anos de escolaridad a nivel 1-6
    out['escoacum'] = pd.cut(anios,
        bins=[-1, 0, 6, 9, 12, 16, 100],
        labels=[1, 2, 3, 4, 5, 6]).astype(int)
else:
    out['escoacum'] = 3

# Edad
if 'edad' in col_map:
    out['edad'] = pd.to_numeric(df[col_map['edad']], errors='coerce').fillna(35)
    out['edad'] = out['edad'].clip(15, 70)
else:
    out['edad'] = 35

# Sector SCIAN -> etiqueta
RAMA_TO_SECTOR = {
    1: 'Agricultura', 2: 'Agricultura', 3: 'Manufactura', 4: 'Manufactura',
    5: 'Construccion', 6: 'Comercio', 7: 'Servicios', 8: 'Servicios',
    9: 'Servicios', 10: 'Educacion', 11: 'Salud', 12: 'Finanzas',
    13: 'Tecnologia', 14: 'Transporte',
}
if 'sector_raw' in col_map:
    rama = pd.to_numeric(df[col_map['sector_raw']], errors='coerce').fillna(7)
    out['scian_sector'] = rama.map(RAMA_TO_SECTOR).fillna('Servicios')
else:
    out['scian_sector'] = 'Servicios'

# Formalidad — seg_soc: 1=tiene seguridad social (formal), 2=no tiene (informal)
# Si la columna es tip_con: 1=contrato escrito (formal)
# Si la columna es pos_ocu: 1=empleado privado, 2=empleado publico (pueden ser formal/informal)
if 'formalidad_raw' in col_map:
    col_name = col_map['formalidad_raw']
    val = pd.to_numeric(df[col_name], errors='coerce').fillna(2)
    if col_name in ('seg_soc', 'SEG_SOC'):
        # seg_soc=1 → formal (tiene seguridad social)
        out['formalidad'] = np.where(val == 1, 'Formal', 'Informal')
    elif col_name in ('tip_con', 'TIP_CON'):
        # tip_con=1 → contrato escrito = formal
        out['formalidad'] = np.where(val == 1, 'Formal', 'Informal')
    else:
        # pos_ocu fallback
        out['formalidad'] = np.where(val <= 2, 'Formal', 'Informal')
else:
    out['formalidad'] = 'Informal'

# Tamano empresa — emple7c en ENOE SDEM:
# 1=solo, 2=2-5, 3=6-10, 4=11-15, 5=16-50, 6=51-250, 7=251+
if 'tamanio_raw' in col_map:
    tam_raw = pd.to_numeric(df[col_map['tamanio_raw']], errors='coerce').fillna(2)
    TAM_MAP = {1: 1, 2: 4, 3: 8, 4: 13, 5: 30, 6: 100, 7: 250}
    out['tamanio_empresa'] = tam_raw.map(TAM_MAP).fillna(8).astype(int)
else:
    out['tamanio_empresa'] = 8

# ------------------------------------------------------------------
# 6. Filtrar solo ocupados con ingreso
# ------------------------------------------------------------------
out = out.dropna(subset=['sinco_major'])
out = out[out['ingocup'] > 0]
print(f'\nRegistros ocupados con ingreso: {len(out):,}')

# ------------------------------------------------------------------
# 7. Unir con scores O*NET via grupo SINCO
# ------------------------------------------------------------------
sinco_scores = pd.read_csv(f'{PROC}/sinco_group_scores.csv')
sinco_scores['sinco_major'] = sinco_scores['sinco_major'].astype(str)
out['sinco_major'] = out['sinco_major'].astype(str)

out = out.merge(sinco_scores[['sinco_major','cognitive_demand','social_interaction',
                               'creativity','manual_dexterity','rti',
                               'automation_risk','frey_osborne_score']],
                on='sinco_major', how='left')

# Llenar con mediana global si no hay match
for col in ['cognitive_demand','social_interaction','creativity',
            'manual_dexterity','rti','automation_risk','frey_osborne_score']:
    out[col] = out[col].fillna(out[col].median())

# sinco_code final (usar codigo SINCO raw si disponible, sino usar major*1000+100)
out['sinco_code'] = out['sinco_code_raw'].fillna(
    out['sinco_major'].astype(float) * 1000 + 100
).astype(int)

# ------------------------------------------------------------------
# 8. IRA = ingocup_anual / activo_fijo_amortizado (por sector)
# ------------------------------------------------------------------
ACTIVO_FIJO = {
    'Agricultura': 18_000, 'Manufactura': 95_000, 'Construccion': 45_000,
    'Comercio': 38_000, 'Servicios': 28_000, 'Tecnologia': 72_000,
    'Educacion': 22_000, 'Salud': 110_000, 'Finanzas': 85_000, 'Transporte': 98_000,
}
out['activo_fijo'] = out['scian_sector'].map(ACTIVO_FIJO).fillna(28_000)
out['ira'] = ((out['ingocup'] * 12) / out['activo_fijo']).round(4)

# riesgo_categoria
out['riesgo_categoria'] = pd.cut(out['frey_osborne_score'],
    bins=[-0.01, 0.4, 0.7, 1.01],
    labels=['Bajo', 'Medio', 'Alto'])

# ------------------------------------------------------------------
# 9. Sample para SQL (max 2000 registros para el proyecto)
# ------------------------------------------------------------------
sample_size = min(2000, len(out))
out_sample = out.sample(sample_size, random_state=42)

cols_final = ['sinco_code', 'escoacum', 'ingocup', 'scian_sector', 'formalidad',
              'edad', 'tamanio_empresa', 'ira', 'cognitive_demand',
              'social_interaction', 'creativity', 'manual_dexterity',
              'rti', 'automation_risk', 'frey_osborne_score', 'riesgo_categoria']

out_sample[cols_final].to_csv(f'{PROC}/enoe_jalisco.csv', index=False)

print(f'\nGuardado: {PROC}/enoe_jalisco.csv ({sample_size:,} trabajadores)')
print('\n=== ESTADISTICAS ENOE JALISCO ===')
print(f'Ingreso promedio mensual : ${out_sample["ingocup"].mean():,.0f}')
print(f'Edad promedio            : {out_sample["edad"].mean():.1f} anos')
print(f'Formales                 : {(out_sample["formalidad"]=="Formal").mean()*100:.1f}%')
print(f'Riesgo promedio          : {out_sample["automation_risk"].mean():.3f}')
print(f'\nDistribucion por sector:')
print(out_sample['scian_sector'].value_counts())
print(f'\nDistribucion riesgo_categoria:')
print(out_sample['riesgo_categoria'].value_counts())
