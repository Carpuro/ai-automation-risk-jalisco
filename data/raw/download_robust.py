"""
download_robust.py
Descarga datos para robustecer el analisis multi-LLM e internacional:

  1. ILO WP140 2025 — Final_Scores_ISCO08 (scores GenAI por ocupacion ISCO-08 a 4 digitos)
  2. ILO WP140 2025 — 4digits_with_tasks (scores a nivel de tarea ISCO-08)
  3. ILO Radial Plot JSON — scores ISCO anteriores (Gmyrek et al. 2023)
  4. OECD/PIAAC codigo Stata — riesgo de automatizacion Frey-Osborne por pais OCDE
"""

import os, time, requests

OUT = os.path.dirname(__file__)

FILES = [
    # ─── ILO Gmyrek et al. 2025 — Scores ISCO-08 actualizados ────────────────
    {
        'name': 'ILO_WP140_Final_Scores_ISCO08_2025.xlsx',
        'url':  'https://raw.githubusercontent.com/pgmyrek/2025_GenAI_scores_ISCO08/main/Final_Scores_ISCO08_Gmyrek_et_al_2025.xlsx',
        'desc': 'ILO WP140 (2025): Scores finales de exposicion GenAI por ocupacion ISCO-08 a 4 digitos — Gmyrek, Berg, Bescond',
    },
    {
        'name': 'ILO_WP140_4digits_with_tasks_2025.xlsx',
        'url':  'https://raw.githubusercontent.com/pgmyrek/2025_GenAI_scores_ISCO08/main/4digits_with_tasks.xlsx',
        'desc': 'ILO WP140 (2025): Scores por tarea a nivel ISCO-08 4 digitos',
    },
    {
        'name': 'ILO_WP140_output_data.json',
        'url':  'https://raw.githubusercontent.com/pgmyrek/2025_GenAI_scores_ISCO08/main/output_data.json',
        'desc': 'ILO WP140 (2025): Datos de visualizacion — todos los scores ISCO-08 en JSON',
    },
    # ─── ILO Radial Plot — scores 2023 (version anterior, comparacion) ────────
    {
        'name': 'ILO_WP096_ISCO08_scores_2023.json',
        'url':  'https://raw.githubusercontent.com/pgmyrek/GenAI_Exposure_Radial_Plot/main/output_data.json',
        'desc': 'ILO WP096 (2023): Scores de exposicion GenAI por ISCO-08 — version original Gmyrek et al.',
    },
    # ─── OECD PIAAC — riesgo Frey-Osborne por pais OCDE ─────────────────────
    {
        'name': 'OECD_PIAAC_foclass.do',
        'url':  'https://raw.githubusercontent.com/LjubicaN/Risk-of-automation/main/03_foclass.do',
        'desc': 'Codigo Stata para calcular riesgo Frey-Osborne por ocupacion — Nedelkoska & Quintini (2018)',
    },
    {
        'name': 'OECD_PIAAC_readme.md',
        'url':  'https://raw.githubusercontent.com/LjubicaN/Risk-of-automation/main/README.md',
        'desc': 'Documentacion del dataset PIAAC y metodologia de calculo',
    },
]

# ─── INEGI Marco Geoestadistico 2020 — Municipios Jalisco (para mapa) ────────
# Fuente: sbl-sdsc/mexico-boundaries (parquet con CVE_MUN oficial INEGI)
# Output: INEGI_Jalisco_municipios.geojson (125 municipios, CVE_ENT=14)
def download_jalisco_geo():
    import io
    from shapely import wkb
    import geopandas as gpd

    dest = os.path.join(OUT, 'INEGI_Jalisco_municipios.geojson')
    if os.path.exists(dest):
        print(f'[ya existe] INEGI_Jalisco_municipios.geojson ({os.path.getsize(dest)/1024:.0f} KB)')
        return

    print('Descargando INEGI Marco Geoestadistico 2020 (municipios Mexico)...')
    url = 'https://raw.githubusercontent.com/sbl-sdsc/mexico-boundaries/main/data/mexico_admin2.parquet'
    import pandas as pd
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=120, stream=True)
    r.raise_for_status()
    df = pd.read_parquet(io.BytesIO(r.content))
    df['geometry'] = df['geometry'].apply(lambda b: wkb.loads(b) if isinstance(b, (bytes, bytearray)) else b)
    jal = df[df['CVE_ENT'] == '14'].copy().reset_index(drop=True)
    gpd.GeoDataFrame(jal, geometry='geometry', crs='EPSG:4326').to_file(dest, driver='GeoJSON')
    print(f'  [OK] {os.path.getsize(dest)/1024:.0f} KB — {len(jal)} municipios Jalisco con CVE_MUN oficial')

try:
    download_jalisco_geo()
except Exception as e:
    print(f'  [ERROR] INEGI GeoJSON: {e}')

for f in FILES:
    dest = os.path.join(OUT, f['name'])
    if os.path.exists(dest):
        size = os.path.getsize(dest) / 1024
        print(f'[ya existe] {f["name"]} ({size:.0f} KB)')
        continue

    print(f'Descargando: {f["name"]}')
    print(f'  {f["desc"]}')
    try:
        t0 = time.time()
        r  = requests.get(f['url'], headers={'User-Agent': 'Mozilla/5.0'},
                          timeout=60, stream=True)
        r.raise_for_status()
        total = 0
        with open(dest, 'wb') as fh:
            for chunk in r.iter_content(65536):
                fh.write(chunk)
                total += len(chunk)
        print(f'  [OK] {total/1024:.0f} KB — {time.time()-t0:.1f}s')
    except Exception as e:
        print(f'  [ERROR] {e}')

print('\n=== Resumen ===')
for f in FILES:
    dest = os.path.join(OUT, f['name'])
    if os.path.exists(dest):
        print(f'  ✓ {f["name"]:<50} {os.path.getsize(dest)/1024:>8.0f} KB')
    else:
        print(f'  ✗ {f["name"]:<50} NO DESCARGADO')
