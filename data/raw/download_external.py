"""
download_external.py
Descarga datos externos de Block 3 con URLs verificadas:
  1. AIOE_DataAppendix.xlsx       — Felten et al. (AI Occupational Exposure por SOC)
  2. Language Modeling AIOE.xlsx  — Felten et al. (exposicion especifica a LLMs)
  3. task_penetration.csv         — Anthropic Economic Index (uso real de Claude por tarea O*NET)
  4. job_exposure.csv             — Anthropic Economic Index (exposicion por ocupacion)
"""

import os, time
import requests

OUT = os.path.dirname(__file__)

FILES = [
    {
        'name': 'AIOE_DataAppendix.xlsx',
        'url':  'https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx',
        'desc': 'AIOE scores por SOC code — Felten, Raj & Seamans (2021/2023)',
    },
    {
        'name': 'Language_Modeling_AIOE_AIIE.xlsx',
        'url':  'https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Language%20Modeling%20AIOE%20and%20AIIE.xlsx',
        'desc': 'AIOE especifico para aplicaciones de language modeling (LLMs)',
    },
    {
        'name': 'anthropic_task_penetration.csv',
        'url':  'https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/labor_market_impacts/task_penetration.csv?download=true',
        'desc': 'Penetracion de Claude por tarea O*NET — Anthropic Economic Index 2025-2026',
    },
    {
        'name': 'anthropic_job_exposure.csv',
        'url':  'https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/labor_market_impacts/job_exposure.csv?download=true',
        'desc': 'Exposicion por ocupacion — Anthropic Economic Index 2025-2026',
    },
]

for f in FILES:
    dest = os.path.join(OUT, f['name'])
    if os.path.exists(dest):
        size = os.path.getsize(dest) / 1024
        print(f'[ya existe] {f["name"]} ({size:.0f} KB) — omitiendo')
        continue

    print(f'Descargando {f["name"]}...')
    print(f'  Fuente: {f["desc"]}')
    t0 = time.time()
    resp = requests.get(f['url'], stream=True,
                        headers={'User-Agent': 'Mozilla/5.0'},
                        timeout=120)
    resp.raise_for_status()

    total = 0
    with open(dest, 'wb') as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            total += len(chunk)

    size_kb = total / 1024
    print(f'  [OK] {size_kb:.0f} KB en {time.time()-t0:.1f}s → {dest}')

print('\n=== Archivos descargados ===')
for f in FILES:
    dest = os.path.join(OUT, f['name'])
    if os.path.exists(dest):
        print(f'  ✓ {f["name"]:45s} {os.path.getsize(dest)/1024:>8.0f} KB')
    else:
        print(f'  ✗ {f["name"]:45s} NO ENCONTRADO')
