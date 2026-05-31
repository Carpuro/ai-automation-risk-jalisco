"""
build_ira.py
Recalcula el Indice de Rentabilidad de la Automatizacion (IRA) por sector SCIAN
usando el archivo COMPLETO de Censos Economicos 2024 (INEGI_CE2024_jalisco_completo.csv),
que incluye las variables que el archivo viejo no tenia:
  Q000B Depreciacion real de activos fijos
  Q000C Compra/adquisicion de activos fijos (inversion del anio)
  Q400A Acervo de equipo de computo y perifericos (capital tecnologico)

Tres variantes (millones de pesos en numerador y denominador, ratio adimensional):
  ira_base = J000A / (Q000A / 5)              # original: remuneraciones / acervo amortizado a 5 anios
  ira_real = (J000A + J300A + J400A) / Q000B  # costo laboral total / depreciacion real
  ira_tech = (J000A + J300A + J400A) / Q400A  # costo laboral total / capital tecnologico

Salida: tabla SQL Server external_ira_by_sector_full (NO toca external_ira_by_sector).
Nivel: sectores SCIAN a nivel estatal Jalisco. El archivo completo contiene los
CINCO censos economicos (2003, 2008, 2013, 2018, 2023), asi que la tabla es una
serie longitudinal sector x anio — sirve para Block 4 (usar anio=2023) y para el
analisis longitudinal capital/trabajo.
"""

import os
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)
CE_FILE = os.path.join(BASE, "INEGI_CE2024_jalisco_completo.csv")
engine = create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes&TrustServerCertificate=yes"
)

# Codigos INEGI requeridos (el header trae el codigo como prefijo del nombre).
CODES = {
    "H001A": "personal_ocupado",
    "H010A": "personal_remunerado",
    "J000A": "remuneraciones",
    "J300A": "contrib_seg_social",
    "J400A": "otras_prestaciones",
    "Q000A": "acervo_activos",
    "Q000B": "depreciacion",
    "Q000C": "inversion_activos",
    "Q400A": "acervo_computo",
    "A131A": "valor_agregado",
}

print("=== Cargando CE2024 completo ===")
ce = pd.read_csv(CE_FILE, skiprows=4, dtype=str, low_memory=False, encoding="latin-1")
actividad = ce.iloc[:, 3]   # 'Actividad economica' (posicional por encoding)
municipio = ce.iloc[:, 2]   # 'Municipio'

# Mapear cada codigo a su nombre de columna real (prefijo "CODE ").
colmap = {}
for code, alias in CODES.items():
    match = [c for c in ce.columns if str(c).startswith(code + " ")]
    if not match:
        raise SystemExit(f"Columna {code} no encontrada en CE2024")
    colmap[alias] = match[0]

# Filtrar: nivel estatal (Municipio vacio) + filas de Sector SCIAN.
mask = municipio.fillna("").str.strip().eq("") & \
    actividad.fillna("").str.startswith("Sector ")
sec = ce[mask].copy()
sec["anio"] = pd.to_numeric(ce.iloc[:, 0][mask], errors="coerce").astype("Int64")
sec["sector_code"] = actividad[mask].str.extract(r"Sector (\S+)")[0]
sec["sector"] = actividad[mask].str.replace(r"^Sector \S+\s*", "", regex=True)

for alias, col in colmap.items():
    sec[alias] = pd.to_numeric(sec[col], errors="coerce")

print(f"  Filas sector x anio: {len(sec)} | anios: {sorted(sec['anio'].dropna().unique())}")

# ── Calcular IRA (tres variantes) ────────────────────────────────────────────
labor_total = sec["remuneraciones"] + sec["contrib_seg_social"] + sec["otras_prestaciones"]

sec["salario_anual_prom"] = (sec["remuneraciones"] * 1e6) / sec["personal_remunerado"]
sec["ira_base"] = sec["remuneraciones"] / (sec["acervo_activos"] / 5)
sec["ira_real"] = labor_total / sec["depreciacion"].where(sec["depreciacion"] > 0)
sec["ira_tech"] = labor_total / sec["acervo_computo"].where(sec["acervo_computo"] > 0)

out_cols = ["anio", "sector_code", "sector", "personal_ocupado", "personal_remunerado",
            "remuneraciones", "contrib_seg_social", "otras_prestaciones",
            "acervo_activos", "depreciacion", "inversion_activos", "acervo_computo",
            "valor_agregado", "salario_anual_prom", "ira_base", "ira_real", "ira_tech"]
ira = (sec[out_cols].round(4)
       .sort_values(["sector_code", "anio"]).reset_index(drop=True))

print("\n=== IRA por sector SCIAN — Jalisco 2023 (ordenado por ira_base) ===")
latest = ira[ira["anio"] == 2023].sort_values("ira_base")
print(latest[["sector_code", "sector", "ira_base", "ira_real", "ira_tech"]]
      .to_string(index=False))

ira.to_sql("external_ira_by_sector_full", engine, if_exists="replace", index=False)
print(f"\n[OK] -> external_ira_by_sector_full ({len(ira)} filas sector x anio)")

with engine.connect() as conn:
    n = conn.execute(text("SELECT COUNT(*) FROM dbo.external_ira_by_sector_full")).scalar()
    print(f"  Verificado: {n} filas")
