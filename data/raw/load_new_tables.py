"""
load_new_tables.py
Carga a SQL Server los datos nuevos recolectados el 2026-05-30, reusando el
mismo patron de conexion y to_sql que process_external.py:

  1. dynamic_aioe_scores   — DBOE por SOC (5 anios + z) — indice propio
  2. sinco_dboe_scores     — DBOE agregado a grupo SINCO mayor
  3. imss_empleo_sector    — empleo formal IMSS Jalisco 2000-2024 (tidy long)
  4. latinobarometro_mx    — subset Mexico, percepcion IA/empleo (4 oleadas)

No modifica las tablas nucleo existentes (trabajadores, ocupaciones_onet, etc.).
"""

import os
import re
import glob
import zipfile
import tempfile
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)
PROC = os.path.join(os.path.dirname(BASE), "processed")
engine = create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes&TrustServerCertificate=yes"
)

# ── 1-2. DBOE (indice propio) ────────────────────────────────────────────────
print("\n=== 1. DBOE por SOC (dynamic_aioe_scores) ===")
dboe_soc = pd.read_csv(os.path.join(PROC, "dynamic_aioe_scores.csv"))
print(f"  {len(dboe_soc):,} ocupaciones SOC | cols: {list(dboe_soc.columns)}")
dboe_soc.to_sql("dynamic_aioe_scores", engine, if_exists="replace", index=False)
print("  [OK] -> dynamic_aioe_scores")

print("\n=== 2. DBOE por SINCO (sinco_dboe_scores) ===")
dboe_sinco = pd.read_csv(os.path.join(PROC, "sinco_dboe_scores.csv"))
print(f"  {len(dboe_sinco)} grupos SINCO")
dboe_sinco.to_sql("sinco_dboe_scores", engine, if_exists="replace", index=False)
print("  [OK] -> sinco_dboe_scores")

# ── 3. IMSS empleo formal Jalisco por sector (wide -> tidy long) ─────────────
print("\n=== 3. IMSS empleo Jalisco por sector ===")
MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
imss_raw = pd.read_excel(
    os.path.join(BASE, "IMSS_empleo_jalisco_por_sector.xlsx"), header=None
)
# Locate the year-header row (col0 == 'Division Economica' followed by years).
year_row = None
for i in range(len(imss_raw)):
    cell = str(imss_raw.iloc[i, 0]).strip().lower()
    if cell.startswith("divisi") and cell.endswith("ica"):
        if pd.to_numeric(imss_raw.iloc[i, 1:], errors="coerce").notna().sum() > 10:
            year_row = i
            break
month_row = year_row + 1
years = pd.to_numeric(imss_raw.iloc[year_row, 1:], errors="coerce")
months = imss_raw.iloc[month_row, 1:].astype(str).str.strip().str.lower()

records = []
for r in range(month_row + 1, len(imss_raw)):
    sector = str(imss_raw.iloc[r, 0]).strip()
    if not sector or sector.lower().startswith(("fuente", "nan")):
        continue
    for c in range(1, imss_raw.shape[1]):
        y = years.iloc[c - 1]
        m = MONTHS.get(months.iloc[c - 1])
        val = pd.to_numeric(imss_raw.iloc[r, c], errors="coerce")
        if pd.notna(y) and m and pd.notna(val):
            records.append((sector, int(y), m, int(val)))

imss = pd.DataFrame(records, columns=["sector", "anio", "mes", "trabajadores"])
imss["fecha"] = pd.to_datetime(
    dict(year=imss["anio"], month=imss["mes"], day=1)
)
print(f"  {len(imss):,} filas | sectores: {imss['sector'].nunique()} | "
      f"rango: {imss['fecha'].min():%Y-%m} a {imss['fecha'].max():%Y-%m}")
imss.to_sql("imss_empleo_sector", engine, if_exists="replace", index=False,
            chunksize=2000)
print("  [OK] -> imss_empleo_sector")

# ── 4. Latinobarometro — subset Mexico, percepcion IA/empleo ─────────────────
print("\n=== 4. Latinobarometro Mexico (percepcion IA) ===")
MEXICO = 484
WAVES = {
    2017: "Latinobarometro_2017_stata.zip",
    2018: "Latinobarometro_2018_stata.zip",
    2020: "Latinobarometro_2020_stata.zip",
    2023: "Latinobarometro_2023_stata.zip",
}

# Robots/AI-displacing-jobs item, verified manually from each wave's labels
# (Stata truncates labels at 80 chars, defeating keyword auto-detection):
#   2017 P56N_A  "AI and robotics will make most jobs disappear" (general)
#   2018 P61N    "robots are going to take your job"             (personal)
#   2020 p29n_e  "in 10 years robots will have taken my work place"
#   2023 P30STIN_A "robots will take away my job in ten years"   (personal)
# Note: 2017 is societal-framed; 2018/2020/2023 are personal — scales differ,
# so cross-wave comparison needs care (see docs/SQL_SERVER_SCHEMA.md).
WAVE_ROBOT_VAR = {2017: "P56N_A", 2018: "P61N", 2020: "p29n_e", 2023: "P30STIN_A"}


def _find_by_name(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _find_by_label(cols, labels, keywords):
    for var in cols:
        lab = str(labels.get(var, "")).lower()
        if any(k in lab for k in keywords):
            return var
    return None


def _find_robot_jobs(cols, labels):
    """Robots/AI displacing jobs — require an automation term AND a job term
    so we skip unrelated robot items (e.g. willingness to ride a robot car)."""
    auto = ("robot", "artificial intelligence", "inteligencia artificial")
    jobs = ("job", "work", "trabajo", "empleo", "puesto", "disappear",
            "desaparec", "take away", "replace")
    for var in cols:
        lab = str(labels.get(var, "")).lower()
        if any(a in lab for a in auto) and any(j in lab for j in jobs):
            return var
    return None


def extract_wave(df, labels, year):
    country = _find_by_name(df.columns, ["idenpa", "numinves"])
    mx = df[df[country] == MEXICO].copy() if country else df.copy()
    out = pd.DataFrame(index=mx.index)
    out["year"] = year
    out["country_code"] = MEXICO

    weight = _find_by_name(mx.columns, ["wt", "ponder", "pondera", "peso"])
    age = _find_by_name(mx.columns, ["edad", "reedad", "reeda"])
    sex = _find_by_name(mx.columns, ["sexo", "reers", "s2"])
    # Verified override first; fall back to keyword auto-detection.
    robot = WAVE_ROBOT_VAR.get(year)
    if robot not in mx.columns:
        robot = _find_robot_jobs(mx.columns, labels)
    internet = _find_by_label(mx.columns, labels, ["internet"])

    out["weight"] = pd.to_numeric(mx[weight], errors="coerce") if weight else np.nan
    out["age"] = pd.to_numeric(mx[age], errors="coerce") if age else np.nan
    out["sex"] = pd.to_numeric(mx[sex], errors="coerce") if sex else np.nan
    out["robot_jobs_perception"] = (
        pd.to_numeric(mx[robot], errors="coerce") if robot else np.nan
    )
    out["robot_var"] = robot or ""
    out["internet_home"] = (
        pd.to_numeric(mx[internet], errors="coerce") if internet else np.nan
    )
    return out, robot


frames = []
for year, zname in WAVES.items():
    zpath = os.path.join(BASE, zname)
    if not os.path.exists(zpath):
        print(f"  WARN: {zname} no encontrado, se omite")
        continue
    with zipfile.ZipFile(zpath) as zf:
        dtas = [n for n in zf.namelist() if n.lower().endswith(".dta")]
        eng = [n for n in dtas if "eng" in n.lower()]
        pick = (eng or dtas)[0]
        tmp = tempfile.mkdtemp()
        zf.extract(pick, tmp)
    path = os.path.join(tmp, pick)
    labels = pd.read_stata(path, iterator=True).variable_labels()
    df = pd.read_stata(path, convert_categoricals=False)
    wave, robot = extract_wave(df, labels, year)
    frames.append(wave)
    print(f"  {year}: Mexico n={len(wave):,} | robot var={robot or 'N/A'}")

latino = pd.concat(frames, ignore_index=True)
print(f"  Total Mexico (todas las oleadas): {len(latino):,} filas")
latino.to_sql("latinobarometro_mx", engine, if_exists="replace", index=False,
              chunksize=2000)
print("  [OK] -> latinobarometro_mx")

# ── Verificacion final ───────────────────────────────────────────────────────
print("\n=== Tablas cargadas ===")
with engine.connect() as conn:
    for t in ["dynamic_aioe_scores", "sinco_dboe_scores",
              "imss_empleo_sector", "latinobarometro_mx"]:
        try:
            n = conn.execute(text(f"SELECT COUNT(*) FROM dbo.{t}")).scalar()
            print(f"  {t:<28} {n:>10,} filas")
        except Exception as e:
            print(f"  {t:<28} ERROR: {str(e)[:60]}")
