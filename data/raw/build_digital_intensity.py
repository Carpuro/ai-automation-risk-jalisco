"""
build_digital_intensity.py
Construye una medida REAL de intensidad digital por ocupacion a partir de O*NET
Technology Skills (tabla onet_technology_skills, ya cargada en SQL Server).

Reemplaza el `digital_access` sintetico del Block 3. La idea (buffer/bottleneck,
ILO/World Bank WP121): cuantas herramientas digitales/software requiere la
ocupacion. Mas tecnologias = mayor intensidad digital.

Medidas por ocupacion:
  tech_skill_count  — numero de tecnologias/software distintas listadas
  hot_tech_count    — de esas, cuantas son "hot technology" (en demanda actual)
  in_demand_count   — cuantas marcadas "in demand"

Agregacion: O*NET-SOC (8 digitos) -> SOC (6) -> ISCO mayor (ESCO) -> SINCO mayor,
misma logica que build_crosswalk.py / build_dynamic_aioe.py.

Salidas:
  - tabla sinco_digital_intensity (10 grupos SINCO + z-score)
  - columna ocupaciones_onet.digital_intensity_z (no destructivo; conserva el
    digital_access sintetico previo)
"""

import os
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)
engine = create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes&TrustServerCertificate=yes"
)

# SINCO mayor -> ISCO-08 mayor (identico a build_crosswalk.py).
SINCO_TO_ISCO = {
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "5", "7": "6", "8": "7", "9": "8", "0": "9",
}
SINCO_LABELS = {
    "1": "Directivos y gerentes",
    "2": "Profesionistas y tecnicos de alto nivel",
    "3": "Tecnicos especializados",
    "4": "Trabajadores de apoyo administrativo",
    "5": "Comerciantes y vendedores",
    "6": "Trabajadores en servicios personales",
    "7": "Trabajadores agropecuarios",
    "8": "Artesanos y trabajadores en manufactura",
    "9": "Operadores de maquinaria y transporte",
    "0": "Trabajadores en ocupaciones elementales",
}


def zscore(s):
    return (s - s.mean()) / s.std()


# ── 1. Intensidad digital por O*NET-SOC -> SOC de 6 digitos ──────────────────
print("=== Technology Skills por ocupacion ===")
tech = pd.read_sql("SELECT onetsoc_code, hot_technology, in_demand "
                   "FROM onet_technology_skills", engine)
tech["hot"] = tech["hot_technology"].str.upper().eq("Y").astype(int)
tech["dem"] = tech["in_demand"].str.upper().eq("Y").astype(int)

by_soc8 = tech.groupby("onetsoc_code").agg(
    tech_skill_count=("onetsoc_code", "size"),
    hot_tech_count=("hot", "sum"),
    in_demand_count=("dem", "sum"),
).reset_index()
by_soc8["soc6"] = by_soc8["onetsoc_code"].str.slice(0, 7)
by_soc6 = by_soc8.groupby("soc6")[
    ["tech_skill_count", "hot_tech_count", "in_demand_count"]
].mean().reset_index()
print(f"  {len(by_soc6)} SOC (6 digitos) con intensidad digital")

# ── 2. SOC -> ISCO mayor via crosswalk ESCO ──────────────────────────────────
esco = pd.read_sql("SELECT soc_code, isco_4 FROM external_esco_crosswalk", engine)
esco["soc6"] = esco["soc_code"].astype(str).str.slice(0, 7)
esco["isco_major"] = esco["isco_4"].astype(str).str[0]
esco_soc = esco.merge(by_soc6, on="soc6", how="inner")

# ── 3. Agregar a grupo SINCO mayor ───────────────────────────────────────────
metrics = ["tech_skill_count", "hot_tech_count", "in_demand_count"]
rows = []
for sinco, isco in SINCO_TO_ISCO.items():
    sub = esco_soc[esco_soc["isco_major"] == isco]
    if sub.empty:
        sub = esco_soc
    means = sub[metrics].mean()
    means["sinco_major"] = sinco
    means["n_soc"] = sub["soc6"].nunique()
    rows.append(means)

sinco_df = pd.DataFrame(rows)
sinco_df["digital_intensity_z"] = zscore(sinco_df["tech_skill_count"])
sinco_df["sinco_label"] = sinco_df["sinco_major"].map(SINCO_LABELS)
sinco_df = sinco_df[["sinco_major", "sinco_label", "n_soc",
                     "tech_skill_count", "hot_tech_count", "in_demand_count",
                     "digital_intensity_z"]].round(4).sort_values("sinco_major")

print("\n=== Intensidad digital por SINCO mayor ===")
print(sinco_df[["sinco_major", "sinco_label", "tech_skill_count",
                "digital_intensity_z"]].to_string(index=False))

sinco_df.to_sql("sinco_digital_intensity", engine, if_exists="replace", index=False)
print("\n[OK] -> sinco_digital_intensity")

# ── 4. Integrar en ocupaciones_onet (no destructivo) ─────────────────────────
CHK = ("SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE "
       "TABLE_NAME='ocupaciones_onet' AND COLUMN_NAME='digital_intensity_z'")
UPD = ("UPDATE o SET o.digital_intensity_z = s.digital_intensity_z "
       "FROM ocupaciones_onet o JOIN sinco_digital_intensity s "
       "ON CAST(o.sinco_code AS VARCHAR(2)) = CAST(s.sinco_major AS VARCHAR(2))")
with engine.begin() as cx:
    if not cx.execute(text(CHK)).scalar():
        cx.execute(text("ALTER TABLE ocupaciones_onet ADD digital_intensity_z FLOAT"))
        print("  Columna digital_intensity_z agregada")
    n = cx.execute(text(UPD)).rowcount
    print(f"  [OK] {n} filas de ocupaciones_onet actualizadas")
