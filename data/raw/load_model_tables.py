"""
Load all locally-available data that was missing or only coarsely present in
SQL Server, so the server becomes the single source of truth for modeling.

Creates / replaces (non-destructive to core tables trabajadores / ocupaciones_onet):
  1. enoe_jalisco_workers     -- Jalisco occupied workers with GRANULAR SINCO-4d
                                 occupation (COE1.p3), joined SDEMT+COE1. Replaces
                                 the 11-group coarseness that was loaded before.
  2. crosswalk_isco4_onet_scores  -- ISCO-4d O*NET-derived Frey-Osborne features
  3. crosswalk_onet_scores        -- O*NET SOC-level derived scores
  4. crosswalk_sinco_group_scores -- SINCO major-group scores
  5. model_exposure_soc       -- consolidated Level-1 modeling table: 667 SOC
                                 occupations x 5 AI-exposure indices on a common
                                 6-digit SOC grain (cognitive + physical axes).

Reproduction order: run AFTER process_external.py / load_new_tables.py.
"""

import os
import pandas as pd
import numpy as np
import sqlalchemy as sa

RAW = os.path.dirname(__file__)
PROC = os.path.join(RAW, "..", "processed")
ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)


def load(df, name):
    df.to_sql(name, ENGINE, if_exists="replace", index=False, chunksize=1000)
    print(f"  -> {name}: {len(df):,} rows, {len(df.columns)} cols")


# ---------------------------------------------------------------------------
# 1. ENOE Jalisco workers with granular SINCO-4d occupation (SDEMT + COE1.p3)
# ---------------------------------------------------------------------------
def build_enoe_workers():
    print("1. ENOE Jalisco workers (SDEMT + COE1, granular occupation)")
    key = ["ent", "con", "upm", "v_sel", "n_hog", "h_mud", "n_ren", "per"]

    sd = pd.read_csv(f"{RAW}/enoe/ENOE_SDEMT324.csv", encoding="latin-1", low_memory=False)
    sd.columns = [c.lower() for c in sd.columns]
    sd = sd[sd["ent"].astype(str).str.zfill(2) == "14"].copy()

    co = pd.read_csv(
        f"{RAW}/enoe/ENOE_COE1T324.csv", encoding="latin-1", low_memory=False,
        usecols=lambda c: c.lower() in key + ["p3"],
    )
    co.columns = [c.lower() for c in co.columns]
    co = co.drop_duplicates(subset=key, keep="first")  # one occupation per person

    m = sd.merge(co, on=key, how="left")

    # keep occupied workers (clase1 == 1) with a valid 4-digit SINCO occupation
    m["sinco4"] = pd.to_numeric(m["p3"], errors="coerce")
    w = m[(m["clase1"] == 1) & (m["sinco4"] >= 1000)].copy()
    w["sinco4"] = w["sinco4"].astype(int)
    w["sinco_major"] = (w["sinco4"] // 1000).astype(int)

    cols = [
        "ent", "con", "v_sel", "n_hog", "n_ren", "per", "fac_tri",
        "sex", "eda", "anios_esc", "niv_ins", "cs_p17",
        "sinco4", "sinco_major", "c_ocu11c", "scian", "rama_est2",
        "pos_ocu", "seg_soc", "emple7c", "ing7c", "hrsocup", "ingocup", "ur",
    ]
    cols = [c for c in cols if c in w.columns]
    out = w[cols].copy()
    load(out, "enoe_jalisco_workers")
    print(f"     distinct SINCO-4d occupations in Jalisco: {out['sinco4'].nunique()}")
    return out


# ---------------------------------------------------------------------------
# 2-4. Processed crosswalk / score CSVs not yet in the server
# ---------------------------------------------------------------------------
def load_processed_csvs():
    print("2-4. Processed crosswalk tables")
    mapping = {
        "isco4_onet_scores.csv": "crosswalk_isco4_onet_scores",
        "onet_scores.csv": "crosswalk_onet_scores",
        "sinco_group_scores.csv": "crosswalk_sinco_group_scores",
    }
    for fname, tname in mapping.items():
        path = os.path.join(PROC, fname)
        if os.path.exists(path):
            load(pd.read_csv(path), tname)
        else:
            print(f"  !! missing {path}")


# ---------------------------------------------------------------------------
# 5. Consolidated Level-1 modeling table (667 SOC x 5 exposure indices)
# ---------------------------------------------------------------------------
def build_model_exposure_soc():
    print("5. model_exposure_soc (consolidated 2-axis exposure, SOC grain)")
    c = ENGINE.connect()

    dboe = pd.read_sql("SELECT SOC AS soc6, dboe_2026, dboe_2026_z FROM dynamic_aioe_scores", c)
    aioe = pd.read_sql("SELECT soc_code AS soc6, aioe_score, lm_aioe_score FROM external_aioe", c)
    anth = pd.read_sql("SELECT soc_code AS soc6, anthropic_observed_exposure FROM external_anthropic_job", c)
    # physical axis: O*NET-detail (8-digit) aggregated to 6-digit SOC
    mor = pd.read_sql(
        "SELECT LEFT(soc_code,7) AS soc6, AVG(auto_w) AS moravec_auto_w, "
        "AVG(agree_w) AS moravec_agree_w FROM external_moravec_occ GROUP BY LEFT(soc_code,7)", c)
    rl = pd.read_sql(
        "SELECT LEFT(soc_code,7) AS soc6, AVG(rl_index_mean) AS rl_index_mean, "
        "AVG(rl_index_max) AS rl_index_max FROM external_rl_feasibility_occ GROUP BY LEFT(soc_code,7)", c)
    title = pd.read_sql("SELECT soc_code AS soc6, occupation_title FROM external_aioe", c)
    c.close()

    df = (dboe.merge(aioe, on="soc6").merge(anth, on="soc6")
              .merge(mor, on="soc6").merge(rl, on="soc6")
              .merge(title, on="soc6", how="left"))

    # z-standardize the physical-axis components for comparability with dboe_z
    for col in ["moravec_auto_w", "rl_index_mean"]:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

    load(df, "model_exposure_soc")
    print(f"     N = {len(df)} occupations with all 5 indices")
    print("     index correlations:")
    print(df[["dboe_2026_z", "aioe_score", "anthropic_observed_exposure",
              "moravec_auto_w", "rl_index_mean"]].corr().round(2).to_string())
    return df


if __name__ == "__main__":
    build_enoe_workers()
    load_processed_csvs()
    build_model_exposure_soc()
    print("\nDone. Server updated.")
