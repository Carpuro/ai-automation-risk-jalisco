"""
SINCO-level exposure scores (DBOE + DEOE) via the OFFICIAL INEGI crosswalk.

Supersedes the SINCO aggregation inside build_dynamic_aioe.py (and the
`sinco_dboe_scores` table), which used a crude first-digit SINCO->ISCO bridge.
This version chains the validated path:

  model_exposure_soc (SOC detailed) -> SOC broad (last digit -> 0)
    -> crosswalk_sinco_soc (official INEGI comparative tables, 4-digit)
    -> SINCO 4-digit -> SINCO major group

Two major-group aggregates are produced:
  * occupational mean  -- unweighted over SINCO 4-digit codes (US occupational
    structure projected onto SINCO; comparable to the old table)
  * Jalisco worker-weighted -- weighted by ENOE fac_tri over actual Jalisco
    workers (enoe_jalisco_workers), i.e. the exposure profile of the real
    Jalisco labor force

Outputs: tables `sinco4_exposure_scores` and `sinco_exposure_scores` +
data/processed/sinco_exposure_scores.csv. The old `sinco_dboe_scores` table
is left in place but should be considered superseded.

Run after load_model_tables.py, build_embodied_exposure.py and
build_sinco_crosswalk.py.
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
SCORES = ["dboe_2026_z", "embodied_exposure_z"]

# Official SINCO 2011 divisions (INEGI). NOTE: the labels used by the old
# build_dynamic_aioe.py aggregation were shifted from division 3 onward and
# included a non-existent division "0" -- verified against the sinco_title
# column of the official INEGI comparative tables.
SINCO_LABELS = {
    "1": "Funcionarios, directores y jefes",
    "2": "Profesionistas y tecnicos",
    "3": "Trabajadores auxiliares administrativos",
    "4": "Comerciantes, empleados en ventas y agentes",
    "5": "Trabajadores en servicios personales y vigilancia",
    "6": "Trabajadores agricolas, ganaderos, forestales y pesca",
    "7": "Trabajadores artesanales, construccion y oficios",
    "8": "Operadores de maquinaria industrial y transporte",
    "9": "Trabajadores en actividades elementales y de apoyo",
}


def main() -> None:
    c = ENGINE.connect()
    exp = pd.read_sql(
        f"SELECT soc6, {', '.join(SCORES)} FROM model_exposure_soc", c)
    xw = pd.read_sql("SELECT sinco4, soc_code FROM crosswalk_sinco_soc", c)
    wrk = pd.read_sql(
        "SELECT sinco4, fac_tri FROM enoe_jalisco_workers", c)
    c.close()

    # Detailed SOC (11-1011) -> broad (11-1010): exposure mean per broad code
    exp["soc_broad"] = exp.soc6.str.slice(0, 6) + "0"
    broad = exp.groupby("soc_broad")[SCORES].mean().reset_index()

    # SINCO 4-digit scores: mean over the broad SOC codes each SINCO maps to
    xw["sinco4"] = xw.sinco4.astype(str).str.zfill(4)
    s4 = (xw.merge(broad, left_on="soc_code", right_on="soc_broad", how="inner")
            .groupby("sinco4")[SCORES].mean().reset_index())
    s4["n_soc_broad"] = (xw.merge(broad, left_on="soc_code",
                                  right_on="soc_broad", how="inner")
                           .groupby("sinco4")["soc_broad"].nunique().values)
    print(f"SINCO 4-digit codes with exposure attached: {len(s4)}")

    # Major-group aggregate 1: unweighted occupational mean
    s4["sinco_major"] = s4.sinco4.str[0]
    occ = s4.groupby("sinco_major")[SCORES].mean()

    # Major-group aggregate 2: Jalisco worker-weighted (ENOE fac_tri)
    wrk["sinco4"] = wrk.sinco4.astype(str).str.zfill(4)
    ww = wrk.merge(s4[["sinco4"] + SCORES], on="sinco4", how="inner")
    covered = ww.fac_tri.sum() / wrk.fac_tri.sum()
    print(f"Jalisco workers covered (weighted): {covered:.1%}")
    ww["sinco_major"] = ww.sinco4.str[0]
    jal = ww.groupby("sinco_major").apply(
        lambda g: pd.Series({s: (g[s] * g.fac_tri).sum() / g.fac_tri.sum()
                             for s in SCORES}))

    major = occ.add_suffix("_occ").join(jal.add_suffix("_jal")).reset_index()
    major["sinco_label"] = major.sinco_major.map(SINCO_LABELS)
    major["n_sinco4"] = major.sinco_major.map(
        s4.groupby("sinco_major").size())
    order = (["sinco_major", "sinco_label", "n_sinco4"]
             + [f"{s}_occ" for s in SCORES] + [f"{s}_jal" for s in SCORES])
    major = major[order].sort_values("sinco_major").reset_index(drop=True)

    print("\nSINCO major-group exposure (official crosswalk):")
    print(f"{'grp':<4}{'label':<42}{'DBOE_occ':>9}{'DEOE_occ':>9}"
          f"{'DBOE_jal':>9}{'DEOE_jal':>9}")
    for _, r in major.iterrows():
        print(f"{r.sinco_major:<4}{r.sinco_label[:40]:<42}"
              f"{r.dboe_2026_z_occ:>+9.2f}{r.embodied_exposure_z_occ:>+9.2f}"
              f"{r.dboe_2026_z_jal:>+9.2f}{r.embodied_exposure_z_jal:>+9.2f}")

    # Comparison against the superseded crude-bridge table
    c = ENGINE.connect()
    old = pd.read_sql(
        "SELECT sinco_major, dboe_2026_z FROM sinco_dboe_scores", c)
    c.close()
    old["sinco_major"] = old.sinco_major.astype(str)
    cmp = major[["sinco_major", "dboe_2026_z_occ"]].merge(old, on="sinco_major")
    r = cmp.dboe_2026_z_occ.corr(cmp.dboe_2026_z)
    print(f"\nDBOE gradient: corr(official crosswalk, old crude bridge) = {r:.3f}")

    # Persist
    s4_out = s4[["sinco4", "sinco_major", "n_soc_broad"] + SCORES]
    s4_out.to_sql("sinco4_exposure_scores", ENGINE,
                  if_exists="replace", index=False, chunksize=1000)
    major.to_sql("sinco_exposure_scores", ENGINE,
                 if_exists="replace", index=False, chunksize=1000)
    major.to_csv(PROCESSED_DIR / "sinco_exposure_scores.csv", index=False)
    print(f"\n-> sinco4_exposure_scores: {len(s4_out)} rows")
    print(f"-> sinco_exposure_scores: {len(major)} rows (+ CSV in processed/)")


if __name__ == "__main__":
    main()
