"""
Level-2: Jalisco sector projection of AI automation pressure, 2025-2030.

Scenario analysis anchored in real data -- NOT a trained forecaster (no
observed automation panel exists for Jalisco). The design chains:

  1. Sector exposure profile. Employment-weighted DBOE (cognitive) and DEOE
     (embodied) per ENOE SCIAN sector: workers (enoe_jalisco_workers, fac_tri)
     -> sinco4_exposure_scores (official INEGI crosswalk).
  2. Technology curves, both indexed to 2024 = 1.00 (the last observed year
     for both), so pressure growth reads as "growth from the present" and the
     index is not distorted by the level of an arbitrary earlier base year:
       cognitive c(t) -- mean frontier LLM capability (Epoch benchmarks,
         reusing build_dynamic_aioe.load_application_capabilities), observed
         2022-2026, scenario-extended to 2030. Capability is bounded at 1.0,
         so the cognitive curve SATURATES (~2028 under baseline) -- reported
         as a finding, not hidden: the LLM frontier on current benchmarks is
         approaching its ceiling while robot adoption keeps compounding.
       embodied r(t) -- world robot operational stock (robot_capability_curve).
  3. Economic moderation (H4). IRA per sector (external_ira_by_sector_full,
     census 2023): pressure materializes where automation is also profitable.
  4. IMSS anchor. Formal-employment trend per sector (imss_empleo_sector,
     2000-2024 monthly) gives the realized-trajectory context each projection
     row is read against.

Pressure definition (per sector s, year t, scenario k):
    pressure_cog(s,t,k) = pct(DBOE_s) * c_k(t)
    pressure_emb(s,t,k) = pct(DEOE_s) * r_k(t)
    *_ira variants multiply by pct(IRA_s)  -- the H4 moderation
where pct() is the percentile rank across sectors (0-1), so pressure reads as
"relative exposure position x technology growth", comparable across axes.

Output: tables `sector_exposure_profile` (one row per sector) and
`sector_pressure_projection` (sector x year x scenario) + CSVs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data" / "raw"))
from build_dynamic_aioe import load_application_capabilities  # noqa: E402

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

PROCESSED_DIR = ROOT / "data" / "processed"
BASE_YEAR = 2024            # last observed year for both curves; index = 1.00
YEARS = list(range(2022, 2031))
SCENARIOS = ["conservative", "baseline", "accelerated"]

# ENOE SDEMT `scian` recode -> (SCIAN sector code, label, IMSS sector).
# The recode enumerates SCIAN 2018 sectors in standard order (ENOE
# classification); validated below by cross-tab against rama_est2.
ENOE_SCIAN = {
    1:  ("11", "Agricultura y aprovechamiento forestal", "Agricultura"),
    2:  ("21", "Mineria", "Extractivas"),
    3:  ("22", "Electricidad, agua y gas", "Electrica"),
    4:  ("23", "Construccion", "Construccion"),
    5:  ("31-33", "Industrias manufactureras", "Transformacion"),
    6:  ("43", "Comercio al por mayor", "Comercio"),
    7:  ("46", "Comercio al por menor", "Comercio"),
    8:  ("48-49", "Transportes y almacenamiento", "Transportes"),
    9:  ("51", "Informacion en medios masivos", "Transportes"),
    10: ("52", "Servicios financieros y de seguros", "Servicios"),
    11: ("53", "Servicios inmobiliarios", "Servicios"),
    12: ("54", "Servicios profesionales y tecnicos", "Servicios"),
    13: ("55", "Corporativos", "Servicios"),
    14: ("56", "Servicios de apoyo a negocios", "Servicios"),
    15: ("61", "Servicios educativos", "Servicios"),
    16: ("62", "Servicios de salud", "Servicios"),
    17: ("71", "Esparcimiento y cultura", "Servicios"),
    18: ("72", "Alojamiento y preparacion de alimentos", "Servicios"),
    19: ("81", "Otros servicios", "Servicios"),
    20: ("93", "Actividades gubernamentales", "Servicios"),
    21: (None, "Organismos internacionales", "Servicios"),
}
IMSS_NAME = {  # our IMSS table's sector strings (latin-1 accents stripped on match)
    "Agricultura": "Agricultura, ganader",
    "Extractivas": "extractivas",
    "Electrica": "ctrica",
    "Construccion": "construcci",
    "Transformacion": "transformaci",
    "Comercio": "Comercio",
    "Transportes": "Transportes",
    "Servicios": "Servicios",
}


def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def validate_scian_recode(w: pd.DataFrame) -> None:
    """Cross-tab modal rama_est2 per scian code against known anchors."""
    modal = (w[w.scian.between(1, 21)]
             .groupby("scian").rama_est2.agg(lambda x: x.mode().iloc[0]))
    anchors = {1: 1, 4: 4, 5: 3, 18: 6}   # agro, construccion, manufactura, alojamiento
    for k, v in anchors.items():
        assert modal.get(k) == v, (
            f"ENOE scian recode anchor failed: scian={k} modal rama_est2="
            f"{modal.get(k)} (expected {v}) -- mapping needs review")
    print("ENOE scian recode validated against rama_est2 anchors "
          f"(modal: {dict(modal)})")


def cognitive_curve() -> pd.DataFrame:
    """c(t) on the capability scale (0-1), scenarios to 2030, indexed to 2024.

    Scenario extension works on the bounded capability scale: the observed
    mean annual increment 2022-2026 continues (x0 / x1 / x1.5) and is capped
    at 1.0 -- the frontier cannot exceed a solved benchmark suite. Under
    baseline this saturates around 2028: a substantive result (current-
    benchmark LLM frontier nearing its ceiling), not an artifact.
    """
    caps = load_application_capabilities()       # apps x [2022..2026]
    c = dict(caps.mean(axis=0))                  # mean frontier capability, 0-1
    inc = (c[2026] - c[2022]) / 4                # observed annual increment
    rows = []
    for k, mult in [("conservative", 0.0), ("baseline", 1.0), ("accelerated", 1.5)]:
        v = dict(c)
        for y in range(2027, 2031):
            v[y] = min(v[y - 1] + inc * mult, 1.0)
        rows += [{"year": y, "scenario": k, "c_t": v[y] / c[BASE_YEAR]}
                 for y in YEARS]
    return pd.DataFrame(rows)


def embodied_curve() -> pd.DataFrame:
    """r(t) from robot_capability_curve, rebased to 2024 = 1.00."""
    c = ENGINE.connect()
    rc = pd.read_sql("SELECT year, scenario, r_t FROM robot_capability_curve", c)
    c.close()
    obs = rc[rc.scenario == "observed"][["year", "r_t"]]
    base = obs.set_index("year").r_t[BASE_YEAR]
    rows = []
    for k in SCENARIOS:
        proj = rc[rc.scenario == k][["year", "r_t"]]
        full = pd.concat([obs, proj]).drop_duplicates("year").sort_values("year")
        full = full[full.year.isin(YEARS)]
        rows.append(full.assign(scenario=k, r_t=full.r_t / base))
    return pd.concat(rows)


def main() -> None:
    c = ENGINE.connect()
    w = pd.read_sql(
        "SELECT sinco4, scian, rama_est2, fac_tri FROM enoe_jalisco_workers", c)
    s4 = pd.read_sql(
        "SELECT sinco4, dboe_2026_z, embodied_exposure_z "
        "FROM sinco4_exposure_scores", c)
    ira = pd.read_sql(
        "SELECT sector_code, ira_base, ira_real, ira_tech "
        "FROM external_ira_by_sector_full WHERE anio = 2023", c)
    imss = pd.read_sql(
        "SELECT sector, anio, AVG(trabajadores) AS workers "
        "FROM imss_empleo_sector WHERE anio IN (2014, 2024) "
        "GROUP BY sector, anio", c)
    c.close()

    validate_scian_recode(w)

    # --- 1. sector exposure profile (employment-weighted) -------------------
    w["sinco4"] = w.sinco4.astype(str).str.zfill(4)
    s4["sinco4"] = s4.sinco4.astype(str).str.zfill(4)
    ws = w.merge(s4, on="sinco4", how="inner")
    ws = ws[ws.scian.between(1, 20)]            # drop 0 (n.e.) and 21 (orgs intl)

    prof = ws.groupby("scian").apply(lambda g: pd.Series({
        "workers": g.fac_tri.sum(),
        "dboe": (g.dboe_2026_z * g.fac_tri).sum() / g.fac_tri.sum(),
        "deoe": (g.embodied_exposure_z * g.fac_tri).sum() / g.fac_tri.sum(),
    })).reset_index()
    prof["sector_code"] = prof.scian.map(lambda k: ENOE_SCIAN[k][0])
    prof["sector_label"] = prof.scian.map(lambda k: ENOE_SCIAN[k][1])
    prof["imss_sector"] = prof.scian.map(lambda k: ENOE_SCIAN[k][2])
    prof = prof.merge(ira, on="sector_code", how="left")

    # IMSS 10-year employment CAGR per matched sector (anchor context)
    piv = imss.pivot_table(index="sector", columns="anio", values="workers")
    cagr = ((piv[2024] / piv[2014]) ** (1 / 10) - 1).rename("imss_cagr_10y")
    def match_imss(name):
        pat = IMSS_NAME[name]
        hit = [s_ for s_ in cagr.index if pat.lower() in s_.lower()]
        return cagr[hit[0]] if hit else np.nan
    prof["imss_cagr_10y"] = prof.imss_sector.map(match_imss)

    # percentile ranks (0-1) for the pressure formula
    prof["pct_dboe"] = pct_rank(prof.dboe)
    prof["pct_deoe"] = pct_rank(prof.deoe)
    prof["pct_ira"] = pct_rank(prof.ira_real)

    print(f"\nSector exposure profile -- {len(prof)} sectors, "
          f"{prof.workers.sum():,.0f} weighted workers")
    show = prof.sort_values("dboe", ascending=False)
    print(f"{'sector':<42}{'workers':>10}{'DBOE':>7}{'DEOE':>7}"
          f"{'IRA_real':>9}{'IMSS10y':>8}")
    for _, r in show.iterrows():
        print(f"{r.sector_label[:40]:<42}{r.workers:>10,.0f}{r.dboe:>+7.2f}"
              f"{r.deoe:>+7.2f}{r.ira_real if pd.notna(r.ira_real) else float('nan'):>9.2f}"
              f"{r.imss_cagr_10y:>8.1%}")

    # --- 2-3. pressure projection -------------------------------------------
    cog = cognitive_curve()
    emb = embodied_curve()
    curves = cog.merge(emb, on=["year", "scenario"])

    proj = prof.assign(key=1).merge(curves.assign(key=1), on="key").drop(columns="key")
    proj["pressure_cog"] = proj.pct_dboe * proj.c_t
    proj["pressure_emb"] = proj.pct_deoe * proj.r_t
    proj["pressure_total"] = proj.pressure_cog + proj.pressure_emb
    proj["pressure_cog_ira"] = proj.pressure_cog * proj.pct_ira
    proj["pressure_emb_ira"] = proj.pressure_emb * proj.pct_ira
    proj["pressure_total_ira"] = proj.pressure_cog_ira + proj.pressure_emb_ira

    out_cols = ["scian", "sector_code", "sector_label", "imss_sector", "workers",
                "year", "scenario", "c_t", "r_t",
                "pressure_cog", "pressure_emb", "pressure_total",
                "pressure_cog_ira", "pressure_emb_ira", "pressure_total_ira"]
    proj = proj[out_cols].sort_values(["scenario", "year", "scian"])

    print("\nTop-6 sectors by total pressure 2030 (baseline, IRA-moderated):")
    top = (proj.query("year == 2030 and scenario == 'baseline'")
               .sort_values("pressure_total_ira", ascending=False).head(6))
    for _, r in top.iterrows():
        dom = "COG" if r.pressure_cog_ira > r.pressure_emb_ira else "EMB"
        print(f"  {r.sector_label[:40]:<42} total={r.pressure_total_ira:.2f} "
              f"(cog {r.pressure_cog_ira:.2f} / emb {r.pressure_emb_ira:.2f} -> {dom}) "
              f"workers={r.workers:,.0f}")

    print(f"\nCurve indices ({BASE_YEAR}=1.00):")
    for k in SCENARIOS:
        cc = curves[(curves.scenario == k) & (curves.year == 2030)]
        print(f"  {k:<14} c(2030)={cc.c_t.iloc[0]:.2f}  r(2030)={cc.r_t.iloc[0]:.2f}")

    prof.drop(columns=["pct_dboe", "pct_deoe", "pct_ira"]).to_sql(
        "sector_exposure_profile", ENGINE, if_exists="replace", index=False)
    proj.to_sql("sector_pressure_projection", ENGINE,
                if_exists="replace", index=False, chunksize=1000)
    prof.to_csv(PROCESSED_DIR / "sector_exposure_profile.csv", index=False)
    proj.to_csv(PROCESSED_DIR / "sector_pressure_projection.csv", index=False)
    print(f"\n-> sector_exposure_profile: {len(prof)} rows")
    print(f"-> sector_pressure_projection: {len(proj)} rows (+ CSVs in processed/)")


if __name__ == "__main__":
    main()
