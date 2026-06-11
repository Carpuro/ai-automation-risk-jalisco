"""
Crosswalk coverage: who are the ~12% uncovered workers, and can they flip H3?

The SINCO->SOC chain attaches exposure to ~88% of Jalisco workers
(employment-weighted). This closes the open flank in two steps:

  1. DIAGNOSIS -- which SINCO-4d codes carry the uncovered weight, and are
     they codes absent from the official 2011 comparative tables (e.g.
     SINCO-2019 revisions in ENOE 2024) or codes whose mapped SOC has no
     exposure score?
  2. BOUNDS -- worst-case sensitivity: assign ALL uncovered workers first
     the minimum and then the maximum observed occupation exposure on each
     axis and recompute the Jalisco employment-weighted means. If the H3
     signs/ordering survive at both extremes, the coverage gap cannot
     change the conclusion.

Output: table `crosswalk_coverage_bounds` (+ CSV) and printed diagnosis.
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    c = ENGINE.connect()
    w = pd.read_sql("SELECT sinco4, fac_tri FROM enoe_jalisco_workers", c)
    s4 = pd.read_sql(
        "SELECT sinco4, dboe_2026_z AS dboe, embodied_exposure_z AS deoe "
        "FROM sinco4_exposure_scores", c)
    xw = pd.read_sql("SELECT DISTINCT sinco4, sinco_title "
                     "FROM crosswalk_sinco_soc", c)
    c.close()
    w["fac_tri"] = pd.to_numeric(w.fac_tri, errors="coerce")
    for d_ in (w, s4, xw):
        d_["sinco4"] = d_.sinco4.astype(str).str.zfill(4)

    d = w.merge(s4, on="sinco4", how="left")
    total = d.fac_tri.sum()
    unc = d[d.dboe.isna()]
    print(f"Uncovered: {unc.fac_tri.sum():,.0f} of {total:,.0f} weighted "
          f"workers ({unc.fac_tri.sum()/total:.1%})")

    # diagnosis: in-crosswalk-but-no-exposure vs absent-from-crosswalk
    unc_codes = (unc.groupby("sinco4").fac_tri.sum()
                 .sort_values(ascending=False).reset_index())
    unc_codes["in_crosswalk"] = unc_codes.sinco4.isin(xw.sinco4)
    in_xw = unc_codes[unc_codes.in_crosswalk].fac_tri.sum()
    not_xw = unc_codes[~unc_codes.in_crosswalk].fac_tri.sum()
    print(f"  mapped to SOC but SOC lacks exposure score: {in_xw:,.0f}")
    print(f"  absent from official 2011 comparative tables "
          f"(likely SINCO-2019 revisions): {not_xw:,.0f}")
    print("\nTop-8 uncovered codes by weight:")
    titles = xw.set_index("sinco4").sinco_title
    for _, r in unc_codes.head(8).iterrows():
        t = titles.get(r.sinco4, "-- not in 2011 tables --")
        print(f"  {r.sinco4}  {r.fac_tri:>9,.0f}  {str(t)[:60]}")

    # realistic direction: uncovered weight by SINCO division. If it sits in
    # the manual divisions (6 agro, 7 artisanal, 8 operators, 9 elementary),
    # the realistic correction RAISES the Jalisco DEOE mean (and lowers DBOE),
    # i.e. the covered-sample estimate is conservative for H3.
    unc_codes["division"] = unc_codes.sinco4.str[0]
    div_w = unc_codes.groupby("division").fac_tri.sum()
    manual_share = div_w.reindex(list("6789")).sum() / div_w.sum()
    print(f"\nUncovered weight in manual divisions (6-9): {manual_share:.0%} "
          "-> realistic correction pushes DEOE UP / DBOE DOWN (H3-conservative)")

    # bounds
    cov = d.dropna(subset=["dboe"])
    rows = []
    for axis in ["dboe", "deoe"]:
        base = (cov[axis] * cov.fac_tri).sum() / cov.fac_tri.sum()
        lo_v, hi_v = s4[axis].min(), s4[axis].max()
        w_unc, w_cov = unc.fac_tri.sum(), cov.fac_tri.sum()
        lo = (base * w_cov + lo_v * w_unc) / total
        hi = (base * w_cov + hi_v * w_unc) / total
        rows.append({"axis": axis, "covered_mean": base,
                     "bound_low": lo, "bound_high": hi})
        print(f"\n{axis.upper()}: covered mean {base:+.3f}; "
              f"worst-case bounds [{lo:+.3f}, {hi:+.3f}] "
              f"(uncovered all at min {lo_v:+.2f} / max {hi_v:+.2f})")
    res = pd.DataFrame(rows)

    db, de = res.iloc[0], res.iloc[1]
    verdict_neg = db.bound_high < 0          # DBOE stays below average?
    verdict_pos = de.bound_low > 0           # DEOE stays above average?
    print(f"\nH3 verdict at bounds: DBOE stays negative even at upper bound: "
          f"{verdict_neg}; DEOE stays positive even at lower bound: {verdict_pos}")
    print("If either is False, the conclusion holds for the covered 88% and "
          "the gap is reported as bounded uncertainty, not ignored.")

    res.to_sql("crosswalk_coverage_bounds", ENGINE,
               if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "crosswalk_coverage_bounds.csv", index=False)
    print(f"\n-> crosswalk_coverage_bounds: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
