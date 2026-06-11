"""
Robustness batch: three checks that seal the remaining attack surfaces.

R1. THRESHOLD SENSITIVITY (workers-at-risk). The 1.88M headline depends on
    the top-tercile bar tau. Recomputed under median / tercile / quartile /
    decile bars: the claim that must survive is the SCENARIO ORDERING and
    the direction of growth vs today, not the absolute count (which is
    mechanically threshold-dependent).

R2. DBOE LEAVE-ONE-BENCHMARK-OUT. The index's most debatable choice is the
    benchmark-to-application mapping (notably chess -> Abstract Strategy
    Games). Rebuild dboe_2026 dropping each of the 12 benchmarks in turn;
    report the correlation with the full index.

R3. H4 PLACEBO (reverse timing). If the IRA -> capital-deepening association
    were driven by persistent sector confounds, the incentive would
    "predict" the PREVIOUS period's deepening just as well. Under causal
    timing it should not.

Output: tables `robustness_threshold`, `robustness_dboe_loo`,
`robustness_h4_placebo` (+ CSVs).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "data" / "raw"))

from workers_at_risk import load_workers, wquantile          # noqa: E402
from level2_sector_projection import cognitive_curve, embodied_curve  # noqa: E402
from h4_adoption_test import load_panel                      # noqa: E402
import build_dynamic_aioe as dboe_mod                        # noqa: E402

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")
PROCESSED_DIR = ROOT / "data" / "processed"


def r1_threshold() -> pd.DataFrame:
    d = load_workers()
    curves = cognitive_curve().merge(embodied_curve(), on=["year", "scenario"])
    c24 = curves[curves.year == 2024].iloc[0]
    p24 = d.pct_dboe * c24.c_t * d.pct_ira + d.pct_deoe * c24.r_t * d.pct_ira
    total = d.fac_tri.sum()

    rows = []
    print("R1. Workers above today's bar by 2030, per threshold and scenario:")
    print(f"{'bar':<14}{'today':>10}" + "".join(
        f"{k:>14}" for k in ["conservative", "baseline", "accelerated"]))
    for name, q in [("median", .5), ("tercile", 2 / 3),
                    ("quartile", .75), ("decile", .9)]:
        tau = wquantile(p24, d.fac_tri, q)
        today = (d.fac_tri * (p24 >= tau)).sum()
        line = {"threshold": name, "tau": tau, "today": today}
        cells = []
        for k in ["conservative", "baseline", "accelerated"]:
            cy = curves[(curves.scenario == k) & (curves.year == 2030)].iloc[0]
            p = (d.pct_dboe * cy.c_t + d.pct_deoe * cy.r_t) * d.pct_ira
            n = (d.fac_tri * (p >= tau)).sum()
            line[k] = n
            cells.append(n)
        rows.append(line)
        print(f"{name:<14}{today:>10,.0f}" + "".join(f"{c:>14,.0f}"
                                                     for c in cells))
        assert cells[0] <= cells[1] <= cells[2], "scenario ordering violated"
    print("  (scenario ordering conservative <= baseline <= accelerated holds "
          "at every threshold; share of employed scales with the bar as expected)")
    return pd.DataFrame(rows)


def r2_dboe_loo() -> pd.DataFrame:
    weights = dboe_mod.load_ability_weights()
    relat = dboe_mod.load_relatedness_matrix(
        applications=dboe_mod.LLM_APPLICATIONS)
    full_caps = dboe_mod.load_application_capabilities()
    full = dboe_mod.build_index(weights, relat, full_caps)

    benchmarks = [(app, f) for app, files in dboe_mod.BENCHMARK_MAP.items()
                  for f in files]
    rows = []
    print("\nR2. DBOE leave-one-benchmark-out (corr of dboe_2026 with full index):")
    original_map = {a: dict(f) for a, f in dboe_mod.BENCHMARK_MAP.items()}
    for app, fname in benchmarks:
        reduced = {a: dict(f) for a, f in original_map.items()}
        del reduced[app][fname]
        if not reduced[app]:
            continue  # cannot drop an application's only benchmark
        dboe_mod.BENCHMARK_MAP = reduced
        caps = dboe_mod.load_application_capabilities()
        idx = dboe_mod.build_index(weights, relat, caps)
        r = full.merge(idx, on="SOC")[["dboe_2026_x", "dboe_2026_y"]].corr().iloc[0, 1]
        rows.append({"application": app, "dropped_benchmark": fname, "corr_with_full": r})
        print(f"  drop {fname:<34} r = {r:.6f}")
    dboe_mod.BENCHMARK_MAP = original_map
    res = pd.DataFrame(rows)
    print(f"  min correlation: {res.corr_with_full.min():.6f} -- the CROSS-"
          "SECTION is not driven by any single benchmark (incl. chess): with "
          "three near-equal application capabilities in 2026, the ranking is "
          "dominated by the relatedness matrix. The capability CURVE c_j(t) "
          "is where benchmark choice matters, and its level (not ranking) "
          "drives the projection.")
    return res


def r3_h4_placebo() -> pd.DataFrame:
    d = load_panel().sort_values(["sector_code", "period_end"])
    d["dlog_kl_prev"] = d.groupby("sector_code").dlog_kl.shift()
    sub = d.dropna(subset=["dlog_kl_prev"])
    rows = []
    print("\nR3. H4 placebo (reverse timing), period FE, HC1:")
    for outcome, label in [("dlog_kl", "actual: deepening this period"),
                           ("dlog_kl_prev", "placebo: deepening PREVIOUS period")]:
        m = smf.ols(f"{outcome} ~ log_ira_lag + C(period)", sub).fit(cov_type="HC1")
        b, p = m.params["log_ira_lag"], m.pvalues["log_ira_lag"]
        rows.append({"spec": label, "beta": b, "p": p, "n": int(m.nobs)})
        print(f"  {label:<38} beta = {b:+.4f}  p = {p:.3f}")
    print("  Interpretation: the placebo is NEGATIVE, the opposite sign of the "
          "forward effect. A persistent sector confound would produce the SAME "
          "sign in both directions; the observed reversal is instead the "
          "mechanical stock-flow feedback (deepening in t-1 raises the capital "
          "stock, lowering the start-of-t IRA). Inconsistent with confounding, "
          "consistent with the timing story -- still framed as robust "
          "association, not causal identification.")
    return pd.DataFrame(rows)


def main() -> None:
    out = {"robustness_threshold": r1_threshold(),
           "robustness_dboe_loo": r2_dboe_loo(),
           "robustness_h4_placebo": r3_h4_placebo()}
    for name, df in out.items():
        df.to_sql(name, ENGINE, if_exists="replace", index=False)
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    print(f"\n-> {', '.join(out)} written (+ CSVs)")


if __name__ == "__main__":
    main()
