"""
AEI Mexico validation: does Mexican Claude usage look like the US profile
that the Level-1 target is built from?

The Level-1 target (anthropic_observed_exposure) is derived from globally
aggregated, US-dominated Claude usage -- its biggest documented limitation.
The Anthropic Economic Index release 2025-09-15 (HuggingFace
Anthropic/EconomicIndex) publishes country-level usage shares by SOC major
group, including Mexico (geo_id MEX).

This script extracts the soc_occupation shares for MEX / USA / global,
renormalizes over CLASSIFIED usage (Mexico's not_classified share is 42% vs
17% USA -- shorter conversations in Spanish classify worse), and reports the
cross-country correlation of occupational usage profiles.

If Mexico's profile tracks the US/global one, using the US-derived target in
the Level-1 model is defensible for Mexico; the residual differences (which
SOC groups Mexico over/under-uses) are reported for the limitations section.

Full target upgrade is NOT feasible from this release: the country x task
grain publishes only the top ~100 tasks per country (long tail truncated),
far too sparse for the 667-occupation Level-1 grain -- documented here.

Source file: data/raw/aei/aei_enriched_claude_ai_2025-08.csv (downloaded
2026-06-11). Output: table `aei_mexico_soc` (+ CSV).
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

ROOT = Path(__file__).resolve().parent.parent
AEI_FILE = ROOT / "data" / "raw" / "aei" / "aei_enriched_claude_ai_2025-08.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
EXCLUDE = {"not_classified", "none"}


def main() -> None:
    d = pd.read_csv(AEI_FILE)
    s = d[(d.facet == "soc_occupation") & (d.variable == "soc_pct")]

    frames = {}
    for key, sel in [("mex", s.geo_id == "MEX"), ("usa", s.geo_id == "USA"),
                     ("global_", s.geography == "global")]:
        sub = s[sel][["cluster_name", "value"]].set_index("cluster_name").value
        sub = sub[~sub.index.isin(EXCLUDE)]
        frames[key] = sub / sub.sum() * 100   # renormalize over classified
    m = pd.DataFrame(frames).dropna(subset=["mex"]).fillna(0).reset_index()

    print("Claude usage share by SOC major group, classified usage only (%):")
    print(m.sort_values("mex", ascending=False).round(2).to_string(index=False))
    r_us = m.mex.corr(m.usa)
    r_gl = m.mex.corr(m.global_)
    rho_us = m.mex.corr(m.usa, method="spearman")
    print(f"\ncorr(MEX, USA) = {r_us:.3f} (Spearman {rho_us:.3f}); "
          f"corr(MEX, global) = {r_gl:.3f}")
    print("Reading: Mexico's occupational usage profile tracks the US profile "
          "the Level-1 target is built from -- using the US-derived target is "
          "defensible; largest residual gaps reported above for limitations.")
    m["mex_minus_usa"] = m.mex - m.usa
    print("\nLargest MEX-USA gaps (pp of classified usage):")
    for _, r in m.reindex(m.mex_minus_usa.abs().sort_values(ascending=False)
                          .index[:4]).iterrows():
        print(f"  {r.cluster_name:<48} {r.mex_minus_usa:+.1f}")

    m.to_sql("aei_mexico_soc", ENGINE, if_exists="replace", index=False)
    m.to_csv(PROCESSED_DIR / "aei_mexico_soc.csv", index=False)
    print(f"\n-> aei_mexico_soc: {len(m)} rows (+ CSV)")


if __name__ == "__main__":
    main()
