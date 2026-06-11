"""
Informality as a severity multiplier on automation risk.

Informality is not a separate risk -- it multiplies the consequences of the
same exposure: an informal worker displaced by automation has no IMSS
coverage, no severance, no unemployment protection. This script crosses the
workers-at-risk classification (baseline 2030) with formality status to
quantify the UNPROTECTED at-risk population, the number the policy chapter
needs.

Conceptual note for the thesis (the formal-bias asymmetry): robot adoption
requires formal capital -- a firm buys the machine -- so formal-sector
outcome instruments (CE, IMSS) measure the embodied channel correctly. LLM
adoption diffuses person-by-person with no firm investment, so the cognitive
channel can reach informal workers and micro-businesses that those
instruments never see. Exposure here comes from ENOE (the full labor force,
informal included), so the worker-level numbers are unbiased; it is the
REALIZED-outcome evidence that is formal-only.

Output: table `informality_severity` (+ CSV).

Run after workers_at_risk prerequisites (imports its loader and curves).
"""

import sys
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workers_at_risk import load_workers, wquantile  # noqa: E402
from level2_sector_projection import cognitive_curve, embodied_curve  # noqa: E402

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    d = load_workers()
    curves = cognitive_curve().merge(embodied_curve(), on=["year", "scenario"])
    c24 = curves[curves.year == 2024].iloc[0]
    cy = curves[(curves.scenario == "baseline") & (curves.year == 2030)].iloc[0]

    p24 = d.pct_dboe * c24.c_t * d.pct_ira + d.pct_deoe * c24.r_t * d.pct_ira
    tau = wquantile(p24, d.fac_tri, 2 / 3)
    p_cog = d.pct_dboe * cy.c_t * d.pct_ira
    p_emb = d.pct_deoe * cy.r_t * d.pct_ira
    d["at_risk"] = (p_cog + p_emb) >= tau
    d["pole"] = (p_cog > p_emb).map({True: "cognitive", False: "embodied"})

    d = d.dropna(subset=["formal"])
    rows = []
    print("At-risk workers (baseline 2030) by formality x pole x education:\n")
    print(f"{'formality':<10}{'pole':<11}{'educ':<10}{'workers':>10}{'share_at_risk':>15}")
    total_risk = (d.fac_tri * d.at_risk).sum()
    for (f, pole, educ), g in d[d.at_risk].groupby(
            ["formal", "pole", "educ"], observed=True):
        n = g.fac_tri.sum()
        rows.append({"formal": f, "pole": pole, "educ": str(educ),
                     "workers": n, "share_at_risk": n / total_risk})
        print(f"{f:<10}{pole:<11}{str(educ):<10}{n:>10,.0f}{n/total_risk:>15.1%}")

    res = pd.DataFrame(rows)
    informal_risk = res[res.formal == "Informal"].workers.sum()
    informal_emb = res[(res.formal == "Informal")
                       & (res.pole == "embodied")].workers.sum()
    low_ed_informal = res[(res.formal == "Informal")
                          & (res.educ == "<=9 anos")].workers.sum()
    print(f"\nHEADLINE: {informal_risk:,.0f} at-risk workers are INFORMAL "
          f"({informal_risk/total_risk:.0%} of the at-risk population) -- "
          "same exposure, zero safety net (no IMSS, no severance).")
    print(f"  Of them, {informal_emb:,.0f} face the robot pole and "
          f"{low_ed_informal:,.0f} have <=9 years of schooling -- the "
          "triple-vulnerability group (exposed + unprotected + hard to retrain).")

    res.to_sql("informality_severity", ENGINE, if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "informality_severity.csv", index=False)
    print(f"\n-> informality_severity: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
