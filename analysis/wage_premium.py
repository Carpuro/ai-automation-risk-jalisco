"""
Wage-premium test: is AI-exposed work already paid more?

Rhetorical link to H4: the wage attached to exposed work IS the automation
incentive -- if LLM-exposed occupations command a premium conditional on
human capital, that premium is precisely the labor cost an adopter saves,
connecting the worker-level profile to the IRA mechanism.

Mincer-style weighted OLS on ENOE Jalisco workers (fac_tri, HC1):

  log(hourly wage) ~ schooling + age + age^2 + female + informal + rural
                     + DBOE + DEOE

Sample: income reporters with positive hours (hourly wage =
ingocup / (hrsocup x 4.33)). Income non-response is 38.5% -- noted as a
selection caveat; results describe the reporting sample.

Output: table `wage_premium` (+ CSV) with coefficients for three nested
specs (human capital only; + exposure; + exposure x education interaction).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def load() -> pd.DataFrame:
    c = ENGINE.connect()
    w = pd.read_sql(
        "SELECT sinco4, sex, eda, anios_esc, seg_soc, ur, ingocup, hrsocup, "
        "fac_tri FROM enoe_jalisco_workers", c)
    s4 = pd.read_sql(
        "SELECT sinco4, dboe_2026_z AS dboe, embodied_exposure_z AS deoe "
        "FROM sinco4_exposure_scores", c)
    c.close()
    for col in ["sex", "eda", "anios_esc", "seg_soc", "ur", "ingocup",
                "hrsocup", "fac_tri"]:
        w[col] = pd.to_numeric(w[col], errors="coerce")
    w["sinco4"] = w.sinco4.astype(str).str.zfill(4)
    s4["sinco4"] = s4.sinco4.astype(str).str.zfill(4)
    d = w.merge(s4, on="sinco4", how="inner")

    d = d[(d.ingocup > 0) & (d.hrsocup > 0) & (d.anios_esc != 99)].copy()
    d["log_wage_h"] = np.log(d.ingocup / (d.hrsocup * 4.33))
    d["mujer"] = (d.sex == 2).astype(int)
    d["informal"] = (d.seg_soc == 2).astype(int)
    d["rural"] = (d.ur == 2).astype(int)
    d["eda2"] = d.eda ** 2
    d["educ13"] = (d.anios_esc >= 13).astype(int)
    return d


def report(m, name, terms):
    print(f"\n{name} (N = {int(m.nobs)}, R2 = {m.rsquared:.3f}):")
    for t in terms:
        star = ("***" if m.pvalues[t] < .01 else
                "**" if m.pvalues[t] < .05 else
                "*" if m.pvalues[t] < .1 else "")
        print(f"  {t:<16} {m.params[t]:+.4f}{star:<4} (se {m.bse[t]:.4f})")
    return [{"spec": name, "term": t, "beta": m.params[t], "se": m.bse[t],
             "p": m.pvalues[t], "r2": m.rsquared, "n": int(m.nobs)}
            for t in terms]


def main() -> None:
    d = load()
    print(f"Wage sample: {len(d):,} reporting workers "
          f"({d.fac_tri.sum():,.0f} weighted); median hourly wage "
          f"{np.exp(np.median(d.log_wage_h)):.0f} MXN")

    base = "log_wage_h ~ anios_esc + eda + eda2 + mujer + informal + rural"
    rows = []
    m1 = smf.wls(base, d, weights=d.fac_tri).fit(cov_type="HC1")
    rows += report(m1, "M1 human capital", ["anios_esc", "mujer", "informal"])

    m2 = smf.wls(base + " + dboe + deoe", d, weights=d.fac_tri).fit(cov_type="HC1")
    rows += report(m2, "M2 + exposure",
                   ["anios_esc", "dboe", "deoe"])
    prem = np.exp(m2.params["dboe"]) - 1
    print(f"  -> 1 SD of LLM exposure pays {prem:+.1%} per hour, conditional "
          "on schooling/age/sex/formality")

    m3 = smf.wls(base + " + dboe * educ13 + deoe", d,
                 weights=d.fac_tri).fit(cov_type="HC1")
    rows += report(m3, "M3 + DBOE x educ13",
                   ["dboe", "dboe:educ13", "deoe"])

    res = pd.DataFrame(rows)
    res.to_sql("wage_premium", ENGINE, if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "wage_premium.csv", index=False)
    print(f"\n-> wage_premium: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
