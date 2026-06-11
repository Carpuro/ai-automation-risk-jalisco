"""
H4 adoption test: does the market actually pursue automation where it pays?

Until now H4 (the IRA economic incentive moderates whether technical exposure
becomes real adoption pressure) was ASSUMED by the Level-2 pressure formula,
never tested. This script tests it against REALIZED substitution behavior in
the Economic Censuses panel: Jalisco, 19 SCIAN sectors x 5 censuses
(2003-2023, fully balanced, external_ira_by_sector_full), giving 4
census-to-census transitions per sector (N = 76).

Outcomes (census-to-census changes within sector):
  dlog_kl         capital deepening, dlog(acervo_activos / personal_ocupado)
  dlog_compl      computing-capital deepening, dlog(acervo_computo / personal)
  d_labor_share   change in (remuneraciones + contrib + prestaciones) / VA
  dlog_emp        employment growth, dlog(personal_ocupado)

Predictors:
  log_ira_lag     log IRA (base variant) at the START of each transition --
                  the incentive observed before the substitution decision
  dboe / deoe     sector exposure (time-invariant, from sector_exposure_profile)

Axis-specific predictions if the market seeks automation where profitable:
  P1  dlog_kl rises with log_ira_lag (capital replaces expensive labor)
  P2  dlog_compl rises with log_ira_lag and with DBOE (computing capital
      flows to cognitively exposed sectors)
  P3  dlog_emp falls with log_ira_lag x exposure (incentive + feasibility
      slow employment)

Estimation: OLS with period fixed effects, cluster-robust SEs by sector
(19 clusters; small panel -- report effect sizes, not just stars). Values are
nominal pesos; period FE absorb economy-wide inflation, so identification is
cross-sector within period.

Output: tidy results table `h4_adoption_test` (+ CSV) and
figures/h4_adoption_test.png. Also prints an employment-CAGR quadrant
descriptive (exposure x IRA) as the readable summary.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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
FIGURES_DIR = ROOT / "figures"


def load_panel() -> pd.DataFrame:
    c = ENGINE.connect()
    ce = pd.read_sql(
        "SELECT anio, sector_code, personal_ocupado, remuneraciones, "
        "contrib_seg_social, otras_prestaciones, acervo_activos, "
        "acervo_computo, valor_agregado, ira_base "
        "FROM external_ira_by_sector_full", c)
    exp = pd.read_sql(
        "SELECT sector_code, sector_label, dboe, deoe "
        "FROM sector_exposure_profile", c)
    c.close()

    ce = ce.sort_values(["sector_code", "anio"])
    g = ce.groupby("sector_code")
    ce["kl"] = ce.acervo_activos / ce.personal_ocupado
    ce["compl"] = ce.acervo_computo / ce.personal_ocupado
    ce["labor_share"] = ((ce.remuneraciones + ce.contrib_seg_social
                          + ce.otras_prestaciones) / ce.valor_agregado)

    d = pd.DataFrame({
        "sector_code": ce.sector_code,
        "period_end": ce.anio,
        "dlog_kl": g.kl.transform(lambda s: np.log(s).diff()),
        "dlog_compl": g.compl.transform(lambda s: np.log(s).diff()),
        "d_labor_share": g.labor_share.transform("diff"),
        "dlog_emp": g.personal_ocupado.transform(lambda s: np.log(s).diff()),
        "log_ira_lag": g.ira_base.transform(
            lambda s: np.log(s.clip(lower=1e-3)).shift()),
    }).dropna()
    d = d.merge(exp, on="sector_code", how="inner")
    d["period"] = d.period_end.astype(int).astype(str)
    return d


def fit(formula: str, d: pd.DataFrame):
    return smf.ols(formula, d).fit(
        cov_type="cluster", cov_kwds={"groups": d.sector_code})


def main() -> None:
    d = load_panel()
    print(f"H4 adoption test -- CE panel: {d.sector_code.nunique()} sectors x "
          f"{d.period.nunique()} transitions = {len(d)} obs\n")

    specs = {
        "P1 dlog_kl ~ IRA":
            ("dlog_kl ~ log_ira_lag + C(period)", "log_ira_lag"),
        "P2 dlog_compl ~ IRA":
            ("dlog_compl ~ log_ira_lag + C(period)", "log_ira_lag"),
        "P2b dlog_compl ~ IRA + DBOE":
            ("dlog_compl ~ log_ira_lag + dboe + C(period)", "dboe"),
        "P3a dlog_emp ~ IRA x DBOE":
            ("dlog_emp ~ log_ira_lag * dboe + C(period)", "log_ira_lag:dboe"),
        "P3b dlog_emp ~ IRA x DEOE":
            ("dlog_emp ~ log_ira_lag * deoe + C(period)", "log_ira_lag:deoe"),
        "labor share ~ IRA":
            ("d_labor_share ~ log_ira_lag + C(period)", "log_ira_lag"),
    }

    rows = []
    print(f"{'spec':<30}{'term':>16}{'beta':>9}{'se':>8}{'p':>9}{'R2':>7}")
    for name, (formula, key) in specs.items():
        m = fit(formula, d)
        for term in m.params.index:
            if term.startswith("C(period)") or term == "Intercept":
                continue
            rows.append({"spec": name, "term": term,
                         "beta": m.params[term], "se": m.bse[term],
                         "p": m.pvalues[term], "r2": m.rsquared,
                         "n": int(m.nobs)})
        k = m.params.index[m.params.index.str.replace(" ", "") == key]
        kk = k[0] if len(k) else key
        print(f"{name:<30}{kk:>16}{m.params[kk]:>+9.3f}{m.bse[kk]:>8.3f}"
              f"{m.pvalues[kk]:>9.3f}{m.rsquared:>7.2f}")
    res = pd.DataFrame(rows)

    # --- readable descriptive: employment CAGR 2003-2023 by quadrant --------
    c = ENGINE.connect()
    lvl = pd.read_sql(
        "SELECT anio, sector_code, personal_ocupado, ira_base "
        "FROM external_ira_by_sector_full WHERE anio IN (2003, 2023)", c)
    c.close()
    piv = lvl.pivot(index="sector_code", columns="anio",
                    values="personal_ocupado")
    q = pd.DataFrame({
        "emp_cagr": (piv[2023] / piv[2003]) ** (1 / 20) - 1,
        "ira_2003": lvl[lvl.anio == 2003].set_index("sector_code").ira_base,
    }).join(d.groupby("sector_code")[["dboe", "deoe"]].first())
    q = q.dropna()
    q["exposure_total"] = q[["dboe", "deoe"]].max(axis=1)  # nearest threat axis
    hi_ira = q.ira_2003 > q.ira_2003.median()
    hi_exp = q.exposure_total > q.exposure_total.median()
    print("\nEmployment CAGR 2003-2023 by quadrant (medians split):")
    for (ei, ii), label in [((True, True), "exposed & profitable-to-automate"),
                            ((True, False), "exposed, low incentive"),
                            ((False, True), "low exposure, high incentive"),
                            ((False, False), "low exposure, low incentive")]:
        sel = q[(hi_exp == ei) & (hi_ira == ii)]
        print(f"  {label:<36} {sel.emp_cagr.mean():+6.1%}  (n={len(sel)})")

    # --- figure --------------------------------------------------------------
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(d.log_ira_lag, d.dlog_kl, alpha=0.6, color="#3b6ea5")
    b = np.polyfit(d.log_ira_lag, d.dlog_kl, 1)
    xs = np.linspace(d.log_ira_lag.min(), d.log_ira_lag.max(), 10)
    ax.plot(xs, np.polyval(b, xs), color="#b3452c")
    ax.set_xlabel("log IRA at period start (lagged incentive)")
    ax.set_ylabel("dlog(capital / worker) over the census period")
    ax.set_title("P1: capital deepening vs lagged incentive")
    ax = axes[1]
    sc = q.reset_index()
    ax.scatter(sc.dboe, sc.emp_cagr * 100, s=40 + 200 * (sc.ira_2003 /
               sc.ira_2003.max()), alpha=0.6, color="#b3452c")
    ax.set_xlabel("Cognitive exposure (DBOE) -- bubble size = IRA 2003")
    ax.set_ylabel("Employment CAGR 2003-2023 (%)")
    ax.set_title("Employment growth vs exposure and incentive")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("H4: realized substitution behavior, CE Jalisco panel 2003-2023")
    fig.tight_layout()
    out_png = FIGURES_DIR / "h4_adoption_test.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n-> {out_png}")

    res.to_sql("h4_adoption_test", ENGINE, if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "h4_adoption_test.csv", index=False)
    print(f"-> h4_adoption_test: {len(res)} coefficient rows (+ CSV)")


if __name__ == "__main__":
    main()
