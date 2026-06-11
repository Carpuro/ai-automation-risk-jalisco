"""
Absorption test: does informality mask displacement that IMSS cannot see?

The IMSS-based ChatGPT test found no displacement in FORMAL employment. The
classic Latin American objection: displaced formal workers do not become
unemployed, they reappear as informal workers -- so a formal-only null could
mask real displacement. This tests that directly on the full-employment ENOE
panel (enoe_jalisco_quarterly: 2022T1-2024T3, sector x quarter x formality,
informal included).

Two diff-in-trends regressions (sector + quarter FE, HC1; post = 2023T1+,
first full quarter after ChatGPT):

  A. log TOTAL employment ~ post x DBOE + post x DEOE
     -- the IMSS test repeated without the formal-only blindspot.
  B. INFORMAL SHARE of sector employment ~ post x DBOE + post x DEOE
     -- the absorption signature: if exposed sectors push workers into
     informality, their informal share should RISE post-event.

Output: table `absorption_test` (+ CSV), figure
figures/absorption_informality.png.
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


def load() -> pd.DataFrame:
    c = ENGINE.connect()
    p = pd.read_sql("SELECT * FROM enoe_jalisco_quarterly", c)
    exp = pd.read_sql("SELECT scian, dboe, deoe FROM sector_exposure_profile", c)
    c.close()

    wide = (p.pivot_table(index=["year", "quarter", "scian"],
                          columns="formality", values="workers", aggfunc="sum")
              .reset_index().fillna(0))
    wide["total"] = wide.formal + wide.informal
    wide["informal_share"] = wide.informal / wide.total
    wide = wide.merge(exp, on="scian", how="inner")
    wide = wide[wide.total > 5000]          # drop tiny unstable sector cells
    wide["t"] = wide.year.astype(str) + "T" + wide.quarter.astype(str)
    wide["post"] = ((wide.year > 2023) | ((wide.year == 2023))).astype(int)
    wide.loc[wide.year == 2022, "post"] = 0
    for col in ["dboe", "deoe"]:
        m_ = wide.groupby("scian")[col].first()
        wide[col + "_z"] = (wide[col] - m_.mean()) / m_.std(ddof=0)
    wide["log_total"] = np.log(wide.total)
    return wide


def did(d, formula, label, terms):
    m = smf.ols(formula, d).fit(cov_type="HC1")
    rows = []
    print(f"\n{label} (N = {int(m.nobs)}, R2 = {m.rsquared:.3f}):")
    for t in terms:
        star = ("***" if m.pvalues[t] < .01 else
                "**" if m.pvalues[t] < .05 else
                "*" if m.pvalues[t] < .1 else "")
        print(f"  {t:<16} {m.params[t]:+.4f}{star:<4} (se {m.bse[t]:.4f})")
        rows.append({"test": label, "term": t, "beta": m.params[t],
                     "se": m.bse[t], "p": m.pvalues[t], "n": int(m.nobs)})
    return rows


def main() -> None:
    d = load()
    print(f"ENOE full-employment panel: {d.scian.nunique()} sectors x "
          f"{d.t.nunique()} quarters (informal included)")
    inf0 = d[d.year == 2022].informal.sum() / d[d.year == 2022].total.sum()
    inf1 = d[d.year == 2024].informal.sum() / d[d.year == 2024].total.sum()
    print(f"Jalisco informality: {inf0:.1%} (2022) -> {inf1:.1%} (2024)")

    rows = did(d, "log_total ~ C(scian) + C(t) + post:dboe_z + post:deoe_z",
               "A. log TOTAL employment", ["post:dboe_z", "post:deoe_z"])
    rows += did(d, "informal_share ~ C(scian) + C(t) + post:dboe_z + post:deoe_z",
                "B. informal share (absorption)", ["post:dboe_z", "post:deoe_z"])
    res = pd.DataFrame(rows)

    # --- figure ---------------------------------------------------------------
    FIGURES_DIR.mkdir(exist_ok=True)
    d["grp"] = np.where(d.dboe_z > 0, "High cognitive exposure",
                        "Low cognitive exposure")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = sorted(d.t.unique())
    for grp, color in [("High cognitive exposure", "#b3452c"),
                       ("Low cognitive exposure", "#3b6ea5")]:
        g = d[d.grp == grp].groupby("t")[["total", "informal"]].sum().loc[order]
        base = g.total.iloc[3]  # 2022T4
        axes[0].plot(order, g.total / base * 100, label=grp, color=color)
        axes[1].plot(order, g.informal / g.total * 100, label=grp, color=color)
    for ax, (ttl, yl) in zip(axes, [
            ("Total employment (formal + informal)", "Index (2022T4 = 100)"),
            ("Informal share of sector employment", "% informal")]):
        ax.axvline("2022T4", color="grey", ls="--", lw=1)
        ax.set_title(ttl)
        ax.set_ylabel(yl)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].annotate("ChatGPT", ("2022T4", axes[0].get_ylim()[0]),
                     textcoords="offset points", xytext=(5, 10),
                     fontsize=9, color="grey")
    fig.suptitle("Jalisco: full-employment view around the generative-AI shock "
                 "(ENOE, informal included)")
    fig.tight_layout()
    out_png = FIGURES_DIR / "absorption_informality.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n-> {out_png}")

    res.to_sql("absorption_test", ENGINE, if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "absorption_test.csv", index=False)
    print(f"-> absorption_test: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
