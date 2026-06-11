"""
Worker-profile chapter: WHO holds the AI-exposed jobs in Jalisco?

Joins ENOE Jalisco workers (Q3 2024, fac_tri weights) to occupation-level
exposure (sinco4_exposure_scores, official INEGI crosswalk; ~88% coverage)
and profiles BOTH axes across worker characteristics:

  education (anios_esc bands), income (ingocup quintiles among reporters),
  formality (seg_soc access to social security), sex, age group, urban/rural.

Policy question this answers: the cognitive (LLM) threat concentrates on
educated / higher-income / formal workers, while the embodied (robot) threat
concentrates on the opposite profile -- so WHICH workers does each frontier
put at risk, and how many are they?

Codes (ENOE SDEMT): anios_esc 99 = unspecified (dropped); ingocup 0 = income
not reported (38.5%, kept in group profiles, excluded from quintiles);
seg_soc 1 = with access, 2 = without (other codes dropped); ur 1 urban /
2 rural; eda 12-98.

Outputs: table `worker_exposure_profile` (+ CSV) and
figures/worker_exposure_profile.png. Plus weighted OLS of each exposure axis
on worker characteristics as a compact summary.
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


def wmean(s, w):
    return (s * w).sum() / w.sum()


def load() -> pd.DataFrame:
    c = ENGINE.connect()
    w = pd.read_sql(
        "SELECT sinco4, sex, eda, anios_esc, seg_soc, ingocup, ur, fac_tri "
        "FROM enoe_jalisco_workers", c)
    s4 = pd.read_sql(
        "SELECT sinco4, dboe_2026_z AS dboe, embodied_exposure_z AS deoe "
        "FROM sinco4_exposure_scores", c)
    c.close()
    w["sinco4"] = w.sinco4.astype(str).str.zfill(4)
    s4["sinco4"] = s4.sinco4.astype(str).str.zfill(4)
    for col in ["sex", "eda", "anios_esc", "seg_soc", "ingocup", "ur", "fac_tri"]:
        w[col] = pd.to_numeric(w[col], errors="coerce")
    d = w.merge(s4, on="sinco4", how="inner")

    d = d[d.anios_esc != 99]
    d["educ"] = pd.cut(d.anios_esc, [-1, 6, 9, 12, 30],
                       labels=["0-6 (primaria)", "7-9 (secundaria)",
                               "10-12 (media sup.)", "13+ (superior)"])
    d["age_group"] = pd.cut(d.eda, [14, 29, 44, 59, 98],
                            labels=["15-29", "30-44", "45-59", "60+"])
    d["sexo"] = d.sex.map({1: "Hombre", 2: "Mujer"})
    d["formal"] = d.seg_soc.map({1: "Formal (seg. social)", 2: "Informal"})
    d["zona"] = d.ur.map({1: "Urbano", 2: "Rural"})
    rep = d.ingocup > 0
    d.loc[rep, "ing_q"] = pd.qcut(d[rep].ingocup, 5,
                                  labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return d


def main() -> None:
    d = load()
    print(f"Workers with exposure attached: {len(d):,} "
          f"({d.fac_tri.sum():,.0f} weighted)\n")

    dims = [("educacion", "educ"), ("ingreso_quintil", "ing_q"),
            ("formalidad", "formal"), ("sexo", "sexo"),
            ("edad", "age_group"), ("zona", "zona")]
    rows = []
    for dim, col in dims:
        for level, g in d.dropna(subset=[col]).groupby(col, observed=True):
            rows.append({
                "dimension": dim, "level": str(level),
                "workers": g.fac_tri.sum(),
                "dboe": wmean(g.dboe, g.fac_tri),
                "deoe": wmean(g.deoe, g.fac_tri),
            })
    prof = pd.DataFrame(rows)

    print(f"{'dimension':<17}{'level':<22}{'workers':>10}{'DBOE':>8}{'DEOE':>8}")
    for _, r in prof.iterrows():
        print(f"{r.dimension:<17}{r.level:<22}{r.workers:>10,.0f}"
              f"{r.dboe:>+8.2f}{r.deoe:>+8.2f}")

    # --- compact weighted OLS summary (both axes, same covariates) ----------
    reg = d.dropna(subset=["educ", "formal"]).copy()
    reg["educ_yrs"] = reg.anios_esc
    reg["mujer"] = (reg.sex == 2).astype(int)
    reg["informal"] = (reg.formal == "Informal").astype(int)
    reg["rural"] = (reg.zona == "Rural").astype(int)
    print("\nWeighted OLS (standardized exposure on worker traits):")
    for axis in ["dboe", "deoe"]:
        m = smf.wls(f"{axis} ~ educ_yrs + mujer + informal + rural + eda",
                    reg, weights=reg.fac_tri).fit(cov_type="HC1")
        terms = {t: f"{m.params[t]:+.3f}{'***' if m.pvalues[t] < .01 else '*' if m.pvalues[t] < .05 else ''}"
                 for t in ["educ_yrs", "mujer", "informal", "rural", "eda"]}
        print(f"  {axis.upper()}: " + "  ".join(f"{k}={v}" for k, v in terms.items())
              + f"  R2={m.rsquared:.2f}")

    # --- figure: dot plot per dimension --------------------------------------
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_dims = [("educacion", "Educación (años)"),
                 ("ingreso_quintil", "Ingreso (quintil)"),
                 ("formalidad", "Formalidad"), ("sexo", "Sexo")]
    fig, axes = plt.subplots(1, len(plot_dims), figsize=(16, 4.5), sharey=False)
    for ax, (dim, title) in zip(axes, plot_dims):
        sub = prof[prof.dimension == dim]
        y = np.arange(len(sub))
        ax.scatter(sub.dboe, y, color="#b3452c", label="DBOE (LLM)", zorder=3)
        ax.scatter(sub.deoe, y, color="#3b6ea5", label="DEOE (robots)", zorder=3)
        for yi, (db, de) in enumerate(zip(sub.dboe, sub.deoe)):
            ax.plot([db, de], [yi, yi], color="grey", lw=1, zorder=2)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y, sub.level)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Jalisco: exposición media ponderada por perfil del trabajador "
                 "(z, ENOE Q3 2024)", fontsize=13)
    fig.tight_layout()
    out_png = FIGURES_DIR / "worker_exposure_profile.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n-> {out_png}")

    prof.to_sql("worker_exposure_profile", ENGINE,
                if_exists="replace", index=False)
    prof.to_csv(PROCESSED_DIR / "worker_exposure_profile.csv", index=False)
    print(f"-> worker_exposure_profile: {len(prof)} rows (+ CSV)")


if __name__ == "__main__":
    main()
