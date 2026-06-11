"""
Workers at risk: the headline numbers of the policy chapter.

Synthesizes everything built so far at the level of individual Jalisco
workers (ENOE Q3 2024, fac_tri weights):

  worker pressure(t, scenario) per axis
      = pct(occupation exposure among workers)   [sinco4_exposure_scores]
      x technology curve, 2024 = 1.00            [c(t) / r(t), Level-2]
      x pct(sector IRA)                          [H4-evidenced moderation]

Threshold anchored in the present: tau = the employment-weighted TOP-TERCILE
bar of total pressure in 2024. As the curves grow to 2030, more workers'
pressure exceeds today's bar -- the count of workers above tau by scenario is
the headline ("workers whose 2030 pressure exceeds today's high bar").

Outputs: table `workers_at_risk` (+ CSV): headline counts by scenario, and the
baseline-2030 profile of the at-risk population (dominant axis, education,
sex, formality) vs the employed population. Figure
figures/workers_at_risk.png.

Run after level2_sector_projection.py and worker_profile prerequisites.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level2_sector_projection import (cognitive_curve, embodied_curve,
                                      ENOE_SCIAN)  # noqa: E402

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
SCENARIOS = ["conservative", "baseline", "accelerated"]


def wpct(s: pd.Series, w: pd.Series) -> pd.Series:
    """Employment-weighted percentile rank (0-1) of s."""
    order = s.sort_values().index
    cum = w.loc[order].cumsum() / w.sum()
    return cum.reindex(s.index)


def wquantile(s: pd.Series, w: pd.Series, q: float) -> float:
    order = s.sort_values().index
    cum = w.loc[order].cumsum() / w.sum()
    return s.loc[order][cum >= q].iloc[0]


def load_workers() -> pd.DataFrame:
    c = ENGINE.connect()
    w = pd.read_sql(
        "SELECT sinco4, scian, sex, anios_esc, seg_soc, fac_tri "
        "FROM enoe_jalisco_workers", c)
    s4 = pd.read_sql(
        "SELECT sinco4, dboe_2026_z AS dboe, embodied_exposure_z AS deoe "
        "FROM sinco4_exposure_scores", c)
    prof = pd.read_sql(
        "SELECT scian, ira_real FROM sector_exposure_profile", c)
    c.close()
    for col in ["scian", "sex", "anios_esc", "seg_soc", "fac_tri"]:
        w[col] = pd.to_numeric(w[col], errors="coerce")
    w["sinco4"] = w.sinco4.astype(str).str.zfill(4)
    s4["sinco4"] = s4.sinco4.astype(str).str.zfill(4)

    d = w.merge(s4, on="sinco4", how="inner").merge(prof, on="scian", how="inner")
    d = d.dropna(subset=["ira_real", "fac_tri"]).reset_index(drop=True)

    # worker-level percentiles (weighted): occupation exposure + sector IRA
    d["pct_dboe"] = wpct(d.dboe, d.fac_tri)
    d["pct_deoe"] = wpct(d.deoe, d.fac_tri)
    d["pct_ira"] = wpct(d.ira_real, d.fac_tri)

    d["educ"] = pd.cut(d.anios_esc.where(d.anios_esc != 99),
                       [-1, 9, 12, 30],
                       labels=["<=9 anos", "10-12", "13+"])
    d["sexo"] = d.sex.map({1: "Hombre", 2: "Mujer"})
    d["formal"] = d.seg_soc.map({1: "Formal", 2: "Informal"})
    return d


def main() -> None:
    d = load_workers()
    total_w = d.fac_tri.sum()
    print(f"Workers with exposure + sector IRA attached: {len(d):,} "
          f"({total_w:,.0f} weighted)\n")

    curves = cognitive_curve().merge(embodied_curve(), on=["year", "scenario"])
    c24 = curves[curves.year == 2024].iloc[0]   # = 1.00 by construction

    # pressure today (2024) and the top-tercile bar tau
    p24_cog = d.pct_dboe * c24.c_t * d.pct_ira
    p24_emb = d.pct_deoe * c24.r_t * d.pct_ira
    p24 = p24_cog + p24_emb
    tau = wquantile(p24, d.fac_tri, 2 / 3)
    base_count = (d.fac_tri * (p24 >= tau)).sum()
    print(f"tau (top-tercile total-pressure bar, 2024) = {tau:.3f}; "
          f"workers above today: {base_count:,.0f} ({base_count/total_w:.0%})")

    rows = []
    for k in SCENARIOS:
        cy = curves[(curves.scenario == k) & (curves.year == 2030)].iloc[0]
        p_cog = d.pct_dboe * cy.c_t * d.pct_ira
        p_emb = d.pct_deoe * cy.r_t * d.pct_ira
        p = p_cog + p_emb
        above = p >= tau
        n_above = (d.fac_tri * above).sum()
        cog_dom = above & (p_cog > p_emb)
        rows.append({
            "scenario": k, "year": 2030,
            "workers_above_bar": n_above,
            "share_of_employed": n_above / total_w,
            "new_vs_2024": n_above - base_count,
            "cognitive_dominant": (d.fac_tri * cog_dom).sum(),
            "embodied_dominant": (d.fac_tri * (above & ~cog_dom)).sum(),
        })
    res = pd.DataFrame(rows)
    print("\nWorkers whose 2030 pressure exceeds today's high bar:")
    for _, r in res.iterrows():
        print(f"  {r.scenario:<14} {r.workers_above_bar:>10,.0f} "
              f"({r.share_of_employed:.0%} of employed; "
              f"+{r.new_vs_2024:,.0f} vs today)  "
              f"[cog-dom {r.cognitive_dominant:,.0f} / "
              f"emb-dom {r.embodied_dominant:,.0f}]")

    # --- profile of the at-risk population, baseline 2030 -------------------
    cy = curves[(curves.scenario == "baseline") & (curves.year == 2030)].iloc[0]
    p_cog = d.pct_dboe * cy.c_t * d.pct_ira
    p_emb = d.pct_deoe * cy.r_t * d.pct_ira
    d["at_risk"] = (p_cog + p_emb) >= tau
    d["pole"] = np.where(p_cog > p_emb, "cognitive", "embodied")

    prof_rows = []
    print("\nProfile of at-risk workers (baseline 2030) vs all employed:")
    for dim in ["educ", "sexo", "formal", "pole"]:
        for level, g in d.dropna(subset=[dim]).groupby(dim, observed=True):
            share_risk = (g.fac_tri * g.at_risk).sum() / (d.fac_tri * d.at_risk).sum()
            share_all = g.fac_tri.sum() / total_w
            prof_rows.append({"dimension": dim, "level": str(level),
                              "share_at_risk": share_risk,
                              "share_employed": share_all})
            print(f"  {dim:<8} {str(level):<12} at-risk {share_risk:5.0%}  "
                  f"vs employed {share_all:5.0%}")
    prof = pd.DataFrame(prof_rows)

    # --- figure ---------------------------------------------------------------
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(res))
    ax.bar(x - 0.18, res.cognitive_dominant / 1e3, width=0.36,
           color="#b3452c", label="Cognitive-dominant (LLM)")
    ax.bar(x + 0.18, res.embodied_dominant / 1e3, width=0.36,
           color="#3b6ea5", label="Embodied-dominant (robots)")
    ax.axhline(base_count / 1e3, color="grey", ls="--", lw=1,
               label=f"Above bar today ({base_count/1e3:,.0f}k)")
    ax.set_xticks(x, res.scenario)
    ax.set_ylabel("Workers above today's high-pressure bar, 2030 (thousands)")
    ax.set_title("Jalisco: workers at high AI-automation pressure by 2030\n"
                 "(exposure x technology curve x IRA; bar = top tercile 2024)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = FIGURES_DIR / "workers_at_risk.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n-> {out_png}")

    res.to_sql("workers_at_risk", ENGINE, if_exists="replace", index=False)
    prof.to_sql("workers_at_risk_profile", ENGINE, if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "workers_at_risk.csv", index=False)
    print(f"-> workers_at_risk: {len(res)} rows; profile: {len(prof)} rows (+ CSV)")


if __name__ == "__main__":
    main()
