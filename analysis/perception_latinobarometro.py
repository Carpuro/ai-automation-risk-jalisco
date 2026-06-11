"""
Perception chapter: do Mexican workers expect robots/AI to take their jobs?
Latinobarometro Mexico subsets, 4 waves (2017, 2018, 2020, 2023).

Harmonization (scales verified against the Stata value labels of each wave;
see data/raw/load_new_tables.py for the item provenance):

  2017 P56N_A    SOCIETAL frame ("AI and robotics will make most jobs
                 disappear"), 4-pt agree: 1 very agree .. 4 very disagree.
                 expects_displacement = codes 1-2.
  2018 P61N      PERSONAL frame with a TIMELINE scale ("robots are going to
                 take your job?"): 1 = no, 2 = yes within 1 year, 3 = yes in
                 5 years, 4 = yes in 10+ years. expects = codes 2-4.
  2020 p29n_e    PERSONAL, 10-yr horizon, 4-pt agree. expects = codes 1-2.
  2023 P30STIN_A PERSONAL, 10-yr horizon, 4-pt agree. expects = codes 1-2.

Comparability rules applied:
  * 2017 is reported as a separate SOCIETAL series, never chained with the
    personal series.
  * 2018's "yes at any horizon" is a more lenient threshold than agreeing on
    a 10-year horizon; flagged on every output.
  * Negative codes (DK/NA/missing) are excluded from denominators.
  * All shares use the survey weight (wt).

Outputs: table `perception_trend` (+ CSV) and a breakdown table
`perception_breakdown_2023` (sex, age group, internet at home), and
figures/perception_trend.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

WAVE_DESIGN = {
    2017: {"frame": "societal", "expects": {1, 2}, "scale": "agree4"},
    2018: {"frame": "personal", "expects": {2, 3, 4}, "scale": "timeline"},
    2020: {"frame": "personal", "expects": {1, 2}, "scale": "agree4"},
    2023: {"frame": "personal", "expects": {1, 2}, "scale": "agree4"},
}


def wshare(flag: pd.Series, wt: pd.Series) -> float:
    return (flag * wt).sum() / wt.sum()


def main() -> None:
    c = ENGINE.connect()
    d = pd.read_sql("SELECT * FROM latinobarometro_mx", c)
    c.close()

    # valid responses only (positive codes), harmonized expects flag
    d = d[d.robot_jobs_perception > 0].copy()
    d["expects"] = d.apply(
        lambda r: int(r.robot_jobs_perception in WAVE_DESIGN[r.year]["expects"]),
        axis=1)
    d["frame"] = d.year.map(lambda y: WAVE_DESIGN[y]["frame"])
    d["scale"] = d.year.map(lambda y: WAVE_DESIGN[y]["scale"])

    # --- trend table ---------------------------------------------------------
    trend = d.groupby(["year", "frame", "scale"]).apply(
        lambda g: pd.Series({
            "n_valid": len(g),
            "pct_expects": wshare(g.expects, g.weight),
        })).reset_index()
    print("Expects robot/AI job displacement (weighted, valid responses):")
    for _, r in trend.iterrows():
        note = " [lenient: yes at ANY horizon]" if r.scale == "timeline" else ""
        print(f"  {int(r.year)}  {r.frame:<9} {r.pct_expects:6.1%}  "
              f"(n={int(r.n_valid)}){note}")

    # --- 2023 breakdowns -----------------------------------------------------
    w23 = d[d.year == 2023].copy()
    w23["age_group"] = pd.cut(w23.age, [15, 29, 44, 59, 120],
                              labels=["16-29", "30-44", "45-59", "60+"])
    w23["sex_label"] = w23.sex.map({1: "Hombre", 2: "Mujer"})
    w23["internet"] = np.select(
        [w23.internet_home == 1, w23.internet_home.isin([2, 0])],
        ["Con internet", "Sin internet"], default=None)

    rows = []
    for dim, col in [("sexo", "sex_label"), ("edad", "age_group"),
                     ("internet_hogar", "internet")]:
        for level, g in w23.dropna(subset=[col]).groupby(col, observed=True):
            rows.append({"dimension": dim, "level": str(level),
                         "n_valid": len(g),
                         "pct_expects": wshare(g.expects, g.weight)})
    bd = pd.DataFrame(rows)
    print("\n2023 personal displacement expectation by group:")
    for _, r in bd.iterrows():
        print(f"  {r.dimension:<15} {r.level:<14} {r.pct_expects:6.1%} "
              f"(n={int(r.n_valid)})")

    # --- figure --------------------------------------------------------------
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    pers = trend[trend.frame == "personal"]
    soc = trend[trend.frame == "societal"]
    ax.plot(pers.year, pers.pct_expects * 100, "o-", color="#b3452c",
            label="Personal: \"robots will take MY job\"")
    ax.scatter(soc.year, soc.pct_expects * 100, marker="s", s=70,
               color="#3b6ea5", zorder=3,
               label="Societal: \"AI will make most jobs disappear\" (2017)")
    for _, r in trend.iterrows():
        ax.annotate(f"{r.pct_expects:.0%}", (r.year, r.pct_expects * 100),
                    textcoords="offset points", xytext=(0, 9), ha="center")
    ax.annotate("2018: lenient threshold\n(yes at any horizon)",
                (2018, float(pers[pers.year == 2018].pct_expects) * 100),
                textcoords="offset points", xytext=(12, -28), fontsize=8,
                color="grey")
    ax.set_ylabel("% expects displacement (weighted)")
    ax.set_ylim(0, 100)
    ax.set_xticks(trend.year.unique())
    ax.set_title("Mexico: perceived robot/AI job displacement "
                 "(Latinobarometro)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = FIGURES_DIR / "perception_trend.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n-> {out_png}")

    trend.to_sql("perception_trend", ENGINE, if_exists="replace", index=False)
    bd.to_sql("perception_breakdown_2023", ENGINE,
              if_exists="replace", index=False)
    trend.to_csv(PROCESSED_DIR / "perception_trend.csv", index=False)
    print(f"-> perception_trend: {len(trend)} rows; "
          f"perception_breakdown_2023: {len(bd)} rows (+ CSV)")


if __name__ == "__main__":
    main()
