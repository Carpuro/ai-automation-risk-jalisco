"""
Reinstatement check (Acemoglu-Restrepo side B): do exposed occupations also
GAIN new tasks?

The displacement evidence (H4, Level-2) covers only one side of the
race-between-man-and-machine framing. O*NET's Emerging Tasks file
(onet_emerging_tasks, loaded and previously unused) records tasks added to
occupations because incumbents report performing them -- a direct, if
sparse, measure of task creation.

Honest scope: the file is a small curated set (186 "New" tasks across all
occupations, 2007-2023, mostly pre-LLM), so this measures HISTORICAL
task-renewal capacity, not generative-AI-era reinstatement. It answers: do
the occupations most exposed to each frontier have a track record of gaining
new tasks (adaptation), or are they static (pure displacement candidates)?

Design: per SOC-6 occupation, count of New emerging tasks (zero for most);
logit P(any new task) ~ DBOE + DEOE plus group comparison of mean exposure
for occupations with vs without new tasks.

Output: table `reinstatement_emerging` (+ CSV).
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


def main() -> None:
    c = ENGINE.connect()
    em = pd.read_sql(
        "SELECT onetsoc_code, category FROM onet_emerging_tasks "
        "WHERE category = 'New'", c)
    exp = pd.read_sql(
        "SELECT soc6, occupation_title, dboe_2026_z AS dboe, "
        "embodied_exposure_z AS deoe FROM model_exposure_soc", c)
    c.close()

    em["soc6"] = em.onetsoc_code.str.slice(0, 7)
    counts = em.groupby("soc6").size().rename("new_tasks")
    d = exp.merge(counts, on="soc6", how="left")
    d["new_tasks"] = d.new_tasks.fillna(0).astype(int)
    d["has_new"] = (d.new_tasks > 0).astype(int)
    d = d.dropna(subset=["dboe", "deoe"])
    print(f"Occupations: {len(d)}; with >=1 New emerging task: "
          f"{d.has_new.sum()} ({d.has_new.mean():.1%})\n")

    print("Mean exposure, occupations WITH vs WITHOUT new tasks:")
    cmp = d.groupby("has_new")[["dboe", "deoe"]].mean().round(2)
    print(cmp.rename(index={0: "without", 1: "with new tasks"}).to_string())

    m = smf.logit("has_new ~ dboe + deoe", d).fit(disp=0)
    print("\nLogit P(any new task):")
    rows = []
    for t in ["dboe", "deoe"]:
        orr = np.exp(m.params[t])
        star = ("***" if m.pvalues[t] < .01 else
                "**" if m.pvalues[t] < .05 else
                "*" if m.pvalues[t] < .1 else "")
        print(f"  {t}: beta = {m.params[t]:+.3f}{star} "
              f"(OR = {orr:.2f}, p = {m.pvalues[t]:.3f})")
        rows.append({"term": t, "beta": m.params[t], "odds_ratio": orr,
                     "p": m.pvalues[t], "n": int(m.nobs),
                     "pseudo_r2": m.prsquared})
    res = pd.DataFrame(rows)

    print("\nTop occupations by new-task count:")
    for _, r in d.nlargest(6, "new_tasks").iterrows():
        print(f"  {r.new_tasks:>2}  {str(r.occupation_title)[:52]:<54} "
              f"dboe={r.dboe:+.2f} deoe={r.deoe:+.2f}")

    res.to_sql("reinstatement_emerging", ENGINE,
               if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "reinstatement_emerging.csv", index=False)
    print(f"\n-> reinstatement_emerging: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
