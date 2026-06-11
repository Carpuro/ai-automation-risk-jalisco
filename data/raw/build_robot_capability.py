"""
DEOE dynamic layer: robotics adoption curve r(t), the embodied mirror of the
DBOE's LLM capability curve c_j(t).

Design
------
The DBOE separates WHO is exposed (occupation weights, cross-section) from HOW
CAPABLE the technology is over time (c_j(t) from benchmarks). The DEOE mirror
keeps the validated static cross-section (embodied_exposure_z) and adds a
time curve of robotization:

  r(t) = world operational stock of industrial robots, indexed to the base
         year (2022 = 1.0, matching the DBOE curve's first year)

Rationale: no public benchmark series measures robot *capability* the way
Epoch measures LLM capability, but the IFR operational stock is the standard
adoption measure in the robots-and-jobs literature (Acemoglu & Restrepo 2020,
Graetz & Michaels 2018) and is smooth, public and citable. Cost-decline
evidence (Graetz & Michaels: quality-adjusted robot prices fell ~80%
1990-2005) is cited in the thesis text as context, not fabricated into data.

Projection 2025-2030: three scenarios for world-stock growth, bracketing the
observed 2019-2024 CAGR. Mexico's own installations series (IFR, archived
PDFs -- see robotics/README.md) anchors the local adoption level and is
carried as a Mexico-specific index (cumulative installations).

Output: table `robot_capability_curve` + data/processed/robot_capability_curve.csv
(one row per year 2012-2030: world stock, r(t) index, scenario variants,
Mexico installations and cumulative index).

Run anytime; depends only on files in data/raw/robotics/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

RAW_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = RAW_DIR.parent / "processed"

BASE_YEAR = 2022          # matches the DBOE c_j(t) base year
PROJECTION_YEARS = range(2025, 2031)

# Mexico annual installations, primary IFR sources (robotics/README.md)
MX_INSTALLATIONS = {
    2019: 4600, 2020: 3363, 2021: 5401,
    2022: 6000, 2023: 5832, 2024: 5594,
}

# Scenario growth rates for world operational stock 2025-2030, derived from
# the data: baseline = observed CAGR of the last 3 years (post-COVID
# normalization, avoids the 2021 rebound spike); conservative = half that
# rate; accelerated = 1.5x (AI-driven robotics investment wave).
SCENARIOS = {"conservative": None, "baseline": None, "accelerated": None}


def main() -> None:
    world = (pd.read_csv(RAW_DIR / "robotics" / "owid_robots_global_stock.csv")
               .query("entity == 'World'")
               .loc[:, ["year", "industrial_robot_stock"]]
               .rename(columns={"industrial_robot_stock": "world_stock"})
               .dropna().astype({"world_stock": int}))

    # Observed world-stock CAGR over the last 3 years -> baseline scenario
    last = world.year.max()
    cagr = (world.set_index("year").world_stock[last]
            / world.set_index("year").world_stock[last - 3]) ** (1 / 3) - 1
    SCENARIOS["baseline"] = round(cagr, 4)
    SCENARIOS["conservative"] = round(cagr / 2, 4)
    SCENARIOS["accelerated"] = round(cagr * 1.5, 4)
    print(f"World stock {last - 3}->{last} CAGR = {cagr:.1%} (baseline scenario)")
    print(f"Scenarios 2025-2030: {SCENARIOS}")

    # Project world stock per scenario
    rows = [{"year": int(y), "world_stock": int(s), "scenario": "observed"}
            for y, s in world.values]
    for name, g in SCENARIOS.items():
        stock = world.set_index("year").world_stock[last]
        for y in PROJECTION_YEARS:
            stock = stock * (1 + g)
            rows.append({"year": y, "world_stock": int(stock), "scenario": name})
    curve = pd.DataFrame(rows)

    # r(t): index world stock to the base year (mirror of c_j(t) anchoring)
    base = world.set_index("year").world_stock[BASE_YEAR]
    curve["r_t"] = curve.world_stock / base

    # Mexico layer: installations + cumulative index (2019 = 1.0)
    mx = pd.Series(MX_INSTALLATIONS, name="mx_installations")
    curve = curve.merge(mx.rename_axis("year").reset_index(), on="year", how="left")
    mx_cum = mx.cumsum()
    curve = curve.merge(
        (mx_cum / mx_cum.iloc[0]).rename("mx_cum_index").rename_axis("year").reset_index(),
        on="year", how="left")

    print("\nr(t) world-stock index (base 2022 = 1.00):")
    obs = curve[curve.scenario == "observed"]
    for _, r in obs[obs.year >= 2019].iterrows():
        mxs = f"  MX inst = {int(r.mx_installations):,}" if pd.notna(r.mx_installations) else ""
        print(f"  {int(r.year)}  r = {r.r_t:.2f}{mxs}")
    print("\nProjected r(t) 2030 by scenario:")
    for name in SCENARIOS:
        v = curve[(curve.scenario == name) & (curve.year == 2030)].r_t.iloc[0]
        print(f"  {name:<14} r(2030) = {v:.2f}")

    curve.to_sql("robot_capability_curve", ENGINE,
                 if_exists="replace", index=False, chunksize=500)
    curve.to_csv(PROCESSED_DIR / "robot_capability_curve.csv", index=False)
    print(f"\n-> robot_capability_curve: {len(curve)} rows (+ CSV in processed/)")


if __name__ == "__main__":
    main()
