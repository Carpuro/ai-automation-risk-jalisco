"""
ChatGPT natural experiment: has the generative-AI shock registered in Jalisco
formal employment?

Event: ChatGPT release (2022-11), the discontinuity in public LLM capability
and adoption. If LLM substitution were already operating, employment growth in
COGNITIVELY exposed sectors should slow after the event relative to its own
pre-trend and relative to robot-exposed sectors.

Data: imss_empleo_sector (monthly formal employment, 9 IMSS sectors,
2000-2024); sector exposure aggregated from sector_exposure_profile to the
IMSS grain (employment-weighted DBOE/DEOE).

Design: difference-in-trends panel
    log(emp_{s,t}) = sector FE + month FE + post_t x dboe_s + post_t x deoe_s
with post = month >= 2022-11. Main window 2021-01..2024-10 (post-COVID
recovery, balanced ~22 pre / 24 post months); robustness window 2017-01+.
Month FE absorb common shocks; identification is cross-sector. Only 9
clusters -- HC1 SEs reported with that caveat; read effect sizes, not stars.

INTERPRETATION -- "not yet" is not "safe" (the policy window):
  * This test measures one narrow thing: the STOCK of FORMAL jobs at a
    coarse 8-sector grain over ~24 months. Early displacement appears first
    in hiring slowdowns, entry-level postings and within-job task
    composition (US evidence: effects concentrate in entry-level workers
    WITHIN exposed occupations, invisible in sector aggregates), and a
    sector can grow while occupations inside it hollow out.
  * The informality objection is tested and rejected separately
    (absorption_informality.py): including the informal ~55%, total
    employment in exposed sectors still did not fall, and their informal
    share did not rise -- the null is not a formal-only artifact.
  * Nearshoring confounder (nearshoring_channel.py): the post-2022 window
    coincides with an FDI wave (+27% vs 2015-2019) that pushes formal
    employment UP -- the positive coefficient partly reflects investment-
    driven labor demand, not absence of automation pressure.
  * Mexico's adoption is early (AEI: usage at 0.44x the expected level), and
    H4 shows the realized form of automation here is MISSING JOB GROWTH over
    decades (+0.6%/yr vs +3.7%/yr), which a 24-month event study cannot see.
  Read with the perception result (+12.5pp expectation jump): the capability
  shock arrived, the labor-market shock has not -- that lag is the policy
  window, not evidence of safety.

Outputs: table `chatgpt_event_imss` (+ CSV), figure
figures/chatgpt_event_imss.png (employment indexed to 2022-10, high- vs
low-DBOE sector groups).
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
EVENT = pd.Timestamp("2022-11-01")


def load() -> pd.DataFrame:
    c = ENGINE.connect()
    imss = pd.read_sql(
        "SELECT sector, fecha, trabajadores FROM imss_empleo_sector "
        "WHERE sector <> 'Total Jalisco'", c)
    prof = pd.read_sql(
        "SELECT imss_sector, workers, dboe, deoe FROM sector_exposure_profile", c)
    c.close()

    # exposure at the IMSS grain: employment-weighted over ENOE sectors
    exp = prof.groupby("imss_sector").apply(lambda g: pd.Series({
        "dboe": (g.dboe * g.workers).sum() / g.workers.sum(),
        "deoe": (g.deoe * g.workers).sum() / g.workers.sum(),
    })).reset_index()

    # match IMSS sector strings to the profile's imss_sector keys
    patterns = {"Agricultura": "Agricultura, ganader",
                "Extractivas": "extractivas", "Electrica": "ctrica",
                "Construccion": "construcci", "Transformacion": "transformaci",
                "Comercio": "Comercio", "Transportes": "Transportes",
                "Servicios": "Servicios"}
    def match(s):
        for key, pat in patterns.items():
            if pat.lower() in s.lower():
                return key
        return None
    imss["imss_sector"] = imss.sector.map(match)
    d = imss.dropna(subset=["imss_sector"]).merge(exp, on="imss_sector")
    d["fecha"] = pd.to_datetime(d.fecha)
    d["log_emp"] = np.log(pd.to_numeric(d.trabajadores))
    d["post"] = (d.fecha >= EVENT).astype(int)
    # standardize exposure across the 8 IMSS sectors for readable betas
    for col in ["dboe", "deoe"]:
        d[col + "_z"] = (d[col] - d[col].mean()) / d[col].std(ddof=0)
    return d


def did(d: pd.DataFrame, start: str, label: str):
    sub = d[d.fecha >= start].copy()
    sub["month"] = sub.fecha.dt.strftime("%Y-%m")
    m = smf.ols("log_emp ~ C(sector) + C(month) + post:dboe_z + post:deoe_z",
                sub).fit(cov_type="HC1")
    out = []
    for term, axis in [("post:dboe_z", "cognitive"), ("post:deoe_z", "embodied")]:
        out.append({"window": label, "axis": axis, "beta": m.params[term],
                    "se": m.bse[term], "p": m.pvalues[term], "n": int(m.nobs)})
        print(f"  {label:<14} post x {axis:<10} beta = {m.params[term]:+.4f} "
              f"(se {m.bse[term]:.4f}, p = {m.pvalues[term]:.3f})")
    return out


def main() -> None:
    d = load()
    print(f"IMSS panel: {d.sector.nunique()} sectors x "
          f"{d.fecha.nunique()} months ({d.fecha.min():%Y-%m}..{d.fecha.max():%Y-%m})")
    print(f"Event: ChatGPT release {EVENT:%Y-%m}\n")
    print("Diff-in-trends (log employment; month + sector FE):")
    rows = did(d, "2021-01-01", "2021+ (main)")
    rows += did(d, "2017-01-01", "2017+ (robust)")
    res = pd.DataFrame(rows)

    # --- figure: high vs low cognitive-exposure groups, indexed -------------
    hi = d[d.dboe_z > 0]
    lo = d[d.dboe_z <= 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    for g, label, color in [(hi, "High cognitive exposure (DBOE > median)",
                             "#b3452c"),
                            (lo, "Low cognitive exposure", "#3b6ea5")]:
        ts = g.groupby("fecha").trabajadores.apply(
            lambda s: pd.to_numeric(s).sum())
        ts = ts[ts.index >= "2021-01-01"]
        base = ts[ts.index == "2022-10-01"].iloc[0]
        ax.plot(ts.index, ts / base * 100, label=label, color=color)
    ax.axvline(EVENT, color="grey", ls="--", lw=1)
    ax.annotate("ChatGPT\n(2022-11)", (EVENT, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(6, 12), fontsize=9,
                color="grey")
    ax.set_ylabel("Formal employment (2022-10 = 100)")
    ax.set_title("Jalisco IMSS employment around the generative-AI shock")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = FIGURES_DIR / "chatgpt_event_imss.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n-> {out_png}")

    res.to_sql("chatgpt_event_imss", ENGINE, if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "chatgpt_event_imss.csv", index=False)
    print(f"-> chatgpt_event_imss: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
