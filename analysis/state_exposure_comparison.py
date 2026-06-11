"""
National context for H3: exposure profiles for all 32 Mexican states.

Is Jalisco UNUSUALLY robot-leaning, or just average? This replicates the
Jalisco worker chain nationally from the cached ENOE 2024T3 microdata
(national SDEMT + COE1 person-level join, the same key and filters as
load_model_tables.py) and computes the employment-weighted DBOE / DEOE for
every state, positioning Jalisco in the national distribution.

Output: table `state_exposure_comparison` (32 states: weighted DBOE, DEOE,
coverage, workers) + figures/state_exposure_map.png (scatter DBOE x DEOE
with Jalisco highlighted).
"""

import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "raw" / "enoe" / "quarters" / "enoe_2024_trim3_csv.zip"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

KEY = ["ent", "con", "upm", "v_sel", "n_hog", "h_mud", "n_ren", "per"]
STATES = {
    1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur",
    4: "Campeche", 5: "Coahuila", 6: "Colima", 7: "Chiapas", 8: "Chihuahua",
    9: "CDMX", 10: "Durango", 11: "Guanajuato", 12: "Guerrero", 13: "Hidalgo",
    14: "Jalisco", 15: "Edo. de Mexico", 16: "Michoacan", 17: "Morelos",
    18: "Nayarit", 19: "Nuevo Leon", 20: "Oaxaca", 21: "Puebla",
    22: "Queretaro", 23: "Quintana Roo", 24: "San Luis Potosi", 25: "Sinaloa",
    26: "Sonora", 27: "Tabasco", 28: "Tamaulipas", 29: "Tlaxcala",
    30: "Veracruz", 31: "Yucatan", 32: "Zacatecas",
}


def read_member(z: zipfile.ZipFile, token: str, cols: set) -> pd.DataFrame:
    member = [n for n in z.namelist() if token in n.lower()][0]
    with z.open(member) as f:
        d = pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"),
                        usecols=lambda c: c.strip().lower() in cols,
                        low_memory=False)
    d.columns = [c.strip().lower() for c in d.columns]
    return d


def main() -> None:
    z = zipfile.ZipFile(ZIP)
    sd = read_member(z, "sdemt", set(KEY) | {"clase1", "fac_tri"})
    co = read_member(z, "coe1", set(KEY) | {"p3"})
    co = co.drop_duplicates(subset=KEY, keep="first")
    m = sd.merge(co, on=KEY, how="left")
    m["sinco4"] = pd.to_numeric(m.p3, errors="coerce")
    w = m[(m.clase1 == 1) & (m.sinco4 >= 1000)].copy()
    w["sinco4"] = w.sinco4.astype(int).astype(str).str.zfill(4)
    print(f"National occupied workers with SINCO-4d: {len(w):,} "
          f"({w.fac_tri.sum():,.0f} weighted)")

    c = ENGINE.connect()
    s4 = pd.read_sql(
        "SELECT sinco4, dboe_2026_z AS dboe, embodied_exposure_z AS deoe "
        "FROM sinco4_exposure_scores", c)
    c.close()
    s4["sinco4"] = s4.sinco4.astype(str).str.zfill(4)
    d = w.merge(s4, on="sinco4", how="left")

    res = d.groupby("ent").apply(lambda g: pd.Series({
        "workers": g.fac_tri.sum(),
        "coverage": g[g.dboe.notna()].fac_tri.sum() / g.fac_tri.sum(),
        "dboe": (g.dboe * g.fac_tri).sum() / (g.dboe.notna() * g.fac_tri).sum(),
        "deoe": (g.deoe * g.fac_tri).sum() / (g.deoe.notna() * g.fac_tri).sum(),
    })).reset_index()
    res["state"] = res.ent.map(STATES)
    res["rank_dboe"] = res.dboe.rank(ascending=False).astype(int)
    res["rank_deoe"] = res.deoe.rank(ascending=False).astype(int)

    jal = res[res.ent == 14].iloc[0]
    print(f"\nJalisco: DBOE {jal.dboe:+.3f} (rank {jal.rank_dboe}/32), "
          f"DEOE {jal.deoe:+.3f} (rank {jal.rank_deoe}/32), "
          f"coverage {jal.coverage:.0%}")
    print(f"National coverage range: {res.coverage.min():.0%}-"
          f"{res.coverage.max():.0%}")
    print("\nTop-5 cognitive (DBOE) states:")
    for _, r in res.nlargest(5, "dboe").iterrows():
        print(f"  {r.state:<20} dboe={r.dboe:+.3f} deoe={r.deoe:+.3f}")
    print("Top-5 embodied (DEOE) states:")
    for _, r in res.nlargest(5, "deoe").iterrows():
        print(f"  {r.state:<20} deoe={r.deoe:+.3f} dboe={r.dboe:+.3f}")

    # --- figure ---------------------------------------------------------------
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(res.dboe, res.deoe, s=res.workers / 3e4, alpha=0.5,
               color="#3b6ea5")
    for _, r in res.iterrows():
        if r.ent == 14:
            ax.scatter(r.dboe, r.deoe, s=r.workers / 3e4, color="#b3452c",
                       zorder=3)
        label = r.state if (r.ent == 14 or r.rank_dboe <= 3
                            or r.rank_deoe <= 3) else None
        if label:
            ax.annotate(label, (r.dboe, r.deoe), textcoords="offset points",
                        xytext=(7, 4), fontsize=9,
                        color="#b3452c" if r.ent == 14 else "black")
    ax.axhline(0, color="grey", lw=0.7)
    ax.axvline(0, color="grey", lw=0.7)
    ax.set_xlabel("Cognitive exposure (DBOE, employment-weighted z)")
    ax.set_ylabel("Embodied exposure (DEOE, employment-weighted z)")
    ax.set_title("Mexico's 32 states on the two automation axes (ENOE 2024T3)\n"
                 "bubble = employed workers; Jalisco highlighted")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = FIGURES_DIR / "state_exposure_map.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n-> {out_png}")

    res.to_sql("state_exposure_comparison", ENGINE,
               if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "state_exposure_comparison.csv", index=False)
    print(f"-> state_exposure_comparison: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
