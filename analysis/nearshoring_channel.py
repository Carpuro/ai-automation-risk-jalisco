"""
Nearshoring channel: foreign capital inflow as the importer of the robot
frontier into Jalisco.

Jalisco is a nearshoring hub (electronics/semiconductor manufacturing,
agro-food). Three roles in the thesis:

  1. SCENARIO JUSTIFICATION: greenfield plants ("nuevas inversiones") are
     built at the global automation frontier, not at legacy local levels --
     nearshoring-era FDI is the concrete mechanism behind the ACCELERATED
     r(t) robot-adoption scenario.
  2. CONFOUNDER NOTE for the ChatGPT employment test: the post-2022 window
     coincides with the nearshoring investment wave, which pushes formal
     employment UP in manufacturing/services -- one more reason the no-
     displacement result must be read as "not yet", not "safe".
  3. EXPOSURE COMPOSITION: FDI flows into the sectors whose occupations sit
     on the embodied pole (manufacturing, logistics), growing the robot-
     exposed workforce even as it creates jobs.

Data: Secretaria de Economia, FDI flows by state and investment type
(datos.gob.mx, ied_entidad_tipo.csv, quarterly 2006+; millones de dolares).
The by-sector state file is behind an access wall (documented); state
exports (ETEF) have no open CSV endpoint -- noted as future work.

Output: table `nearshoring_fdi` (Jalisco annual FDI, total + greenfield +
national share) + figures/nearshoring_fdi.png.
"""

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
IED_FILE = ROOT / "data" / "raw" / "nearshoring" / "ied_entidad_tipo.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"


def main() -> None:
    d = pd.read_csv(IED_FILE, encoding="latin-1")
    d["mdd"] = pd.to_numeric(d.fn_millones_de_dolares, errors="coerce")
    d = d.dropna(subset=["mdd"])
    last_full = int(d[d.trimestre == 4].anio.max())
    d = d[d.anio <= last_full]

    jal = d[d.entidad == "Jalisco"]
    nat = d.groupby("anio").mdd.sum()
    tot = jal.groupby("anio").mdd.sum()
    green = (jal[jal.tipo_de_inversion == "Nuevas inversiones"]
             .groupby("anio").mdd.sum())
    res = pd.DataFrame({"fdi_total_mdd": tot, "fdi_greenfield_mdd": green,
                        "national_share": tot / nat}).reset_index()

    def era(a, b):
        sel = res[(res.anio >= a) & (res.anio <= b)]
        return (sel.fdi_total_mdd.mean(), sel.fdi_greenfield_mdd.mean(),
                sel.national_share.mean())

    pre = era(2015, 2019)
    post = era(2021, last_full)
    print(f"Jalisco FDI (annual averages, millions USD; data through {last_full}):")
    print(f"  2015-2019 (pre):          total {pre[0]:>8,.0f}  "
          f"greenfield {pre[1]:>7,.0f}  natl share {pre[2]:.1%}")
    print(f"  2021-{last_full} (nearshoring): total {post[0]:>8,.0f}  "
          f"greenfield {post[1]:>7,.0f}  natl share {post[2]:.1%}")
    print(f"  change: total {post[0]/pre[0]-1:+.0%}, "
          f"greenfield {post[1]/pre[1]-1:+.0%}")

    # tie-in: IMSS manufacturing employment + Mexico robot installations
    c = ENGINE.connect()
    imss = pd.read_sql(
        "SELECT anio, AVG(trabajadores) w FROM imss_empleo_sector "
        "WHERE sector LIKE '%transformaci%' GROUP BY anio", c)
    rob = pd.read_sql(
        "SELECT year, mx_installations FROM robot_capability_curve "
        "WHERE scenario='observed' AND mx_installations IS NOT NULL", c)
    c.close()
    m15 = imss[imss.anio == 2015].w.iloc[0]
    m24 = imss[imss.anio == 2024].w.iloc[0]
    print(f"\nTie-ins: IMSS manufacturing employment {m15:,.0f} (2015) -> "
          f"{m24:,.0f} (2024, {m24/m15-1:+.0%});")
    print(f"  Mexico robot installations sustained at "
          f"{rob.mx_installations.mean():,.0f}/yr (2019-2024, IFR) -- "
          "the capital arriving is automation-intensive.")

    # figure
    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(res.anio, res.fdi_total_mdd, color="#c9d7e4", label="Total FDI")
    ax.bar(res.anio, res.fdi_greenfield_mdd, color="#3b6ea5",
           label="Greenfield (nuevas inversiones)")
    ax.axvspan(2020.5, last_full + 0.5, color="orange", alpha=0.10,
               label="Nearshoring era")
    ax.set_ylabel("FDI inflow, millions USD")
    ax.set_title("Jalisco FDI by year: the capital channel of the robot frontier")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = FIGURES_DIR / "nearshoring_fdi.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n-> {out_png}")

    res.to_sql("nearshoring_fdi", ENGINE, if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "nearshoring_fdi.csv", index=False)
    print(f"-> nearshoring_fdi: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
