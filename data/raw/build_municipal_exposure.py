"""
Municipal automation-exposure map for Jalisco (125 municipios).

Chains the CE2024 municipal sector composition with the sector exposure
profiles built at Level-2:

  CE2024 (2023 census year, municipio x SCIAN sector, personal ocupado H001A)
    x sector_exposure_profile (employment-weighted DBOE / DEOE per sector)
    x sector_pressure_projection (2030 baseline, IRA-moderated pressure)
  -> employment-weighted municipal DBOE, DEOE and 2030 pressure
  -> choropleths (INEGI municipal geojson)

Known limitations (report in the thesis):
  * CE covers FORMAL ESTABLISHMENTS only -- agriculture and informal work are
    undercovered, so agro-heavy municipios reflect their formal-sector mix.
  * Occupational composition within a sector is assumed constant across
    municipios (exposure enters at sector grain).
  * Small-municipio cells may be censored (blank) by INEGI confidentiality.
  * Small-municipio composition artifact: in municipios with very few formal
    establishments (e.g. < 1,000 CE workers) the sector mix collapses to a
    handful of service units (school, government, clinic), inflating relative
    cognitive exposure. Read rankings restricted to larger municipios (e.g.
    >= 20k CE workers: metro core most cognitive -- Zapotlan el Grande,
    Tonala, Guadalajara, Zapopan; industrial corridor most embodied --
    El Salto, Tlajomulco, Lagos de Moreno), or filter by ce_workers.

Output: table `municipal_exposure` + data/processed/municipal_exposure.csv +
figures/municipal_exposure_map.png (3 panels: DBOE / DEOE / pressure 2030).

Run after level2_sector_projection.py.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

RAW_DIR = Path(__file__).resolve().parent
ROOT = RAW_DIR.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

CE_FILE = RAW_DIR / "INEGI_CE2024_jalisco_completo.csv"
GEOJSON = RAW_DIR / "INEGI_Jalisco_municipios.geojson"


def load_ce_municipal() -> pd.DataFrame:
    """CE 2023: one row per municipio x SCIAN sector with personal ocupado."""
    ce = pd.read_csv(CE_FILE, skiprows=4, dtype=str,
                     low_memory=False, encoding="latin-1")
    anio = pd.to_numeric(ce.iloc[:, 0], errors="coerce")
    mun = ce.iloc[:, 2].fillna("").str.strip()
    act = ce.iloc[:, 3].fillna("")
    h001a = [c for c in ce.columns if str(c).startswith("H001A ")][0]

    m = (anio == 2023) & mun.ne("") & act.str.startswith("Sector ")
    d = pd.DataFrame({
        "mun_code": mun[m].str.slice(0, 3),
        "mun_name": mun[m].str.slice(4),
        "sector_code": act[m].str.extract(r"Sector (\S+)", expand=False),
        "workers": pd.to_numeric(ce.loc[m, h001a], errors="coerce"),
    }).dropna(subset=["workers"])
    return d


def main() -> None:
    ce = load_ce_municipal()
    c = ENGINE.connect()
    prof = pd.read_sql(
        "SELECT sector_code, dboe, deoe FROM sector_exposure_profile", c)
    press = pd.read_sql(
        "SELECT sector_code, pressure_total_ira FROM sector_pressure_projection "
        "WHERE year = 2030 AND scenario = 'baseline'", c)
    c.close()

    d = ce.merge(prof, on="sector_code", how="inner").merge(
        press, on="sector_code", how="left")
    matched = d.workers.sum() / ce.workers.sum()
    print(f"CE2023 municipal rows: {len(ce)}; employment matched to "
          f"exposure profile: {matched:.1%}")

    mun = d.groupby(["mun_code", "mun_name"]).apply(lambda g: pd.Series({
        "ce_workers": g.workers.sum(),
        "dboe_mun": (g.dboe * g.workers).sum() / g.workers.sum(),
        "deoe_mun": (g.deoe * g.workers).sum() / g.workers.sum(),
        "pressure_2030": (g.pressure_total_ira * g.workers).sum() / g.workers.sum(),
    })).reset_index()
    print(f"Municipios with exposure: {len(mun)}")

    print("\nTop-5 municipios by cognitive exposure (DBOE):")
    for _, r in mun.nlargest(5, "dboe_mun").iterrows():
        print(f"  {r.mun_name:<28} dboe={r.dboe_mun:+.2f} deoe={r.deoe_mun:+.2f} "
              f"workers={r.ce_workers:,.0f}")
    print("Top-5 municipios by embodied exposure (DEOE):")
    for _, r in mun.nlargest(5, "deoe_mun").iterrows():
        print(f"  {r.mun_name:<28} deoe={r.deoe_mun:+.2f} dboe={r.dboe_mun:+.2f} "
              f"workers={r.ce_workers:,.0f}")

    # --- map ----------------------------------------------------------------
    geo = gpd.read_file(GEOJSON)
    geo = geo.merge(mun, left_on="CVE_MUN", right_on="mun_code", how="left")
    missing = geo.dboe_mun.isna().sum()
    if missing:
        print(f"\nMunicipios without CE data on map (grey): {missing}")

    FIGURES_DIR.mkdir(exist_ok=True)
    panels = [
        ("dboe_mun", "Cognitive exposure (DBOE)", "RdBu_r"),
        ("deoe_mun", "Embodied exposure (DEOE)", "RdBu_r"),
        ("pressure_2030", "AI pressure 2030 (baseline, IRA-mod.)", "YlOrRd"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (col, title, cmap) in zip(axes, panels):
        # aspect="equal" set explicitly: the INEGI geojson carries projected
        # coordinates, which break geopandas' latitude-based auto-aspect
        geo.plot(column=col, cmap=cmap, legend=True, ax=ax, aspect="equal",
                 missing_kwds={"color": "lightgrey"},
                 legend_kwds={"shrink": 0.6})
        ax.set_title(f"Jalisco — {title}", fontsize=12)
        ax.set_axis_off()
    fig.suptitle("Municipal AI-automation exposure, formal establishments "
                 "(CE 2023 sector mix x occupational exposure)", fontsize=13)
    fig.tight_layout()
    out_png = FIGURES_DIR / "municipal_exposure_map.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n-> {out_png}")

    mun.to_sql("municipal_exposure", ENGINE, if_exists="replace", index=False)
    mun.to_csv(PROCESSED_DIR / "municipal_exposure.csv", index=False)
    print(f"-> municipal_exposure: {len(mun)} rows (+ CSV in processed/)")


if __name__ == "__main__":
    main()
