"""
ENOE quarterly panel for Jalisco: TOTAL employment (formal + informal) by
sector, 2022T1-2024T3.

Closes the formal-only bias of the outcome evidence: IMSS sees only formal
jobs (~45% of Jalisco employment). This builds the full-employment panel from
ENOE SDEMT microdata -- one quarter at a time, ent=14, occupied population
(clase2 == 1), survey-weighted (fac_tri) -- split by the official
formal/informal classification of the main job (emp_ppal: 1 = informal,
2 = formal) and by the ENOE SCIAN sector recode (same grain as
sector_exposure_profile).

Sources (INEGI microdata, downloaded on first run and cached locally):
  2022 quarters:  enoe_n_2022_trimQ_csv.zip   (ENOE-N naming)
  2023-2024:      enoe_YYYY_trimQ_csv.zip
Zips are cached in data/raw/enoe/quarters/ (gitignored, ~350 MB total).

Output: table `enoe_jalisco_quarterly` (quarter x scian x formality:
weighted workers) + CSV. The absorption test consumes this table.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

RAW_DIR = Path(__file__).resolve().parent
CACHE = RAW_DIR / "enoe" / "quarters"
PROCESSED_DIR = RAW_DIR.parent / "processed"
BASE = "https://www.inegi.org.mx/contenidos/programas/enoe/15ymas/microdatos/"

QUARTERS = ([(2022, q) for q in (1, 2, 3, 4)]
            + [(2023, q) for q in (1, 2, 3, 4)]
            + [(2024, q) for q in (1, 2, 3)])

NEEDED = {"ent", "clase2", "scian", "emp_ppal", "fac_tri", "fac"}


def zip_name(year: int, q: int) -> str:
    prefix = "enoe_n" if year == 2022 else "enoe"
    return f"{prefix}_{year}_trim{q}_csv.zip"


def fetch(year: int, q: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / zip_name(year, q)
    if not path.exists():
        url = BASE + zip_name(year, q)
        print(f"  downloading {url} ...")
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def read_sdemt(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        member = [n for n in z.namelist() if "sdem" in n.lower()
                  and n.lower().endswith(".csv")][0]
        with z.open(member) as f:
            d = pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"),
                            usecols=lambda c: c.strip().lower() in NEEDED,
                            low_memory=False)
    d.columns = [c.strip().lower() for c in d.columns]
    return d


def main() -> None:
    frames = []
    for year, q in QUARTERS:
        d = read_sdemt(fetch(year, q))
        w = "fac_tri" if "fac_tri" in d.columns else "fac"
        for col in ["ent", "clase2", "scian", "emp_ppal", w]:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d[(d.ent == 14) & (d.clase2 == 1) & d.emp_ppal.isin([1, 2])]
        g = (d.groupby(["scian", "emp_ppal"])[w].sum()
               .rename("workers").reset_index())
        g["year"], g["quarter"] = year, q
        frames.append(g)
        tot = g.workers.sum()
        inf = g[g.emp_ppal == 1].workers.sum() / tot
        print(f"  {year}T{q}: occupied {tot:,.0f} weighted, "
              f"informality {inf:.1%}")

    panel = pd.concat(frames, ignore_index=True)
    panel["formality"] = panel.emp_ppal.map({1: "informal", 2: "formal"})
    panel = panel[["year", "quarter", "scian", "formality", "workers"]]

    panel.to_sql("enoe_jalisco_quarterly", ENGINE,
                 if_exists="replace", index=False, chunksize=2000)
    panel.to_csv(PROCESSED_DIR / "enoe_jalisco_quarterly.csv", index=False)
    print(f"\n-> enoe_jalisco_quarterly: {len(panel)} rows "
          f"({panel.year.nunique()} years x sectors x formality; + CSV)")


if __name__ == "__main__":
    main()
