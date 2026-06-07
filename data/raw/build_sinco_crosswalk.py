"""
Build the missing SINCO-2011 <-> ISCO-08 (CIUO) and SINCO-2011 <-> SOC crosswalks
from INEGI's official comparison workbook (`mexico_sinco_tablas_comparativas.xlsx`,
sheets SINCO-CIUO and SINCO-SOC).

This closes the only external data gap: it lets the Jalisco side (ENOE occupation
in SINCO-4-digit) reach the occupation-level AI-exposure indices (DBOE cognitive,
DEOE physical) that live on a SOC / ISCO grain.

Source provenance: the workbook is INEGI's "Tablas comparativas del SINCO 2011";
obtained via the open `occupationcross` R package (Guidowe), data-raw/cross/.

Output: server tables `crosswalk_sinco_ciuo` and `crosswalk_sinco_soc`
(long format, one row per SINCO-4d -> target mapping; relationships are 1-to-many).
"""

import os
import re
import pandas as pd
import sqlalchemy as sa

RAW = os.path.dirname(__file__)
XLSX = os.path.join(RAW, "crosswalks", "mexico_sinco_tablas_comparativas.xlsx")
ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

SINCO_CODE = re.compile(r"^\d{2,6}$")          # SINCO source code (plain digits)


def is_target_code(x):
    """A target code (CIUO '1344', SOC '11-9110', '0110') has digits and no
    letters; titles and 'No tiene correspondencia' have letters -> rejected."""
    return (isinstance(x, str) and bool(re.search(r"\d", x))
            and not re.search(r"[A-Za-zÀ-ÿ]", x))


def parse_sheet(sheet, target_name):
    """Cols: 3=SINCO code, 4=SINCO title, 5=target code, 6=target title.
    Forward-carry the current SINCO unit group so continuation rows (blank SINCO,
    populated target) attach as additional 1-to-many mappings."""
    d = pd.read_excel(XLSX, sheet_name=sheet, header=None, dtype=str)
    cur_code = cur_title = None
    rows = []
    for _, r in d.iterrows():
        s, st = r[3], r[4]
        t, tt = r[5], r[6]
        s = s.strip() if isinstance(s, str) else s
        t = t.strip() if isinstance(t, str) else t
        if isinstance(s, str) and SINCO_CODE.match(s):  # new SINCO group starts
            cur_code, cur_title = s, (st.strip() if isinstance(st, str) else st)
        if cur_code and is_target_code(t):
            rows.append({
                "sinco4": cur_code, "sinco_title": cur_title,
                f"{target_name}_code": t,
                f"{target_name}_title": (tt.strip() if isinstance(tt, str) else tt),
            })
    out = pd.DataFrame(rows)
    out = out[out.sinco4.str.len() == 4]               # keep SINCO unit groups only
    return out.drop_duplicates().reset_index(drop=True)


def main():
    ciuo = parse_sheet("SINCO-CIUO", "ciuo")
    soc = parse_sheet("SINCO-SOC", "soc")
    print(f"SINCO->CIUO: {len(ciuo)} mappings, {ciuo.sinco4.nunique()} SINCO-4d, "
          f"{ciuo.ciuo_code.nunique()} CIUO codes")
    print(f"SINCO->SOC : {len(soc)} mappings, {soc.sinco4.nunique()} SINCO-4d, "
          f"{soc.soc_code.nunique()} SOC codes")

    # coverage of the Jalisco workforce occupations actually observed
    c = ENGINE.connect()
    enoe = pd.read_sql("SELECT DISTINCT sinco4 FROM enoe_jalisco_workers", c)
    c.close()
    enoe["s4"] = enoe.sinco4.astype("Int64").astype(str).str.zfill(4)
    cov_c = enoe.s4.isin(ciuo.sinco4).mean()
    cov_s = enoe.s4.isin(soc.sinco4).mean()
    print(f"\nENOE Jalisco distinct SINCO-4d: {len(enoe)}")
    print(f"  covered by SINCO->CIUO: {cov_c:.0%}")
    print(f"  covered by SINCO->SOC : {cov_s:.0%}")

    ciuo.to_sql("crosswalk_sinco_ciuo", ENGINE, if_exists="replace", index=False, chunksize=1000)
    soc.to_sql("crosswalk_sinco_soc", ENGINE, if_exists="replace", index=False, chunksize=1000)
    print("\n-> crosswalk_sinco_ciuo, crosswalk_sinco_soc loaded.")


if __name__ == "__main__":
    main()
