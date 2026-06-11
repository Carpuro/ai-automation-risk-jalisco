"""
Permutation inference for the small-N sector tests.

The ChatGPT diff-in-trends (8 IMSS sectors), the absorption test (19 ENOE
sectors) and the H4 adoption test (19 CE sectors) all rest on asymptotic
robust SEs with few cross-sectional units -- the most predictable referee
objection. This recomputes their key coefficients under PERMUTATION
inference: the sector-level treatment (exposure scores; the sector's IRA
path) is randomly reassigned across sectors B times, the model refit, and
the empirical two-sided p-value is the share of permuted |beta| >= observed
|beta|. Exact under the sharp null, valid at any N.

Output: table `permutation_inference` (+ CSV) comparing analytic and
permutation p-values for each headline coefficient.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatgpt_event_imss import load as load_imss            # noqa: E402
from absorption_informality import load as load_enoe        # noqa: E402
from h4_adoption_test import load_panel as load_h4          # noqa: E402

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
B = 1000
RNG = np.random.default_rng(14)  # Jalisco


def perm_test(d, unit_col, time_col, treat_cols, formula, term, label):
    """Permute the unit -> treatment-PATH assignment across units.

    Treatment may vary over time within a unit (H4's lagged IRA): the whole
    path is reassigned, aligned by period, so the time structure of the
    treatment is preserved under the permutation.
    """
    d = d.copy()
    obs = smf.ols(formula, d).fit().params[term]
    p_analytic = smf.ols(formula, d).fit(cov_type="HC1").pvalues[term]

    units = list(d[unit_col].unique())
    treat = d.set_index([unit_col, time_col])[treat_cols].sort_index()
    hits = 0
    for _ in range(B):
        mapping = dict(zip(units, RNG.permutation(units)))
        idx = pd.MultiIndex.from_arrays(
            [d[unit_col].map(mapping), d[time_col]])
        dd = d.drop(columns=treat_cols)
        vals = treat.reindex(idx)
        for col_ in treat_cols:
            dd[col_] = vals[col_].values
        dd = dd.dropna(subset=treat_cols)  # unmatched (unit,time) cells
        b = smf.ols(formula, dd).fit().params[term]
        if abs(b) >= abs(obs):
            hits += 1
    p_perm = (hits + 1) / (B + 1)
    print(f"{label:<46} beta={obs:+.4f}  p_HC1={p_analytic:.3f}  "
          f"p_perm={p_perm:.3f}")
    return {"test": label, "term": term, "beta": obs,
            "p_analytic": p_analytic, "p_permutation": p_perm, "B": B}


def main() -> None:
    rows = []
    print(f"Permutation inference (B = {B}, seed 14):\n")

    d1 = load_imss()
    d1["month"] = d1.fecha.dt.strftime("%Y-%m")
    d1 = d1[d1.fecha >= "2021-01-01"]
    rows.append(perm_test(
        d1, "sector", "month", ["dboe_z", "deoe_z"],
        "log_emp ~ C(sector) + C(month) + post:dboe_z + post:deoe_z",
        "post:dboe_z", "ChatGPT IMSS: post x DBOE (formal emp.)"))

    d2 = load_enoe()
    rows.append(perm_test(
        d2, "scian", "t", ["dboe_z", "deoe_z"],
        "log_total ~ C(scian) + C(t) + post:dboe_z + post:deoe_z",
        "post:dboe_z", "Absorption: post x DBOE (total emp.)"))
    rows.append(perm_test(
        d2, "scian", "t", ["dboe_z", "deoe_z"],
        "informal_share ~ C(scian) + C(t) + post:dboe_z + post:deoe_z",
        "post:dboe_z", "Absorption: post x DBOE (informal share)"))

    d3 = load_h4()
    rows.append(perm_test(
        d3, "sector_code", "period", ["log_ira_lag"],
        "dlog_kl ~ log_ira_lag + C(period)",
        "log_ira_lag", "H4: lagged IRA -> capital deepening"))
    rows.append(perm_test(
        d3, "sector_code", "period", ["log_ira_lag"],
        "d_labor_share ~ log_ira_lag + C(period)",
        "log_ira_lag", "H4: lagged IRA -> labor share"))

    res = pd.DataFrame(rows)
    res.to_sql("permutation_inference", ENGINE,
               if_exists="replace", index=False)
    res.to_csv(PROCESSED_DIR / "permutation_inference.csv", index=False)
    print(f"\n-> permutation_inference: {len(res)} rows (+ CSV)")


if __name__ == "__main__":
    main()
