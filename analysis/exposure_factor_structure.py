"""
Exploratory factor analysis of the occupation-level AI-exposure battery.

Question this answers (the core empirical claim of the two-axis design):
  Do AI-exposure indices collapse into ONE dominant factor, or do they split
  into TWO distinct factors -- cognitive (LLM/AI) vs embodied (robots/physical)?

Earlier work wrongly concluded "one dominant factor" because the supposed
physical indices (Moravec/RL) were actually cognitive. With the validated DEOE
(embodied) axis added, this re-runs the EFA on a balanced battery.

Uses sklearn (no factor_analyzer dependency). Run after build_embodied_exposure.py
and after Comparison of Indices is loaded to external_index_comparison.
"""

import numpy as np
import pandas as pd
import sqlalchemy as sa
from factor_analyzer import FactorAnalyzer
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

# Battery: a priori labels (COG=cognitive/AI, PHY=embodied/physical, AMB=claimed
# physical but found cognitive). EFA must recover the split without being told.
# De-duplicated: AIOE (r~.96 w/ DBOE) and Eloundou (r~.9 w/ SML) dropped so the
# cognitive cluster's internal redundancy does not bloat the first factor.
VARS = {
    "dboe_2026_z": "COG  DBOE (LLM frontier)",
    "anthropic_observed_exposure": "COG  Anthropic observed",
    "sml": "COG  Suitability for ML",
    "moravec_auto_w": "AMB  Moravec auto_w",
    "rl_index_mean": "AMB  RL feasibility",
    "embodied_exposure_z": "PHY  DEOE (embodied)",
    "webb_robot": "PHY  Webb robot (patents)",
    "routine_manual": "PHY  Routine manual",
}


def load_battery():
    c = ENGINE.connect()
    m = pd.read_sql(
        "SELECT soc6, dboe_2026_z, anthropic_observed_exposure, "
        "moravec_auto_w, rl_index_mean, embodied_exposure_z FROM model_exposure_soc", c)
    x = pd.read_sql(
        "SELECT soc6_key, sml, webb_robot, routine_manual "
        "FROM external_index_comparison", c)
    c.close()
    m["k"] = m.soc6.astype(str).str.replace("-", "", regex=False).str.zfill(6)
    d = m.merge(x, left_on="k", right_on="soc6_key", how="inner")
    return d[list(VARS)].dropna()


def main():
    d = load_battery()
    print(f"EFA battery: {d.shape[1]} indices x {len(d)} occupations (complete cases)\n")
    labels = [VARS[v] for v in d.columns]

    # --- how many factors? eigenvalues (Kaiser>1) ---
    fa0 = FactorAnalyzer(n_factors=d.shape[1], rotation=None)
    fa0.fit(d.values)
    eig = fa0.get_eigenvalues()[0]
    print("Eigenvalues (scree):", np.round(eig, 2))
    print(f"Factors with eigenvalue > 1 (Kaiser): {(eig > 1).sum()}\n")

    # --- 2-factor EFA, OBLIQUE rotation (factors are allowed to correlate) ---
    fa = FactorAnalyzer(n_factors=2, rotation="oblimin")
    fa.fit(d.values)
    L = pd.DataFrame(fa.loadings_, index=labels, columns=["Factor1", "Factor2"])
    L["communality"] = fa.get_communalities()
    L["assigned"] = np.where(L.Factor1.abs() > L.Factor2.abs(), "F1", "F2")
    print("Pattern loadings (oblimin, oblique):")
    print(L.round(2).to_string())

    var = fa.get_factor_variance()  # (variance, proportional, cumulative)
    print("\nProportion variance per factor:", np.round(var[1], 2))
    if getattr(fa, "phi_", None) is not None:
        print(f"Inter-factor correlation (phi): {fa.phi_[0,1]:+.2f}")

    print("\nInterpretation:")
    for f in ["F1", "F2"]:
        members = [idx for idx in L.index if L.loc[idx, "assigned"] == f]
        print(f"  {f}: " + " | ".join(members))


if __name__ == "__main__":
    main()
