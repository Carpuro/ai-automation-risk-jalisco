"""
Level-1 occupation model: do the thesis indices (DBOE cognitive, DEOE embodied)
explain OBSERVED AI usage beyond the Frey-Osborne baseline and the published
cognitive-exposure indices?

Target choice (non-circular): anthropic_observed_exposure -- real-world Claude
usage share by occupation (Anthropic Economic Index). It is the only index in
the battery built from observed behavior rather than expert or task-rating
scores, so predicting it is an external test, not re-learning a formula.

Target distribution (measured): ~52% of occupations have zero observed usage,
median = 0, skew = 2.4. A single OLS on this target is mis-specified, so the
model is a TWO-PART (hurdle) specification:

  Part 1 (extensive margin): Logit  P(usage > 0)        -- WHO gets touched
  Part 2 (intensive margin): OLS    log(usage) | usage>0 -- HOW MUCH, given touch

Each part is fit hierarchically: M0 frey_osborne -> M1 +DBOE -> M2 +DEOE, with
likelihood-ratio tests (logit) / incremental F-tests (OLS).

Rival comparison: DBOE is also benchmarked against the published cognitive
indices (Felten AIOE, SML, Eloundou beta) as single predictors on both margins.
The AIOE rival is our own aioe_score (reproduces the published Appendix A at
r = 0.94); the `felten` column in Comparison of Indices.csv is a DIFFERENT
variant (r = 0.26 with the published index, different scale) and is not used.
Note: DBOE and AIOE correlate r ~ 0.96 by construction (DBOE extends AIOE), so
a joint regression of the two is uninformative (collinear); the fair claim is
"matches the published index while adding the temporal dimension".

Run after load_model_tables.py and the external_index_comparison load.
"""

import numpy as np
import pandas as pd
import sqlalchemy as sa
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server")

TARGET = "anthropic_observed_exposure"
BASELINE = "frey_osborne"
COGNITIVE = "dboe_2026_z"
EMBODIED = "embodied_exposure_z"
RIVALS = ["aioe_score", "sml", "eloundou_beta"]


def load_data() -> pd.DataFrame:
    c = ENGINE.connect()
    m = pd.read_sql(
        f"SELECT soc6, {TARGET}, {COGNITIVE}, {EMBODIED}, aioe_score "
        "FROM model_exposure_soc", c)
    x = pd.read_sql(
        f"SELECT soc6_key, {BASELINE}, sml, eloundou_beta "
        "FROM external_index_comparison", c)
    c.close()
    m["k"] = m.soc6.astype(str).str.replace("-", "", regex=False).str.zfill(6)
    d = m.merge(x, left_on="k", right_on="soc6_key", how="inner")
    d = d[[TARGET, BASELINE, COGNITIVE, EMBODIED] + RIVALS].dropna()
    # standardize predictors (not the target) so coefficients are comparable
    for col in d.columns:
        if col != TARGET:
            d[col] = (d[col] - d[col].mean()) / d[col].std(ddof=0)
    return d


def auc(y, p) -> float:
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    r = stats.rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def hierarchical_logit(d: pd.DataFrame, order: list[str]) -> None:
    y = (d[TARGET] > 0).astype(int)
    print(f"\n--- Part 1 (extensive margin): Logit P(usage > 0), "
          f"N = {len(d)}, positives = {y.sum()} ---")
    print(f"{'step':<26}{'pseudoR2':>10}{'AUC':>8}{'LLF':>10}{'LR_p':>12}")
    prev = None
    models = []
    for i in range(len(order)):
        X = sm.add_constant(d[order[: i + 1]])
        m = sm.Logit(y, X).fit(disp=0)
        models.append(m)
        label = ("+ " + order[i]) if prev is not None else order[i] + " (base)"
        if prev is None:
            print(f"{label:<26}{m.prsquared:>10.3f}{auc(y.values, m.fittedvalues.values):>8.3f}"
                  f"{m.llf:>10.1f}{'':>12}")
        else:
            lr = 2 * (m.llf - prev.llf)
            p = stats.chi2.sf(lr, df=1)
            print(f"{label:<26}{m.prsquared:>10.3f}{auc(y.values, m.fittedvalues.values):>8.3f}"
                  f"{m.llf:>10.1f}{p:>12.2e}")
        prev = m
    full = models[-1]
    print("Full-model logit coefficients (standardized predictors):")
    coefs = pd.DataFrame({"beta": full.params, "z": full.tvalues, "p": full.pvalues})
    print(coefs.drop(index="const").round(3).to_string())


def hierarchical_ols_positive(d: pd.DataFrame, order: list[str]) -> None:
    pos = d[d[TARGET] > 0].copy()
    pos["y"] = np.log(pos[TARGET])
    print(f"\n--- Part 2 (intensive margin): OLS log(usage) on positives, "
          f"N = {len(pos)} ---")
    print(f"{'step':<26}{'R2':>8}{'adjR2':>8}{'dR2':>8}{'F_change':>10}{'p':>12}")
    prev = None
    models = []
    for i in range(len(order)):
        X = sm.add_constant(pos[order[: i + 1]])
        m = sm.OLS(pos["y"], X).fit()
        models.append(m)
        label = ("+ " + order[i]) if prev is not None else order[i] + " (base)"
        if prev is None:
            print(f"{label:<26}{m.rsquared:>8.3f}{m.rsquared_adj:>8.3f}{'':>8}{'':>10}{'':>12}")
        else:
            cmp = anova_lm(prev, m)
            print(f"{label:<26}{m.rsquared:>8.3f}{m.rsquared_adj:>8.3f}"
                  f"{m.rsquared - prev.rsquared:>8.3f}{cmp.F[1]:>10.1f}"
                  f"{cmp['Pr(>F)'][1]:>12.2e}")
        prev = m
    full = models[-1]
    print("Full-model OLS coefficients (standardized predictors):")
    coefs = pd.DataFrame({"beta": full.params, "t": full.tvalues, "p": full.pvalues})
    print(coefs.drop(index="const").round(3).to_string())


def rival_comparison(d: pd.DataFrame) -> None:
    """Single-predictor performance of DBOE vs published cognitive indices."""
    y_bin = (d[TARGET] > 0).astype(int)
    pos = d[d[TARGET] > 0].copy()
    pos["y"] = np.log(pos[TARGET])
    print("\n--- Rival comparison: single cognitive predictor per margin ---")
    print(f"{'index':<18}{'logit_AUC':>10}{'positive_R2':>13}")
    for idx in [COGNITIVE] + RIVALS + [BASELINE]:
        m1 = sm.Logit(y_bin, sm.add_constant(d[idx])).fit(disp=0)
        a = auc(y_bin.values, m1.fittedvalues.values)
        m2 = sm.OLS(pos["y"], sm.add_constant(pos[idx])).fit()
        print(f"{idx:<18}{a:>10.3f}{m2.rsquared:>13.3f}")
    print("(DBOE vs aioe_score: r ~ 0.96 by construction -- near-identical "
          "performance expected; DBOE's added value is the temporal c_j(t) dimension.)")


def main() -> None:
    d = load_data()
    share_zero = (d[TARGET] == 0).mean()
    print(f"Level-1 hurdle model -- N = {len(d)} occupations (complete cases)")
    print(f"Target: {TARGET}; share with zero usage = {share_zero:.1%}")
    print("\nSpearman correlations with target (robust to zero inflation):")
    rho = {c: stats.spearmanr(d[TARGET], d[c]).statistic
           for c in d.columns if c != TARGET}
    print(pd.Series(rho).round(3).to_string())

    order = [BASELINE, COGNITIVE, EMBODIED]
    hierarchical_logit(d, order)
    hierarchical_ols_positive(d, order)
    rival_comparison(d)


if __name__ == "__main__":
    main()
