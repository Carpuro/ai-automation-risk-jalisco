"""
DEOE -- Dynamic Embodied Occupational Exposure (static core).

Physical / embodied-AI exposure index, built as the structural mirror of the
DBOE (cognitive / LLM) index so the thesis covers BOTH faces of AI automation
-- embodied (robots, drones, autonomous vehicles, physical manipulation) AND
cognitive -- with the same methodological rigor.

Construction (mirrors AIOE/DBOE):
  * Physical task composition from O*NET, weighted by importance x level
    (both z-standardized, as in DBOE's W_ok -- skipping either inverts the rank).
  * Grouped into 5 theory-grounded subdomains (Frey & Osborne 2017 engineering
    bottlenecks: robots are limited by perception/manipulation in unstructured,
    cramped settings; routine/structured physical work is roboticizable).
  * Internal consistency (Cronbach alpha) reported per subdomain.
  * Convergent validation against Moravec auto_w, RL feasibility, AIOE.

Output: table `embodied_exposure_soc` (soc6 grain) + embodied_* columns merged
into `model_exposure_soc`. Static core; Webb (2020) robot calibration and the
robotics capability time-curve (dynamic layer) are added in later stages.

Robustness (documented 2026-06-10): phys_routine is the weakest subdomain
(alpha = 0.57, 2 items). Dropping it and recomputing PC1 over the remaining 4
subdomains yields r = 0.987 with the full 5-subdomain PC1 (73.5% variance) --
the summary index is insensitive to the weak subdomain.

Run AFTER load_model_tables.py.
"""

import os
import pandas as pd
import numpy as np
import sqlalchemy as sa

ENGINE = sa.create_engine(
    "mssql+pyodbc://localhost,1433/ai_automation_risk_jalisco"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)

# --- O*NET element design -------------------------------------------------
# Work Activities (scale: importance IM + level LV, combined as in DBOE W_ok)
ACT = {
    "4.A.3.a.1": "general_physical",
    "4.A.3.a.2": "handling_objects",
    "4.A.3.a.3": "controlling_machines",
    "4.A.3.a.4": "operating_vehicles",
    "4.A.3.b.4": "repair_mechanical",
    "4.A.3.b.5": "repair_electronic",
}
# Work Context (scale: CX 1-5)
CTX = {
    "4.C.3.b.2": "degree_automation",
    "4.C.3.d.3": "pace_by_equipment",
    "4.C.2.a.1.c": "outdoors_weather",
    "4.C.2.a.1.e": "open_vehicle",
    "4.C.2.a.1.f": "enclosed_vehicle",
    "4.C.3.b.7": "repeating_tasks",
    "4.C.2.d.1.i": "repetitive_motions",
    "4.C.2.d.1.g": "hands_handle_feel",
    "4.C.2.b.1.e": "cramped_awkward",
}
# Subdomains -> member feature names (post-merge column names)
SUBDOMAINS = {
    "phys_manual": ["general_physical", "handling_objects"],
    "phys_machine": ["controlling_machines", "degree_automation",
                     "pace_by_equipment", "repair_mechanical", "repair_electronic"],
    "phys_vehicle_field": ["operating_vehicles", "outdoors_weather",
                           "open_vehicle", "enclosed_vehicle"],
    "phys_routine": ["repeating_tasks", "repetitive_motions"],
    "phys_dexterity": ["hands_handle_feel", "cramped_awkward"],
}


def z(s):
    return (s - s.mean()) / s.std()


def cronbach_alpha(df_items):
    """Standardized Cronbach's alpha on a set of (already z-scored) items."""
    k = df_items.shape[1]
    if k < 2:
        return np.nan
    item_var = df_items.var(axis=0, ddof=1).sum()
    total_var = df_items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var / total_var)


def load_features():
    """Pull O*NET physical descriptors, aggregate 8-digit -> 6-digit SOC."""
    c = ENGINE.connect()
    # Activities: importance + level, combined to a single weight per element
    a = pd.read_sql(
        "SELECT LEFT(onetsoc_code,7) AS soc6, element_id, scale_id, "
        "AVG(data_value) AS v FROM onet_work_activities_detail "
        "WHERE element_id IN ({}) AND scale_id IN ('IM','LV') "
        "GROUP BY LEFT(onetsoc_code,7), element_id, scale_id".format(
            ",".join(f"'{k}'" for k in ACT)), c)
    # Context: CX scale
    x = pd.read_sql(
        "SELECT LEFT(onetsoc_code,7) AS soc6, element_id, AVG(data_value) AS v "
        "FROM onet_work_context WHERE element_id IN ({}) AND scale_id='CX' "
        "GROUP BY LEFT(onetsoc_code,7), element_id".format(
            ",".join(f"'{k}'" for k in CTX)), c)
    c.close()

    # Activities -> W = z(importance) + z(level), per element, then name it
    feats = {}
    for eid, name in ACT.items():
        sub = a[a.element_id == eid].pivot_table(index="soc6", columns="scale_id", values="v")
        if {"IM", "LV"}.issubset(sub.columns):
            feats[name] = z(sub["IM"]) + z(sub["LV"])
    # Context -> z(CX)
    for eid, name in CTX.items():
        sub = x[x.element_id == eid].set_index("soc6")["v"]
        feats[name] = z(sub)

    F = pd.DataFrame(feats)
    return F


def build():
    print("DEOE static core -- building embodied exposure from O*NET\n")
    F = load_features()
    print(f"features: {F.shape[1]} O*NET descriptors x {F.shape[0]} SOC (6-digit)")
    F = F.dropna(how="any")
    print(f"complete-case SOC: {len(F)}\n")

    # z-standardize each feature so subdomains average comparable scales
    Fz = F.apply(z)

    # Subdomain scores + internal consistency
    out = pd.DataFrame(index=Fz.index)
    print("Cronbach's alpha by subdomain:")
    for sub, items in SUBDOMAINS.items():
        out[sub] = Fz[items].mean(axis=1)
        print(f"  {sub:28s} a={cronbach_alpha(Fz[items]):.2f}  (k={len(items)})")

    # Summary DEOE = first principal component of the 5 subdomains.
    # Empirically (vs Webb 2020 robot patents) ALL subdomains -- including manual
    # dexterity -- load positively on real robot exposure; the Frey-Osborne (2017)
    # manipulation "bottleneck" no longer protects from robots. So no hand signs:
    # PC1 is the data-driven single embodied-exposure factor.
    subs = list(SUBDOMAINS)
    S = out[subs].apply(z)
    Xc = S.values - S.values.mean(axis=0)
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Xc @ Vt[0]
    if np.corrcoef(pc1, S["phys_manual"])[0] [1] < 0:  # orient: high = more physical
        pc1, Vt = -pc1, -Vt
    var_exp = sv[0] ** 2 / (sv ** 2).sum()
    out["embodied_exposure"] = pc1
    out["embodied_exposure_z"] = z(pd.Series(pc1, index=out.index))
    out = out.reset_index().rename(columns={"index": "soc6"})
    print(f"\nPC1 (embodied_exposure): {var_exp:.0%} of variance; loadings:")
    for s_, w_ in zip(subs, Vt[0]):
        print(f"    {s_:28s} {w_:+.2f}")

    # --- convergent / discriminant validation against external benchmarks ---
    cmp = pd.read_csv(os.path.join(os.path.dirname(__file__),
                      "moravec_index", "Comparison of Indices.csv"))
    cmp["soc6"] = cmp["soc6"].astype(str).str.replace("-", "", regex=False).str.zfill(6)
    v = out.copy()
    v["soc6n"] = v["soc6"].astype(str).str.replace("-", "", regex=False).str.zfill(6)
    v = v.merge(cmp.rename(columns={"soc6": "soc6n"}), on="soc6n", how="inner")
    print(f"\nValidation N (joined to Comparison of Indices): {len(v)}")
    print("embodied_exposure_z correlations:")
    print("  CONVERGENT (real robots / manual, expect HIGH +):")
    for col in ["webb_robot", "routine_manual"]:
        print(f"    {col:20s} r={v['embodied_exposure_z'].corr(v[col]):+.2f}")
    print("  DISCRIMINANT (cognitive / AI, expect LOW or -):")
    for col in ["webb_ai", "webb_software", "felten", "sml", "routine_cognitive"]:
        print(f"    {col:20s} r={v['embodied_exposure_z'].corr(v[col]):+.2f}")

    # --- persist ----------------------------------------------------------
    out.to_sql("embodied_exposure_soc", ENGINE, if_exists="replace", index=False, chunksize=1000)
    print(f"\n-> embodied_exposure_soc: {len(out)} rows, {out.shape[1]} cols")

    # merge embodied columns into model_exposure_soc (additive, non-destructive)
    c = ENGINE.connect()
    m = pd.read_sql("SELECT * FROM model_exposure_soc", c)
    c.close()
    emb_cols = [col for col in out.columns if col.startswith("phys_") or col.startswith("embodied_")]
    m = m[[col for col in m.columns  # drop any prior embodied cols if re-running
           if not (col.startswith("phys_") or col.startswith("embodied_"))]]
    m = m.merge(out[["soc6"] + emb_cols], on="soc6", how="left")
    m.to_sql("model_exposure_soc", ENGINE, if_exists="replace", index=False, chunksize=1000)
    print(f"-> model_exposure_soc updated: {len(m)} rows, {m.shape[1]} cols "
          f"(+{len(emb_cols)} embodied cols)")
    return v


if __name__ == "__main__":
    build()
    print("\nDone. DEOE static core built and validated.")
