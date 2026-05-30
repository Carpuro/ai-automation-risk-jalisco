"""Build the Dynamic LLM Occupational Exposure index (DBOE).

This script extends the static AI Occupational Exposure (AIOE) of
Felten, Raj & Seamans (2021) with a temporal dimension driven by real
LLM benchmark scores from the Epoch AI Capabilities dataset.

Method
------
Felten's AIOE combines, for each occupation, the O*NET importance and
level of 52 abilities with a fixed 52x10 ability-to-application
relatedness matrix (Appendix D). The published index is reproduced
(r = 0.94 against Appendix A) by the standardized form:

    A_k        = sum_j rel[k, j]                 (ability AI exposure)
    W_ok       = z_occ(importance_ok) + z_occ(level_ok)
    AIOE_o     = sum_k z_abil(A_k) * W_ok

where z_occ standardizes an ability across occupations and z_abil
standardizes across abilities. Standardizing A_k is essential: without
it the relatedness floor (~0.2 on every cell) makes occupations with
many high-level abilities dominate, which inverts the ranking.

We make the application capability time-varying. For each year t we
estimate the frontier capability c_j(t) of three LLM-relevant
applications from benchmark scores and rebuild the ability exposure:

    c_j(t)     = mean over the benchmarks of application j of the
                 frontier (max) score among models released up to t
    A_k(t)     = sum_j rel[k, j] * c_j(t)        (LLM apps only)
    DBOE_o(t)  = sum_k z_abil(A_k(t)) * W_ok

We restrict to the three applications LLMs actually perform
(Language Modeling, Reading Comprehension, Abstract Strategy Games as a
reasoning/math proxy). The remaining seven applications in the AIOE
matrix are vision/audio/robotics modalities outside the LLM scope of
this thesis and are excluded so the index measures LLM exposure cleanly.
As reasoning/math capability catches up to language capability over
2022-2026, occupations weighted toward mathematical reasoning gain
relative exposure - the re-ranking is the temporal signal of interest.

Outputs
-------
data/processed/dynamic_aioe_scores.csv
    One row per SOC occupation with the DBOE score for each year. The
    raw application-capability curve c_j(t) is printed for the temporal
    narrative (it is the clean way to show absolute capability growth).

Usage
-----
    python data/raw/build_dynamic_aioe.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = RAW_DIR.parent / "processed"
ABILITIES_FILE = RAW_DIR / "onet" / "db_28_3_text" / "Abilities.txt"
AIOE_FILE = RAW_DIR / "AIOE_DataAppendix.xlsx"
EPOCH_ZIP = RAW_DIR / "epoch_ai_capabilities.zip"
ESCO_FILE = RAW_DIR / "ESCO_to_ONET_SOC_crosswalk.xlsx"
OUTPUT_FILE = PROCESSED_DIR / "dynamic_aioe_scores.csv"
SINCO_OUTPUT_FILE = PROCESSED_DIR / "sinco_dboe_scores.csv"

# SINCO major -> ISCO-08 major bridge, identical to build_crosswalk.py so
# the DBOE aggregation matches the SINCO groups already loaded in the DB.
SINCO_TO_ISCO = {
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "5", "7": "6", "8": "7", "9": "8", "0": "9",
}
SINCO_LABELS = {
    "1": "Directivos y gerentes",
    "2": "Profesionistas y tecnicos de alto nivel",
    "3": "Tecnicos especializados",
    "4": "Trabajadores de apoyo administrativo",
    "5": "Comerciantes y vendedores",
    "6": "Trabajadores en servicios personales",
    "7": "Trabajadores agropecuarios",
    "8": "Artesanos y trabajadores en manufactura",
    "9": "Operadores de maquinaria y transporte",
    "0": "Trabajadores en ocupaciones elementales",
}

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
YEARS = [2022, 2023, 2024, 2025, 2026]

# AIOE applications kept (the LLM-relevant subset of the 10-column matrix).
LLM_APPLICATIONS = [
    "Language Modeling",
    "Reading Comprehension",
    "Abstract Strategy Games",
]

# Mapping from each AIOE application to its Epoch benchmark files and the
# score column to read. Scores are 0-1 unless noted in SCORE_SCALE.
BENCHMARK_MAP = {
    "Language Modeling": {
        "mmlu_external.csv": "EM",
        "bbh_external.csv": "Average",
        "live_bench_external.csv": "Global average",
        "wino_grande_external.csv": "Accuracy",
    },
    "Reading Comprehension": {
        "gpqa_diamond.csv": "mean_score",
        "hle_external.csv": "Accuracy",
        "bool_q_external.csv": "Score",
        "open_book_qa_external.csv": "Accuracy",
    },
    "Abstract Strategy Games": {
        "math_level_5.csv": "mean_score",
        "otis_mock_aime_2024_2025.csv": "mean_score",
        "frontiermath.csv": "mean_score",
        "chess_puzzles.csv": "mean_score",
    },
}

# Benchmarks reported on a 0-100 scale that must be divided by 100.
SCORE_SCALE = {"live_bench_external.csv": 100.0}

# One ability is named differently in the two sources.
ABILITY_RENAME = {"Visual Color Determination": "Visual Color Discrimination"}


def log(msg: str) -> None:
    print(f"[build_dynamic_aioe] {msg}")


def zscore(frame: pd.DataFrame | pd.Series, axis: int = 0):
    """Standardize to zero mean and unit std along the given axis."""
    return (frame - frame.mean(axis=axis)) / frame.std(axis=axis)


# --------------------------------------------------------------------------
# Step 1 - O*NET ability weights per occupation
# --------------------------------------------------------------------------
def load_ability_weights() -> pd.DataFrame:
    """Return a SOC x ability matrix of occupation weights W_ok.

    W_ok = z_occ(importance) + z_occ(level), standardizing each ability
    across occupations. The O*NET-SOC code (e.g. 11-1011.00) is truncated
    to the 6-digit SOC code (11-1011) to align with the AIOE appendix,
    averaging detailed occupations into their SOC parent.
    """
    if not ABILITIES_FILE.exists():
        sys.exit(f"Missing O*NET Abilities file: {ABILITIES_FILE}")

    df = pd.read_csv(ABILITIES_FILE, sep="\t")
    df = df[["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"]].copy()
    df["SOC"] = df["O*NET-SOC Code"].str.slice(0, 7)

    importance = (
        df[df["Scale ID"] == "IM"]
        .groupby(["SOC", "Element Name"])["Data Value"]
        .mean()
        .unstack("Element Name")
    )
    level = (
        df[df["Scale ID"] == "LV"]
        .groupby(["SOC", "Element Name"])["Data Value"]
        .mean()
        .unstack("Element Name")
    )

    weights = zscore(importance, axis=0) + zscore(level, axis=0)

    log(f"Ability weights: {weights.shape[0]} SOC occupations, "
        f"{weights.shape[1]} abilities")
    return weights


# --------------------------------------------------------------------------
# Step 2 - AIOE ability-to-application relatedness matrix
# --------------------------------------------------------------------------
def load_relatedness_matrix(applications: list[str] | None = None) -> pd.DataFrame:
    """Return the 52 x len(applications) relatedness matrix (Appendix D).

    Defaults to the LLM application subset. Pass a different list (or the
    full 10 columns) for validation.
    """
    if not AIOE_FILE.exists():
        sys.exit(f"Missing AIOE data appendix: {AIOE_FILE}")

    matrix = pd.read_excel(AIOE_FILE, sheet_name="Appendix D")
    matrix = matrix.rename(columns={"Unnamed: 0": "Ability"}).set_index("Ability")
    matrix = matrix.rename(index=ABILITY_RENAME)

    if applications is not None:
        matrix = matrix[applications]

    log(f"Relatedness matrix: {matrix.shape[0]} abilities x "
        f"{matrix.shape[1]} applications")
    return matrix


# --------------------------------------------------------------------------
# Step 3 - Frontier benchmark capability per application per year
# --------------------------------------------------------------------------
def load_application_capabilities() -> pd.DataFrame:
    """Return an application x year matrix of frontier capability (0-1)."""
    if not EPOCH_ZIP.exists():
        sys.exit(f"Missing Epoch capabilities zip: {EPOCH_ZIP}")

    rows = []
    with zipfile.ZipFile(EPOCH_ZIP) as z:
        for app, benchmarks in BENCHMARK_MAP.items():
            for fname, score_col in benchmarks.items():
                with z.open(fname) as f:
                    bench = pd.read_csv(f)
                score = pd.to_numeric(bench[score_col], errors="coerce")
                score = score / SCORE_SCALE.get(fname, 1.0)
                date = pd.to_datetime(bench["Release date"], errors="coerce")
                tmp = pd.DataFrame({"app": app, "benchmark": fname,
                                    "score": score, "date": date})
                rows.append(tmp.dropna(subset=["score", "date"]))

    obs = pd.concat(rows, ignore_index=True)

    # Frontier (max) score per benchmark among models released up to each
    # year-end, then average across the benchmarks of the application.
    records = {}
    for year in YEARS:
        cutoff = pd.Timestamp(f"{year}-12-31")
        up_to = obs[obs["date"] <= cutoff]
        frontier = up_to.groupby(["app", "benchmark"])["score"].max()
        app_cap = frontier.groupby("app").mean()
        records[year] = app_cap

    capabilities = pd.DataFrame(records).reindex(LLM_APPLICATIONS)

    # A year with no released benchmark (e.g. frontier-math/AIME before
    # 2023) means frontier models scored ~0 on that capability; treat as 0.
    if capabilities.isna().any().any():
        log("Filling missing early-year capabilities with 0 "
            "(benchmark did not yet exist; frontier score ~0).")
        capabilities = capabilities.fillna(0.0)

    log("Frontier application capability by year:")
    for app in LLM_APPLICATIONS:
        vals = ", ".join(f"{y}={capabilities.loc[app, y]:.3f}" for y in YEARS)
        log(f"    {app:28s} {vals}")
    return capabilities


# --------------------------------------------------------------------------
# Step 4 - Assemble the Dynamic AIOE index
# --------------------------------------------------------------------------
def build_index(weights: pd.DataFrame, relatedness: pd.DataFrame,
                capabilities: pd.DataFrame) -> pd.DataFrame:
    """Combine the three components into per-occupation DBOE scores.

    DBOE_o(t) = sum_k z_abil(A_k(t)) * W_ok, where
    A_k(t) = sum_j rel[k, j] * c_j(t) over the LLM applications.
    """
    # Align abilities present in both sources (expected: 52).
    common = weights.columns.intersection(relatedness.index)
    if len(common) != 52:
        log(f"WARNING: {len(common)} abilities matched (expected 52). "
            "Check ability-name alignment between O*NET and AIOE.")
    weights = weights[common]
    relatedness = relatedness.loc[common]

    result = pd.DataFrame(index=weights.index)
    for year in YEARS:
        c = capabilities[year]                           # (apps,)
        ability_exposure = relatedness.mul(c, axis=1).sum(axis=1)  # A_k(t)
        ability_z = zscore(ability_exposure, axis=0)     # across abilities
        result[f"dboe_{year}"] = weights.values @ ability_z.values

    result = result.reset_index().rename(columns={"index": "SOC"})
    return result


# --------------------------------------------------------------------------
# Step 5 - Validation against the static Felten AIOE
# --------------------------------------------------------------------------
def validate(weights: pd.DataFrame, result: pd.DataFrame) -> None:
    """Two checks: reproduce the published AIOE and the LM-specific AIOE."""
    # Check 1: reconstruct the full static AIOE (all 10 applications,
    # unit capability) and compare to the published Appendix A index.
    full = load_relatedness_matrix(applications=None)
    common = weights.columns.intersection(full.index)
    a_static = zscore(full.loc[common].sum(axis=1), axis=0)
    static_aioe = pd.DataFrame({
        "SOC": weights.index,
        "recon": weights[common].values @ a_static.values,
    })
    pub = pd.read_excel(AIOE_FILE, sheet_name="Appendix A").rename(
        columns={"SOC Code": "SOC", "AIOE": "pub_aioe"})
    m1 = static_aioe.merge(pub[["SOC", "pub_aioe"]], on="SOC")
    r1 = m1["recon"].corr(m1["pub_aioe"])
    log(f"Validation 1: corr(reconstructed static AIOE, published) "
        f"= {r1:.3f} over {len(m1)} occupations")

    # Check 2: the dynamic LLM index should track Felten's
    # language-modeling-specific AIOE (same application family).
    lm = pd.read_excel(
        RAW_DIR / "Language_Modeling_AIOE_AIIE.xlsx", sheet_name="LM AIOE"
    ).rename(columns={"SOC Code": "SOC",
                      "Language Modeling AIOE": "lm_aioe"})
    m2 = result.merge(lm[["SOC", "lm_aioe"]], on="SOC", how="inner")
    r2 = m2["dboe_2026"].corr(m2["lm_aioe"])
    log(f"Validation 2: corr(DBOE 2026, Felten LM AIOE) "
        f"= {r2:.3f} over {len(m2)} occupations")

    if r1 < 0.85 or r2 < 0.5:
        log("WARNING: validation below expected threshold - review method.")


def aggregate_to_sinco(result: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SOC-level DBOE to the 10 SINCO major groups.

    Uses the ESCO ISCO-08 <-> SOC crosswalk and the SINCO->ISCO bridge,
    replicating build_crosswalk.py so the result joins to the SINCO
    groups already loaded as ocupaciones_onet in the database.
    """
    if not ESCO_FILE.exists():
        sys.exit(f"Missing ESCO crosswalk: {ESCO_FILE}")

    esco = pd.read_excel(ESCO_FILE, header=2)
    esco.columns = ["isco_code", "isco_title", "soc_code", "soc_title"]
    esco = esco.dropna(subset=["isco_code", "soc_code"])
    esco = esco[~esco["isco_code"].astype(str).str.contains("ESCO", na=True)]
    esco["isco_major"] = esco["isco_code"].astype(str).str[0]
    esco["SOC"] = esco["soc_code"].astype(str).str.slice(0, 7)

    esco_dboe = esco.merge(result, on="SOC", how="inner")
    score_cols = [f"dboe_{y}" for y in YEARS] + ["dboe_2026_z"]

    rows = []
    for sinco_major, isco_major in SINCO_TO_ISCO.items():
        subset = esco_dboe[esco_dboe["isco_major"] == isco_major]
        if subset.empty:
            subset = esco_dboe  # fallback: global mean
        means = subset[score_cols].mean()
        means["sinco_major"] = sinco_major
        means["n_soc"] = subset["SOC"].nunique()
        rows.append(means)

    sinco_df = pd.DataFrame(rows)
    sinco_df["sinco_label"] = sinco_df["sinco_major"].map(SINCO_LABELS)
    ordered = ["sinco_major", "sinco_label", "n_soc"] + score_cols
    return sinco_df[ordered].sort_values("sinco_major").reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    weights = load_ability_weights()
    relatedness = load_relatedness_matrix(applications=LLM_APPLICATIONS)
    capabilities = load_application_capabilities()
    result = build_index(weights, relatedness, capabilities)

    validate(weights, result)

    # Standardized 2026 score across occupations, for a clean ENOE merge.
    result["dboe_2026_z"] = zscore(result["dboe_2026"], axis=0)

    result.to_csv(OUTPUT_FILE, index=False)
    log(f"Wrote {len(result)} occupations to {OUTPUT_FILE}")

    top = result.nlargest(8, "dboe_2026")[["SOC", "dboe_2026"]]
    log("Top-8 most LLM-exposed SOC codes (2026):")
    for _, row in top.iterrows():
        log(f"    {row['SOC']}  {row['dboe_2026']:.4f}")

    # Aggregate to SINCO major groups for the thesis ENOE join.
    sinco_df = aggregate_to_sinco(result)
    sinco_df.to_csv(SINCO_OUTPUT_FILE, index=False)
    log(f"Wrote {len(sinco_df)} SINCO groups to {SINCO_OUTPUT_FILE}")
    log("DBOE 2026 by SINCO major group (z-standardized):")
    for _, row in sinco_df.iterrows():
        log(f"    {row['sinco_major']} {row['sinco_label'][:38]:38s} "
            f"z={row['dboe_2026_z']:+.3f}  (n={int(row['n_soc'])})")


if __name__ == "__main__":
    main()
