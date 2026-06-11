# Legacy layer — superseded coursework demo

Everything in this directory comes from the original course project
(Programación II / early thesis demo) and is **superseded by the real thesis
methodology**. It is kept for provenance only and must not be cited as the
thesis method.

Why it is superseded:

- `src/automation_analyzer.py` *constructs* the `automation_risk` target from a
  fixed-weight formula and then "predicts" it with a Random Forest. The
  reported R² ≈ 0.75 measures re-learning that formula, not any real-world
  signal (circularity).
- The pipeline runs on **simulated data** (n = 5,000 Gaussian draws), not on
  the ENOE/O*NET/SQL Server data of the thesis.
- `docs/METHODOLOGY.md` and `docs/ANALYSIS_GUIDE.md` describe that demo design
  (old hypotheses, "GPT Exposure Score", fixed weights) and contradict the
  current methodology.

The real methodology lives in the repository root:

- Index construction: `data/raw/build_dynamic_aioe.py` (DBOE),
  `data/raw/build_embodied_exposure.py` (DEOE)
- Factor structure: `analysis/exposure_factor_structure.py`
- Level-1 model: `analysis/level1_exposure_model.py`
- Data inventory: `docs/SQL_SERVER_SCHEMA.md`, `data/DATA_INDEX.md`
