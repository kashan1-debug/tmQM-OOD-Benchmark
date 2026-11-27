# tmQM benchmark — verification report (2026-11-27)

What was checked in the uploaded bundle (`tmQM_Research` zip, 28 files), what
matches, and what does not. The main README now reports only numbers these
files support.

## Files in the bundle

- Feature files (parquet, verified shapes with the project venv):
  - `X_scalar_FINAL.parquet` — (100,831 × 5): `Electronic_E`,
    `Dispersion_E`, `Dipole_M`, `Metal_q`, `Polarizability`.
  - `X_morgan.parquet` — (100,831 × 1024): Morgan fingerprints, 1024 bits.
  - `X_rdkit_final.parquet` — (100,831 × 14): RDKit 2D descriptors.
  - `y_FINAL.parquet` — (100,831 × 1): `HL_Gap_eV`, mean 2.97 eV, std 0.91
    eV, min 0.06, max 8.37 eV.
  - `X_rdkit.csv` — same 14 RDKit descriptors as a CSV (8.7 MB).
  - `X_rdkit_FINAL.parquet(1)` — duplicate of `X_rdkit_final.parquet`.
- Result tables (CSV): `benchmark_results.csv` (24 rows),
  `benchmark_summary.csv` (24 rows, same numbers, one row per run),
  `results_cv.csv` (16 rows), `model_results.csv` (12 rows),
  `model_results_progress.csv` (duplicate of `model_results.csv`),
  `model_summary.csv` = `model_performance_summary.csv` (same 4-row table),
  `feature_importance.csv` (5 rows), `rf_predictions.csv` (21,709 rows).
- Figures (PNG, 13): `model_comparison_r2.png`,
  `model_comparison_rmse.png`, `parity_plot.png`,
  `parity_plot_final.png`, `per_metal_r2.png`, `shap_summary.png`,
  `shap_summary_final.png`, `feature_importance.png`,
  `actual_vs_predicted.png`, `applicability_domain.png`,
  `correlation.png`, `correlation_heatmap.png`, `gap_distribution.png`,
  `target_distribution.png`.

## Checks that pass

- `benchmark_results.csv` is internally consistent: 3 seeds × 2 feature sets
  × 4 models = 24 rows; train 86,832 + test 21,709 = 108,541; seed-to-seed
  variation is small (RF R² 0.322–0.336 DFT-assisted).
- `model_results.csv` has exactly the same R² values as
  `benchmark_results.csv` (RF 0.3346, XGB 0.2739, …) with RMSE/MAE scaled by
  1/27.21 — i.e. the same runs with the target in Hartree. Not a
  contradiction, but the two unit systems must not be mixed.
- `results_cv.csv` random rows agree with `benchmark_results.csv` to ~0.005
  (5 scalars RF 0.330 vs 0.334; RDKit RF 0.379).
- `model_comparison_r2.png` matches `benchmark_results.csv` (Dummy 0.00,
  Ridge ≈ 0.083, Random Forest ≈ 0.331, XGBoost ≈ 0.269).
- `per_metal_r2.png` matches the per-metal values quoted in the old README
  (Hg ≈ −0.77, Ag ≈ −0.70, Mn ≈ 0.35, median −0.048, 29 metals shown).
- `feature_importance.csv` sums to 1.0 and its ranking (Metal_q first) is
  consistent with the SHAP figure's story.

## Mismatches found — the old README's tables do not come from these files

The previous README reported a random-CV table (5 scalars RF 0.2983, RDKit RF
0.3074, Morgan RF 0.3860) and a GroupKFold table (5 scalars RF 0.0946, RDKit
RF 0.1425, Morgan RF 0.1836 / XGB 0.2009). **None of those numbers appear in
this bundle.** The closest files give 0.330 / 0.379 (random) and 0.070 / 0.149
(grouped). The old numbers come from different runs, and the Morgan numbers
have no source table at all in the bundle. The new README therefore reports
only the numbers in `results/` and drops the Morgan table.

## What is missing from the bundle

- Morgan-fingerprint results table (features exist, results do not).
- Per-metal results table (only the figure survives).
- xTB feature file (the `xTB_only` rows cannot be reproduced from here).
- Leave-one-metal-out results table (the old audit noted a 16-metal file;
  the figure here shows 29 metals).
- The pipeline notebook / exact split definitions for `results_cv.csv`
  (whether "random" is 5-fold CV or a single split is not recorded in the
  file).

## Code audit — `run_pipeline.py` (287 lines, checked 2026-11-27)

The code is clean and readable, it compiles, and the grouped-CV leakage
assertion (`train`/`test` metal sets must be disjoint) is genuinely good.
But **it does not reproduce the tables in `results/`**. Three concrete
mismatches:

1. **It writes a file that does not exist.** `main()` runs random + grouped
   5-fold CV and saves `results_final.csv`. Neither the repo nor the uploaded
   bundle contains `results_final.csv`. Its split labels would be
   `random_known` / `grouped`, while `results_cv.csv` says `random` /
   `grouped` and reports N = 108,541 for the 5 scalars — this pipeline aligns
   all three feature sets on the common valid index first (≈ 100,831 rows),
   so it can never report N = 108,541.
2. **There is no xTB in the code.** Nothing computes or loads xTB features,
   so the `xTB_only` rows in `benchmark_results.csv` (RF −0.20) come from a
   different script that is not in this repo. Same for the 80/20 splits with
   seeds 42 / 123 / 2024: the only 80/20 split in this code is inside
   `generate_figures()` for the parity plot, on the aligned set — which would
   give a 20,167-row test set, not the 21,709 rows in `rf_predictions.csv`.
3. **Morgan fingerprints are 256-bit here, 1024-bit in the bundle.**
   `compute_fingerprints()` uses `nBits=256`; the surviving
   `X_morgan.parquet` has 1024 columns. Different features, different
   numbers — another reason the old README's Morgan table cannot be traced to
   this code.

Smaller findings:

- `requirements.txt` said `scikit-learn>=1.3.0`, but the code imports
  `root_mean_squared_error`, which only exists from scikit-learn 1.4. Fixed
  to `>=1.4.0`.
- `seaborn` is listed in `requirements.txt` but imported nowhere in the
  pipeline. Harmless; left as is.
- `generate_figures()` trains the per-metal LOMO models with 100 XGBoost
  trees while `get_models()` uses 200 — the figure and the CV table use
  different model settings. The figure's values still match the bundle.
- `extract_metal()` (regex over SMILES) is fragile — it returns the first
  metal-like token anywhere in the string — but no wrong-metal case was found
  in the verified files. Rows it cannot parse become `UNKNOWN` and are
  dropped from the grouped runs, which is the honest handling.
- The `tmqm/tmQM/tmQM_y.csv` input path (after cloning
  `github.com/uiocompcat/tmqm`) was not re-verified by cloning here; it
  matches the path used in the original runs.

The README's "Reproducing the results" section now says this plainly instead
of claiming the pipeline produced the tables.

## Scientific reading (unchanged)

- The 5 DFT scalars come from the same TPSSh calculation as the gap. Using
  them is a DFT-to-DFT mapping, not screening from structure alone. The
  feature importances (all five scalars, nothing else) show the model leans
  on them entirely.
- Even with those oracle features, the best random-split R² is 0.33–0.38 and
  the grouped R² is 0.07–0.15. xTB-only features fail (RF −0.20).
  Leave-one-metal-out median is −0.048.
- Honest conclusion: the tmQM HOMO–LUMO gap is not predictable from these
  inputs, and it does not transfer to unseen metals. That is the benchmark's
  result — a negative result, recorded as such, not a paper.
