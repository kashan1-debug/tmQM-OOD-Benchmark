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
