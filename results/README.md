# Results files — guide

All files in this folder are the verified result tables from the tmQM runs
(uploaded 2026-11-27). Every number in the main README comes from these files.

| File | Rows | What it is |
|---|---|---|
| `benchmark_results.csv` | 24 + header | 80/20 random splits, seeds 42 / 123 / 2024. Feature sets `DFT_assisted` and `xTB_only`; models Dummy / Ridge / RandomForest / XGBoost. Errors in **eV**. Train 86,832 / test 21,709. |
| `results_cv.csv` | 16 + header | Older cross-validation summary: `random` and `grouped` (by metal) splits × `5 Scalars` / `RDKit 2D` × 4 models. Errors in **eV**. A different run from `results_cv_reproduced.csv` — do not mix them. |
| `results_cv_reproduced.csv` | 24 + header | 5-fold CV rerun of 2026-11-27 (pasted run output): `random_known` and `grouped` × `5 Scalars` / `RDKit 2D` / `Morgan FP (256)` × Mean baseline / Ridge / Random Forest / XGBoost. Aligned set, 100,831 rows. Errors in **eV**. This is the run that reproduces the old README's numbers. |
| `results_3d_test.csv` | 24 + header | 3D geometry test of 2026-11-27: 2 seeds (0 = Colab run, 1 = independent reproduction) × `random` / `grouped` × `3D only` / `Morgan FP (256)` / `Morgan + 3D` × Random Forest / XGBoost. Random 5,000-complex samples (N = 4,640 / 4,634 after SMILES filtering). Errors in **eV**. Grouped 3D R² 0.03–0.16: 3D does not fix transfer. |
| `model_results.csv` | 12 + header | The same runs as `benchmark_results.csv` (same R²), but with the target in **Hartree** — so RMSE ≈ 0.027 Ha and MAE ≈ 0.021 Ha instead of ≈ 0.75 / 0.56 eV. Do not mix the two unit systems. |
| `model_summary.csv` | 4 + header | Mean ± std of `model_results.csv` over the three seeds (Hartree units). |
| `feature_importance.csv` | 5 + header | Random Forest importances for the 5 DFT scalars. They sum to 1.0. |
| `rf_predictions.csv` | 21,709 + header | Random Forest actual vs predicted gap, in **eV**, for the test split. |
| `results_all_runs.csv` | 72 + header | Final 3-seed CV run of 2026-11-27 (Colab): `random_known` and `grouped` × `5 Scalars` / `RDKit 2D` / `Morgan FP (256)` × Mean baseline / Ridge / Random Forest / XGBoost × seeds 0–2. Errors in **eV**. |
| `results_final.csv` | 24 + header | Mean ± std over the 3 seeds of `results_all_runs.csv`. Errors in **eV**. |
| `results_table.md` | — | The same 3-seed summary as Markdown. |
| `per_metal_results.csv` | 120 + header | Leave-one-metal-out, final run of 2026-11-27: 30 metals × Mean baseline / Ridge / Random Forest / XGBoost, 5-scalar features only, min N = 100. Errors in **eV**. Median R²: Random Forest −0.025, XGBoost −0.012. |
| `metal_distribution_check.csv` | 30 + header | Metal counts before/after SMILES filtering for the final run (La retained only 27.3%). |

## Units

- 1 Hartree = 27.2114 eV. `model_results.csv` / `model_summary.csv` are in
  Hartree; everything else is in eV. R² is unit-free and identical in both.
- Target: HOMO–LUMO gap. In the feature files it is `HL_Gap_eV`
  (mean 2.97 eV, std 0.91 eV, range 0.06–8.37 eV, N = 100,831).

## Known gaps in this bundle

- Per-metal results table supplied 2026-11-27 (`per_metal_results.csv`, 120 rows). The parity figure (`parity_plot_final.png`) was not recovered — the seed-0 out-of-fold predictions it needs were never saved.
- No xTB feature file (the `xTB_only` rows in `benchmark_results.csv` cannot
  be reproduced from the files here).
- `model_results_progress.csv` and `model_performance_summary.csv` from the
  upload are exact duplicates of `model_results.csv` and `model_summary.csv`
  and were not copied.
- `X_rdkit_FINAL.parquet(1)` in the upload is a duplicate of
  `X_rdkit_final.parquet` and was not copied.
- The large feature files (`.parquet`, `.csv` descriptors) are not in this
  repo — regenerate them with `run_pipeline.py`.
