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

## Units

- 1 Hartree = 27.2114 eV. `model_results.csv` / `model_summary.csv` are in
  Hartree; everything else is in eV. R² is unit-free and identical in both.
- Target: HOMO–LUMO gap. In the feature files it is `HL_Gap_eV`
  (mean 2.97 eV, std 0.91 eV, range 0.06–8.37 eV, N = 100,831).

## Known gaps in this bundle

- No per-metal results table (the leave-one-metal-out values in the README
  are read from `per_metal_r2.png`). A full `--seeds 3` rerun is in progress
  (2026-11-27) and will supply it.
- No xTB feature file (the `xTB_only` rows in `benchmark_results.csv` cannot
  be reproduced from the files here).
- `model_results_progress.csv` and `model_performance_summary.csv` from the
  upload are exact duplicates of `model_results.csv` and `model_summary.csv`
  and were not copied.
- `X_rdkit_FINAL.parquet(1)` in the upload is a duplicate of
  `X_rdkit_final.parquet` and was not copied.
- The large feature files (`.parquet`, `.csv` descriptors) are not in this
  repo — regenerate them with `run_pipeline.py`.
