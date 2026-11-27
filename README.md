# Benchmarking the Out-of-Distribution Extrapolation Limits of Non-Spatial ML Models on tmQM

> Predicting the HOMO–LUMO gap of transition-metal complexes from composition alone — and where that fails.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Status:** benchmark archive. The results below are the verified numbers from the
runs in `results/`. There is no paper in preparation from this project — the work
is kept here as a complete, honest benchmark, including its negative result.

## Core finding

Non-spatial models barely predict the tmQM HOMO–LUMO gap, even when they are
given help. The best model (Random Forest) reaches **R² ≈ 0.33–0.38 on a random
split** and collapses to **R² ≈ 0.07–0.15 on grouped (by-metal) splits**. Under
leave-one-metal-out, the median metal scores **R² = −0.048** — worse than
predicting the mean. Cheap xTB-only features fail outright (R² ≈ 0.03, Random
Forest −0.20). The gap is not predictable from these inputs, and it does not
transfer to unseen metals.

## Read this first — what the numbers do and do not mean

- **The 5 "DFT scalars" are oracle features, not screening features.**
  `Electronic_E`, `Dispersion_E`, `Dipole_M`, `Metal_q` (and `Polarizability`)
  come from the same TPSSh calculation as the HOMO–LUMO gap itself. A model
  using them is a DFT-to-DFT mapping: it cannot be used to screen new complexes
  without first running the expensive calculation it is supposed to replace.
  The feature-importance file proves the point — the model leans almost
  entirely on these five numbers (see below).
- **The honest screening numbers are the structure-only ones** (RDKit 2D
  descriptors, Morgan fingerprints). They are no better: RDKit 2D Random
  Forest reaches R² 0.379 random / 0.149 grouped.
- **Morgan-fingerprint results are not in this bundle.** An earlier version of
  this README quoted Morgan FP numbers (R² 0.386 random / 0.201 grouped), but
  no Morgan results table survives in the uploaded files — only the feature
  file (`X_morgan.parquet`, 1024 bits, not included here). Those numbers are
  therefore **not** repeated below. If the Morgan results table turns up, it
  can be added back with its source file.
- **Two unit systems are mixed in the files.** `benchmark_results.csv` and
  `results_cv.csv` report errors in **eV**; `model_results.csv` reports the
  same runs with the target in **Hartree** (RMSE ≈ 0.027 Ha ≈ 0.75 eV).
  R² is identical in both; only the error scale differs.

## Dataset and features

- **Source:** tmQM dataset, HOMO–LUMO gap (`HL_Gap`) from TPSSh calculations.
- **Complexes:** 108,541 in total; **100,831** with valid SMILES (the rest
  drop out of any SMILES-based featurisation, which is why the N differs
  between runs below).
- **Target:** `HL_Gap_eV` = `HL_Gap` × 27.2114. Mean 2.97 eV, std 0.91 eV,
  range 0.06–8.37 eV (N = 100,831).
- **Feature sets:**
  - 5 DFT scalars: `Electronic_E`, `Dispersion_E`, `Dipole_M`, `Metal_q`,
    `Polarizability` (same calculation as the target — see the caveat above).
  - RDKit 2D: 14 descriptors (MolWt, LogP, TPSA, …).
  - Morgan fingerprints: 1024 bits (feature file only; no results table here).
  - xTB-only: as recorded in `benchmark_results.csv`; its feature file is not
    in this bundle.

## Benchmark results

### Random 80/20 splits, seeds 42 / 123 / 2024 (N = 108,541; train 86,832, test 21,709)

From `results/benchmark_results.csv` (seed 42 shown; the range over the three
seeds is in brackets).

| Features | Model | R² | RMSE (eV) | MAE (eV) |
|---|---|---|---|---|
| DFT-assisted (5 scalars) | Dummy (mean) | 0.000 | 0.916 | 0.708 |
| DFT-assisted (5 scalars) | Ridge | 0.086 | 0.876 | 0.671 |
| DFT-assisted (5 scalars) | Random Forest | **0.334** [0.322–0.336] | 0.748 | 0.560 |
| DFT-assisted (5 scalars) | XGBoost | 0.274 [0.263–0.277] | 0.781 | 0.600 |
| xTB-only | Dummy (mean) | 0.000 | 0.916 | 0.708 |
| xTB-only | Ridge | 0.019 | 0.908 | 0.703 |
| xTB-only | Random Forest | −0.200 | 1.004 | 0.781 |
| xTB-only | XGBoost | 0.031 | 0.902 | 0.698 |

### Cross-validation summary (from `results/results_cv.csv`)

| Split | Features | Model | N | R² | RMSE (eV) |
|---|---|---|---|---|---|
| random | 5 scalars | Ridge | 108,541 | 0.082 | 0.875 |
| random | 5 scalars | Random Forest | 108,541 | 0.330 | 0.748 |
| random | 5 scalars | XGBoost | 108,541 | 0.271 | 0.780 |
| random | RDKit 2D | Ridge | 100,831 | 0.116 | 0.852 |
| random | RDKit 2D | Random Forest | 100,831 | 0.379 | 0.715 |
| random | RDKit 2D | XGBoost | 100,831 | 0.271 | 0.774 |
| grouped (by metal) | 5 scalars | Ridge | 100,849 | 0.032 | 0.892 |
| grouped (by metal) | 5 scalars | Random Forest | 100,849 | 0.070 | 0.874 |
| grouped (by metal) | 5 scalars | XGBoost | 100,849 | 0.126 | 0.848 |
| grouped (by metal) | RDKit 2D | Ridge | 100,831 | 0.056 | 0.881 |
| grouped (by metal) | RDKit 2D | Random Forest | 100,831 | 0.149 | 0.836 |
| grouped (by metal) | RDKit 2D | XGBoost | 100,831 | 0.149 | 0.837 |

Note the N differences: the 5-scalar runs use all 108,541 rows (scalars need no
SMILES); RDKit runs use the 100,831 rows with valid SMILES. The grouped
5-scalar N (100,849) differs slightly from the random N — the runs come from
different pipeline stages, so treat the two tables as two recorded runs, not
one perfectly matched experiment. See `docs/tmqm_benchmark_report.md`.

### Leave-one-metal-out (XGBoost, from `per_metal_r2.png`)

Median R² over held-out metals: **−0.048**. Worst: Hg ≈ −0.77, Ag ≈ −0.70,
Au ≈ −0.56. Best: Mn ≈ 0.35. These values are read from the figure — no
per-metal results table survives in this bundle.

### Feature importance (Random Forest, DFT-assisted)

From `results/feature_importance.csv` — the five scalars take all of the
credit, which is the oracle problem in one table:

| Feature | Importance |
|---|---|
| Metal_q | 0.265 |
| Electronic_E | 0.212 |
| Polarizability | 0.185 |
| Dipole_M | 0.177 |
| Dispersion_E | 0.161 |

## Figures

### 1. Gap distribution

![Gap Distribution](gap_distribution.png)

### 2. Random Forest parity plot

*The model fails to predict extreme gaps (> 5 eV) and collapses to the mean.*

![Parity Plot](parity_plot_final.png)

### 3. SHAP feature importance

*SHAP confirms the model relies on global atomic properties rather than local
geometry.*

![SHAP Summary](shap_summary_final.png)

### 4. Applicability domain

*Error increases significantly as test molecules move further from the training
manifold (scaled kNN distance).*

![Applicability Domain](applicability_domain.png)

### 5. Leave-one-metal-out analysis

*The model catastrophically fails on unseen metals like Hg (R² = −0.77) and Ag
(R² = −0.70), while maintaining some accuracy on Mn (R² = 0.35).*

![Per-Metal Breakdown](per_metal_r2.png)

## Reproducing the results

```bash
pip install -r requirements.txt
python run_pipeline.py
```

The pipeline in `run_pipeline.py` is the code that produced these runs. The
large feature files (parquet) are not in this repo — regenerate them with the
pipeline.

## Files

```
├── run_pipeline.py          # Pipeline code
├── requirements.txt
├── results/
│   ├── benchmark_results.csv       # 80/20 splits, seeds 42/123/2024, eV
│   ├── results_cv.csv              # Random + grouped CV summary, eV
│   ├── model_results.csv           # Same runs as benchmark_results, in Hartree
│   ├── model_summary.csv           # Mean ± std over seeds
│   ├── feature_importance.csv      # RF importances (5 DFT scalars)
│   ├── rf_predictions.csv          # RF actual vs predicted, eV (test split)
│   └── README.md                   # Guide to the files and their units
└── docs/
    └── tmqm_benchmark_report.md    # Verification report: what was checked,
                                    # what matches, what is missing
```

## Author

**Muhammad Kashan Ali** — Department of Chemistry, Bahauddin Zakariya
University, Multan, Pakistan.

## License

MIT — see [LICENSE](LICENSE).
