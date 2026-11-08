# Quantifying the Applicability Domain of Non-Spatial ML Models on tmQM

This repository contains the data preprocessing, model training, and evaluation pipeline for our study on the 2024 release of the tmQM dataset. 

**Core Finding:** We demonstrate that non-spatial machine learning models (global scalars and 2D descriptors) reach an empirical ceiling of \(R^2 \approx 0.31\) under random cross-validation. Crucially, when evaluated on unseen transition metals via GroupKFold validation, performance catastrophically collapses, proving these models have near-zero extrapolative power.

## Dataset
- **Source:** [tmQM 2024 release](https://github.com/uiocompcat/tmqm) — `tmQM/tmQM_y.csv`
- **Complexes:** 108,541
- **Unit Conversion:** The target `HL_Gap` is stored in Hartree. We apply the standard 27.2114 conversion factor to eV.

## Benchmark Results (Aligned Data, N = 100,831)

**Random 5-Fold Cross-Validation (In-Distribution)**
| Features | Model | R² | RMSE (eV) |
|---|---|---|---|
| 5 Scalars | Random Forest | **0.2983** | 0.7595 |
| RDKit 2D | Random Forest | **0.3074** | 0.7545 |

**GroupKFold Validation (Out-of-Distribution by Metal)**
| Features | Model | R² | RMSE (eV) |
|---|---|---|---|
| 5 Scalars | Random Forest | **0.0946** | 0.8627 |
| RDKit 2D | Random Forest | **0.1425** | 0.8396 |

## Reproducing the Results
```bash
pip install -r requirements.txt
python run_pipeline.py
