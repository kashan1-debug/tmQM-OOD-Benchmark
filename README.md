# Cautionary Note on the tmQM Dataset: Unit Mislabeling and the Catastrophic Out-of-Distribution Failure of Non-Spatial Models

This repository contains the dataset preprocessing, model training, and evaluation pipeline for our study on the 2024 release of the tmQM (Transition Metal Quantum Mechanics) dataset. 

**Critical Finding:** We demonstrate that recent literature has significantly inflated the performance of non-spatial machine learning models on this dataset due to target leakage and a unit mislabeling error in the public data file. We provide the corrected, leak-free baseline and expose the catastrophic out-of-distribution (OOD) failure of these models.

## Dataset

- **Source:** [tmQM 2024 release](https://github.com/uiocompcat/tmqm) — `tmQM/tmQM_y.csv`
- **Complexes:** 108,541 mononuclear transition metal complexes
- **Target:** `HL_Gap` (HOMO–LUMO gap)

### ⚠️ Critical Unit Correction
The target variable `HL_Gap` in the public `tmQM_y.csv` is stored in **Hartree**, not electron-volts (eV). Applying a conversion factor of **27.2114** yields physically realistic gaps (median = 2.97 eV, max = 8.36 eV). Failing to apply this conversion results in an artificially deflated Root Mean Square Error (RMSE) by a factor of 27.2. 

### Feature Sets
To ensure a completely fair comparison, all models were evaluated on a strictly aligned dataset of $N = 100,831$ valid complexes. We benchmarked two feature sets:
1. **5 Scalars:** `Electronic_E`, `Dispersion_E`, `Dipole_M`, `Metal_q`, `Polarizability`.
2. **RDKit 2D:** 14 topological descriptors (e.g., MolWt, LogP, TPSA). Gasteiger charge descriptors were excluded due to incompatibility with transition metals.

## Benchmark Results (Aligned Data)

**Random 5-Fold Cross-Validation (In-Distribution)**
| Features | Model | $R^2$ | RMSE (eV) |
|---|---|---|---|
| 5 Scalars | Random Forest | **0.2983** | 0.7595 |
| 5 Scalars | XGBoost | 0.2712 | 0.7740 |
| RDKit 2D | Random Forest | **0.3074** | 0.7545 |
| RDKit 2D | XGBoost | 0.2702 | 0.7746 |

**GroupKFold Validation (Out-of-Distribution by Metal)**
| Features | Model | $R^2$ | RMSE (eV) |
|---|---|---|---|
| 5 Scalars | Random Forest | **0.0946** | 0.8627 |
| 5 Scalars | XGBoost | 0.1183 | 0.8514 |
| RDKit 2D | Random Forest | **0.1425** | 0.8396 |
| RDKit 2D | XGBoost | 0.1465 | 0.8377 |

### Key Findings
- **Information Ceiling:** Global scalar descriptors and 14 RDKit descriptors reach an empirical ceiling of $R^2 \approx 0.31$.
- **OOD Collapse:** When tested on unseen metals (GroupKFold), model performance catastrophically collapses, proving non-spatial models have near-zero extrapolative power.
- **SHAP Analysis:** Feature importance confirms the models rely on `Polarizability` and `Electronic_E`, exploiting global atomic properties rather than true spatial topology.

## Repository Structure
