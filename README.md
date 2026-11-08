# Quantifying the Applicability Domain of Non-Spatial ML Models on tmQM

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Citation
If you use this code or the corrected data alignment in your research, please cite our paper:

> **Kashan.** *Quantifying the Applicability Domain of Non-Spatial Machine Learning Models for HOMO–LUMO Gap Prediction on the tmQM Dataset.* (in preparation, 2024).

---

## 📌 Overview
This repository contains the data preprocessing, model training, and evaluation pipeline for our study on the 2024 release of the tmQM dataset. 

**Core Finding:** We demonstrate that non-spatial machine learning models (global scalars and 2D descriptors) reach an empirical ceiling of \(R^2 \approx 0.31\) under random cross-validation. Crucially, when evaluated on unseen transition metals via GroupKFold validation, performance catastrophically collapses, proving these models have near-zero extrapolative power.

## 📊 Dataset & Features
- **Source:** [tmQM 2024 release](https://github.com/uiocompcat/tmqm) — `tmQM/tmQM_y.csv`
- **Complexes:** 108,541
- **Unit Conversion:** The target `HL_Gap` is stored in Hartree. To ensure physical interpretability, we apply the standard **27.2114** conversion factor to eV (yielding a median gap of 2.97 eV).
- **Aligned Dataset:** To ensure a fair comparison, all models were evaluated on exactly 100,831 complexes (removing those without valid SMILES).

**Feature Sets Benchmarked:**
1. **5 Scalars:** `Electronic_E`, `Dispersion_E`, `Dipole_M`, `Metal_q`, `Polarizability`.
2. **RDKit 2D:** 14 topological descriptors (e.g., MolWt, LogP, TPSA).

## 🏆 Benchmark Results

**Random 5-Fold Cross-Validation (In-Distribution)**
| Features | Model | R² | RMSE (eV) |
|---|---|---|---|
| 5 Scalars | Random Forest | **0.2983** | 0.7595 |
| 5 Scalars | XGBoost | 0.2712 | 0.7740 |
| RDKit 2D | Random Forest | **0.3074** | 0.7545 |
| RDKit 2D | XGBoost | 0.2702 | 0.7746 |

**GroupKFold Validation (Out-of-Distribution by Metal)**
| Features | Model | R² | RMSE (eV) |
|---|---|---|---|
| 5 Scalars | Random Forest | **0.0946** | 0.8627 |
| 5 Scalars | XGBoost | 0.1183 | 0.8514 |
| RDKit 2D | Random Forest | **0.1425** | 0.8396 |
| RDKit 2D | XGBoost | 0.1465 | 0.8377 |

## 📂 Repository Structure
```text
├── gap_distribution.png          # Corrected HOMO-LUMO gap distribution (eV)
├── parity_plot_final.png         # Random Forest parity plot (R² = 0.305)
├── shap_summary_final.png        # SHAP summary plot for 5-scalar model
├── run_pipeline.py               # Main execution pipeline
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md
