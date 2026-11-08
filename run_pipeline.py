#!/usr/bin/env python3
"""
Quantifying the Applicability Domain of Non-Spatial ML Models on tmQM.
Main execution pipeline.

Fully self-contained. Clones data, preprocesses, evaluates, and generates figures.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import KFold, GroupKFold, cross_val_predict, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

# --- Configuration ---
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_DIR = os.getcwd()
UNIT_FACTOR = 27.2114
SCALAR_FEATURES = ["Electronic_E", "Dispersion_E", "Dipole_M", "Metal_q", "Polarizability"]

def download_dataset():
    if not os.path.exists("tmqm"):
        print("Cloning tmQM repository (2024 release)...")
        # Using os.system guarantees the progress bar streams to the console (no deadlocks)
        exit_code = os.system("git clone --depth 1 https://github.com/uiocompcat/tmqm.git tmqm")
        if exit_code != 0:
            raise RuntimeError("Failed to clone tmQM. Check your internet connection.")
        print("Download complete.")
    else:
        print("tmQM repository already exists.")

def load_and_clean_data():
    print("\nLoading and cleaning dataset...")
    df = pd.read_csv("tmqm/tmQM/tmQM_y.csv", sep=";")
    
    def extract_metal(smi):
        if not isinstance(smi, str): return "UNKNOWN"
        METALS = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}
        for m in re.finditer(r"\[([A-Z][a-z]?)", smi):
            if m.group(1) in METALS: return m.group(1)
        for m in re.finditer(r"[A-Z][a-z]?", smi):
            if m.group(0) in METALS: return m.group(0)
        return "UNKNOWN"

    df["Metal"] = df["SMILES"].map(extract_metal)
    valid_mask = df[SCALAR_FEATURES].notna().all(axis=1) & df["SMILES"].notna() & df["HL_Gap"].notna()
    df_clean = df[valid_mask].reset_index(drop=True)

    y = df_clean["HL_Gap"] * UNIT_FACTOR
    metal = df_clean["Metal"].values
    X_scalar_full = df_clean[SCALAR_FEATURES].astype(float)
    
    return df_clean, X_scalar_full, y, metal

def compute_rdkit_descriptors(df_clean):
    def smiles_to_desc(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        try:
            vals = [
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol), mol.GetNumHeavyAtoms(),
                Descriptors.FractionCSP3(mol), Descriptors.NumValenceElectrons(mol),
                rdMolDescriptors.CalcNumHeteroatoms(mol), rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol)
            ]
            for v in vals:
                if v is None or not np.isfinite(v): return None
            return vals
        except:
            return None

    print("\nComputing RDKit descriptors (~10 mins)...")
    desc_rows, desc_idx = [], []
    for i, smi in enumerate(df_clean["SMILES"].tolist()):
        d = smiles_to_desc(smi)
        if d is not None:
            desc_rows.append(d)
            desc_idx.append(i)
        if (i + 1) % 20000 == 0:
            print(f"  scanned {i+1:,}/{len(df_clean):,}   kept {len(desc_rows):,}")
            
    return pd.DataFrame(desc_rows).astype(float), desc_idx

def get_models():
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1, tree_method="hist"),
    }

def run_cv(split_label, feature_label, X, y_, cv, groups, results):
    print(f"\n=== {split_label} | {feature_label} (n={len(y_):,}) ===")
    
    # FORCE STRICT ALIGNMENT: Reset index to prevent scikit-learn indexing errors
    X_aligned = X.reset_index(drop=True)
    y_aligned = pd.Series(y_).reset_index(drop=True)
    groups_aligned = pd.Series(groups).reset_index(drop=True) if groups is not None else None

    for name, model in get_models().items():
        if groups_aligned is not None:
            yp = cross_val_predict(model, X_aligned, y_aligned, cv=cv, groups=groups_aligned, n_jobs=-1)
        else:
            yp = cross_val_predict(model, X_aligned, y_aligned, cv=cv, n_jobs=-1)
            
        r2, rmse = r2_score(y_aligned, yp), root_mean_squared_error(y_aligned, yp)
        results.append({"Split": split_label, "Features": feature_label, "Model": name, "R2": r2, "RMSE": rmse})
        print(f"  {name:14s} R2={r2: .4f}  RMSE={rmse: .4f}")

def generate_figures(X_scalar, y_clean):
    print("\nGenerating figures...")
    
    # 1. Gap Distribution Plot
    plt.figure(figsize=(7, 4), dpi=300)
    plt.hist(y_clean, bins=60, color="teal", edgecolor="black", alpha=0.7)
    plt.xlabel("HOMO-LUMO Gap (eV)", fontweight='bold')
    plt.ylabel("Frequency", fontweight='bold')
    plt.title("Distribution of HOMO-LUMO Gaps (tmQM 2024)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "gap_distribution.png"), dpi=300)
    plt.close()
    print("  Saved gap_distribution.png")

    # 2. Parity Plot
    X_train, X_test, y_train, y_test = train_test_split(X_scalar, y_clean, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(y_test, y_pred, s=10, alpha=0.3, edgecolors="none", color="#1f77b4")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal Fit")
    plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes,
             fontsize=11, fontweight='bold', bbox=dict(facecolor="white", edgecolor="black"))
    plt.xlabel("DFT HOMO-LUMO Gap (eV)", fontweight='bold')
    plt.ylabel("Predicted Gap (eV)", fontweight='bold')
    plt.title("Random Forest Parity Plot", fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "parity_plot_final.png"), dpi=300)
    plt.close()
    print("  Saved parity_plot_final.png")

    # 3. SHAP Plot
    print("  Generating SHAP values...")
    xgb = XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_train, y_train)
    explainer = shap.TreeExplainer(xgb)
    X_sample = X_test.sample(n=1000, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(8, 5), dpi=300)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "shap_summary_final.png"), dpi=300)
    plt.close()
    print("  Saved shap_summary_final.png")

def main():
    download_dataset()
    df_clean, X_scalar_full, y, metal = load_and_clean_data()
    
    X_rdkit, desc_idx = compute_rdkit_descriptors(df_clean)
    
    # STRICT ALIGNMENT
    y_clean = y.loc[desc_idx].reset_index(drop=True)
    metal_clean = pd.Series(metal).loc[desc_idx].reset_index(drop=True)
    X_scalar = X_scalar_full.loc[desc_idx].reset_index(drop=True)
    
    print(f"\nAligned Dataset Size: {len(y_clean):,}")

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)

    run_cv("random", "5 Scalars", X_scalar, y_clean, kf, None, results)
    run_cv("random", "RDKit 2D", X_rdkit, y_clean, kf, None, results)

    # EXPLICIT MASK FOR GROUPED CV
    ms = metal_clean != "UNKNOWN"
    run_cv("grouped", "5 Scalars", X_scalar[ms].reset_index(drop=True), y_clean[ms].reset_index(drop=True), gkf, metal_clean[ms].reset_index(drop=True), results)
    run_cv("grouped", "RDKit 2D", X_rdkit[ms].reset_index(drop=True), y_clean[ms].reset_index(drop=True), gkf, metal_clean[ms].reset_index(drop=True), results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(PROJECT_DIR, "results_final.csv"), index=False)
    
    print("\n--- FINAL RESULTS ---")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved results_final.csv")
    
    generate_figures(X_scalar, y_clean)
    
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
