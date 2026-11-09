#!/usr/bin/env python3
"""
Benchmarking the Out-of-Distribution Extrapolation Limits of 
Non-Spatial Machine Learning Models on the tmQM Dataset.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# --- Configuration ---
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_DIR = os.getcwd()
UNIT_FACTOR = 27.2114
SCALAR_FEATURES = ["Electronic_E", "Dispersion_E", "Dipole_M", "Metal_q", "Polarizability"]

def download_dataset():
    """Clones the tmQM repository if it doesn't exist (avoids deadlocks)."""
    if not os.path.exists("tmqm"):
        print("Cloning tmQM repository (2024 release)...")
        exit_code = os.system("git clone --depth 1 https://github.com/uiocompcat/tmqm.git tmqm")
        if exit_code != 0:
            raise RuntimeError("Failed to clone tmQM. Check your internet connection.")
        print("Download complete.")
    else:
        print("tmQM repository already exists.")

def load_and_clean_data():
    """Loads data, parses metals, converts units, and drops NaNs."""
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

    y = (df_clean["HL_Gap"] * UNIT_FACTOR).values
    metal = df_clean["Metal"].values
    X_scalar = df_clean[SCALAR_FEATURES].astype(np.float32)
    
    return df_clean, X_scalar, y, metal

def compute_fingerprints(df_clean):
    """Generates 256-bit Morgan Fingerprints efficiently in memory."""
    print("\nGenerating 256-bit Morgan Fingerprints...")
    fps, valid_idx = [], []
    for i, smi in enumerate(df_clean["SMILES"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        fps.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)))
        valid_idx.append(i)
        if (i + 1) % 20000 == 0:
            print(f"  scanned {i+1:,}/{len(df_clean):,}   kept {len(fps):,}")
            
    return pd.DataFrame(fps).astype(np.float32), valid_idx

def get_models():
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1, tree_method="hist"),
    }

def run_cv(split_label, feature_label, X, y_, cv, groups, results):
    """Runs cross-validation and strictly aligns indices to prevent crashes."""
    print(f"\n=== {split_label} | {feature_label} (n={len(y_):,}) ===")
    
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

def generate_figures(X_scalar, y_clean, metal_clean):
    """Generates all publication figures."""
    print("\nGenerating figures...")
    
    # 1. Gap Distribution
    plt.figure(figsize=(7, 4), dpi=300)
    plt.hist(y_clean, bins=60, color="teal", edgecolor="black", alpha=0.7)
    plt.xlabel("HOMO-LUMO Gap (eV)", fontweight='bold')
    plt.ylabel("Frequency", fontweight='bold')
    plt.title("Distribution of HOMO-LUMO Gaps (tmQM 2024)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "gap_distribution.png"), dpi=300)
    plt.close()
    print("  Saved gap_distribution.png")

    # 2. Parity Plot & SHAP
    X_train, X_test, y_train, y_test = train_test_split(X_scalar, y_clean, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(y_test, y_pred, s=10, alpha=0.3, edgecolors="none", color="#1f77b4")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal Fit")
    plt.text(0.05, 0.9, f"R² = {r2:.3f}", transform=plt.gca().transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor="white", edgecolor="black"))
    plt.xlabel("DFT HOMO-LUMO Gap (eV)", fontweight='bold')
    plt.ylabel("Predicted Gap (eV)", fontweight='bold')
    plt.title("Random Forest Parity Plot", fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "parity_plot_final.png"), dpi=300)
    plt.close()
    print("  Saved parity_plot_final.png")

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

    # 3. Applicability Domain (kNN)
    print("  Generating Applicability Domain plot...")
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)
    distances, _ = nn.kneighbors(X_test)
    errors = np.abs(y_test - y_pred)
    
    plt.figure(figsize=(7, 4), dpi=300)
    plt.scatter(distances.flatten(), errors, s=10, alpha=0.3, color="purple")
    plt.xlabel("Distance to Nearest Training Neighbor", fontweight='bold')
    plt.ylabel("Absolute Error (eV)", fontweight='bold')
    plt.title("Applicability Domain: Error vs Feature Distance", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "applicability_domain.png"), dpi=300)
    plt.close()
    print("  Saved applicability_domain.png")

    # 4. Per-Metal Breakdown (Using XGBoost for speed)
    print("  Generating Per-Metal Breakdown...")
    metal_results = []
    for m in sorted(set(metal_clean)):
        if m == "UNKNOWN": continue
        n = int((metal_clean == m).sum())
        if n < 200: continue
        
        tr = metal_clean != m
        te = metal_clean == m
        
        model = XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, tree_method="hist")
        model.fit(X_scalar[tr], y_clean[tr])
        yp = model.predict(X_scalar[te])
        r2_metal = r2_score(y_clean[te], yp)
        metal_results.append({"Metal": m, "R2": r2_metal, "N": n})

    metal_df = pd.DataFrame(metal_results).sort_values("R2")
    plt.figure(figsize=(10, 5), dpi=300)
    colors = ["#2ca02c" if v >= 0.3 else "#ff7f0e" if v >= 0 else "#d62728" for v in metal_df["R2"]]
    plt.bar(metal_df["Metal"], metal_df["R2"], color=colors, edgecolor="black")
    plt.axhline(0, color="gray", ls=":")
    plt.axhline(metal_df["R2"].median(), color="black", ls="--", label=f"Median R2 = {metal_df['R2'].median():.3f}")
    plt.ylabel("R² on Held-Out Metal", fontweight='bold')
    plt.xlabel("Held-Out Metal", fontweight='bold')
    plt.title("Leave-One-Metal-Out Performance (XGBoost)", fontweight='bold')
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "per_metal_r2.png"), dpi=300)
    plt.close()
    print("  Saved per_metal_r2.png")

def main():
    download_dataset()
    df_clean, X_scalar_full, y, metal = load_and_clean_data()
    
    X_morgan, desc_idx = compute_fingerprints(df_clean)
    y_clean = y[desc_idx]
    metal_clean = metal[desc_idx]
    X_scalar = X_scalar_full.iloc[desc_idx].reset_index(drop=True)
    
    print(f"\nAligned Dataset Size: {len(y_clean):,}")

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)

    run_cv("random", "5 Scalars", X_scalar, y_clean, kf, None, results)
    run_cv("random", "Morgan FP", X_morgan, y_clean, kf, None, results)

    ms = metal_clean != "UNKNOWN"
    run_cv("grouped", "5 Scalars", X_scalar[ms], y_clean[ms], gkf, metal_clean[ms], results)
    run_cv("grouped", "Morgan FP", X_morgan[ms], y_clean[ms], gkf, metal_clean[ms], results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(PROJECT_DIR, "results_final.csv"), index=False)
    print("\n--- FINAL RESULTS ---")
    # Fixed print format to avoid crashing on string columns
    print(results_df.round(4).to_string(index=False))
    print("\nSaved results_final.csv")
    
    generate_figures(X_scalar, y_clean, metal_clean)
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
