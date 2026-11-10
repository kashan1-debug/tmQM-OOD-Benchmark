#!/usr/bin/env python3
"""
Benchmarking the Out-of-Distribution Extrapolation Limits of 
Non-Spatial Machine Learning Models on the tmQM Dataset.
Main execution pipeline.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_DIR = os.getcwd()
UNIT_FACTOR = 27.2114
SCALAR_FEATURES = ["Electronic_E", "Dispersion_E", "Dipole_M", "Metal_q", "Polarizability"]

def download_dataset():
    if not os.path.exists("tmqm"):
        print("Cloning tmQM repository...")
        exit_code = os.system("git clone --depth 1 https://github.com/uiocompcat/tmqm.git tmqm")
        if exit_code != 0:
            raise RuntimeError("Failed to clone tmQM.")
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
    
    n_unknown = (df["Metal"] == "UNKNOWN").sum()
    n_total = len(df)
    print(f"  Total complexes: {n_total:,}")
    print(f"  Parsed metals:   {n_total - n_unknown:,} ({(n_total - n_unknown)/n_total*100:.1f}%)")
    print(f"  UNKNOWN (dropped for grouped CV): {n_unknown:,}")

    valid_mask = df[SCALAR_FEATURES].notna().all(axis=1) & df["SMILES"].notna() & df["HL_Gap"].notna()
    df_clean = df[valid_mask].reset_index(drop=True)

    y = (df_clean["HL_Gap"] * UNIT_FACTOR).values
    metal = df_clean["Metal"].values
    X_scalar_full = df_clean[SCALAR_FEATURES].astype(np.float32)
    
    return df_clean, X_scalar_full, y, metal

def compute_rdkit_descriptors(df_clean):
    print("\nComputing RDKit 2D descriptors...")
    desc_rows, desc_idx = [], []
    for i, smi in enumerate(df_clean["SMILES"].tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
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
            if all(v is not None and np.isfinite(v) for v in vals):
                desc_rows.append(vals)
                desc_idx.append(i)
        except:
            pass
        if (i + 1) % 20000 == 0:
            print(f"  scanned {i+1:,}/{len(df_clean):,}   kept {len(desc_rows):,}")
            
    return pd.DataFrame(desc_rows, index=desc_idx).astype(np.float32), desc_idx

def compute_fingerprints(df_clean):
    print("\nGenerating 256-bit Morgan Fingerprints...")
    fps, valid_idx = [], []
    for i, smi in enumerate(df_clean["SMILES"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        fps.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)))
        valid_idx.append(i)
        if (i + 1) % 20000 == 0:
            print(f"  scanned {i+1:,}/{len(df_clean):,}   kept {len(fps):,}")
            
    return pd.DataFrame(fps, index=valid_idx).astype(np.float32), valid_idx

def get_models():
    # Reviewer Point 7: n_jobs=1 inside models, n_jobs=-1 in cross_val_predict
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=1, tree_method="hist"),
    }

def run_cv(split_label, feature_label, X, y_, cv, groups, results):
    print(f"\n=== {split_label} | {feature_label} (n={len(y_):,}) ===")
    
    X_aligned = X.reset_index(drop=True)
    y_aligned = pd.Series(y_).reset_index(drop=True)
    groups_aligned = pd.Series(groups).reset_index(drop=True) if groups is not None else None

    if groups_aligned is not None:
        for train_idx, test_idx in cv.split(X_aligned, y_aligned, groups_aligned):
            assert set(groups_aligned.iloc[train_idx]).isdisjoint(set(groups_aligned.iloc[test_idx])), "Leakage!"

    for name, model in get_models().items():
        if groups_aligned is not None:
            yp = cross_val_predict(model, X_aligned, y_aligned, cv=cv, groups=groups_aligned, n_jobs=-1)
        else:
            yp = cross_val_predict(model, X_aligned, y_aligned, cv=cv, n_jobs=-1)
            
        r2 = r2_score(y_aligned, yp)
        rmse = root_mean_squared_error(y_aligned, yp)
        mae = mean_absolute_error(y_aligned, yp)
        
        results.append({"Split": split_label, "Features": feature_label, "Model": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"  {name:14s} R2={r2: .4f}  RMSE={rmse: .4f}  MAE={mae: .4f}")

def generate_figures(X_scalar, y_clean, metal_clean):
    print("\nGenerating figures...")
    
    plt.figure(figsize=(7, 4), dpi=300)
    plt.hist(y_clean, bins=60, color="teal", edgecolor="black", alpha=0.7)
    plt.xlabel("HOMO-LUMO Gap (eV)", fontweight='bold')
    plt.ylabel("Frequency", fontweight='bold')
    plt.title("Distribution of HOMO-LUMO Gaps (tmQM 2024)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "gap_distribution.png"), dpi=300)
    plt.close()

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

    print("  Generating SHAP values...")
    xgb = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1).fit(X_train, y_train)
    explainer = shap.TreeExplainer(xgb)
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(8, 5), dpi=300)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "shap_summary_final.png"), dpi=300)
    plt.close()

    print("  Generating Applicability Domain plot...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn = NearestNeighbors(n_neighbors=1).fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_test_scaled)
    errors = np.abs(y_test - y_pred)
    
    plt.figure(figsize=(7, 4), dpi=300)
    plt.scatter(distances.flatten(), errors, s=10, alpha=0.3, color="purple")
    plt.xlabel("Scaled Distance to Nearest Training Neighbor", fontweight='bold')
    plt.ylabel("Absolute Error (eV)", fontweight='bold')
    plt.title("Applicability Domain: Error vs Feature Distance", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "applicability_domain.png"), dpi=300)
    plt.close()

    print("  Generating Per-Metal Breakdown...")
    metal_results = []
    for m in sorted(set(metal_clean)):
        if m == "UNKNOWN": continue
        n = int((metal_clean == m).sum())
        if n < 200: continue
        
        tr = metal_clean != m
        te = metal_clean == m
        
        model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, tree_method="hist")
        model.fit(X_scalar[tr], y_clean[tr])
        yp = model.predict(X_scalar[te])
        
        mean_train = y_clean[tr].mean()
        baseline_rmse = root_mean_squared_error(y_clean[te], np.full_like(y_clean[te], mean_train))
        
        metal_results.append({
            "Metal": m, 
            "R2": r2_score(y_clean[te], yp), 
            "RMSE": root_mean_squared_error(y_clean[te], yp),
            "MAE": mean_absolute_error(y_clean[te], yp),
            "Baseline_RMSE": baseline_rmse,
            "N": n
        })

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

def main():
    download_dataset()
    df_clean, X_scalar_full, y, metal = load_and_clean_data()
    
    X_rdkit, desc_idx = compute_rdkit_descriptors(df_clean)
    X_morgan, fp_idx = compute_fingerprints(df_clean)
    
    common_idx = sorted(set(desc_idx) & set(fp_idx))
    print(f"\nCommon valid index size: {len(common_idx):,}")
    
    y_clean = y[common_idx]
    metal_clean = metal[common_idx]
    X_scalar = X_scalar_full.iloc[common_idx]
    X_rdkit = X_rdkit.loc[common_idx]
    X_morgan = X_morgan.loc[common_idx]
    
    print(f"Aligned Dataset Size: {len(y_clean):,}")

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)

    ms = metal_clean != "UNKNOWN"
    
    # ONLY run on the known-metal subset (fair comparison)
    run_cv("random_known", "5 Scalars", X_scalar[ms], y_clean[ms], kf, None, results)
    run_cv("random_known", "RDKit 2D", X_rdkit[ms], y_clean[ms], kf, None, results)
    run_cv("random_known", "Morgan FP", X_morgan[ms], y_clean[ms], kf, None, results)

    run_cv("grouped", "5 Scalars", X_scalar[ms], y_clean[ms], gkf, metal_clean[ms], results)
    run_cv("grouped", "RDKit 2D", X_rdkit[ms], y_clean[ms], gkf, metal_clean[ms], results)
    run_cv("grouped", "Morgan FP", X_morgan[ms], y_clean[ms], gkf, metal_clean[ms], results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(PROJECT_DIR, "results_final.csv"), index=False)
    
    print("\n--- FINAL RESULTS ---")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved results_final.csv")
    
    generate_figures(X_scalar, y_clean, metal_clean)
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
