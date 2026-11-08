#!/usr/bin/env python3
"""
Quantifying the Applicability Domain of Non-Spatial ML Models on tmQM.
Main execution pipeline.
"""

import os
import re
import warnings
import subprocess
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

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_DIR = "./"
UNIT_FACTOR = 27.2114
SCALAR_FEATURES = ["Electronic_E", "Dispersion_E", "Dipole_M", "Metal_q", "Polarizability"]

def clone_dataset():
    if not os.path.exists("tmqm"):
        print("Cloning tmQM repository...")
        result = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/uiocompcat/tmqm.git", "tmqm"])
        if result.returncode != 0:
            raise RuntimeError("Failed to clone tmQM repository.")

def load_and_preprocess():
    print("Loading tmQM data...")
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

    # Apply Hartree to eV conversion
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
    for name, model in get_models().items():
        kw = {"groups": groups} if groups is not None else {}
        yp = cross_val_predict(model, X, y_, cv=cv, n_jobs=-1, **kw)
        r2, rmse = r2_score(y_, yp), root_mean_squared_error(y_, yp)
        results.append({"Split": split_label, "Features": feature_label, "Model": name, "R2": r2, "RMSE": rmse})
        print(f"  {name:14s} R2={r2: .4f}  RMSE={rmse: .4f}")

def generate_shap(X_scalar, y_clean):
    print("\nGenerating SHAP values...")
    X_train, X_test, y_train, y_test = train_test_split(X_scalar, y_clean, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1).fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.sample(n=1000, random_state=42))
    
    plt.figure()
    shap.summary_plot(shap_values, X_test.sample(n=1000, random_state=42), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "shap_summary_final.png"), dpi=300)
    plt.close()
    print("Saved shap_summary_final.png")

def main():
    clone_dataset()
    df_clean, X_scalar_full, y, metal = load_and_preprocess()
    
    X_rdkit, desc_idx = compute_rdkit_descriptors(df_clean)
    y_clean = y.loc[desc_idx].reset_index(drop=True)
    metal_clean = pd.Series(metal).loc[desc_idx].reset_index(drop=True)
    X_scalar = X_scalar_full.loc[desc_idx].reset_index(drop=True)
    
    print(f"\nAligned Dataset Size: {len(y_clean):,}")

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)

    run_cv("random", "5 Scalars", X_scalar, y_clean, kf, None, results)
    run_cv("random", "RDKit 2D", X_rdkit, y_clean, kf, None, results)

    ms = metal_clean != "UNKNOWN"
    run_cv("grouped", "5 Scalars", X_scalar[ms], y_clean[ms], gkf, metal_clean[ms], results)
    run_cv("grouped", "RDKit 2D", X_rdkit[ms], y_clean[ms], gkf, metal_clean[ms], results)

    pd.DataFrame(results).to_csv(os.path.join(PROJECT_DIR, "results_final.csv"), index=False)
    print("\nSaved results_final.csv")
    
    generate_shap(X_scalar, y_clean)

if __name__ == "__main__":
    main()
