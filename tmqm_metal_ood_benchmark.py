#!/usr/bin/env python3
"""
tmQM metal-OOD benchmark pipeline (v2 - Rigorously Audited & Production Ready).

Question: how far can non-spatial ML models (5 DFT scalars, 14 RDKit 2D
descriptors, 256-bit Morgan FP) extrapolate to transition metals they have
never seen, for HOMO-LUMO gap prediction?
"""

import argparse
import glob
import gzip
import json
import os
import platform
import re
import subprocess
import time
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.DataStructs import cDataStructs
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

UNIT_FACTOR = 27.2114  # Hartree -> eV
SCALARS = ["Electronic_E", "Dispersion_E", "Dipole_M", "Metal_q", "Polarizability"]
F_SCALAR, F_RDKIT, F_MORGAN = "5 Scalars", "RDKit 2D", "Morgan FP (256)"
METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}
HEADER_RE = re.compile(r"CSD_code\s*=\s*(\S+).*?Stoichiometry\s*=\s*(\S+)")

MODEL_FACTORIES = {
    "Mean baseline": lambda s: DummyRegressor(strategy="mean"),
    "Ridge": lambda s: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": lambda s: RandomForestRegressor(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=s),
    "XGBoost": lambda s: XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, tree_method="hist",
        n_jobs=-1, random_state=s),
}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def metrics(y, p):
    return {"R2": float(r2_score(y, p)), "RMSE": rmse(y, p), "MAE": float(mean_absolute_error(y, p))}


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  saved {os.path.basename(path)}")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def ensure_repo(repo_dir, skip_download):
    data_dir = os.path.join(repo_dir, "tmQM")
    if os.path.exists(os.path.join(data_dir, "tmQM_y.csv")):
        print("tmQM data found.")
        return data_dir
    if skip_download:
        raise FileNotFoundError(f"{data_dir}/tmQM_y.csv not found (and --skip-download set).")
    print("Cloning tmQM repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/uiocompcat/tmqm.git", repo_dir],
        check=True)
    return data_dir


def load_table(args, data_dir):
    df = pd.read_csv(os.path.join(data_dir, "tmQM_y.csv"), sep=";")
    needed = SCALARS + ["HL_Gap", "CSD_code"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"tmQM_y.csv is missing columns: {missing}")

    if "SMILES" not in df.columns:
        if args.smiles and os.path.exists(args.smiles):
            sm = pd.read_csv(args.smiles, sep=None, engine="python")
            if not {"CSD_code", "SMILES"} <= set(sm.columns):
                raise SystemExit("--smiles file needs columns CSD_code and SMILES.")
            df = df.merge(sm[["CSD_code", "SMILES"]].drop_duplicates("CSD_code"),
                          on="CSD_code", how="left")
        else:
            raise SystemExit(
                "No SMILES column in tmQM_y.csv. Pass --smiles path/to/file.csv "
                "(columns: CSD_code, SMILES).")

    n0 = len(df)
    df = df.dropna(subset=SCALARS + ["HL_Gap"]).reset_index(drop=True)
    print(f"Loaded {n0:,} rows; {len(df):,} have all scalars and HL_Gap.")
    df["y"] = df["HL_Gap"].astype(float) * UNIT_FACTOR
    return df


def load_metal_map(data_dir):
    """CSD_code -> metal symbol from tmQM_X*.xyz(.gz) headers (Stoichiometry field)."""
    files = sorted(glob.glob(os.path.join(data_dir, "tmQM_X*.xyz*")))
    if not files:
        print("  (no tmQM_X*.xyz files found: metal-distribution check will be skipped)")
        return {}
    print(f"Reading metals from {len(files)} xyz file(s)...")
    out = {}
    for f in files:
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt") as fh:
            for line in fh:
                if "CSD_code" in line and "Stoichiometry" in line:
                    m = HEADER_RE.search(line)
                    if not m:
                        continue
                    els = [e for e in re.findall(r"[A-Z][a-z]?", m.group(2)) if e in METALS]
                    out[m.group(1)] = els[0] if els else "UNKNOWN"
    return out


# --------------------------------------------------------------------------- #
# Featurisation (optimized C++ numpy bindings)
# --------------------------------------------------------------------------- #
def make_morgan(n_bits=256, radius=2):
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        def _get_fp(mol):
            arr = np.zeros((n_bits,), dtype=np.uint8)
            cDataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
            return arr
        return _get_fp
    except Exception:
        from rdkit.Chem import AllChem
        def _get_fp(mol):
            arr = np.zeros((n_bits,), dtype=np.uint8)
            cDataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), arr
            )
            return arr
        return _get_fp


def featurize(smiles):
    n = len(smiles)
    desc = np.full((n, 14), np.nan, dtype=np.float32)
    fp = np.zeros((n, 256), dtype=np.uint8)
    ok = np.zeros(n, dtype=bool)
    mol_metal = np.array(["UNKNOWN"] * n, dtype=object)
    morgan = make_morgan()
    n_parse_fail = n_desc_fail = 0
    t0 = time.time()
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            n_parse_fail += 1
            continue
        try:
            vals = [
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol), mol.GetNumHeavyAtoms(),
                Descriptors.FractionCSP3(mol), Descriptors.NumValenceElectrons(mol),
                rdMolDescriptors.CalcNumHeteroatoms(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
            ]
            if not np.all(np.isfinite(vals)):
                n_desc_fail += 1
                continue
            bits = morgan(mol)
        except Exception:
            n_desc_fail += 1
            continue
        desc[i], fp[i], ok[i] = vals, bits, True
        mets = [a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in METALS]
        mol_metal[i] = mets[0] if mets else "UNKNOWN"
        if (i + 1) % 20000 == 0:
            print(f"  featurised {i + 1:,}/{n:,} ({time.time() - t0:.0f}s)")
    print(f"Featurisation: {ok.sum():,} ok | {n_parse_fail:,} SMILES missing/unparseable | "
          f"{n_desc_fail:,} descriptor failures")
    return desc, fp, ok, mol_metal


# --------------------------------------------------------------------------- #
# Metal-distribution bias check
# --------------------------------------------------------------------------- #
def metal_bias_check(df, ok, out_dir):
    if "Metal_xyz" not in df or df["Metal_xyz"].isna().all():
        print("Metal-distribution check skipped (no xyz metal labels).")
        return None
    before = df["Metal_xyz"].value_counts()
    after = df.loc[ok, "Metal_xyz"].value_counts()
    tab = pd.DataFrame({"n_before": before, "n_after": after}).fillna(0).astype(int)
    tab["retained_frac"] = tab["n_after"] / tab["n_before"]
    tab["share_before"] = tab["n_before"] / tab["n_before"].sum()
    tab["share_after"] = tab["n_after"] / max(tab["n_after"].sum(), 1)
    tab["share_shift"] = tab["share_after"] - tab["share_before"]
    tab = tab.sort_values("n_before", ascending=False)
    tab.to_csv(os.path.join(out_dir, "metal_distribution_check.csv"))

    info = {"n_before": int(tab["n_before"].sum()), "n_after": int(tab["n_after"].sum()),
            "max_abs_share_shift": float(tab["share_shift"].abs().max()),
            "min_retained_frac": float(tab["retained_frac"].min())}
    try:
        from scipy.stats import chi2_contingency
        sub = tab[tab["n_before"] >= 5]
        dropped = sub["n_before"] - sub["n_after"]
        if dropped.sum() > 0:
            chi2, p, _, _ = chi2_contingency(np.c_[sub["n_after"].values, dropped.values])
            info.update(chi2=float(chi2), p_value=float(p))
    except Exception as e:
        print(f"  chi-square skipped: {e}")
    print("Metal distribution before vs after filtering:", json.dumps(info))
    return info


# --------------------------------------------------------------------------- #
# Cross-validation (Audited robust bin packing)
# --------------------------------------------------------------------------- #
def random_folds(n, seed, k):
    return list(KFold(k, shuffle=True, random_state=seed).split(np.arange(n)))


def grouped_folds(groups, seed, k):
    """Randomised, size-balanced group -> fold assignment using structured argsort."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)

    sort_keys = np.empty(len(uniq), dtype=[("count", int), ("rand", float)])
    sort_keys["count"] = -counts  # Descending order of count
    sort_keys["rand"] = rng.random(len(uniq))
    order = np.argsort(sort_keys, order=("count", "rand"))

    uniq_sorted = uniq[order]
    counts_sorted = counts[order]

    sizes = np.zeros(k, dtype=int)
    assign = {}
    for g, c in zip(uniq_sorted, counts_sorted):
        f = int(np.argmin(sizes))
        assign[g] = f
        sizes[f] += c

    fold_of = np.array([assign[g] for g in groups])
    folds = [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(k)]
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te])), "Group leakage!"
    return folds


def oof_predict(factory, X, y, folds, seed):
    yp = np.full(len(y), np.nan)
    for tr, te in folds:
        m = factory(seed)
        m.fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
    assert not np.isnan(yp).any()
    return yp


def run_cv(feature_sets, y, metal, seeds, k, oof_store):
    rows = []
    for split in ["random_known", "grouped"]:
        for feat, X in feature_sets.items():
            for model_name, factory in MODEL_FACTORIES.items():
                for seed in seeds:
                    folds = (random_folds(len(y), seed, k) if split == "random_known"
                             else grouped_folds(metal, seed, k))
                    yp = oof_predict(factory, X, y, folds, seed)
                    rows.append({"Split": split, "Features": feat, "Model": model_name,
                                 "Seed": seed, **metrics(y, yp)})
                    if seed == seeds[0]:
                        oof_store[(split, feat, model_name)] = yp
                r = [x for x in rows if x["Split"] == split and x["Features"] == feat
                     and x["Model"] == model_name]
                print(f"  {split:13s} | {feat:16s} | {model_name:13s} "
                      f"R2={np.mean([x['R2'] for x in r]): .4f} "
                      f"RMSE={np.mean([x['RMSE'] for x in r]):.4f}")
    return pd.DataFrame(rows)


def summarise(runs):
    s = (runs.groupby(["Split", "Features", "Model"], sort=False)
         .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
              RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
              MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
         .reset_index().fillna(0.0))
    return s


def summary_markdown(summary, n_seeds):
    lines = []
    titles = {"random_known": "Random 5-fold CV (in-distribution)",
              "grouped": "Metal-grouped 5-fold CV (out-of-distribution)"}
    for split, title in titles.items():
        lines += [f"**{title}** (mean ± std over {n_seeds} seed(s))", "",
                  "| Features | Model | R² | RMSE (eV) | MAE (eV) |", "|---|---|---|---|---|"]
        for _, r in summary[summary["Split"] == split].iterrows():
            if n_seeds > 1:
                r2_str = f"{r['R2_mean']:.4f} ± {r['R2_std']:.4f}"
                rmse_str = f"{r['RMSE_mean']:.4f} ± {r['RMSE_std']:.4f}"
                mae_str = f"{r['MAE_mean']:.4f} ± {r['MAE_std']:.4f}"
            else:
                r2_str = f"{r['R2_mean']:.4f}"
                rmse_str = f"{r['RMSE_mean']:.4f}"
                mae_str = f"{r['MAE_mean']:.4f}"
            lines.append(f"| {r['Features']} | {r['Model']} | {r2_str} | {rmse_str} | {mae_str} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Leave-one-metal-out + applicability domain
# --------------------------------------------------------------------------- #
def knn_distance(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(sc.transform(Xtr))
    d, _ = nn.kneighbors(sc.transform(Xte))
    return d.ravel()


def leave_one_metal_out(feature_sets, y, metal, lomo_feats, min_n, seed=0, ad_max=2000):
    rows, ad_rows = [], []
    rng = np.random.default_rng(seed)
    counts = pd.Series(metal).value_counts()
    keep = [m for m, c in counts.items() if c >= min_n]
    skipped = [m for m, c in counts.items() if c < min_n]
    print(f"LOMO on {len(keep)} metals (min N={min_n}); skipped: {skipped}")
    for feat in lomo_feats:
        X = feature_sets[feat]
        for m in sorted(keep):
            te = metal == m
            tr = ~te
            for model_name, factory in MODEL_FACTORIES.items():
                mdl = factory(seed)
                mdl.fit(X[tr], y[tr])
                yp = mdl.predict(X[te])
                rows.append({"Features": feat, "Model": model_name, "Metal": m,
                             "N": int(te.sum()), **metrics(y[te], yp)})
                if feat == F_SCALAR and model_name == "XGBoost":
                    idx = rng.choice(te.sum(), size=min(te.sum(), ad_max), replace=False)
                    dist = knn_distance(X[tr], X[te][idx])
                    err = np.abs(y[te][idx] - yp[idx])
                    ad_rows += [{"Metal": m, "dist": d, "abs_err": e} for d, e in zip(dist, err)]
        print(f"  LOMO done: {feat}")
    return pd.DataFrame(rows), pd.DataFrame(ad_rows)


def random_split_ad(X, y, seed=0, max_te=5000):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    mdl = MODEL_FACTORIES["XGBoost"](seed)
    mdl.fit(X[tr], y[tr])
    te = te[:max_te]
    yp = mdl.predict(X[te])
    return pd.DataFrame({"dist": knn_distance(X[tr], X[te]), "abs_err": np.abs(y[te] - yp)})


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_gap_distribution(y, path):
    plt.figure(figsize=(7, 4))
    plt.hist(y, bins=60, color="teal", edgecolor="black", alpha=0.7)
    plt.axvline(np.median(y), color="k", ls="--", label=f"median = {np.median(y):.2f} eV")
    plt.xlabel("HOMO-LUMO gap (eV)", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.title("Distribution of HOMO-LUMO gaps (tmQM)", fontweight="bold")
    plt.legend()
    save_fig(path)


def fig_parity(y, oof, path, feat=F_SCALAR, model="Random Forest", n_plot=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n_plot, len(y)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, split, title in zip(axes, ["random_known", "grouped"],
                                ["Random 5-fold (out-of-fold)", "Metal-grouped 5-fold (out-of-fold)"]):
        yp = oof[(split, feat, model)]
        r2 = r2_score(y, yp)
        ax.scatter(y[idx], yp[idx], s=4, alpha=0.25, edgecolors="none")
        lo, hi = float(min(y.min(), yp.min())), float(max(y.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black"))
        ax.set_title(f"{model}, {feat}\n{title}", fontweight="bold", fontsize=10)
        ax.set_xlabel("DFT gap (eV)", fontweight="bold")
    axes[0].set_ylabel("Predicted gap (eV)", fontweight="bold")
    save_fig(path)


def fig_shap(X_df, y, path, seed=0):
    try:
        import shap
    except ImportError:
        print("  shap not installed: skipping SHAP figure")
        return
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    mdl = MODEL_FACTORIES["XGBoost"](seed).fit(X_df.iloc[tr], y[tr])
    samp = X_df.iloc[te[:1000]]
    sv = shap.TreeExplainer(mdl).shap_values(samp)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, samp, show=False)
    plt.title("SHAP: XGBoost on 5 DFT scalars (random 80/20 split).\n"
              "Shows scalar importance only; no geometry is encoded.", fontsize=9)
    save_fig(path)


def fig_ad(ad_random, ad_lomo, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    out = {}
    for ax, df, title, key, col in [
        (axes[0], ad_random, "Random split (in-distribution)", "random", "purple"),
        (axes[1], ad_lomo, "Held-out metal (leave-one-metal-out)", "lomo", "darkorange"),
    ]:
        rho, p = spearmanr(df["dist"], df["abs_err"])
        ax.scatter(df["dist"], df["abs_err"], s=6, alpha=0.25, color=col, edgecolors="none")
        ax.set_title(f"{title}\nSpearman ρ = {rho:.2f}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Scaled distance to nearest training complex", fontweight="bold")
        out[key] = {"spearman_rho": float(rho), "p_value": float(p), "n": int(len(df))}
    axes[0].set_ylabel("|error| (eV), XGBoost on 5 scalars", fontweight="bold")
    save_fig(path)
    return out


def fig_per_metal(lomo, path, feat=F_SCALAR, model="XGBoost"):
    d = lomo[(lomo["Features"] == feat) & (lomo["Model"] == model)].sort_values("R2")
    if d.empty:
        return
    colors = ["#2ca02c" if v >= 0.3 else "#ff7f0e" if v >= 0 else "#d62728" for v in d["R2"]]
    plt.figure(figsize=(11, 5))
    plt.bar(d["Metal"], d["R2"], color=colors, edgecolor="black")
    plt.axhline(0, color="gray", ls=":")
    med = d["R2"].median()
    plt.axhline(med, color="black", ls="--", label=f"median R² = {med:.3f}")
    for i, (_, r) in enumerate(d.iterrows()):
        plt.text(i, min(r["R2"], 0) - 0.03, f"{int(r['N'])}", ha="center", va="top", fontsize=6, rotation=90)
    plt.ylabel("R² on held-out metal", fontweight="bold")
    plt.xlabel("Held-out metal (N below bars)", fontweight="bold")
    plt.title(f"Leave-one-metal-out: {model}, {feat}", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    save_fig(path)


def fig_cv_comparison(summary, path):
    feats = list(dict.fromkeys(summary["Features"]))
    models = list(dict.fromkeys(summary["Model"]))
    fig, axes = plt.subplots(1, len(feats), figsize=(4.4 * len(feats), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, feats):
        x = np.arange(len(models))
        w = 0.38
        for k, (split, col, lab) in enumerate([("random_known", "#1f77b4", "Random CV"),
                                               ("grouped", "#d62728", "Metal-grouped CV")]):
            s = summary[(summary["Features"] == feat) & (summary["Split"] == split)] \
                .set_index("Model").reindex(models)
            ax.bar(x + (k - 0.5) * w, s["R2_mean"], w, yerr=s["R2_std"], color=col,
                   edgecolor="black", capsize=3, label=lab)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_title(feat, fontweight="bold")
    axes[0].set_ylabel("R² (mean ± std over seeds)", fontweight="bold")
    axes[0].legend()
    save_fig(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-dir", default="tmqm", help="local clone of uiocompcat/tmqm")
    ap.add_argument("--smiles", default=None, help="CSV with CSD_code,SMILES if tmQM_y.csv lacks SMILES")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seeds", type=int, default=3, help="number of repeated CV seeds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-metal-n", type=int, default=100, help="min complexes for LOMO")
    ap.add_argument("--lomo-features", default="scalars",
                    help="comma list of: scalars,rdkit,morgan")
    ap.add_argument("--quick", type=int, default=0, help="subsample N rows (smoke test)")
    ap.add_argument("--skip-download", action="store_true")

    # Use parse_known_args to safely absorb/ignore Jupyter/Colab kernel flags (-f ...)
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"Note: Ignored environment flags: {unknown}")
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)
    t_start = time.time()

    data_dir = ensure_repo(args.repo_dir, args.skip_download)
    df = load_table(args, data_dir)
    if args.quick:
        df = df.sample(n=min(args.quick, len(df)), random_state=0).reset_index(drop=True)
        print(f"--quick: using {len(df):,} rows")

    metal_map = load_metal_map(data_dir)
    if metal_map:
        df["Metal_xyz"] = df["CSD_code"].map(metal_map)

    print("\nFeaturising...")
    desc, fp, ok, mol_metal = featurize(df["SMILES"].tolist())
    df["Metal_mol"] = mol_metal
    if "Metal_xyz" in df:
        df["Metal"] = df["Metal_xyz"].fillna(df["Metal_mol"])
    else:
        df["Metal"] = df["Metal_mol"]
    df["Metal"] = df["Metal"].fillna("UNKNOWN")

    bias_info = metal_bias_check(df, ok, args.out_dir)

    keep = ok & (df["Metal"] != "UNKNOWN").values
    if "Metal_xyz" in df:
        both = keep & df["Metal_xyz"].notna().values & (df["Metal_mol"] != "UNKNOWN").values
        n_dis = int((df.loc[both, "Metal_xyz"] != df.loc[both, "Metal_mol"]).sum())
        print(f"Metal label disagreement (xyz vs RDKit mol): {n_dis:,}/{int(both.sum()):,}")
    else:
        n_dis = None

    d = df[keep].reset_index(drop=True)
    y = d["y"].values
    metal = d["Metal"].values
    feature_sets = {
        F_SCALAR: d[SCALARS].values.astype(np.float32),
        F_RDKIT: desc[keep],
        F_MORGAN: fp[keep].astype(np.float32),
    }
    print(f"\nFinal aligned dataset: {len(y):,} complexes, {len(set(metal))} metals, "
          f"median gap {np.median(y):.2f} eV")

    seeds = list(range(args.seeds))
    print("\n=== Cross-validation ===")
    oof = {}
    runs = run_cv(feature_sets, y, metal, seeds, args.folds, oof)
    runs.to_csv(out("results_all_runs.csv"), index=False)
    summary = summarise(runs)
    summary.to_csv(out("results_final.csv"), index=False)
    with open(out("results_table.md"), "w") as fh:
        fh.write(summary_markdown(summary, len(seeds)))
    print("\n--- SUMMARY ---")
    print(summary.round(4).to_string(index=False))

    print("\n=== Leave-one-metal-out ===")
    name_map = {"scalars": F_SCALAR, "rdkit": F_RDKIT, "morgan": F_MORGAN}
    lomo_feats = [name_map[x.strip()] for x in args.lomo_features.split(",") if x.strip() in name_map]
    lomo, ad_lomo = leave_one_metal_out(feature_sets, y, metal, lomo_feats, args.min_metal_n)
    lomo.to_csv(out("per_metal_results.csv"), index=False)
    if lomo.empty:
        # No metal reached min N (e.g. --quick with a small sample): the
        # groupby below would raise KeyError: 'Features'. The empty file is
        # already written; just report it and carry on.
        print("Median per-metal R2: (no metals passed min N — LOMO empty)")
    else:
        med = lomo.groupby(["Features", "Model"])["R2"].median().round(3)
        print("Median per-metal R2:\n", med.to_string())

    print("\n=== Figures ===")
    fig_gap_distribution(y, out("gap_distribution.png"))
    fig_parity(y, oof, out("parity_plot_final.png"))
    fig_shap(pd.DataFrame(feature_sets[F_SCALAR], columns=SCALARS), y, out("shap_summary_final.png"))
    ad_info = {}
    if not ad_lomo.empty:
        ad_random = random_split_ad(feature_sets[F_SCALAR], y)
        ad_info = fig_ad(ad_random, ad_lomo, out("applicability_domain.png"))
    fig_per_metal(lomo, out("per_metal_r2.png"))
    fig_cv_comparison(summary, out("cv_comparison.png"))

    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "n_complexes_final": int(len(y)),
        "n_metals": int(len(set(metal))),
        "seeds": seeds, "folds": args.folds, "quick": args.quick,
        "metal_label_disagreements": n_dis,
        "metal_bias_check": bias_info,
        "applicability_domain": ad_info,
        "python": platform.python_version(),
        "runtime_min": round((time.time() - t_start) / 60, 1),
    }
    for pkg in ["numpy", "pandas", "sklearn", "xgboost", "rdkit", "shap"]:
        try:
            meta[pkg] = __import__(pkg).__version__
        except Exception:
            meta[pkg] = None
    with open(out("run_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nPipeline complete in {meta['runtime_min']} min.")


if __name__ == "__main__":
    main()
