# tmQM benchmark — verification report (2026-11-27)

What was checked in the uploaded bundle (`tmQM_Research` zip, 28 files), what
matches, and what does not. The main README now reports only numbers these
files support.

## Files in the bundle

- Feature files (parquet, verified shapes with the project venv):
  - `X_scalar_FINAL.parquet` — (100,831 × 5): `Electronic_E`,
    `Dispersion_E`, `Dipole_M`, `Metal_q`, `Polarizability`.
  - `X_morgan.parquet` — (100,831 × 1024): Morgan fingerprints, 1024 bits.
  - `X_rdkit_final.parquet` — (100,831 × 14): RDKit 2D descriptors.
  - `y_FINAL.parquet` — (100,831 × 1): `HL_Gap_eV`, mean 2.97 eV, std 0.91
    eV, min 0.06, max 8.37 eV.
  - `X_rdkit.csv` — same 14 RDKit descriptors as a CSV (8.7 MB).
  - `X_rdkit_FINAL.parquet(1)` — duplicate of `X_rdkit_final.parquet`.
- Result tables (CSV): `benchmark_results.csv` (24 rows),
  `benchmark_summary.csv` (24 rows, same numbers, one row per run),
  `results_cv.csv` (16 rows), `model_results.csv` (12 rows),
  `model_results_progress.csv` (duplicate of `model_results.csv`),
  `model_summary.csv` = `model_performance_summary.csv` (same 4-row table),
  `feature_importance.csv` (5 rows), `rf_predictions.csv` (21,709 rows).
- Figures (PNG, 13): `model_comparison_r2.png`,
  `model_comparison_rmse.png`, `parity_plot.png`,
  `parity_plot_final.png`, `per_metal_r2.png`, `shap_summary.png`,
  `shap_summary_final.png`, `feature_importance.png`,
  `actual_vs_predicted.png`, `applicability_domain.png`,
  `correlation.png`, `correlation_heatmap.png`, `gap_distribution.png`,
  `target_distribution.png`.

## Checks that pass

- `benchmark_results.csv` is internally consistent: 3 seeds × 2 feature sets
  × 4 models = 24 rows; train 86,832 + test 21,709 = 108,541; seed-to-seed
  variation is small (RF R² 0.322–0.336 DFT-assisted).
- `model_results.csv` has exactly the same R² values as
  `benchmark_results.csv` (RF 0.3346, XGB 0.2739, …) with RMSE/MAE scaled by
  1/27.21 — i.e. the same runs with the target in Hartree. Not a
  contradiction, but the two unit systems must not be mixed.
- `results_cv.csv` random rows agree with `benchmark_results.csv` to ~0.005
  (5 scalars RF 0.330 vs 0.334; RDKit RF 0.379).
- `model_comparison_r2.png` matches `benchmark_results.csv` (Dummy 0.00,
  Ridge ≈ 0.083, Random Forest ≈ 0.331, XGBoost ≈ 0.269).
- `per_metal_r2.png` matches the per-metal values quoted in the old README
  (Hg ≈ −0.77, Ag ≈ −0.70, Mn ≈ 0.35, median −0.048, 29 metals shown).
- `feature_importance.csv` sums to 1.0 and its ranking (Metal_q first) is
  consistent with the SHAP figure's story.

## Reproduction — the old README's numbers were real (2026-11-27)

**Correction to this report's earlier claim:** the previous README's
random/grouped tables were described above as coming "from different runs"
with the Morgan numbers having "no source table". That was wrong. On
2026-11-27 the pipeline was rerun end-to-end and reproduced the old README's
numbers to within **0.0011 on every row** (largest diff: RDKit RF random
0.3074 → 0.3085; 5-scalar RF grouped 0.0946 → 0.0957; Morgan XGB grouped
0.2009 → 0.2009 exactly). The old README was a faithful record of this
pipeline. The Morgan table has been restored in the README, sourced from the
rerun (`results/results_cv_reproduced.csv`, transcribed from the pasted run
output).

What the rerun also established:

- Data: 108,541 rows loaded, all with scalars and HL_Gap. Metals read from
  3 xyz files (not from the SMILES regex in `run_pipeline.py` — this rerun
  used a different, newer script; see the code-audit note below).
- Featurisation: 100,831 ok, 7,710 SMILES missing/unparseable, 0 descriptor
  failures. Final aligned set: 100,831 complexes, 30 metals, median gap
  2.98 eV.
- Filtering shifts the metal mix: max share shift 0.0028, but the
  least-retained metal keeps only **27.3%** of its rows (La — matches the
  earlier audit note), chi² = 3666.5, p = 0.0. The grouped results therefore
  rest on a filtered set in which some metals are heavily thinned.
- Metal labels validated: xyz vs RDKit-mol disagreement **0 / 100,831** —
  the metal assignment is trustworthy, which answers the fragility concern
  raised about `extract_metal()` in the code audit.
- The summary table's `R2_std = 0.0` on every row is an artifact: the scores
  come from one pooled `cross_val_predict` pass, so there is no per-fold
  spread to report. Do not quote a spread from this table.
- The rerun's LOMO section ("LOMO on 30 metals, min N=100, skipped: []")
  was cut off in the pasted output — per-metal LOMO numbers from this run
  are not yet on file. The per-metal figure values (Hg −0.77, Ag −0.70,
  Mn 0.35, median −0.048) still rest on the figure alone.
- **The rerun script is not `run_pipeline.py`.** It reads metals from xyz
  files, runs a Mean baseline and Ridge, and has a LOMO section — none of
  which exist in the repo's script. The reproduction gap stands until that
  script (or notebook) is added to the repo.

## Mismatches found — the old README's tables do not come from these files

**[CORRECTED 2026-11-27 — see the Reproduction section above. The rerun
reproduced these numbers to within 0.0011, so the old README was right and
the Morgan table is restored in the README. What follows is the original
note, kept for the record.]**

The previous README reported a random-CV table (5 scalars RF 0.2983, RDKit RF
0.3074, Morgan RF 0.3860) and a GroupKFold table (5 scalars RF 0.0946, RDKit
RF 0.1425, Morgan RF 0.1836 / XGB 0.2009). **None of those numbers appear in
this bundle.** The closest files give 0.330 / 0.379 (random) and 0.070 / 0.149
(grouped). The old numbers come from different runs, and the Morgan numbers
have no source table at all in the bundle. The new README therefore reports
only the numbers in `results/` and drops the Morgan table.

## What is missing from the bundle

- Morgan-fingerprint results table — **resolved 2026-11-27**: the rerun
  reproduced the Morgan numbers and they are now in
  `results/results_cv_reproduced.csv` (256-bit). The surviving
  `X_morgan.parquet` (1024-bit) is still from a different run.
- Per-metal results table (only the figure survives).
- xTB feature file (the `xTB_only` rows cannot be reproduced from here).
- Leave-one-metal-out results table (the old audit noted a 16-metal file;
  the figure here shows 29 metals).
- The pipeline notebook / exact split definitions for `results_cv.csv`
  (whether "random" is 5-fold CV or a single split is not recorded in the
  file).

## Code audit — v2 script `tmqm_metal_ood_benchmark.py` (626 lines, checked 2026-11-27)

This is the script Muhammad ran on 2026-11-27, saved to the repo before it
could be lost with the Colab runtime. It compiles cleanly and it produces
the pasted run output line for line: the "Loaded 108,541 rows" line, the
featurisation counts (100,831 ok / 7,710 bad / 0 failures), the metal-bias
JSON, the 0/100,831 label disagreement, the CV print format and the summary
table all match the code paths exactly.

Findings:

- **The run used a single seed.** The summary's `R2_std = 0.0` on every row
  is the tell: with one seed, pandas' std is NaN and `summarise()` fills it
  with 0.0 (verified by direct test). The rerun was `--seeds 1`, not the
  default 3. The old README's numbers are single-seed too, which is part of
  why they match so closely. Rerun with `--seeds 3` for a real spread.
- **Grouped folds are custom, not sklearn GroupKFold.** `grouped_folds()`
  sorts metals by size (largest first, random tie-break) and assigns each to
  the least-loaded fold. Tested on synthetic data: folds are disjoint (the
  in-code assertion holds), cover all rows, and stay within roughly a factor
  of two in size — but the biggest metals each occupy a fold almost alone
  (e.g. Fe alone in one fold). That is a property of greedy packing, not a
  bug; it is documented here because the grouped numbers depend on it.
- **The LOMO results and v2 figures from the 2026-11-27 run are lost.**
  The script writes `per_metal_results.csv` and the figures only after LOMO
  finishes, and `--out-dir` defaulted to the runtime's disk. The pasted
  output stops at the LOMO header, so the run died during or just after
  LOMO. What survives is the printed CV output (transcribed into
  `results/results_cv_reproduced.csv`) — nothing else from that run.
- **The figures in the old bundle are from the earlier script, not v2.**
  Proof: the bundle's `parity_plot_final.png` is 1500×1500 (single panel,
  as `run_pipeline.py` makes), while v2's `fig_parity` makes a 3000×1500
  two-panel figure; and the bundle's `per_metal_r2.png` shows 29 metals
  with no N labels, while v2's `fig_per_metal` plots all 30 (min N=100,
  nothing skipped) with N printed under each bar. The v2 figures
  (`cv_comparison.png` included) were never saved anywhere.
- Everything else checks out: metal labels come from the xyz headers with
  a MolFromSmiles cross-check (0 disagreements); the bias check, the
  chi-square test and the `results_table.md` writer all do what they claim;
  `parse_known_args` correctly absorbs Colab's `-f` flags. (The
  scikit-learn ≥ 1.4 pin in `requirements.txt` is needed by the older
  `run_pipeline.py`, which imports `root_mean_squared_error`; the v2 script
  itself only uses `mean_squared_error`.)

## Code audit — `run_pipeline.py` (287 lines, checked 2026-11-27)

The code is clean and readable, it compiles, and the grouped-CV leakage
assertion (`train`/`test` metal sets must be disjoint) is genuinely good.
But **it does not reproduce the tables in `results/`**. Three concrete
mismatches:

1. **It writes a file that does not exist.** `main()` runs random + grouped
   5-fold CV and saves `results_final.csv`. Neither the repo nor the uploaded
   bundle contains `results_final.csv`. Its split labels would be
   `random_known` / `grouped`, while `results_cv.csv` says `random` /
   `grouped` and reports N = 108,541 for the 5 scalars — this pipeline aligns
   all three feature sets on the common valid index first (≈ 100,831 rows),
   so it can never report N = 108,541.
2. **There is no xTB in the code.** Nothing computes or loads xTB features,
   so the `xTB_only` rows in `benchmark_results.csv` (RF −0.20) come from a
   different script that is not in this repo. Same for the 80/20 splits with
   seeds 42 / 123 / 2024: the only 80/20 split in this code is inside
   `generate_figures()` for the parity plot, on the aligned set — which would
   give a 20,167-row test set, not the 21,709 rows in `rf_predictions.csv`.
3. **Morgan fingerprints are 256-bit here, 1024-bit in the bundle.**
   `compute_fingerprints()` uses `nBits=256`; the surviving
   `X_morgan.parquet` has 1024 columns. Different features, different
   numbers — another reason the old README's Morgan table cannot be traced to
   this code.

Smaller findings:

- `requirements.txt` said `scikit-learn>=1.3.0`, but the code imports
  `root_mean_squared_error`, which only exists from scikit-learn 1.4. Fixed
  to `>=1.4.0`.
- `seaborn` is listed in `requirements.txt` but imported nowhere in the
  pipeline. Harmless; left as is.
- `generate_figures()` trains the per-metal LOMO models with 100 XGBoost
  trees while `get_models()` uses 200 — the figure and the CV table use
  different model settings. The figure's values still match the bundle.
- `extract_metal()` (regex over SMILES) is fragile — it returns the first
  metal-like token anywhere in the string — but no wrong-metal case was found
  in the verified files. Rows it cannot parse become `UNKNOWN` and are
  dropped from the grouped runs, which is the honest handling.
- The `tmqm/tmQM/tmQM_y.csv` input path (after cloning
  `github.com/uiocompcat/tmqm`) was not re-verified by cloning here; it
  matches the path used in the original runs.

The README's "Reproducing the results" section now says this plainly instead
of claiming the pipeline produced the tables.

## Scientific reading (unchanged)

- The 5 DFT scalars come from the same TPSSh calculation as the gap. Using
  them is a DFT-to-DFT mapping, not screening from structure alone. The
  feature importances (all five scalars, nothing else) show the model leans
  on them entirely.
- Even with those oracle features, the best random-split R² is 0.33–0.38 and
  the grouped R² is 0.07–0.15. xTB-only features fail (RF −0.20).
  Leave-one-metal-out median is −0.048.
- Honest conclusion: the tmQM HOMO–LUMO gap is not predictable from these
  inputs, and it does not transfer to unseen metals. That is the benchmark's
  result — a negative result, recorded as such, not a paper.
