<h1>Quantifying the Applicability Domain of Non-Spatial ML Models on tmQM</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<h2>📖 Citation</h2>
<p>If you use this code or the corrected data alignment in your research, please cite our paper:</p>
<blockquote>
  <p><strong>Muhammad Kashan Ali.</strong> <em>Benchmarking the Out-of-Distribution Extrapolation Limits of Non-Spatial Machine Learning Models on the tmQM Dataset.</em> (in preparation, 2026).</p>
</blockquote>

<hr>

<h2>🖼️ Publication Figures</h2>

<h3>1. Corrected Gap Distribution</h3>
<p><em>The physically realistic distribution of gaps (1-8 eV) confirms the unit conversion.</em></p>
<img src="https://raw.githubusercontent.com/kashan1-debug/tmQM-OOD-Benchmark/main/gap_distribution.png" width="700">

<h3>2. Random Forest Parity Plot</h3>
<p><em>The model fails to predict extreme gaps (> 5 eV) and collapses to the mean.</em></p>
<img src="https://raw.githubusercontent.com/kashan1-debug/tmQM-OOD-Benchmark/main/parity_plot_final.png" width="600">

<h3>3. SHAP Feature Importance</h3>
<p><em>SHAP confirms the model relies on global atomic properties rather than local geometry.</em></p>
<img src="https://raw.githubusercontent.com/kashan1-debug/tmQM-OOD-Benchmark/main/shap_summary_final.png" width="900">

<h3>4. Applicability Domain Analysis</h3>
<p><em>Error increases significantly as test molecules move further from the training manifold (kNN distance).</em></p>
<img src="https://raw.githubusercontent.com/kashan1-debug/tmQM-OOD-Benchmark/main/applicability_domain.png" width="700">

<h3>5. Leave-One-Metal-Out Analysis</h3>
<p><em>The model catastrophically fails on unseen metals like Hg (R² = -0.77) and Ag (R² = -0.70), while maintaining some accuracy on Mn (R² = 0.35).</em></p>
<img src="https://raw.githubusercontent.com/kashan1-debug/tmQM-OOD-Benchmark/main/per_metal_r2.png" width="900">

<hr>

<h2>📌 Overview</h2>
<p>This repository contains the data preprocessing, model training, and evaluation pipeline for our study on the 2024 release of the tmQM dataset.</p>
<p><strong>Core Finding:</strong> We demonstrate that non-spatial machine learning models (global scalars and 2D descriptors) reach an empirical ceiling of \(R^2 \approx 0.37\) under random cross-validation. Crucially, when evaluated on unseen transition metals via GroupKFold validation, performance catastrophically collapses, proving these models have near-zero extrapolative power and justifying the computational cost of 3D GNN architectures.</p>

<h2>📊 Dataset & Features</h2>
<ul>
  <li><strong>Source:</strong> tmQM 2024 release — <code>tmQM/tmQM_y.csv</code></li>
  <li><strong>Complexes:</strong> 108,541</li>
  <li><strong>Unit Conversion:</strong> The target <code>HL_Gap</code> is stored in Hartree. To ensure physical interpretability, we apply the standard <strong>27.2114</strong> conversion factor to eV.</li>
  <li><strong>Aligned Dataset:</strong> To ensure a fair comparison, all models were evaluated on exactly 100,831 complexes (removing those without valid SMILES).</li>
</ul>

<h2>🏆 Benchmark Results</h2>

<h3>Random 5-Fold Cross-Validation (In-Distribution)</h3>
<table>
  <tr><th>Features</th><th>Model</th><th>R²</th><th>RMSE (eV)</th></tr>
  <tr><td>5 Scalars</td><td>Random Forest</td><td><b>0.2983</b></td><td>0.7595</td></tr>
  <tr><td>RDKit 2D</td><td>Random Forest</td><td><b>0.3074</b></td><td>0.7545</td></tr>
  <tr><td>Morgan FP</td><td>XGBoost</td><td><b>0.3732</b></td><td>0.7178</td></tr>
</table>

<h3>GroupKFold Validation (Out-of-Distribution by Metal)</h3>
<table>
  <tr><th>Features</th><th>Model</th><th>R²</th><th>RMSE (eV)</th></tr>
  <tr><td>5 Scalars</td><td>Random Forest</td><td><b>0.0946</b></td><td>0.8627</td></tr>
  <tr><td>RDKit 2D</td><td>Random Forest</td><td><b>0.1425</b></td><td>0.8396</td></tr>
  <tr><td>Morgan FP</td><td>XGBoost</td><td><b>0.2009</b></td><td>0.8105</td></tr>
</table>

<h2>🚀 Reproducing the Results</h2>
<pre><code>pip install -r requirements.txt
python run_pipeline.py
</code></pre>
