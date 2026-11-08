<h1>Quantifying the Applicability Domain of Non-Spatial ML Models on tmQM</h1>

<h2>📖 Citation</h2>
<p>If you use this code or the corrected data alignment in your research, please cite our paper:</p>
<blockquote>
  <p><strong>Muhammad Kashan Ali.</strong> <em>Quantifying the Applicability Domain of Non-Spatial Machine Learning Models for HOMO–LUMO Gap Prediction on the tmQM Dataset.</em> (in preparation, 2024).</p>
</blockquote>

<hr>

<h2>📌 Overview</h2>
<p>This repository contains the data preprocessing, model training, and evaluation pipeline for our study on the 2024 release of the tmQM dataset.</p>
<p><strong>Core Finding:</strong> We demonstrate that non-spatial machine learning models (global scalars and 2D descriptors) reach an empirical ceiling of R² ≈ 0.31 under random cross-validation. Crucially, when evaluated on unseen transition metals via GroupKFold validation, performance catastrophically collapses, proving these models have near-zero extrapolative power.</p>

<h2>📊 Dataset & Features</h2>
<ul>
  <li><strong>Source:</strong> tmQM 2024 release — <code>tmQM/tmQM_y.csv</code></li>
  <li><strong>Complexes:</strong> 108,541</li>
  <li><strong>Unit Conversion:</strong> The target <code>HL_Gap</code> is stored in Hartree. To ensure physical interpretability, we apply the standard <strong>27.2114</strong> conversion factor to eV (yielding a median gap of 2.97 eV).</li>
  <li><strong>Aligned Dataset:</strong> To ensure a fair comparison, all models were evaluated on exactly 100,831 complexes (removing those without valid SMILES).</li>
</ul>

<p><strong>Feature Sets Benchmarked:</strong></p>
<ol>
  <li><strong>5 Scalars:</strong> <code>Electronic_E</code>, <code>Dispersion_E</code>, <code>Dipole_M</code>, <code>Metal_q</code>, <code>Polarizability</code>.</li>
  <li><strong>RDKit 2D:</strong> 14 topological descriptors (e.g., MolWt, LogP, TPSA).</li>
</ol>

<h2>🏆 Benchmark Results</h2>

<p><strong>Random 5-Fold Cross-Validation (In-Distribution)</strong></p>
<table>
  <tr>
    <th>Features</th>
    <th>Model</th>
    <th>R²</th>
    <th>RMSE (eV)</th>
  </tr>
  <tr>
    <td>5 Scalars</td>
    <td>Random Forest</td>
    <td><strong>0.2983</strong></td>
    <td>0.7595</td>
  </tr>
  <tr>
    <td>5 Scalars</td>
    <td>XGBoost</td>
    <td>0.2712</td>
    <td>0.7740</td>
  </tr>
  <tr>
    <td>RDKit 2D</td>
    <td>Random Forest</td>
    <td><strong>0.3074</strong></td>
    <td>0.7545</td>
  </tr>
  <tr>
    <td>RDKit 2D</td>
    <td>XGBoost</td>
    <td>0.2702</td>
    <td>0.7746</td>
  </tr>
</table>

<p><strong>GroupKFold Validation (Out-of-Distribution by Metal)</strong></p>
<table>
  <tr>
    <th>Features</th>
    <th>Model</th>
    <th>R²</th>
    <th>RMSE (eV)</th>
  </tr>
  <tr>
    <td>5 Scalars</td>
    <td>Random Forest</td>
    <td><strong>0.0946</strong></td>
    <td>0.8627</td>
  </tr>
  <tr>
    <td>5 Scalars</td>
    <td>XGBoost</td>
    <td>0.1183</td>
    <td>0.8514</td>
  </tr>
  <tr>
    <td>RDKit 2D</td>
    <td>Random Forest</td>
    <td><strong>0.1425</strong></td>
    <td>0.8396</td>
  </tr>
  <tr>
    <td>RDKit 2D</td>
    <td>XGBoost</td>
    <td>0.1465</td>
    <td>0.8377</td>
  </tr>
</table>

<h2>📂 Repository Structure</h2>
<pre><code>├── gap_distribution.png          # Corrected HOMO-LUMO gap distribution (eV)
├── parity_plot_final.png         # Random Forest parity plot (R² = 0.305)
├── shap_summary_final.png        # SHAP summary plot for 5-scalar model
├── run_pipeline.py               # Main execution pipeline
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md
</code></pre>

<h2>🚀 Reproducing the Results</h2>
<pre><code>pip install -r requirements.txt
python run_pipeline.py
</code></pre>
<p>The pipeline will:</p>
<ol>
  <li>Clone the tmQM repository (2024 release)</li>
  <li>Load and clean <code>tmQM_y.csv</code> (applying the Hartree-to-eV conversion)</li>
  <li>Train Random Forest and XGBoost on 5 Scalars and RDKit 2D descriptors</li>
  <li>Evaluate under both Random KFold and GroupKFold (by metal)</li>
  <li>Generate the three publication figures listed above</li>
</ol>

<hr>

<h2>🖼️ Publication Figures</h2>

<h3>Corrected Gap Distribution</h3>
<p><em>The physically realistic distribution of gaps (1-8 eV) confirms the unit conversion.</em></p>
<p><img src="https://github.com/kashan1-debug/tmQM-OOD-Benchmark/blob/main/gap_distribution.png?raw=true" alt="Gap Distribution" width="700"></p>

<h3>Random Forest Parity Plot</h3>
<p><em>The model fails to predict extreme gaps (> 5 eV) and collapses to the mean (R² = 0.305).</em></p>
<p><img src="https://github.com/kashan1-debug/tmQM-OOD-Benchmark/blob/main/parity_plot_final.png?raw=true" alt="Parity Plot" width="600"></p>

<h3>SHAP Feature Importance</h3>
<p><em>SHAP confirms the model relies on global atomic properties (Polarizability, Electronic_E) rather than local geometry.</em></p>
<p><img src="https://github.com/kashan1-debug/tmQM-OOD-Benchmark/blob/main/shap_summary_final.png?raw=true" alt="SHAP Summary" width="900"></p>
