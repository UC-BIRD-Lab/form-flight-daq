# Nares-inspired pressure sensing enhances aerodynamic state perception for formation flight
Github code for paper: **Nares-inspired pressure sensing enhances aerodynamic state perception for formation flight**

Last modified (this README.md): `7 Nov 2025` by `Huanglun(Adam) Zhu`

---

## Overview
This repository contains all scripts used in the paper for:
- Data acquisition (pressure taps, load cell, pitot tube)
- Post-processing and error/uncertainty calculation
- Statistical summarization into meta datasets
- Mutual information (MI) analysis
- Predictive modeling (including Bayesian hyperparameter tuning for XGBoost)
- Final model training/testing and performance summarization
- Figure/plot generation
- LaTeX table/text generation

> **Reference data:** Figshare — `https://doi.org/10.6084/m9.figshare.c.8065798` (download all input data to same working directory of this code folder to run the code below)

---

## Contents
- [1. Data Acquisition (Wind Tunnel Test Run Code)](#1-data-acquisition-wind-tunnel-test-run-code)
- [2. Converters (Python)](#2-converters-python)
- [3. StatsCal (Python)](#3-statscal-python)
- [4. StatsSum (Python)](#4-statssum-python)
- [5. RepCheck (Python)](#5-repcheck-python)
- [6. Mutual Information (R)](#6-mutual-information-r)
- [7. Model Sensitivity (R)](#7-model-sensitivity-r)
- [8. Predictive Model (Python)](#8-predictive-model-python)
- [9. Plotters (Python)](#9-plotters-python)
- [10. LaTeX Conversion (Python)](#10-latex-conversion-python)
- [Appendix A. Notation & Definitions](#appendix-a-notation--definitions)
- [Appendix B. Paths & Outputs Summary](#appendix-b-paths--outputs-summary)

---

## 1) Data Acquisition (Wind Tunnel Test Run Code)
**Folder:** `Wind Tunnel Test Run Code`  
Scripts for collecting wind-tunnel measurement data.

**Inputs**
1. DAQ system connected to all pressure taps and the pitot tube  
2. DAQ system connected to the load cell  
3. Ambient environmental conditions specified in `Testing_Day_Condition`

**Output**
- `Raw_Data/` containing all measured force and pressure data (available at `https://doi.org/10.6084/m9.figshare.c.8065798`)

> To run the rest of the repository, download `Raw_Data` from `https://doi.org/10.6084/m9.figshare.c.8065798` into the working directory.

---

## 2) Converters (Python)
**Tare definition:** *“Tare” is measured with the model powered on and the wind tunnel turned off.*

1. **`Converter_pressure_tap.py`**
   - Converts raw pressure-tap voltages to calibrated pressures  
   - Subtracts the corresponding tare (per channel)  
   - Writes to `Post_Data/.../Tared/` (mirrors raw structure)

2. **`Converter_load_cell.py`**
   - Converts raw 6-axis load-cell voltages to calibrated forces and moments  
   - Subtracts the corresponding tare (per axis)  
   - Writes to `Post_Data/.../Tared/` (mirrors raw structure)

**Notes**
- Intermediate “Test” and “Bias/Tare” outputs are saved to `Post_Data/.../Test` and `Post_Data/.../Bias`  
- Folder structure/filenames are preserved for cross-referencing

---

## 3) StatsCal (Python)
Computes **true error and statistics** from **time-series** data for both **pressure** and **force**, following: <http://www.metrology.pg.gda.pl/full/2010/M&MS_2010_003.pdf>.

1. **`StatsCal_L_D_stats_csv.py`**
   - Per-sample and aggregate statistics for lift/drag after **subtracting sting aerodynamic effects**  
   - Proper **error propagation**; produces **ΔL/D** summaries  
   - Outputs `*_truestats.csv` into **Tared** folders

2. **`StatsCal_Pressure_stats_csv.py`**
   - Per-sample and aggregate statistics for **pressure** with **error propagation**  
   - Produces **Δp** summaries  
   - Outputs `*_truestats.csv` into **Tared** folders

3. **`StatsSum_polar_force_coefficient_to_csv.py`**
   - Recursively collects **tared load-cell files** from **Polar** runs across all AoA  
   - Uses testing-day conditions to compute **airspeed** and **Reynolds number**  
   - Computes **CL, CD**, attaches airspeed/Reynolds, **propagates errors**, and **concatenates** to a single CSV in `Post_processed_force_coefficient/` (e.g., `Force_coefficient_AoA_*_stats.csv`)

---

## 4) StatsSum (Python)
Collects every `*_truestats.csv` produced by **StatsCal** and **concatenates** them into **meta-summary CSVs**.

1. **`StatsSum_all_force_to_one_csv.py`**
   - Recursively finds force/moment `*_truestats.csv`  
   - Extracts run metadata from folder paths; appends as columns  
   - Writes to `Post_processed_stats_summarized/`:  
     - `L_D_True_Stats_Iter_0.csv`  
     - `L_D_True_Stats_Iter_2.csv`

2. **`StatsSum_all_pressure_to_one_csv.py`**
   - Recursively finds pressure-tap `*_truestats.csv`  
   - Extracts matching metadata; appends as columns  
   - Writes one CSV **per Iter**: `Pressure_True_Stats_Iter_*.csv` in `Post_processed_stats_summarized/`

---

## 5) RepCheck (Python)
Checks **repeatability** between **Iteration 0** and **Iteration 1**.

- **Method:** Compute **RMSD** for each matching data point across iterations, then normalize by data range to report **NRMSD (%)**  
- **Input:** `Post_processed_stats_summarized/` (from StatsSum scripts)  
- **Output:** Text report of **NRMSD for pressure** and **NRMSD for force**

**Notes**
- NRMSD = RMSD / (max − min), per comparable group/configuration  
- Assumes summarized CSVs contain consistent metadata (e.g., `type_of_run`, `X`, `AoA`, `Iter`)

---

## 6) Mutual Information (R)
Quantifies how much information **pressure-set signals** share with **aerodynamic states**.

- **Input:** `Post_processed_stats_summarized/` meta-summary CSVs  
- **Output:**  
  - One **bin-selection** diagnostic plot  
  - Four MI result CSVs in `MI_results/`

**Notes**
- MI with controlled discretization/binning and reproducible seeds  
- Designed to compare across AoA, X, Iter, etc.

---

## 7) Model Sensitivity (R)
**Script:** Sensitivity analysis of different predictive models (LM, GAM, RF, XGBoost) for `delta_L_D_mean` using tap sets, AoA, and relative position (Δx, Δy, Δz). Evaluated via RMSE across feature sets.

**Inputs**
- `Post_processed_stats_summarized/L_D_True_Stats_Iter_0.csv`  
- `Post_processed_stats_summarized/L_D_True_Stats_Iter_1.csv`  
- `Post_processed_stats_summarized/Pressure_True_Stats_Iter_0.csv`  
- `Post_processed_stats_summarized/Pressure_True_Stats_Iter_1.csv`

**Outputs**
- RMSE comparison plot → `Preditive_model_data/plot/Model_performance_RMSE_Plot.svg`  
- (Optional) RMSE summary LaTeX table → `Latex_table_S1_RMSE.txt`

---

## 8) Predictive Model (Python)
XGBoost-based predictive modeling and Bayesian hyperparameter optimization for formation-flight analysis.

### 8.1 `PredModel_Bayes_Optimization_HyperP_XGBoost_P_to_LD.py`
- Bayesian optimization with **6-fold CV**, **1000 iterations**, **20 random initializations**, fixed random seed  
- **Inputs**
  - `Post_processed_stats_summarized/L_D_True_Stats_Iter_0.csv`  
  - `Post_processed_stats_summarized/L_D_True_Stats_Iter_1.csv`  
  - `Post_processed_stats_summarized/Pressure_True_Stats_Iter_0.csv`  
  - `Post_processed_stats_summarized/Pressure_True_Stats_Iter_1.csv`
- **Output**
  - CSV of Bayesian hyperparameter-tuning results
- **Note**
  - Each predictor configuration (tap count, AoA, XYZ) tuned independently  
  - Computationally intensive (~20 hours on Intel i9-10850K at full load)

### 8.2 `PredModel_delta_pressure_interpolator.py`
- Generates interpolated Δpressure dataset for overfitting sanity checks  
- **Input:** `Post_processed_stats_summarized/Pressure_True_Stats_Iter_1.csv`  
- **Output:** `Predictive_model_data/Interpolated_pressure_data/Pressure_True_Stats_Iter_1_Interpolated.csv`

### 8.3 `PredModel_final_train_and_test.py`
- Loads optimized hyperparameters and performs final XGBoost training  
- 6-fold CV, early stopping, multiple fixed seeds for robustness  
- Trained with the best_round from 6-fold CV on iteration 0 dataset, and test on iteration 1 dataset.
- **Inputs**
  - Same preprocessed CSVs as above  
  - Tuning CSVs from **8.1**  
  - Interpolated pressure data from **8.2**
- **Outputs**
  - Predicted vs actual ΔL/D (per tap configuration)  
  - Metrics (RMSE, NRMSE)  
  - Model prediction CSVs (output directory)

### 8.4 `PredModel_RMSE_summary_each_model_to_csv.py`
- Loads files from **8.3** and selects the **best RMSE** across five fixed-seed models  
- **Input:** Prediction CSVs from **8.3**  
- **Output:** One CSV with the best RMSE for each model

---

## 9) Plotters (Python)
Publication-ready plots from processed summary CSVs.

1. **`Plotter_force_coefficient.py`**
   - Plots **CL vs α**, **CD vs α**, combined **CL & CD vs α**, and **drag polar (CD vs CL)**  
   - **Input:** `Post_processed_force_coefficient/Force_Coefficients_AoA_All.csv` (from `StatsSum_polar_force_coefficient_to_csv.py`)  
   - **Outputs:** Saved to `Polar_Plot/`  
     - `CL_vs_alpha.svg`, `CD_vs_alpha.svg`, `CL_CD_vs_alpha.svg`, `Drag_Polar.svg`

2. **`Plotter_MI_taps.py`**
   - MI for individual taps and combinations  
   - **Inputs:** `MI_results/Individual_MI_results.csv`, `MI_results/Combination_MI_results.csv`  
   - **Outputs:** Saved to `MI_results/Plots/`  
     - `Individual_MI_results.svg` (MI per tap)  
     - `Combination_MI_vs_Num_Taps.svg` (MI vs number of taps)  
     - `Combination_MI_Local_Normalized.svg` (locally normalized summed MI)

3. **`Plotter_heatmap_test_L_D_and_RMSE.py`**
   - Predicted heatmaps from XGBoost results  
   - Actual heatmap from Iteration 1 (test set)  
   - Error heatmaps (prediction − actual)  
   - Overall summary RMSE heatmap for each model

4. **`Plotter_heatmap_from_experiment.py`**
   - All ΔL/D heatmaps from experiments (Iterations 0 and 1)  
   - All Δpressure heatmaps from experiments (Iterations 0 and 1)  
   - **Input:** `Post_processed_stats_summarized/`  
   - **Outputs:**  
     - 2 ΔL/D heatmaps  
     - 6 Δpressure heatmaps  
     - Saved under `Post_processed_stats_summarized/Heatmaps_from_Experiment/`

5. **`Plotter_colorbar.py`**
   - Generates standalone colorbars used for figure assembly

---

## 10) LaTeX Conversion (Python)
**`LatexConv_table_HyperP_ModelP.py`**  
Generates plain text from model performance and optimized hyperparameters for direct copy-paste into Overleaf/LaTeX.

---

## Appendix A. Notation & Definitions
- **ΔL/D (delta L/D):** Change in lift-to-drag metric (per StatsCal definitions)  
- **Δp (delta pressure):** Change in measured pressure  
- **Iter:** Iteration index (e.g., 0 or 1)  
- **NRMSD:** Normalized root-mean-square deviation = RMSD / (max − min)

---

## Appendix B. Paths & Outputs Summary
- **Raw data:** `Raw_Data/` (from Figshare `https://doi.org/10.6084/m9.figshare.c.8065798`)  
- **Converted (tared):** `Post_Data/.../Tared/`  
- **Force coefficients:** `Post_processed_force_coefficient/Force_Coefficients_AoA_All.csv`  
- **Meta summaries:** `Post_processed_stats_summarized/`
  - `L_D_True_Stats_Iter_0.csv`, `L_D_True_Stats_Iter_1.csv`
  - `Pressure_True_Stats_Iter_*.csv`
- **MI:** `MI_results/` (+ plots in `MI_results/Plots/`)  
- **Predictive model data:**  
  - Interpolated pressure: `Predictive_model_data/Interpolated_pressure_data/Pressure_True_Stats_Iter_1_Interpolated.csv`
- **Polar plots:** `Polar_Plot/`  
- **Experiment heatmaps:** `Post_processed_stats_summarized/Heatmaps_from_Experiment/`  
- **LaTeX text/tables:** output from `LatexConv_table_HyperP_ModelP.py`
