# Nares-inspired pressure sensing enhances aerodynamic state perception for formation flight
Github code for paper: Nares-inspired pressure sensing enhances aerodynamic state perception for formation flight

## Contents
This repo includes two python scripts, one containing the daqmx class, the other prompts the user for testing information (AoA, X,Y,Z, iteration,...) for file naming convention.

## Purpose
Used for collecting and saving pressure tap, force sensor, and pitot tube data during formation flight experiments. 

## Reference data
Box folder containing the MATLAB files for previous wind tunnel operation: https://ucdavis.box.com/s/df16xfmk344mtkfgysrjifpcb65wsea8

# Code Workflow
## Folder: Wind Tunnel Test Run Code
This folder contains scripts for collecting wind tunnel measurement data.
**Inputs:**  
1. DAQ system connected to all pressure taps and the pitot tube  
2. DAQ system connected to the load cell  
3. Ambient environmental conditions specified in `Testing_Day_Condition`

**Output:**  
A `Raw_Data` folder containing all measured force and pressure data. Avaliable at XXX figshare

## Folder: Raw Data folder

To make the code running, you need to download the `Raw_Data` from XXX to this working directory.

## Script: Converter

**Tare definition:** *“Tare” is measured with the model powered on and the wind tunnel turned off.*
1. **`Converter_pressure_tap.py`**  
   - Converts raw pressure-tap voltages to calibrated pressures.  
   - Subtracts the corresponding tare (per channel) from each test measurement.  
   - Writes the processed files to a `Post_Data/.../Tared` folder (mirroring the raw folder structure).

2. **`Converter_load_cell.py`**  
   - Converts raw 6-axis load-cell voltages to calibrated forces and moments.  
   - Subtracts the corresponding tare (per axis) from each test measurement.  
   - Writes the processed files to a `Post_Data/.../Tared` folder (mirroring the raw folder structure).

**Notes**
- Both scripts also save intermediate “Test” and “Bias/Tare” outputs in `Post_Data/.../Test` and `Post_Data/.../Bias` for traceability.  
- Folder structure and filenames are preserved so runs can be cross-referenced with raw data.

## Script: StatsCal

These scripts compute **true error and statistics** from **time-series** data for both **pressure** and **force** measurements, following the methodology described in this paper: <http://www.metrology.pg.gda.pl/full/2010/M&MS_2010_003.pdf>.

1. **`StatsCal_L_D_stats_csv.py`**  
   - Computes per-sample and aggregate statistics for lift and drag after **subtracting sting aerodynamic effects**.  
   - Performs proper **error propagation** and produces **ΔL/D** (delta L/D) summaries.  
   - Outputs files ending with `*_truestats.csv` in the **Tared** folder.

2. **`StatsCal_Pressure_stats_csv.py`**  
   - Computes per-sample and aggregate statistics for **pressure** with proper **error propagation**.  
   - Produces **Δp** (delta pressure) summaries.  
   - Outputs files ending with `*_truestats.csv` in the **Tared** folder.

3. **`StatsSum_polar_force_coefficient_to_csv.py`**
   - Recursively collects **tared load-cell files** from **Polar runs** across all tested **angles of attack**.
   - Reads **testing-day conditions** to compute **airspeed** from pitot-tube data and the corresponding **Reynolds number**.
   - Computes **lift/drag coefficients (CL, CD)**, attaches airspeed and Reynolds, **propagates errors**, and **concatenates** everything into a single summary CSV under `Post_processed_force_coefficient` folder (e.g., `Force_coefficient_AoA_ * _stats.csv`).

## Script: StatsSum

These scripts collect every `_truestats.csv` produced by the StatsCal step (in each run folder) and **concatenate them into metadata summary CSVs** for easy downstream analysis.

1. **`StatsSum_all_force_to_one_csv.py`**  
   - Recursively finds all `*_truestats.csv` files for **forces/moments** across all tested configurations.  
   - Extracts run metadata from the folder structure and appends it as columns.  
   - Concatenates rows in a consistent column order and writes:  
     - `L_D_True_Stats_Iter_0.csv`  
     - `L_D_True_Stats_Iter_2.csv`  
     into the `Post_processed_stats_summarized` folder.

2. **`StatsSum_all_pressure_to_one_csv.py`**  
   - Recursively finds all `*_truestats.csv` files for **pressure taps** across all tested configurations.  
   - Extracts the same run metadata from the folder structure and appends it as columns.  
   - Concatenates rows into a single table and writes one CSV **per Iter** as `Pressure_True_Stats_Iter_*.csv` under `Post_processed_stats_summarized`.

## Script: RepCheck

Checks the **repeatability** between **Iteration 0** and **Iteration 1** results.

- **Method:** Computes the **root-mean-square deviation (RMSD)** for each matching data point across Iters, then **normalizes by the data range** to report **NRMSD (%)**.
- **Input:** Summarized stats in `Post_processed_stats_summarized/` (from the `StatsSum_*` scripts).
- **Output:** A text report listing **NRMSD for pressure** and **NRMSD for force** datasets.

**Notes**
- NRMSD = RMSD / (max − min), computed per comparable group/configuration.
- Assumes the summarised CSVs contain consistent metadata (e.g., `type_of_run`, `X`, `AoA`, `Iter`) to align pairs between Iter 0 and Iter 1.
