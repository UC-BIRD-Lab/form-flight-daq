import pandas as pd
import numpy as np
import glob
import os
from concurrent.futures import ProcessPoolExecutor

# Constants / ATI calibration matrix (6x6)
RUNTIME_MTX = np.array([
    [ 0.12855,  -0.11201,  0.70170, 12.66216, -0.61082, -11.81956],
    [-0.86417, -14.75965,  0.56375,  7.11891,  0.22851,   7.00026],
    [21.06874,  -0.36473, 20.65335, -0.24244, 20.48729,  -0.64551],
    [-0.00802,  -0.08189,  0.29044,  0.03956, -0.29101,   0.04374],
    [-0.34799,   0.00134,  0.16225, -0.06976,  0.17066,   0.06300],
    [-0.00997,  -0.17803, -0.00990, -0.17682, -0.00807,  -0.16678]
], dtype=np.float64)

USECOLS = [1,2,3,4,5,6]  # skip first index/time column

def load_voltages_csv(path: str) -> np.ndarray:
    """
    Load a single CSV of load-cell voltages and return the 6 measurement columns as a NumPy array.

    Parameters
    ----------
    path : str
        File system path to a CSV containing raw voltage readings. The function
        reads only columns 1–6 (skipping the first index/time column), and casts
        values to float32 for memory efficiency.

    Returns
    -------
    np.ndarray
        Array of shape (N, 6) where N is the number of samples; columns are the
        six bridge voltages in the order expected by the ATI runtime matrix.

    Notes
    -----
    - If your CSVs have headers, `pandas.read_csv` will infer them; if they do not,
      set `header=None` in this function as needed.
    - This function does no filtering or alignment; it is a thin, fast loader.
    """
    df = pd.read_csv(path, usecols=USECOLS, dtype=np.float32)
    return df.values  

def process_one_test_file(test_full_path: str,
                          test_date: str,
                          type_of_run: str,
                          post_bias_path: str,
                          post_test_dir: str,
                          post_tared_dir: str,
                          bias_mean_ft: np.ndarray):
    """
    Convert one test CSV from voltages to forces/torques and write gage and tared outputs.

    Workflow
    --------
    1) Read the test voltages (Nx6) from `test_full_path`.
    2) Multiply by the transposed ATI runtime calibration matrix to get gage forces/torques (Nx6).
    3) Subtract the bias mean vector (1x6) computed from the tare to produce tared data (Nx6).
    4) Write both gage and tared arrays as CSVs to the specified output folders.
    5) Print a progress line (matching your existing console output style).

    Parameters
    ----------
    test_full_path : str
        Path to a single test CSV (raw voltages).
    test_date : str
        Date string appended to the output filename (e.g., '2024_04_04').
    type_of_run : str
        High-level run type label (e.g., 'Polar' or 'Position') used in logging.
    post_bias_path : str
        Path to the saved bias CSV (not read here; kept for symmetry with other APIs).
    post_test_dir : str
        Directory where the gage (un-tared) force/torque CSV will be written.
    post_tared_dir : str
        Directory where the tared force/torque CSV will be written.
    bias_mean_ft : np.ndarray
        Vector of shape (6,) containing the per-axis mean bias in force/torque units.

    Returns
    -------
    None
        Results are written to disk; no value is returned.

    Outputs
    -------
    <post_test_dir>/<basename>_<test_date>.csv   # gage (Nx6)
    <post_tared_dir>/<basename>_<test_date>.csv  # tared (Nx6)

    Notes
    -----
    - Uses vectorized NumPy `@` multiply; no per-row Python loops.
    - Output files are written without headers/index for compactness.
    """
    # Read test 
    test_v = load_voltages_csv(test_full_path)       
    ft_gage = test_v @ RUNTIME_MTX.T                 
    ft_tared = ft_gage - bias_mean_ft[None, :]        

    # Compose output filenames
    test_filename = os.path.basename(test_full_path)
    out_name = f'{os.path.splitext(test_filename)[0]}_{test_date}.csv'
    post_gage_file = os.path.join(post_test_dir, out_name)
    post_tared_file = os.path.join(post_tared_dir, out_name)

    # Write gage & tared 
    pd.DataFrame(ft_gage).to_csv(post_gage_file, index=False, header=False, float_format='%.8f')
    pd.DataFrame(ft_tared).to_csv(post_tared_file, index=False, header=False, float_format='%.8f')

    print(f'Now processing: {type_of_run} - {out_name}')

def process_each_file(type_of_run, test_date, new_tare_csv, tare_full_path,
                      test_files, post_bias_dir, post_test_dir, post_tared_dir,
                      parallel_workers=0):
    """
    End-to-end processing for one (tare, test_files[]) group:
    - Load tare voltages, convert to bias forces/torques, save full bias CSV, compute mean bias.
    - For each test file, convert voltages to gage forces/torques and write both gage and tared CSVs.

    Parameters
    ----------
    type_of_run : str
        Run category label (e.g., 'Polar' or 'Position') used in logging.
    test_date : str
        Date stamp appended to per-test output filenames.
    new_tare_csv : str
        Filename to use when saving the post-processed bias CSV.
    tare_full_path : str
        Path to the tare CSV (raw voltages) corresponding to these test files.
    test_files : list[str]
        List of file paths to test CSVs to process.
    post_bias_dir : str
        Output directory for the bias CSV (per-sample forces/torques from the tare).
    post_test_dir : str
        Output directory for per-test gage forces/torques.
    post_tared_dir : str
        Output directory for per-test tared forces/torques.

    Returns
    -------
    None
        Writes multiple CSVs to disk; no value is returned.

    Side Effects
    ------------
    - Creates `post_bias_dir`, `post_test_dir`, and `post_tared_dir` if they do not exist.
    - Prints a progress line for each processed test file.

    Notes
    -----
    - Bias mean is computed from *all* samples in the tare CSV after conversion to FT units.
    - Gage and tared outputs are written without headers for compact storage.
    """
    os.makedirs(post_bias_dir, exist_ok=True)
    os.makedirs(post_test_dir, exist_ok=True)
    os.makedirs(post_tared_dir, exist_ok=True)

    # Load tare once (all samples, 6 cols)
    tare_v = load_voltages_csv(tare_full_path)          

    ft_bias = tare_v @ RUNTIME_MTX.T                    
    # Save full bias 
    post_bias_file = os.path.join(post_bias_dir, new_tare_csv)
    pd.DataFrame(ft_bias).to_csv(post_bias_file, index=False, header=False, float_format='%.8f')

    bias_mean_ft = ft_bias.mean(axis=0)                 

    for f in test_files:
        process_one_test_file(f, test_date, type_of_run,
                                post_bias_file, post_test_dir, post_tared_dir, bias_mean_ft)

def process_load_cell_automated():
    """
    Automated post-processing for both Polar and Position.

    What it does
    ------------
    - Defines campaign metadata (AoA lists, iteration/date rules, and folder schema).
    - For each (tare, test_folder) combination:
        * Finds all test CSVs via `glob`.
        * Ensures output directory structure exists.
        * Calls `process_each_file` to produce:
            - Bias FT CSV (per-sample)
            - Gage FT CSVs for each test
            - Tared FT CSVs for each test
    - Mirrors your existing folder and filename conventions exactly.

    Constants
    ---------
    Re : int
        Reynolds number used to build tare/test file paths.

    Outputs
    -------
    Writes CSV files under:
      Post_Data/<type>/Bias/Force/...
      Post_Data/<type>/Test/Force/...
      Post_Data/<type>/Tared/Force/...

    Notes
    -----
    """
    Re = 65000
    
    # Polar runs
    aoa_list = [str(a).replace('.', '_') for a in [-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15, 'Nah']]

    for AoA in aoa_list:
        type_of_run = 'Polar'
        xpos = 'Nah'
        Iter = 'Nah' if AoA == 'Nah' else ('1' if AoA in ['-5', '0'] else '0')

        # Match your ternary chain for dates
        if AoA == 'Nah':
            Test_date = '2024_06_20'
        elif AoA == '5':
            Test_date = '2024_03_29'
        elif (AoA in ['0', '-5']) and (Iter == '1'):
            Test_date = '2024_04_02'
        else:
            Test_date = '2024_04_01'

        # Inputs
        tare_name = f'Force_Y_Nah_Z_Nah_Re_{Re}_tare_AoA_{AoA}_Iter_{Iter}.csv'
        tare_path = f'Raw_Data/{type_of_run}/X_{xpos}/Iter_{Iter}_{Test_date}/Tare/Force/AoA_{AoA}/{tare_name}'
        test_files = glob.glob(f'Raw_Data/{type_of_run}/X_{xpos}/Iter_{Iter}_{Test_date}/Test/Force/AoA_{AoA}/*.csv')

        # Outputs
        post_bias_dir = f'Post_Data/{type_of_run}/Bias/Force/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
        post_test_dir = f'Post_Data/{type_of_run}/Test/Force/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
        post_tared_dir = f'Post_Data/{type_of_run}/Tared/Force/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'

        os.makedirs(post_bias_dir, exist_ok=True)
        os.makedirs(post_test_dir, exist_ok=True)
        os.makedirs(post_tared_dir, exist_ok=True)

        print(f'Processing {len(test_files)} files for {type_of_run} - AoA_{AoA} - Iter_{Iter}')

        process_each_file(
            type_of_run=type_of_run,
            test_date=Test_date,
            new_tare_csv=tare_name,
            tare_full_path=tare_path,
            test_files=test_files,
            post_bias_dir=post_bias_dir,
            post_test_dir=post_test_dir,
            post_tared_dir=post_tared_dir,
            parallel_workers=0
        )

    # Position runs
    AoA_values = ['0', '2_5', '5', 'Nah']
    X_positions = ['0', '1', '2']

    Test_dates = {
        'Nah': {'0': {'Nah': '2024_06_18'}, '1': {'Nah': '2024_06_18'}, '2': {'Nah': '2024_06_20'}},
        '0':   {'0': {'0': '2024_04_04', '1': '2024_04_04'},
                '1': {'0': '2024_04_03', '1': '2024_04_04'},
                '2': {'0': '2024_04_03', '1': '2024_04_04'}},
        '2_5': {'0': {'0': '2024_04_05', '1': '2024_04_08'},
                '1': {'0': '2024_04_05', '1': '2024_04_08'},
                '2': {'0': '2024_04_05', '1': '2024_04_09'}},
        '5':   {'0': {'0': '2024_04_10', '1': '2024_04_10'},
                '1': {'0': '2024_04_10', '1': '2024_04_10'},
                '2': {'0': '2024_04_09', '1': '2024_04_10'}}
    }

    for AoA in AoA_values:
        for xpos in X_positions:
            for Iter in (['Nah'] if AoA == 'Nah' else ['0', '1']):
                Test_date = Test_dates[AoA][xpos][Iter]
                type_of_run = 'Position'

                # Inputs
                tare_name = f'Force_Y_Nah_Z_Nah_Re_{Re}_tare_AoA_{AoA}_Iter_{Iter}.csv'
                tare_path = f'Raw_Data/{type_of_run}/X_{xpos}/Iter_{Iter}_{Test_date}/Tare/Force/AoA_{AoA}/{tare_name}'
                test_files = glob.glob(f'Raw_Data/{type_of_run}/X_{xpos}/Iter_{Iter}_{Test_date}/Test/Force/AoA_{AoA}/*.csv')

                # Outputs
                post_bias_dir = f'Post_Data/{type_of_run}/Bias/Force/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
                post_test_dir = f'Post_Data/{type_of_run}/Test/Force/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
                post_tared_dir = f'Post_Data/{type_of_run}/Tared/Force/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'

                print(f'Processing {len(test_files)} files for {type_of_run} - AoA_{AoA} - Iter_{Iter}')

                process_each_file(type_of_run, Test_date, tare_name, tare_path, test_files,
                                  post_bias_dir, post_test_dir, post_tared_dir,
                                  parallel_workers=0)   

process_load_cell_automated()
