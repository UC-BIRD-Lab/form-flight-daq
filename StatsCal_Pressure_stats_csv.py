import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import csv
import glob
import os
import seaborn as sns

def get_info_from_csvname(csv_file):
    # Extract information from file name
    file_name = os.path.basename(csv_file)
    file_info = file_name.split('_')  # Split file name by underscores

    ypos = file_info[2]  
    zpos = file_info[4]  

    if file_info[-4] == '2' and file_info[-3] == '5':
        AoA = '2_5'
    else:
        AoA = file_info[-3]

    return ypos, zpos, AoA

def timeseries_std(timeseries, no_samples):  # input voltage data and number of samples per reading
    data_mean = np.mean(timeseries)  # finds mean of the whole sample data
    data_std_sample = np.std(timeseries, ddof=0)  # finds standard deviation of the whole sample data
    data_std_sample_ddof1 = np.std(timeseries, ddof = 1) # finds STD of N-1
    no_lags = no_samples - 1  # Defining all the possible time lag comparisons
    autocor = pd.Series(sm.tsa.acf(timeseries, nlags=no_lags))


    # http://www.metrology.pg.gda.pl/full/2010/M&MS_2010_003.pdf
    sum_1 = 0  # This creates a variable called sum_1 is tau hat. This is the sum portion of predicted time
    sum_2 = 0  # This creates a variable called sum_2 which is essentially just the variance

    for k in range(0, no_lags):  # want to iterate over all 9,999 comparisons
        # According to EQN 12: 
        # Sum of (n-k)*p[k]/n where k = 1 to n-1
        # autocor[k] = p[k]
        sum_1 = sum_1 + (autocor[k] * ((no_samples - k) / no_samples))  


        # According to EQN 30:
        # Sum of p[k]^2 where k = 1 to n-1
        # autocor[k] = p[k]
        sum_2 = sum_2 + (autocor[k] ** 2)


    # EQN 12: Effective number of observations
    no_effective = no_samples / (1 + (2 * sum_1))  


    # EQN 30: Effective degrees of freedom 
    no_dof = (no_samples / (1 + (2 * sum_2))) - 1  


    # EQN 24b: Sample variance for uncorrelated data s^2 and the correction factor
    correction = no_effective * (no_samples - 1) / (no_samples * (no_effective - 1))


    # NOT SURE we need to use 25 instead of 24b hence the u(x) shown in page 14 is regaring to the variance of the mean.
    # EQN 25: Calculation of S_a^2(x_bar)
    #varience_of_mean = no_samples(data_std_sample_ddof1)**2 / no_effective

    # Page 14: 
    # u(x) = sqrt [ S_a^2(x_bar) ]

    #data_std_true = (test/no_samples)**(0.5)
    data_std_true = (correction * (data_std_sample_ddof1) ** 2) ** (0.5)

    results = {
        'data_mean': data_mean,  
        'data_std_sample': data_std_sample,  
        'data_std_true': data_std_true,
        'no_effctive': no_effective,
    }
    return results

################## ERROR PROPAGATION
### For notes on error propagation see
### https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
### https://www.isobudgets.com/propagation-of-uncertainty-rules/

def sd_add_sub(error1, error2):
    adderror = (error1 ** 2 + error2 ** 2) ** 0.5
    return adderror


def sd_mult_div(meanterm1, error1, meanterm2, error2, meanZ):
    multerror = (((error1 / meanterm1) ** 2 + (error2 / meanterm2) ** 2) ** 0.5) * meanZ
    return multerror


def sd_sin(error1, mean1):
    sinerror = abs(error1) * np.cos(np.radians(mean1))
    sinerrortest = abs(np.cos(np.radians(mean1 + error1)) + np.cos(np.radians(mean1 - error1))) / 2
    return [sinerror, sinerrortest]


def sd_cos(error1, mean1):
    coserror = abs(error1) * -np.sin(np.radians(mean1))
    coserrortest = abs(-np.sin(np.radians(mean1 + error1)) + np.sin(np.radians(mean1 - error1))) / 2
    return [coserror, coserrortest]
def get_ypos_zpos_from_csvname(csv_file):
    # Extract information from file name
    file_name = os.path.basename(csv_file)
    file_info = file_name.split('_')  # Split file name by underscores

    ypos = file_info[2]  
    zpos = file_info[4]  
    
    return ypos, zpos

def calculate_self_pressure_stats(AoA):
    if AoA == "5":
        test_date = '2024_03_29'
        Iter = 0

    elif AoA == "2_5":
        test_date = '2024_04_01'
        Iter = 0

    elif AoA == "0":
        test_date = '2024_04_02'
        Iter = 1

    bias_file_path = f'Post_Data/Polar/Bias/Taps/X_Nah/Iter_{Iter}_{test_date}/AoA_{AoA}/Taps_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv'
    test_file_path = f'Post_Data/Polar/Test/Taps/X_Nah/Iter_{Iter}_{test_date}/AoA_{AoA}/Taps_Y_0_Z_0_Re_65000_Test_AoA_{AoA}_Iter_{Iter}.csv'

    # Number of samples per reading
    no_samples = 10000

    # Read the CSV file
    test_df = pd.read_csv(test_file_path)
    bias_df = pd.read_csv(bias_file_path)

    num_columns = test_df.shape[1]

    # Initialize an empty DataFrame to store the results
    stats_tared_matrix = pd.DataFrame(index=test_df.columns[1:num_columns-1], columns=['self_data_mean', 'self_data_std_sample', 'self_data_std_true'])

    for i in range(1, num_columns):
        stats_bias = timeseries_std(bias_df.iloc[1:, i], no_samples)
        stats_test = timeseries_std(test_df.iloc[1:, i], no_samples)

        stats_tared_mean = stats_test['data_mean'] - stats_bias['data_mean']
        stats_error = sd_add_sub(stats_test['data_std_true'], stats_bias['data_std_true'])
        stats_std = sd_add_sub(stats_test['data_std_sample'], stats_bias['data_std_sample'])

        stats_tared_matrix.loc[test_df.columns[i], 'self_data_mean'] = stats_tared_mean
        stats_tared_matrix.loc[test_df.columns[i], 'self_data_std_sample'] = stats_std
        stats_tared_matrix.loc[test_df.columns[i], 'self_data_std_true'] = stats_error
    
    return stats_tared_matrix


def process_csv_file_to_pressure_stats(testdate, csv_file, empty_matrix, xpos, Iter):
    # Number of samples per reading
    no_samples = 10000

    # Extract information from file name
    ypos, zpos, AoA = get_info_from_csvname(csv_file)

    # Get orginial converted test data and bias data from the information in tared csv files
    
    test_file_path = f'Post_Data/Position/Test/Taps/X_{xpos}/Iter_{Iter}_{testdate}/AoA_{AoA}/Taps_Y_{ypos}_Z_{zpos}_Re_65000_Test_AoA_{AoA}_Iter_{Iter}.csv'
    bias_file_path =  f'Post_Data/Position/Bias/Taps/X_{xpos}/Iter_{Iter}_{testdate}/AoA_{AoA}/Taps_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv'
    
    # Read the CSV file
    test_df = pd.read_csv(test_file_path)
    bias_df = pd.read_csv(bias_file_path)

    num_columns = test_df.shape[1]
    
    # Initialize an empty DataFrame to store the results
    stats_tared_matrix = pd.DataFrame(index=test_df.columns[1:num_columns-1], columns=['data_mean', 'data_std_sample', 'data_std_true', 'delta_pressure_mean', 'delta_pressure_error', 'delta_pressure_std'])

    for i in range(1, num_columns):
        stats_bias = timeseries_std(bias_df.iloc[1:, i], no_samples)
        stats_test = timeseries_std(test_df.iloc[1:, i], no_samples)
 
        stats_tared_mean = stats_test['data_mean'] - stats_bias['data_mean']
        stats_error = sd_add_sub(stats_test['data_std_true'], stats_bias['data_std_true'])
        stats_std = sd_add_sub(stats_test['data_std_sample'], stats_bias['data_std_sample'])

        delta_stats_mean = stats_tared_mean - self_pressure_stats["self_data_mean"].iloc[i-1]
        delta_stats_error = sd_add_sub(stats_error, self_pressure_stats["self_data_std_true"].iloc[i-1])
        delta_stats_std = sd_add_sub(stats_std, self_pressure_stats["self_data_std_sample"].iloc[i-1])

        stats_tared_matrix.loc[test_df.columns[i], 'data_mean'] = stats_tared_mean
        stats_tared_matrix.loc[test_df.columns[i], 'data_std_sample'] = stats_std
        stats_tared_matrix.loc[test_df.columns[i], 'data_std_true'] = stats_error

        stats_tared_matrix.loc[test_df.columns[i], 'delta_pressure_mean'] = delta_stats_mean
        stats_tared_matrix.loc[test_df.columns[i], 'delta_pressure_error'] = delta_stats_error
        stats_tared_matrix.loc[test_df.columns[i], 'delta_pressure_std'] = delta_stats_std


    
    print(f"ypos: {ypos}, zpos: {zpos}, AoA: {AoA}, xpos: {xpos}, Iter: {Iter}")
    # Save the statistics to a new CSV file
    stats_file = csv_file.replace(f"Iter_{Iter}.csv", f"Iter_{Iter}_truestats.csv")

    if os.path.exists(stats_file):
        print(f"{stats_file} already exists")

    stats_tared_matrix.to_csv(stats_file)

    return stats_tared_matrix

def process_csv_files_in_folder(testdate, folder, empty_matrix, xpos, Iter):
    # Iterate over each file in the folder
    for file_name in os.listdir(folder):
        if file_name.endswith(f"Iter_{Iter}.csv"):
            file_path = os.path.join(folder, file_name)
            varying_pressure_matrix = process_csv_file_to_pressure_stats(testdate, file_path, empty_matrix, xpos, Iter)

    return varying_pressure_matrix

if __name__ == "__main__":
    # Get the list of all CSV files in the folder
    # --------------------------------------------
    for AoA in ['0', '2_5', '5']:
        # Pure Pressure value for the given AoA from processed polar data.
        # ---------------- -----------------------------------------------
        if AoA == '0':
            X_0_Test_date_1 = '2024_04_04'
            X_0_Test_date_2 = '2024_04_04'
            X_1_Test_date_1 = '2024_04_03'
            X_1_Test_date_2 = '2024_04_04'
            X_2_Test_date_1 = '2024_04_03'
            X_2_Test_date_2 = '2024_04_04'
        elif AoA == '2_5':
            X_0_Test_date_1 = '2024_04_05'
            X_0_Test_date_2 = '2024_04_08'
            X_1_Test_date_1 = '2024_04_05'
            X_1_Test_date_2 = '2024_04_08'
            X_2_Test_date_1 = '2024_04_05'
            X_2_Test_date_2 = '2024_04_09'
        elif AoA == '5':
            X_0_Test_date_1 = '2024_04_10'
            X_0_Test_date_2 = '2024_04_10'
            X_1_Test_date_1 = '2024_04_10'
            X_1_Test_date_2 = '2024_04_10'
            X_2_Test_date_1 = '2024_04_09'
            X_2_Test_date_2 = '2024_04_10'

        else:
            print("Invalid AoA value. Please enter a valid AoA value.")

        # Initialize L/D matrix for all 6 planes with in 2 iterations.
        # ------------------------------------------------------------

        X_0_empty_matrix_Iter_0 = np.zeros((4,8))
        X_0_empty_matrix_Iter_1 = np.zeros((4,8))
        X_1_empty_matrix_Iter_0 = np.zeros((4,8))
        X_1_empty_matrix_Iter_1 = np.zeros((4,8))
        X_2_empty_matrix_Iter_0 = np.zeros((4,8))
        X_2_empty_matrix_Iter_1 = np.zeros((4,8))

        self_pressure_stats = calculate_self_pressure_stats(AoA)

        # Process all CSV files in the folder to matrix
        # -----------------------------------
        # X_0 ; Iteration 0
        # -----------------
        X_0_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_0/Iter_0_{X_0_Test_date_1}/AoA_{AoA}/'
        X_0_pressure_matrix_Iter_0 = process_csv_files_in_folder(X_0_Test_date_1, X_0_folder_path_Iter_0, X_0_empty_matrix_Iter_0, xpos = 0, Iter=0)

        # X_0 ; Iteration 1
        # -----------------
        X_0_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_0/Iter_1_{X_0_Test_date_2}/AoA_{AoA}/'
        X_0_pressure_matrix_Iter_1 = process_csv_files_in_folder(X_0_Test_date_2, X_0_folder_path_Iter_1, X_0_empty_matrix_Iter_1, xpos = 0, Iter=1)

        # X_1 ; Iteration 1
        # -----------------
        X_1_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_1/Iter_1_{X_1_Test_date_2}/AoA_{AoA}/'
        X_1_pressure_matrix_Iter_1 = process_csv_files_in_folder(X_1_Test_date_2, X_1_folder_path_Iter_1, X_1_empty_matrix_Iter_1, xpos = 1, Iter=1)


        # X_1 ; Iteration 0
        # -----------------
        X_1_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_1/Iter_0_{X_1_Test_date_1}/AoA_{AoA}/'
        X_1_pressure_matrix_Iter_0 = process_csv_files_in_folder(X_1_Test_date_1, X_1_folder_path_Iter_0, X_1_empty_matrix_Iter_0, xpos = 1, Iter=0)

        # X_2 ; Iteration 0
        # -----------------
        X_2_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_2/Iter_0_{X_2_Test_date_1}/AoA_{AoA}/'
        X_2_pressure_matrix_Iter_0 = process_csv_files_in_folder(X_2_Test_date_1, X_2_folder_path_Iter_0, X_2_empty_matrix_Iter_0, xpos = 2, Iter=0)

        # X_2 ; Iteration 1
        # -----------------
        X_2_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_2/Iter_1_{X_2_Test_date_2}/AoA_{AoA}/'
        X_2_pressure_matrix_Iter_1 = process_csv_files_in_folder(X_2_Test_date_2, X_2_folder_path_Iter_1, X_2_empty_matrix_Iter_1, xpos = 2, Iter=1)
