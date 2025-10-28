import pdb
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
        'no_dof': no_dof,
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

def get_info_from_csvname(csv_file):
    # Extract information from file name
    file_name = os.path.basename(csv_file)
    file_info = file_name.split('_')  # Split file name by underscores

    ypos = file_info[2]  
    zpos = file_info[4]  

    if file_info[-7] == '2' and file_info[-6] == '5':
        AoA = '2_5'
    else:
        AoA = file_info[-6]

    return ypos, zpos, AoA

def calculate_self_L_D_stats(XPos, YPos, ZPos, AoA, no_samples):
    if XPos == 0 or XPos == 1:
        test_date = '2024_06_18'
    elif XPos == 2:
        test_date = '2024_06_20'

    sting_test_file_path = f'Post_Data/Polar/Test/Force/X_Nah/Iter_Nah_2024_06_20/AoA_Nah/Force_Y_0_Z_0_Re_65000_Test_AoA_Nah_Iter_Nah_2024_06_20.csv'
    sting_bias_file_path = f'Post_Data/Polar/Bias/Force/X_Nah/Iter_Nah_2024_06_20/AoA_Nah/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_Nah_Iter_Nah.csv'

    sting_test_df = pd.read_csv(sting_test_file_path)
    sting_bias_df = pd.read_csv(sting_bias_file_path)

    sting_lift_test_stats = timeseries_std(sting_test_df.iloc[:,1], no_samples)
    sting_lift_bias_stats = timeseries_std(sting_bias_df.iloc[:,1], no_samples)

    sting_drag_test_stats = timeseries_std(sting_test_df.iloc[:,0], no_samples)
    sting_drag_bias_stats = timeseries_std(sting_bias_df.iloc[:,0], no_samples)

    sting_lift_force_mean = sting_lift_test_stats['data_mean'] - sting_lift_bias_stats['data_mean']
    sting_lift_force_error = sd_add_sub(sting_lift_test_stats['data_std_true'], sting_lift_bias_stats['data_std_true'])
    sting_lift_force_std = sd_add_sub(sting_lift_test_stats['data_std_sample'], sting_lift_bias_stats['data_std_sample'])

    sting_drag_force_mean = sting_drag_test_stats['data_mean'] - sting_drag_bias_stats['data_mean']
    sting_drag_force_error = sd_add_sub(sting_drag_test_stats['data_std_true'], sting_drag_bias_stats['data_std_true'])
    sting_drag_force_std = sd_add_sub(sting_drag_test_stats['data_std_sample'], sting_drag_bias_stats['data_std_sample'])

    if AoA == "5":
        Iter = 0
        self_bias_file_path = f"Post_Data/Polar/bias/Force/X_nah/Iter_{Iter}_2024_03_29/AoA_{AoA}/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv"
        self_test_file_path = f"Post_Data/Polar/Test/Force/X_nah/Iter_{Iter}_2024_03_29/AoA_{AoA}/Force_Y_0_Z_0_Re_65000_Test_AoA_{AoA}_Iter_{Iter}_2024_03_29.csv"
    elif AoA == "-5" or AoA == "0":
        Iter = 1
        self_bias_file_path = f"Post_Data/Polar/Bias/Force/X_nah/Iter_{Iter}_2024_04_02/AoA_{AoA}/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv"
        self_test_file_path = f"Post_Data/Polar/Test/Force/X_nah/Iter_{Iter}_2024_04_02/AoA_{AoA}/Force_Y_0_Z_0_Re_65000_Test_AoA_{AoA}_Iter_{Iter}_2024_04_02.csv"
    else:
        Iter = 0
        self_bias_file_path = f"Post_Data/Polar/Bias/Force/X_nah/Iter_{Iter}_2024_04_01/AoA_{AoA}/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv"
        self_test_file_path = f"Post_Data/Polar/Test/Force/X_nah/Iter_{Iter}_2024_04_01/AoA_{AoA}/Force_Y_0_Z_0_Re_65000_Test_AoA_{AoA}_Iter_{Iter}_2024_04_01.csv"

    self_test_df = pd.read_csv(self_test_file_path)
    self_bias_df = pd.read_csv(self_bias_file_path)

    self_lift_test_stats = timeseries_std(self_test_df.iloc[:,1], no_samples)
    self_lift_bias_stats = timeseries_std(self_bias_df.iloc[:,1], no_samples)

    self_drag_test_stats = timeseries_std(self_test_df.iloc[:,0], no_samples)
    self_drag_bias_stats = timeseries_std(self_bias_df.iloc[:,0], no_samples)

    self_lift_force = self_test_df.iloc[:,1] - self_lift_bias_stats['data_mean']
    self_lift_force_mean = self_lift_test_stats['data_mean'] - self_lift_bias_stats['data_mean']
    self_lift_force_error = sd_add_sub(self_lift_test_stats['data_std_true'], self_lift_bias_stats['data_std_true'])
    self_lift_force_std = sd_add_sub(self_lift_test_stats['data_std_sample'], self_lift_bias_stats['data_std_sample'])

    self_drag_force = self_test_df.iloc[:,0] - self_drag_bias_stats['data_mean']
    self_drag_force_mean = self_drag_test_stats['data_mean'] - self_drag_bias_stats['data_mean']
    self_drag_force_error = sd_add_sub(self_drag_test_stats['data_std_true'], self_drag_bias_stats['data_std_true'])
    self_drag_force_std = sd_add_sub(self_drag_test_stats['data_std_sample'], self_drag_bias_stats['data_std_sample'])

    self_pure_lift_force_mean = self_lift_force_mean - sting_lift_force_mean
    self_pure_lift_force_true_error = sd_add_sub(self_lift_force_error, sting_lift_force_error)
    self_pure_lift_force_std = sd_add_sub(self_lift_force_std, sting_lift_force_std)

    self_pure_drag_force_mean = self_drag_force_mean - sting_drag_force_mean
    self_pure_drag_force_true_error = sd_add_sub(self_drag_force_error, sting_drag_force_error)
    self_pure_drag_force_std = sd_add_sub(self_drag_force_std, sting_drag_force_std)

    self_L_D_mean = self_pure_lift_force_mean / self_pure_drag_force_mean
    self_L_D_error = sd_mult_div(self_pure_lift_force_mean, self_pure_lift_force_true_error, self_pure_drag_force_mean, self_pure_drag_force_true_error, self_L_D_mean)
    self_L_D_std = sd_mult_div(self_pure_lift_force_mean, self_pure_lift_force_std, self_pure_drag_force_mean, self_pure_drag_force_std, self_L_D_mean)

    result = {
        'Self_L/D_mean': self_L_D_mean,
        'Self_L/D_error': self_L_D_error,
        'Self_L/D_std': self_L_D_std
    }
    
    return result
def process_csv_file_to_L_D(testdate, csv_file, empty_matrix, xpos, Iter):
    # Number of samples per reading
    no_samples = 10000

    # Extract information from file name
    ypos, zpos, AoA = get_info_from_csvname(csv_file)

    # Get the self L/D stats
    self_L_D_stats = calculate_self_L_D_stats(xpos, ypos, zpos, AoA, no_samples)
    self_L_D_mean = self_L_D_stats['Self_L/D_mean']
    self_L_D_error = self_L_D_stats['Self_L/D_error']
    self_L_D_std = self_L_D_stats['Self_L/D_std']

    # Get orginial converted test data and bias data from the information in tared csv files
    test_file_path = f'Post_Data/Position/Test/Force/X_{xpos}/Iter_{Iter}_{testdate}/AoA_{AoA}/Force_Y_{ypos}_Z_{zpos}_Re_65000_Test_AoA_{AoA}_Iter_{Iter}_{testdate}.csv'
    bias_file_path =  f'Post_Data/Position/Bias/Force/X_{xpos}/Iter_{Iter}_{testdate}/AoA_{AoA}/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv'
    
    if xpos == 0 or xpos == 1:
        sting_test_file_path = f'Post_Data/Position/Test/Force/X_{xpos}/Iter_Nah_2024_06_18/AoA_Nah/Force_Y_{ypos}_Z_{zpos}_Re_65000_Test_AoA_Nah_Iter_Nah_2024_06_18.csv'
        sting_bias_file_path = f'Post_Data/Position/Bias/Force/X_{xpos}/Iter_Nah_2024_06_18/AoA_Nah/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_Nah_Iter_Nah.csv'
    elif xpos == 2:
        sting_test_file_path = f'Post_Data/Position/Tared/Force/X_{xpos}/Iter_Nah_2024_06_20/AoA_Nah/Force_Y_{ypos}_Z_{zpos}_Re_65000_Test_AoA_Nah_Iter_Nah_2024_06_20.csv'
        sting_bias_file_path = f'Post_Data/Position/Bias/Force/X_{xpos}/Iter_Nah_2024_06_20/AoA_Nah/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_Nah_Iter_Nah.csv'
    

    # Read the CSV file
    test_data = pd.read_csv(test_file_path)
    bias_data = pd.read_csv(bias_file_path)
    sting_test_data = pd.read_csv(sting_test_file_path)
    sting_bias_data = pd.read_csv(sting_bias_file_path)

    stat_lift_test = timeseries_std(test_data.iloc[:, 1], no_samples)
    stat_lift_bias = timeseries_std(bias_data.iloc[:, 1], no_samples)
    stat_lift_sting_test = timeseries_std(sting_test_data.iloc[:, 1], no_samples)
    stat_lift_sting_bias = timeseries_std(sting_bias_data.iloc[:, 1], no_samples)

    stat_drag_test = timeseries_std(test_data.iloc[:, 0], no_samples)
    stat_drag_bias = timeseries_std(bias_data.iloc[:, 0], no_samples)
    stat_drag_sting_test = timeseries_std(sting_test_data.iloc[:, 0], no_samples)
    stat_drag_sting_bias = timeseries_std(sting_bias_data.iloc[:, 0], no_samples)

    # Calculate the lift force by subtracting the mean bias from the test data
    lift_force = test_data.iloc[:, 1] - stat_lift_bias['data_mean']
    lift_force_mean = stat_lift_test['data_mean'] - stat_lift_bias['data_mean']
    lift_force_error = sd_add_sub(stat_lift_test['data_std_true'], stat_lift_bias['data_std_true'])
    lift_force_std = sd_add_sub(stat_lift_test['data_std_sample'], stat_lift_bias['data_std_sample'])

    # Calculate the drag forces by subtracting the mean bias from the test data
    drag_force = test_data.iloc[:, 0] - stat_drag_bias['data_mean']
    drag_force_mean = stat_drag_test['data_mean'] - stat_drag_bias['data_mean']
    drag_force_error = sd_add_sub(stat_drag_test['data_std_true'], stat_drag_bias['data_std_true'])
    drag_force_std = sd_add_sub(stat_drag_test['data_std_sample'], stat_drag_bias['data_std_sample'])

    # Calculate the lift forces caused by the sting by subtracting the mean bias from the test data
    sting_lift_force = sting_test_data.iloc[:, 1] - stat_lift_sting_bias['data_mean']
    sting_lift_force_mean = stat_lift_sting_test['data_mean'] - stat_lift_sting_bias['data_mean']
    sting_lift_force_error = sd_add_sub(stat_lift_sting_test['data_std_true'], stat_lift_sting_bias['data_std_true'])
    sting_lift_force_std = sd_add_sub(stat_lift_sting_test['data_std_sample'], stat_lift_sting_bias['data_std_sample'])

    # Calculate the drag forces caused by the sting by subtracting the mean bias from the test data
    sting_drag_force = sting_test_data.iloc[:, 0] - stat_drag_sting_bias['data_mean']
    sting_drag_force_mean = stat_drag_sting_test['data_mean'] - stat_drag_sting_bias['data_mean']
    sting_drag_force_error = sd_add_sub(stat_drag_sting_test['data_std_true'], stat_drag_sting_bias['data_std_true'])
    sting_drag_force_std = sd_add_sub(stat_drag_sting_test['data_std_sample'], stat_drag_sting_bias['data_std_sample'])

    # Calculate the pure lift force by subtracting the mean sting lift force from the lift force
    pure_lift_force = lift_force - sting_lift_force_mean
    pure_lift_force_mean = lift_force_mean - sting_lift_force_mean
    pure_lift_force_true_error = sd_add_sub(lift_force_error, sting_lift_force_error)

    pure_drag_force = drag_force - sting_drag_force_mean
    pure_drag_force_mean = drag_force_mean - sting_drag_force_mean
    pure_drag_force_true_error = sd_add_sub(drag_force_error, sting_drag_force_error)

    # Calculate the L/D ratio
    L_D_mean = (pure_lift_force_mean) / (pure_drag_force_mean)
    L_D_true_error = sd_mult_div(pure_lift_force_mean, pure_lift_force_true_error, pure_drag_force_mean, pure_drag_force_true_error, L_D_mean)

    ######ONLY FOR CALCULATING THE REGULAR STANDARD DEVIATION OF THE LIFT AND DRAG FORCES#######
    pure_lift_force_std = sd_add_sub(lift_force_std, sting_lift_force_std)
    pure_drag_force_std = sd_add_sub(drag_force_std, sting_drag_force_std)
    L_D_std = sd_mult_div(pure_lift_force_mean, pure_lift_force_std, pure_drag_force_mean, pure_drag_force_std, L_D_mean)

    delta_L_D_mean = L_D_mean - self_L_D_mean
    delta_L_D_error = sd_add_sub(L_D_true_error, self_L_D_error)
    delta_L_D_std = sd_add_sub(L_D_std, self_L_D_std)

    print(f"ypos: {ypos}, zpos: {zpos}, AoA: {AoA}, xpos: {xpos}, Iter: {Iter}")
    #print(f"delta_L_D_mean: {delta_L_D_mean}")
    
    # Create the true_stats dataframe
    true_stats = pd.DataFrame({'Lift Mean': pure_lift_force_mean,
                               'Lift True Error': pure_lift_force_true_error,
                               'Lift Std': pure_lift_force_std,
                               'Drag Mean': pure_drag_force_mean,
                               'Drag True Error': pure_drag_force_true_error,
                               'Drag Std': pure_drag_force_std,
                               'L/D Mean': L_D_mean,
                               'L/D True Error': L_D_true_error,
                               'L/D std': L_D_std, 
                               'Self_L/D_mean':self_L_D_mean,
                               'Self_L/D_error':self_L_D_error,
                               'Self_L/D_std':self_L_D_std,
                               'delta_L_D_mean':delta_L_D_mean,
                               'delta_L_D_error':delta_L_D_error,
                               'delta_L_D_std':delta_L_D_std},    
                               index = [0])

    # Save the statistics to a new CSV file
    true_stats_file = csv_file.replace(f"{testdate}.csv", f"{testdate}_truestats.csv")

    # if os.path.exists(true_stats_file):
    #     print(f"{true_stats_file} already exists")
    # else:
    true_stats.to_csv(true_stats_file)

    empty_matrix[int(zpos)+1, int(ypos)] = L_D_mean
    
    return empty_matrix

def process_csv_files_in_folder(testdate, folder, empty_matrix, xpos, Iter):
    # Iterate over each file in the folder
    for file_name in os.listdir(folder):
        if file_name.endswith(f"{testdate}.csv"):
            file_path = os.path.join(folder, file_name)
            L_D_matrix = process_csv_file_to_L_D(testdate, file_path, empty_matrix, xpos, Iter)

    return L_D_matrix


if __name__ == "__main__":
    # Get the list of all CSV files in the folder
    # --------------------------------------------
    for AoA in ['0', '2_5', '5']:
        no_samples = 10000

        # Getting information for all 6 planes with in 2 iterations.
        # ----------------------------------------------------------
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

        # Process all CSV files in the folder to matrix
        # -----------------------------------
        # X_0 ; Iteration 0
        # -----------------
        X_0_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_0/Iter_0_{X_0_Test_date_1}/AoA_{AoA}/'
        X_0_L_D_matrix_Iter_0 = process_csv_files_in_folder(X_0_Test_date_1, X_0_folder_path_Iter_0, X_0_empty_matrix_Iter_0, xpos = 0, Iter=0)

        # X_0 ; Iteration 1
        # -----------------
        X_0_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_0/Iter_1_{X_0_Test_date_2}/AoA_{AoA}/'
        X_0_L_D_matrix_Iter_1 = process_csv_files_in_folder(X_0_Test_date_2, X_0_folder_path_Iter_1, X_0_empty_matrix_Iter_1, xpos = 0, Iter=1)

        # X_1 ; Iteration 0
        # -----------------
        X_1_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_1/Iter_0_{X_1_Test_date_1}/AoA_{AoA}/'
        X_1_L_D_matrix_Iter_0 = process_csv_files_in_folder(X_1_Test_date_1, X_1_folder_path_Iter_0, X_1_empty_matrix_Iter_0, xpos = 1, Iter=0)

        # X_1 ; Iteration 1
        # -----------------
        X_1_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_1/Iter_1_{X_1_Test_date_2}/AoA_{AoA}/'
        X_1_L_D_matrix_Iter_1 = process_csv_files_in_folder(X_1_Test_date_2, X_1_folder_path_Iter_1, X_1_empty_matrix_Iter_1, xpos = 1, Iter=1)

        # X_2 ; Iteration 0
        # -----------------
        X_2_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_2/Iter_0_{X_2_Test_date_1}/AoA_{AoA}/'
        X_2_L_D_matrix_Iter_0 = process_csv_files_in_folder(X_2_Test_date_1, X_2_folder_path_Iter_0, X_2_empty_matrix_Iter_0, xpos = 2, Iter=0)

        # X_2 ; Iteration 1
        # -----------------
        X_2_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_2/Iter_1_{X_2_Test_date_2}/AoA_{AoA}/'
        X_2_L_D_matrix_Iter_1 = process_csv_files_in_folder(X_2_Test_date_2, X_2_folder_path_Iter_1, X_2_empty_matrix_Iter_1, xpos = 2, Iter=1)





