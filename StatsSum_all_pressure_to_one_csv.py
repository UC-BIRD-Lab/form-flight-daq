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

    if file_info[-5] == '2' and file_info[-4] == '5':
        AoA = '2_5'
    else:
        AoA = file_info[-4]

    return ypos, zpos, AoA

def reshape_tap_data(file_path):
    """
    Reads a CSV file containing tap and pitot data, reshapes the data, and returns a DataFrame with one row.

    Parameters:
    file_path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame: A reshaped DataFrame with one row containing the data.
    """
    
    # Read the CSV file into a DataFrame and ensure the index is set correctly
    df = pd.read_csv(file_path, index_col=0)

    # Initialize an empty dictionary to store the new columns
    reshaped_data = {}

    # Iterate over each row (tap/pitot) in the original DataFrame
    for tap_name, row in df.iterrows():
        reshaped_data[f'{tap_name}_data_mean'] = row['data_mean']
        reshaped_data[f'{tap_name}_data_std_sample'] = row['data_std_sample']
        reshaped_data[f'{tap_name}_data_std_true'] = row['data_std_true']
        reshaped_data[f'{tap_name}_delta_pressure_mean'] = row['delta_pressure_mean']
        reshaped_data[f'{tap_name}_delta_pressure_error'] = row['delta_pressure_error']
        reshaped_data[f'{tap_name}_delta_pressure_std'] = row['delta_pressure_std']

    # Convert the dictionary into a DataFrame with one row
    reshaped_df = pd.DataFrame([reshaped_data])

    return reshaped_df

def process_csv_files(testdate, folder_path, AoA, xpos, Iter):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(f'Iter_{Iter}_truestats.csv')]
    data = []
    columns = ['XPos', 'YPos', 'ZPos', 'AoA']

    chord_length = 11 # inches
    xpos = (int(xpos)+1) * chord_length * 0.75  # Convert to inches
    
    
    for csv_file in csv_files:
        ypos, zpos, AoA_value = get_info_from_csvname(csv_file)
        ypos = int(ypos) * chord_length * 0.25  # Convert to inches
        zpos = int(zpos) * chord_length * 0.25 
        truestat_filename = f'{folder_path}{csv_file}'
        truestat_reshaped_df = reshape_tap_data(truestat_filename)
        print(f'Processed {truestat_filename}')
        # Combine static columns with CSV columns
        if len(columns) == 4:  # Only update once
            columns.extend(truestat_reshaped_df.columns)
        
        for row in truestat_reshaped_df.itertuples(index=False):
            data.append([xpos, ypos, zpos, AoA_value, *row])
    
    new_df = pd.DataFrame(data, columns=columns)
    new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')]
    
    return new_df

if __name__ == "__main__":
    AoA = ['0', '2_5', '5']
    for Iter in ['0', '1']:
        for AoA_str in AoA:
            # Create a new folder called 
            truestats_folder = 'Post_processed_stats_summarized'
            os.makedirs(truestats_folder, exist_ok=True)
            for i in range(len(AoA)):
                AoA_str = str(AoA[i])
                if AoA_str == '0':
                    X_0_Test_date_1 = '2024_04_04'
                    X_0_Test_date_2 = '2024_04_04'
                    X_1_Test_date_1 = '2024_04_03'
                    X_1_Test_date_2 = '2024_04_04'
                    X_2_Test_date_1 = '2024_04_03'
                    X_2_Test_date_2 = '2024_04_04'
                    if Iter == '0':
                        X_0_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_0/Iter_0_{X_0_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_1, X_0_folder_path_Iter_0, AoA_str, xpos = "0", Iter = Iter)

                        X_1_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_1/Iter_0_{X_1_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_1, X_1_folder_path_Iter_0, AoA_str, xpos = "1", Iter = Iter)

                        X_2_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_2/Iter_0_{X_2_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_1, X_2_folder_path_Iter_0, AoA_str, xpos = "2", Iter = Iter)

                        new_df_X_0.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)

                    elif Iter == '1':
                        X_0_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_0/Iter_1_{X_0_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_2, X_0_folder_path_Iter_1, AoA_str, xpos = "0", Iter = Iter)

                        X_1_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_1/Iter_1_{X_1_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_2, X_1_folder_path_Iter_1, AoA_str, xpos = "1", Iter = Iter)

                        X_2_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_2/Iter_1_{X_2_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_2, X_2_folder_path_Iter_1, AoA_str, xpos = "2", Iter = Iter)

                        new_df_X_0.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        
                elif AoA_str == '2_5':
                    X_0_Test_date_1 = '2024_04_05'
                    X_0_Test_date_2 = '2024_04_08'
                    X_1_Test_date_1 = '2024_04_05'
                    X_1_Test_date_2 = '2024_04_08'
                    X_2_Test_date_1 = '2024_04_05'
                    X_2_Test_date_2 = '2024_04_09'
                    if Iter == '0':
                        X_0_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_0/Iter_0_{X_0_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_1, X_0_folder_path_Iter_0, AoA_str, xpos = "0", Iter = Iter)

                        X_1_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_1/Iter_0_{X_1_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_1, X_1_folder_path_Iter_0, AoA_str, xpos = "1", Iter = Iter)

                        X_2_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_2/Iter_0_{X_2_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_1, X_2_folder_path_Iter_0, AoA_str, xpos = "2", Iter = Iter)

                        new_df_X_0.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)

                    elif Iter == '1':
                        X_0_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_0/Iter_1_{X_0_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_2, X_0_folder_path_Iter_1, AoA_str, xpos = "0", Iter = Iter)

                        X_1_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_1/Iter_1_{X_1_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_2, X_1_folder_path_Iter_1, AoA_str, xpos = "1", Iter = Iter)

                        X_2_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_2/Iter_1_{X_2_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_2, X_2_folder_path_Iter_1, AoA_str, xpos = "2", Iter = Iter)

                        new_df_X_0.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)

                elif AoA_str == '5':
                    X_0_Test_date_1 = '2024_04_10'
                    X_0_Test_date_2 = '2024_04_10'
                    X_1_Test_date_1 = '2024_04_10'
                    X_1_Test_date_2 = '2024_04_10'
                    X_2_Test_date_1 = '2024_04_09'
                    X_2_Test_date_2 = '2024_04_10'
                    if Iter == '0':
                        X_0_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_0/Iter_0_{X_0_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_1, X_0_folder_path_Iter_0, AoA_str, xpos = "0", Iter = Iter)

                        X_1_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_1/Iter_0_{X_1_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_1, X_1_folder_path_Iter_0, AoA_str, xpos = "1", Iter = Iter)

                        X_2_folder_path_Iter_0 = f'Post_Data/Position/Tared/Taps/X_2/Iter_0_{X_2_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_1, X_2_folder_path_Iter_0, AoA_str, xpos = "2", Iter = Iter)

                        new_df_X_0.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)

                    elif Iter == '1':
                        X_0_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_0/Iter_1_{X_0_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_2, X_0_folder_path_Iter_1, AoA_str, xpos = "0", Iter = Iter)

                        X_1_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_1/Iter_1_{X_1_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_2, X_1_folder_path_Iter_1, AoA_str, xpos = "1", Iter = Iter)

                        X_2_folder_path_Iter_1 = f'Post_Data/Position/Tared/Taps/X_2/Iter_1_{X_2_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_2, X_2_folder_path_Iter_1, AoA_str, xpos = "2", Iter = Iter)

                        new_df_X_0.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/Pressure_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                print(f'Completed AoA {AoA_str} for Iteration {Iter}')
    print('All CSV files have been processed and summarized into a single CSV file per iteration.')







