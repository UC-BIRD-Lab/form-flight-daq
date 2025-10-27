import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import glob

def Px2650_5BD5V_Pressure_Sensor_Convertion(voltage):
    """
    [General]
    Note: This funtion is for converting the raw Voltage reading to Preesure reading
    Model: PX2650-5BD5V 
    Sensing Range: +/- 5 inH2O
    Voltage Range: 0.2V - 5.2V

     Parameters
    ------------
        Voltage : Voltage value
    Return
    -----------
        Pressure : Pressure value 
    """
    # pressure = stagnation_pressure - static_pressure
    # P = M * (V - V_offset) + P_low
    # M = (Span of the transducer's pressure range)/(Span of the transducer's voltage range)
    # these values can be found on the pressure transducer's specification sheet for PX2650-5BD5V

    pressure = (voltage - 0.2)*(1244.2 + 1244.2)/(5.2 - 0.2) - 1244.2
    return pressure

def Px165_001B_Pressure_Sensor_Convertion(voltage):
    """
    [General]
    Note: This funtion is for converting the raw Voltage reading to Preesure reading
    Model: PX165-001B
    Sensing Range: +/- 1 inH2O
    Voltage Range: 0.5V - 4.5V

     Parameters
    ------------
        Voltage: Voltage value
    Return
    -----------
        Pressure : Pressure value
    """
    # pressure = stagnation_pressure - static_pressure
    # P = M * (V - V_offset) + P_low
    # M = (Span of the transducer's pressure range)/(Span of the transducer's voltage range)
    # these values can be found on the pressure transducer's specification sheet for PX165-001B

    pressure = (voltage - 0.5)*(248.84 + 248.84)/(4.5 - 0.5) - 248.84
    return pressure


# [CAUTION]: In order to calculate all the files, the following 3 functions must be created. If not, it won't work.
def process_and_save_tare_file(tare_full_path, post_bias_folder):
    """
    [Only for Formation Flight Project] : The first AI reading is for the Pitot_Pressure, and the remaining AIs are for Pressure_Tap_Pressure
    Note: This function is for reading each tare csv file, then post-processing and save to new bias_csv file under post_bias_folder
        
    Parameters
    ------------
        tare_full_path : the tare csv file path
        post_bias_folder : the post-processed csv file will be generated under this folder
    Return
    -----------
        tare_csv : bias_csv data file saved under post_bias_folder
    """
    # Ensure output directory exists
    if not os.path.exists(post_bias_folder):
        os.makedirs(post_bias_folder)

    # Read raw pressure tap voltage data from tare_full_path
    tare_csv = pd.read_csv(tare_full_path)

    # Apply the pitot tube pressure (first column in csv) calculation by using Px2650_5BD5V_Pressure_Sensor_Convertion function 
    tare_csv[tare_csv.columns[1]] = tare_csv[tare_csv.columns[1]].apply(Px2650_5BD5V_Pressure_Sensor_Convertion)

    # Apply the pressure taps pressure (second column to the remaining in csv) calculation by using Px165_001B_Pressure_Sensor_Convertion function 
    for column in tare_csv.columns[2:]:
        tare_csv[column] = tare_csv[column].apply(Px165_001B_Pressure_Sensor_Convertion)
    
    # Save processed Tare file
    tare_csv.to_csv(os.path.join(post_bias_folder, os.path.basename(tare_full_path)), index=False)
    return tare_csv

def process_and_save_test_file(test_full_path, post_test_folder):
    """
    [Only for Formation Flight Project] : The first AI reading is for the Pitot_Pressure, and the remaining AIs are for Pressure_Tap_Pressure
    Note: This function is for reading each test csv file, then post-processing and save to new test_csv file under post_test_folder
        
    Parameters
    ------------
        test_full_path : the test csv file path
        post_test_folder : the post-processed csv file will be generated under this folder
    Return
    -----------
        test_csv : test_csv data file saved under post_test_folder
    """

    # Ensure output directory exists
    if not os.path.exists(post_test_folder):
        os.makedirs(post_test_folder)

    # Read raw pressure tap voltage data from tare_full_path
    test_csv = pd.read_csv(test_full_path)

    # Apply the pitot tube pressure (first column in csv) calculation by using Px2650_5BD5V_Pressure_Sensor_Convertion function 
    test_csv[test_csv.columns[1]] = test_csv[test_csv.columns[1]].apply(Px2650_5BD5V_Pressure_Sensor_Convertion)

    # Apply the pressure taps pressure (second column to the remaining in csv) calculation by using Px165_001B_Pressure_Sensor_Convertion function 
    for column in test_csv.columns[2:]:
        test_csv[column] = test_csv[column].apply(Px165_001B_Pressure_Sensor_Convertion)
    
    # Save processed Test file
    test_csv.to_csv(os.path.join(post_test_folder, os.path.basename(test_full_path)), index=False)
    return test_csv

def process_all_pressure_taps_tests(type_of_run, tare_full_path, test_folder_path, post_bias_folder, post_test_folder, post_tared_folder):
    """
    [Only for Formation Flight Project] : The first AI reading is for the Pitot_Pressure, and the remaining AIs are for Pressure_Tap_Pressure
    Note: This function is for processing all test files and subtracting the average value in single Tare file. Then, saving all tared (subtracted) data file under post_tared_folder
    
    Parameters
    ------------
        tare_full_path : the tare csv file path
        test_folder_path : the test csv file folder path; this funtion will scan each csv file under this folder path
        post_test_folder : the post-processed test csv files will be generated under this folder
        post_bias_folder : the post-processed bias csv files will be generated under this folder
        post_tared_folder : the post-processed tared csv files will be generated under this folder
    Return
    -----------
        Nah
    """

    # Process and save the Tare file
    tare_csv = process_and_save_tare_file(tare_full_path, post_bias_folder)
    
    # Ensure output directory for Tared files exists
    if not os.path.exists(post_tared_folder):
        os.makedirs(post_tared_folder)

    # Calculate the mean value for each column in tare_csv
    mean_tare_values = tare_csv.mean()
    
    # Get the column order from the Tare DataFrame, excluding the first non-numeric column if necessary
    column_order = tare_csv.columns[1:]  # Adjust if the first column should be included
    
    # Process each Test file under the test_folder_path
    for filename in os.listdir(test_folder_path):
        if filename.endswith(".csv"):
            test_full_path = os.path.join(test_folder_path, filename)
            test_csv = process_and_save_test_file(test_full_path, post_test_folder)
            
            # Reorder test_csv columns to match tare_csv before subtraction
            test_csv = test_csv[column_order]
            
            # Subtract Mean Tare values from Test values, now that columns are aligned by name
            tared_taps_csv = test_csv - mean_tare_values[column_order]
            
            # Save the resulting subtracted DataFrame
            tared_taps_filename = filename.replace('Test', 'Tared')
            tared_taps_csv.to_csv(os.path.join(post_tared_folder, tared_taps_filename), index=False)
            print(f'Now processing: {type_of_run} - {tared_taps_filename}')


def process_pressure_taps_automated():
    # [Main Run Code]
    # [Hard Code] Reynolds number for this experiment
    Re = 65000
    
    # Polar runs
    for AoA in [str(a).replace('.', '_') for a in [-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]]:
        type_of_run = 'Polar'
        xpos = 'Nah'
        Iter = '1' if AoA in ['-5', '0'] else '0'
        Test_date = '2024_03_29' if AoA == '5' else '2024_04_02' if AoA in ["0", "-5"] and Iter == '1' else '2024_04_01'
        
        # Tare file
        tare_pressure_filename = f'Taps_Y_Nah_Z_Nah_Re_{Re}_tare_AoA_{AoA}_Iter_{Iter}.csv'
        tare_pressure_full_path = f'Raw_Data/{type_of_run}/X_{xpos}/Iter_{Iter}_{Test_date}/Tare/Taps/AoA_{AoA}/{tare_pressure_filename}'
        
        # Test folder
        test_pressure_folder_path = f'Raw_Data/{type_of_run}/X_{xpos}/Iter_{Iter}_{Test_date}/Test/Taps/AoA_{AoA}'

        file_count = len([f for f in os.listdir(test_pressure_folder_path) if f.endswith('.csv')])
        print(f'Processing {file_count} files for {type_of_run} - AoA_{AoA} - Iter_{Iter}')
        
        # Post-processed folders
        post_pressure_bias_folder = f'Post_Data/{type_of_run}/Bias/Taps/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
        post_pressure_test_folder = f'Post_Data/{type_of_run}/Test/Taps/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
        post_pressure_tared_folder = f'Post_Data/{type_of_run}/Tared/Taps/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
        
        
        process_all_pressure_taps_tests(type_of_run, tare_pressure_full_path, test_pressure_folder_path, post_pressure_bias_folder, post_pressure_test_folder, post_pressure_tared_folder)


    # Position runs
    AoA_values = ['0', '2_5', '5']
    X_positions = ['0', '1', '2']
    Test_dates = {
        '0': {
            '0': {'0': '2024_04_04', '1': '2024_04_04'},
            '1': {'0': '2024_04_03', '1': '2024_04_04'},
            '2': {'0': '2024_04_03', '1': '2024_04_04'}
        },
        '2_5': {
            '0': {'0': '2024_04_05', '1': '2024_04_08'},
            '1': {'0': '2024_04_05', '1': '2024_04_08'},
            '2': {'0': '2024_04_05', '1': '2024_04_09'}
        },
        '5': {
            '0': {'0': '2024_04_10', '1': '2024_04_10'},
            '1': {'0': '2024_04_10', '1': '2024_04_10'},
            '2': {'0': '2024_04_09', '1': '2024_04_10'}
        }
    }
    
    for AoA in AoA_values:
        for xpos in X_positions:
            for Iter in ['0', '1']:
                Test_date = Test_dates[AoA][xpos][Iter]
                type_of_run = 'Position'
                
                # Tare file
                tare_pressure_filename = f'Taps_Y_Nah_Z_Nah_Re_{Re}_tare_AoA_{AoA}_Iter_{Iter}.csv'
                tare_pressure_full_path = f'Raw_Data/{type_of_run}/X_{xpos}/Iter_{Iter}_{Test_date}/Tare/Taps/AoA_{AoA}/{tare_pressure_filename}'
                
                # Test folder
                test_pressure_folder_path = f'Raw_Data/{type_of_run}/X_{xpos}/Iter_{Iter}_{Test_date}/Test/Taps/AoA_{AoA}'

                # Post-processed folders
                post_pressure_bias_folder = f'Post_Data/{type_of_run}/Bias/Taps/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
                post_pressure_test_folder = f'Post_Data/{type_of_run}/Test/Taps/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
                post_pressure_tared_folder = f'Post_Data/{type_of_run}/Tared/Taps/X_{xpos}/Iter_{Iter}_{Test_date}/AoA_{AoA}/'
                
                file_count = len([f for f in os.listdir(test_pressure_folder_path) if f.endswith('.csv')])
                print(f'Processing {file_count} files for {type_of_run} - X_{xpos} - AoA_{AoA} - Iter_{Iter}')
                process_all_pressure_taps_tests(type_of_run, tare_pressure_full_path, test_pressure_folder_path, post_pressure_bias_folder, post_pressure_test_folder, post_pressure_tared_folder)

process_pressure_taps_automated()

