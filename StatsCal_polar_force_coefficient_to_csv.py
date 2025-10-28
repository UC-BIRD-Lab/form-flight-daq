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

################## Function to calculate the mean and standard deviation of a timeseries data ##################
def timeseries_std(timeseries, no_samples):  # input dataframe and number of samples per reading
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
    # varience_of_mean = no_samples(data_std_sample_ddof1)**2 / no_effective

    # data_std_true = (test/no_samples)**(0.5)
    data_std_true = (correction * (data_std_sample_ddof1) ** 2) ** (0.5)

    results = {
        'data_mean': data_mean,  
        'data_std_sample': data_std_sample,  
        'data_std_true': data_std_true,
        'no_effective': no_effective,
    }
    return results

################## ERROR PROPAGATION ##################
### For notes on error propagation see
### https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
### https://www.isobudgets.com/propagation-of-uncertainty-rules/

################## Function to calculate the standard deviation of addition and subtraction ##################
def sd_add_sub(error1, error2):
    adderror = abs((error1 ** 2 + error2 ** 2) ** 0.5)
    return adderror

################## Function to calculate the standard deviation of multiplication of constant ##################
def sd_mult_const(meanterm1, error1, constant):
    multerror = abs(constant * error1)
    return multerror

################## Function to calculate the standard deviation of multiplication and division ##################
def sd_mult_div(meanterm1, error1, meanterm2, error2, meanZ):
    multerror = abs((((error1 / meanterm1) ** 2 + (error2 / meanterm2) ** 2) ** 0.5) * meanZ)
    return multerror

################## Function to calculate the standard deviation of power ##################
def sd_power(meanterm1, error1, power, meanZ):
    powererror = abs(power * (error1 / meanterm1) * meanZ)
    return powererror

################## Function to calculate the standard deviation of sine and cosine ##################
def sd_sin(error1, mean1):
    sinerror = abs(error1) * np.cos(np.radians(mean1))
    sinerrortest = abs(np.cos(np.radians(mean1 + error1)) + np.cos(np.radians(mean1 - error1))) / 2
    return [sinerror, sinerrortest]

################## Function to calculate the standard deviation of sine and cosine ##################
def sd_cos(error1, mean1):
    coserror = abs(error1) * -np.sin(np.radians(mean1))
    coserrortest = abs(-np.sin(np.radians(mean1 + error1)) + np.sin(np.radians(mean1 - error1))) / 2
    return [coserror, coserrortest]

################## Function to calculate air density and its uncertainty ##################
def get_air_density_and_error(gap_pressure_gauge = 0.02, gap_tempeture_gauge_in_F = 1):
# The function takes two arguments:
#       gap_pressure_gauge: the smallest measurement gap for the pressure gauge in inHg
#       gap_tempeture_gauge_in_F: the smallest measurement gap for the temperature gauge in Fahrenheit
# The function returns two lists:
#       densities: a list of air densities in kg/m^3
#       errors: a list of uncertainties in air densities in kg/m^3
# The function requires a CSV file named 'Testing_Day_Condition.csv' with the following columns:
#       Date: the date of the test
#       Pressure (inHg): the atmospheric pressure in inHg
#       Temperature (C): the atmospheric temperature in Celsius (Can be converted by excel calculation)
#       Temperature (F): the atmospheric temperature in Fahrenheit
# [Note]: The smallest measurement gap for the pressure gauge and temperature gauge in the wind tunnel gauges are: Different experiments may have different smallest measurement gaps
#       gap_pressure_gauge = 0.02 inHg
#       gap_tempeture_gauge_in_F = 1 Fahrenheit
# The function saves the air densities and uncertainties to a new CSV file named 'Testing_Day_Condition_With_Density_And_Error.csv'
    
    # Load the test day conditions from a CSV file
    test_day_condition = pd.read_csv('Testing_Day_Condition.csv')

    # Extract the atmospheric pressure and temperature
    p_atm = test_day_condition.iloc[:, 1]
    t_atm_F = test_day_condition.iloc[:, 3]

    # Create lists to store the air densities and uncertainties
    densities = []
    errors = []
    
    # Calculate the air density and its uncertainty
    for i in range(len(p_atm)):
        p = p_atm[i]
        t_F = t_atm_F[i]
        t_K = (t_F - 32) * 5/9 + 273.15  # Convert to Kelvin directly
        
        # Calculate the air density
        density = p * 3386.39 / (286.9 * t_K)
        densities.append(density)
        
        # Estimate uncertainties
        delta_p = gap_pressure_gauge/ 2
        delta_t_F = gap_tempeture_gauge_in_F / 2
        delta_t_C = delta_t_F * 5 / 9  # Convert error to Celsius
        delta_t_K = delta_t_C  # Error in Celsius is the same in Kelvin
        
        # Error porpagation for air density with the following formula:
        term1 = (3386.39 * delta_p / (286.9 * t_K)) ** 2
        term2 = (p * 3386.39 * delta_t_K / (286.9 * t_K**2)) ** 2
        delta_density = np.sqrt(term1 + term2)
        errors.append(delta_density)

    # Add densities and errors to the DataFrame and save to CSV
    test_day_condition['Density (kg/m^3)'] = densities
    test_day_condition['Density Error (+/- kg/m^3)'] = errors
    test_day_condition.to_csv('Testing_Day_Condition_With_Density_And_Error.csv', index=False)

    return 

################## Function to calculate the velocity and its uncertainty ##################
def airspeed_from_pressure_and_error(dynamic_pressure, dynamic_pressure_error, density, density_error, K_constant=1):
# The function takes 3 arguments:
#       dynamic_pressure: the dynamic pressure in Pa
#       dynamic_pressure_error: the uncertainty in the dynamic pressure in Pa
#       density_error: the uncertainty in the air density in kg/m^3
#       density: the air density in kg/m^3
#       K_constant: the constant for the pitot tube

    # Calculate the airspeed from dynamic pressure and air density
    velocity = np.sqrt(2 * dynamic_pressure / (density * K_constant))
    
    # Calculate the intermediate term
    upper_term = 2 * dynamic_pressure
    upper_term_error = sd_mult_const(dynamic_pressure, dynamic_pressure_error, 2)

    lower_term = density * K_constant
    lower_term_error = sd_mult_const(density, density_error, K_constant)

    intermediate_mean = upper_term / lower_term
    intermediate_error = sd_mult_div(upper_term, upper_term_error, lower_term, lower_term_error, intermediate_mean)
    
    velocity_error = sd_power(intermediate_mean, intermediate_error, 0.5, velocity)

    return velocity, velocity_error

################## Function to calculate the force coefficient and its uncertainty ##################
def calculate_lift_drag_coefficient_stats(density_csv_path,
                                        reference_wing_area, 
                                        no_samples,
                                        AoA):
# The function takes four arguments:
#       density_csv_path: the path to the CSV file containing the air density and uncertainty
#       reference wing area: the reference wing area in m^2
#       no_samples: the number of samples per reading
#       AoA: the angle of attack
    sting_test_file_path = 'Post_Data/Polar/Test/Force/X_Nah/Iter_Nah_2024_06_20/AoA_Nah/Force_Y_0_Z_0_Re_65000_Test_AoA_Nah_Iter_Nah_2024_06_20.csv'
    sting_bias_file_path = 'Post_Data/Polar/Bias/Force/X_Nah/Iter_Nah_2024_06_20/AoA_Nah/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_Nah_Iter_Nah.csv'
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
        test_date    = '2024_03_29'
        test_time    = '2024_03_29_NN'
        Iter = 0
        self_bias_file_path = f"Post_Data/Polar/bias/Force/X_nah/Iter_{Iter}_2024_03_29/AoA_{AoA}/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv"
        self_test_file_path = f"Post_Data/Polar/Test/Force/X_nah/Iter_{Iter}_2024_03_29/AoA_{AoA}/Force_Y_0_Z_0_Re_65000_Test_AoA_{AoA}_Iter_{Iter}_2024_03_29.csv"
    elif AoA == "-5" or AoA == "0":
        test_date    = '2024_04_02'
        test_time    = '2024_04_02_NN'
        Iter = 1
        self_bias_file_path = f"Post_Data/Polar/Bias/Force/X_nah/Iter_{Iter}_2024_04_02/AoA_{AoA}/Force_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv"
        self_test_file_path = f"Post_Data/Polar/Test/Force/X_nah/Iter_{Iter}_2024_04_02/AoA_{AoA}/Force_Y_0_Z_0_Re_65000_Test_AoA_{AoA}_Iter_{Iter}_2024_04_02.csv"
    else:
        test_date    = '2024_04_01'
        test_time    = '2024_04_01_NN'
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

    density_dataframe = pd.read_csv(density_csv_path)
    density = density_dataframe.loc[density_dataframe['Date(YYYY_MM_DD_TT)'] == test_time, 'Density (kg/m^3)'].values[0]
    density_error = density_dataframe.loc[density_dataframe['Date(YYYY_MM_DD_TT)'] == test_time, 'Density Error (+/- kg/m^3)'].values[0]
    
    pressure_test_file = f'Post_Data/Polar/Test/Taps/X_Nah/Iter_{Iter}_{test_date}/AoA_{AoA}/Taps_Y_0_Z_0_Re_65000_Test_AoA_{AoA}_Iter_{Iter}.csv'
    pressure_bias_file = f'Post_Data/Polar/Bias/Taps/X_Nah/Iter_{Iter}_{test_date}/AoA_{AoA}/Taps_Y_Nah_Z_Nah_Re_65000_tare_AoA_{AoA}_Iter_{Iter}.csv'

    pressure_test_df = pd.read_csv(pressure_test_file)
    pressure_bias_df = pd.read_csv(pressure_bias_file)

    pitot_pressure_test_stats = timeseries_std(pressure_test_df.iloc[:,1], no_samples)
    pitot_pressure_bias_stats = timeseries_std(pressure_bias_df.iloc[:,1], no_samples)

    pitot_pressure_mean = pitot_pressure_test_stats['data_mean'] - pitot_pressure_bias_stats['data_mean']
    pitot_pressure_error = sd_add_sub(pitot_pressure_test_stats['data_std_true'], pitot_pressure_bias_stats['data_std_true'])
    pitot_pressure_std = sd_add_sub(pitot_pressure_test_stats['data_std_sample'], pitot_pressure_bias_stats['data_std_sample'])

    dynamic_pressure = pitot_pressure_mean
    dynamic_pressure_error = pitot_pressure_error
    dynamic_pressure_std = pitot_pressure_std

    # Velovity is calculated using the pitot tube constant K = 1
    # Velovity is in m/s
    velocity, velocity_error = airspeed_from_pressure_and_error(dynamic_pressure, dynamic_pressure_error, density, density_error) 

    # Calculate the lift coefficients
    #-------------------------------------------------------------------------------------------------------
    lift_coefficient = 2 * self_pure_lift_force_mean / (density * velocity**2 * reference_wing_area)

    # Intermediate mean values
    velocity_mean_2 = velocity**2
    velocity_error_2 = sd_power(velocity, velocity_error, 2, velocity_mean_2)

    lower_term_mean_1 = density * velocity_mean_2 
    lower_term_error_1 = sd_mult_div(density, density_error, velocity_mean_2, velocity_error_2, lower_term_mean_1)

    lower_term_mean = lower_term_mean_1 * reference_wing_area
    lower_term_error = sd_mult_const(lower_term_mean_1, lower_term_error_1, reference_wing_area)

    upper_term_mean = 2 * self_pure_lift_force_mean
    upper_term_error = sd_mult_const(self_pure_lift_force_mean, self_pure_lift_force_true_error, 2)

    # Calculate the propagated uncertainty for the coefficient
    lift_coefficient_error = sd_mult_div(upper_term_mean, upper_term_error, lower_term_mean, lower_term_error, lift_coefficient)
    
    # Calculate the drag coefficients
    #-------------------------------------------------------------------------------------------------------
    drag_coefficient = 2 * self_pure_drag_force_mean / (density * velocity**2 * reference_wing_area)

    # Intermediate mean values
    upper_term_mean = 2 * self_pure_drag_force_mean
    upper_term_error = sd_mult_const(self_pure_drag_force_mean, self_pure_drag_force_true_error, 2)
    
    # Calculate the propagated uncertainty for the coefficient
    drag_coefficient_error = sd_mult_div(upper_term_mean, upper_term_error, lower_term_mean, lower_term_error, drag_coefficient)

    results = {
        'Date': test_date,
        'AoA': AoA,
        'Density (kg/m^3)': density,
        'Density Error (+/- kg/m^3)': density_error,
        'Velocity (m/s)': velocity,
        'Velocity Error (+/- m/s)': velocity_error,
        'Lift Coefficient': lift_coefficient,
        'Lift Coefficient Error': lift_coefficient_error,
        'Drag Coefficient': drag_coefficient,
        'Drag Coefficient Error': drag_coefficient_error,
    }

    # Save the results to a CSV file
    results_df = pd.DataFrame([results])
    results_file_name = f'Post_processed_force_coefficient/Force_coefficient_AoA_{AoA}_stats.csv'
    if not os.path.exists('Post_processed_force_coefficient'):
        os.makedirs('Post_processed_force_coefficient')
    results_df.to_csv(results_file_name, index=False)

    return results

if __name__ == '__main__':
    # Check if the air density and uncertainty data already exist
    if not os.path.exists('Testing_Day_Condition_With_Density_And_Error.csv'):
        get_air_density_and_error()
    else:
        print('The air density and uncertainty data already exist in the file Testing_Day_Condition_With_Density_And_Error.csv')

    # Load the air density and uncertainty data
    no_samples = 10000
    ref_area = 0.04727702 * 0.2794 # Wing Refarence Area (m^2) S = chord * span due to rectangular wing
    density_info_csv_path = 'Testing_Day_Condition_With_Density_And_Error.csv'

    AoA = (-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15)
    AoA_str = [str(a).replace(".", "_") for a in AoA]

    for i in AoA_str:
        calculate_lift_drag_coefficient_stats(density_info_csv_path, ref_area, no_samples, i)
        print(f'Force coefficient calculation for AoA {i} is done!')
    
    # Combine all AoA results into one CSV file
    combined_results = []

    for i in AoA_str:
        results_file_name = f'Post_processed_force_coefficient/Force_coefficient_AoA_{i}_stats.csv'
        if os.path.exists(results_file_name):
            df = pd.read_csv(results_file_name)
            combined_results.append(df)

    if combined_results:
        combined_df = pd.concat(combined_results, ignore_index=True)
        combined_df.to_csv('Post_processed_force_coefficient/Force_Coefficients_AoA_All.csv', index=False)
        print('All AoA results have been combined into Force_Coefficients_AoA_All.csv')
    else:
        print('No results found to combine.')
    



