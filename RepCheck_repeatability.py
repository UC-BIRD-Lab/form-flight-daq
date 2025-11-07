import numpy as np
import pandas as pd


def calculate_normalized_rmse(data1, data2):
    # Calculate the normalized RMSE between two sets of data
    # data1: the first set of data
    # data2: the second set of data
    # max: the maximum value of the data
    # min: the minimum value of the data

    n = len(data1)

    max_value = np.max([data1, data2])
    min_value = np.min([data1, data2])
    
    summed_diff_squared = 0
    for i in range(len(data1)):
        summed_diff_squared += (data1[i] - data2[i]) ** 2

    numerator = np.sqrt(summed_diff_squared/n)
    denominator = max_value - min_value
    NRMSD = numerator / denominator

    return NRMSD
    

# Load data from CSV files
force_data_0 = pd.read_csv('Post_processed_stats_summarized/L_D_True_Stats_Iter_0.csv')
force_data_1 = pd.read_csv('Post_processed_stats_summarized/L_D_True_Stats_Iter_1.csv')

pressure_data_0 = pd.read_csv('Post_processed_stats_summarized/Pressure_True_Stats_Iter_0.csv')
pressure_data_1 = pd.read_csv('Post_processed_stats_summarized/Pressure_True_Stats_Iter_1.csv')

# Calculate the NRMSD for the force dataset
lift_data_0 = force_data_0['Lift Mean']
lift_data_1 = force_data_1['Lift Mean']

drag_data_0 = force_data_0['Drag Mean']
drag_data_1 = force_data_1['Drag Mean']

lift_NRMSD = calculate_normalized_rmse(lift_data_0, lift_data_1)
drag_NRMSD = calculate_normalized_rmse(drag_data_0, drag_data_1)

force_data_NRMSD = (lift_NRMSD + drag_NRMSD) / 2
print('Force Data NRMSD: ', force_data_NRMSD)

# Calculate the NRMSD for the pressure dataset
pitot_data_0 = pressure_data_0['pitot_data_mean']
tap0_data_0 = pressure_data_0['tap0_data_mean']
tap2_data_0 = pressure_data_0['tap2_data_mean']
tap4_data_0 = pressure_data_0['tap4_data_mean']
tap6_data_0 = pressure_data_0['tap6_data_mean']
tap8_data_0 = pressure_data_0['tap8_data_mean']
tap9_data_0 = pressure_data_0['tap9_data_mean']
tap10_data_0 = pressure_data_0['tap10_data_mean']
tap11_data_0 = pressure_data_0['tap11_data_mean']
tap13_data_0 = pressure_data_0['tap13_data_mean']
tap15_data_0 = pressure_data_0['tap15_data_mean']
tap17_data_0 = pressure_data_0['tap17_data_mean']
tap19_data_0 = pressure_data_0['tap19_data_mean']

pitot_data_1 = pressure_data_1['pitot_data_mean']
tap0_data_1 = pressure_data_1['tap0_data_mean']
tap2_data_1 = pressure_data_1['tap2_data_mean']
tap4_data_1 = pressure_data_1['tap4_data_mean']
tap6_data_1 = pressure_data_1['tap6_data_mean']
tap8_data_1 = pressure_data_1['tap8_data_mean']
tap9_data_1 = pressure_data_1['tap9_data_mean']
tap10_data_1 = pressure_data_1['tap10_data_mean']
tap11_data_1 = pressure_data_1['tap11_data_mean']
tap13_data_1 = pressure_data_1['tap13_data_mean']
tap15_data_1 = pressure_data_1['tap15_data_mean']
tap17_data_1 = pressure_data_1['tap17_data_mean']
tap19_data_1 = pressure_data_1['tap19_data_mean']

pitot_NRMSD = calculate_normalized_rmse(pitot_data_0, pitot_data_1)
tap0_NRMSD = calculate_normalized_rmse(tap0_data_0, tap0_data_1)
tap2_NRMSD = calculate_normalized_rmse(tap2_data_0, tap2_data_1)
tap4_NRMSD = calculate_normalized_rmse(tap4_data_0, tap4_data_1)
tap6_NRMSD = calculate_normalized_rmse(tap6_data_0, tap6_data_1)
tap8_NRMSD = calculate_normalized_rmse(tap8_data_0, tap8_data_1)
tap9_NRMSD = calculate_normalized_rmse(tap9_data_0, tap9_data_1)
tap10_NRMSD = calculate_normalized_rmse(tap10_data_0, tap10_data_1)
tap11_NRMSD = calculate_normalized_rmse(tap11_data_0, tap11_data_1)
tap13_NRMSD = calculate_normalized_rmse(tap13_data_0, tap13_data_1)
tap15_NRMSD = calculate_normalized_rmse(tap15_data_0, tap15_data_1)
tap17_NRMSD = calculate_normalized_rmse(tap17_data_0, tap17_data_1)
tap19_NRMSD = calculate_normalized_rmse(tap19_data_0, tap19_data_1)

pressure_data_NRMSD = (pitot_NRMSD + tap0_NRMSD + tap2_NRMSD + tap4_NRMSD + tap6_NRMSD + tap8_NRMSD + tap9_NRMSD + tap10_NRMSD + tap11_NRMSD + tap13_NRMSD + tap15_NRMSD + tap17_NRMSD + tap19_NRMSD) / 13

print('Pressure Data NRMSD: ', pressure_data_NRMSD)