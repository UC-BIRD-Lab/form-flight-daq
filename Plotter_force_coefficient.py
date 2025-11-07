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

################## Function to calculate the standard deviation of multiplication and division ##################
def sd_mult_div(meanterm1, error1, meanterm2, error2, meanZ):
    multerror = abs((((error1 / meanterm1) ** 2 + (error2 / meanterm2) ** 2) ** 0.5) * meanZ)
    return multerror

def process_force_coefficient_data(force_coefficient_folder_path):
    # List all CSV files in the folder
    force_coefficient_csv_filename = [f for f in os.listdir(force_coefficient_folder_path) if f.endswith('_stats.csv')]

    # Dictionary to hold the DataFrames
    force_coefficient_stats_each_AoA_dataframe = {
        'AoA': [],
        'Drag Coefficient': [],
        'Drag Coefficient Error': [],
        'Lift Coefficient': [],
        'Lift Coefficient Error': []
    }

    # Read each CSV file into a DataFrame
    for file in force_coefficient_csv_filename:
        file_path = os.path.join(force_coefficient_folder_path, file)
        df = pd.read_csv(file_path)

        # Append values to the lists in the dictionary
        force_coefficient_stats_each_AoA_dataframe['AoA'].append(df['AoA'].values[0])

        drag_coefficient = df['Drag Coefficient'].values[0]
        lift_coefficient = df['Lift Coefficient'].values[0]
        drag_error = df['Drag Coefficient Error'].values[0]
        lift_error = df['Lift Coefficient Error'].values[0]

        force_coefficient_stats_each_AoA_dataframe['Drag Coefficient'].append(drag_coefficient)
        force_coefficient_stats_each_AoA_dataframe['Lift Coefficient'].append(lift_coefficient)
        force_coefficient_stats_each_AoA_dataframe['Drag Coefficient Error'].append(drag_error)
        force_coefficient_stats_each_AoA_dataframe['Lift Coefficient Error'].append(lift_error)

    # Convert the dictionary to a pandas DataFrame
    df_combined = pd.DataFrame(force_coefficient_stats_each_AoA_dataframe)
    return df_combined

def plot_force_coefficient_data(force_coefficient_stats_dataframe_each_AoA, save_path):
    # Create a folder to save the plots
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Create a scatter plot of the lift and drag vs AoA with error bars
    # import ipdb; ipdb.set_trace()
    force_coefficient_stats_dataframe_each_AoA = pd.read_csv(force_coefficient_stats_dataframe_each_AoA)
    AoA_str = force_coefficient_stats_dataframe_each_AoA['AoA'].to_numpy()
    AoA = [str(a).replace("_", ".") for a in AoA_str]
    mean_C_L = force_coefficient_stats_dataframe_each_AoA['Lift Coefficient'].to_numpy()
    mean_C_D = force_coefficient_stats_dataframe_each_AoA['Drag Coefficient'].to_numpy()
    C_L_Error = force_coefficient_stats_dataframe_each_AoA['Lift Coefficient Error'].to_numpy()
    C_D_Error = force_coefficient_stats_dataframe_each_AoA['Drag Coefficient Error'].to_numpy()

    plt.figure(1, figsize=(16, 12))
    plt.title('Lift Coefficient & Drag Coefficient vs Angle of Attack', fontsize=30, weight='bold')
    plt.xlabel(r'Angle of Attack ($\degree$)', fontsize=22, weight='bold')
    plt.ylabel('Force Coefficient', fontsize=22, weight='bold')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(AoA, mean_C_L, label=r'$C_L$', color='steelblue', alpha = 1, marker='o')
    plt.plot(AoA, mean_C_D, label=r'$C_D$', color='lightcoral', alpha = 1, marker='*')
    plt.errorbar(AoA, mean_C_L, yerr=C_L_Error, fmt='o', color='steelblue', alpha=0.1)
    plt.errorbar(AoA, mean_C_D, yerr=C_D_Error, fmt='o', color='lightcoral', alpha=0.1)

    plt.fill_between(AoA, mean_C_L - C_L_Error, mean_C_L + C_L_Error, color='steelblue', alpha=0.1)
    plt.fill_between(AoA, mean_C_D - C_D_Error, mean_C_D + C_D_Error, color='lightcoral', alpha=0.1)

    # for i in range(len(AoA)):
    #     plt.annotate(f'{C_L_Error[i]:.4f}', (AoA[i], mean_C_L[i] + C_L_Error[i]), color='blue', ha='center')
    #     plt.annotate(f'{C_D_Error[i]:.4f}', (AoA[i], mean_C_D[i] + C_D_Error[i]), color='purple', ha='center')

    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(save_path + '/froce_coefficient_vs_AoA.svg')

    #-----------------------------------------------------------------------------------------------------------
    plt.figure(2, figsize=(16, 14))
    plt.title('Lift Coefficient vs Angle of attack', fontsize=20)
    plt.xlabel('Angle of Attack (degrees)', fontsize=15)
    plt.ylabel('Lift Coefficient', fontsize=15)
    plt.plot(AoA, mean_C_L, label=r'$C_L$', color='slateblue', alpha = 0.7, marker='o')
    plt.errorbar(AoA, mean_C_L, yerr=C_L_Error, fmt='o', color='slateblue', alpha=0.1)
    plt.fill_between(AoA, mean_C_L - C_L_Error, mean_C_L + C_L_Error, color='slateblue', alpha=0.1)

    # for i in range(len(AoA)):
    #     plt.annotate(f'{C_L_Error[i]:.4f}', (AoA[i], mean_C_L[i] + C_L_Error[i]), color='blue', ha='center')
    
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path + '/C_L_vs_AoA.svg')

    plt.figure(3, figsize=(16, 9))
    plt.title('Drag Coefficient vs Angle of attack', fontsize=20)
    plt.xlabel('Angle of Attack (degrees)', fontsize=15)
    plt.ylabel('Drag Coefficient', fontsize=15)
    plt.plot(AoA, mean_C_D, label=r'$C_D$', color='mediumvioletred', alpha = 0.7, marker='o')
    plt.errorbar(AoA, mean_C_D, yerr=C_D_Error, fmt='o', color='mediumvioletred', alpha=0.1)
    plt.fill_between(AoA, mean_C_D - C_D_Error, mean_C_D + C_D_Error, color='mediumvioletred', alpha=0.1)

    # for i in range(len(AoA)):
    #     plt.annotate(f'{C_D_Error[i]:.4f}', (AoA[i], mean_C_D[i] + C_D_Error[i]), color='purple', ha='center')

    plt.savefig(save_path + '/C_D_vs_AoA.svg')
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

    return None

def L_D_polar_with_error(force_coefficient_stats_dataframe_each_AoA, save_path):
    # Create a folder to save the plots
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    force_coefficient_stats_dataframe_each_AoA = pd.read_csv(force_coefficient_stats_dataframe_each_AoA)
    AoA_str = force_coefficient_stats_dataframe_each_AoA['AoA'].to_numpy()
    AoA = [str(a).replace("_", ".") for a in AoA_str]
    mean_C_L = force_coefficient_stats_dataframe_each_AoA['Lift Coefficient'].to_numpy()
    mean_C_D = force_coefficient_stats_dataframe_each_AoA['Drag Coefficient'].to_numpy()
    C_L_Error = force_coefficient_stats_dataframe_each_AoA['Lift Coefficient Error'].to_numpy()
    C_D_Error = force_coefficient_stats_dataframe_each_AoA['Drag Coefficient Error'].to_numpy()

    # Create a scatter plot of the mean lift and drag polar with error bars
    plt.figure(1, figsize=(6.5/2, 6.5/2))
    # plt.title(r'$C_{L}$ vs $C_{D}$ Polar per Angle of Attack', fontsize=30, weight='bold')
    plt.ylabel(r'$C_{L}$', fontsize=8)
    plt.xlabel(r'$C_{D}$', fontsize=8)

    plt.errorbar(mean_C_D, mean_C_L, xerr=C_D_Error, yerr=C_L_Error, fmt='-', color='black', alpha=0.6, elinewidth=1)
    plt.plot(mean_C_D, mean_C_L, color='black', marker='o', markersize=4, alpha=1)

    # Define tick positions
    x_ticks_positions = np.linspace(0, 0.4, 9)  # Major ticks every 2 units
    x_tick_labels = ['0', '', '0.1', '', '0.2', '', '0.3', '', '0.4']  # Label every other tick

    y_ticks_positions = np.linspace(-0.2, 1, 13)  # Major ticks every 5 units
    y_tick_labels = ['-0.2', '', '0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1']  # Label every other tick

    # Set the ticks on the plot
    plt.xticks(x_ticks_positions, x_tick_labels, fontsize=8)
    plt.yticks(y_ticks_positions, y_tick_labels, fontsize=8)

    # Add text labels for AoA above each point
    for i, txt in enumerate(AoA):
        plt.annotate(fr"${txt}$" + r"$\degree$", (mean_C_D[i], mean_C_L[i]), textcoords="offset points", xytext=(0,5), fontsize = 8)
        # plt.annotate(f'{C_L_Error[i]:.4f}', (mean_C_D[i], mean_C_L[i] + C_L_Error[i]), color='red', ha='right')
        # plt.annotate(f'{C_D_Error[i]:.4f}', (mean_C_D[i] + C_D_Error[i], mean_C_L[i]), color='purple', ha='right')

    plt.tight_layout()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(save_path + '/C_L_vs_C_D_polar.svg', dpi=600)
    plt.show()

    return 


if __name__ == '__main__':
    # Path to the folder containing the force coefficient CSV files
    force_coefficient_stats_dataframe_each_AoA = 'Post_processed_force_coefficient/Force_Coefficients_AoA_All.csv'
    # Create a folder to save the plots
    save_path = 'Polar_Plot'
    plot_force_coefficient_data(force_coefficient_stats_dataframe_each_AoA, save_path)
    L_D_polar_with_error(force_coefficient_stats_dataframe_each_AoA, save_path) 





