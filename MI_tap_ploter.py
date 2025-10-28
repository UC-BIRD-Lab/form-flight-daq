import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
import os
import seaborn as sns

def extract_tap_numbers(row):
    # Find all numbers following 'tap' in the string
    return [int(num) for num in re.findall(r'tap(\d+)_delta_pressure_mean', row)]

# Function to set x-tick values based on a given max value
def generate_x_ticks(max_value):
    x_tick_values = []
    current_value = 1
    
    while current_value <= max_value:
        x_tick_values.append(current_value)
        current_value *= 2  # Increase exponentially (1, 2, 4, 8, ...)
    
    # Ensure the maximum value is included as the last tick
    if x_tick_values[-1] != max_value:
        x_tick_values.append(max_value)
    
    # Generate equally spaced positions for the tick values
    x_tick_positions = np.linspace(0, max_value - 1, len(x_tick_values))
    
    return x_tick_positions, x_tick_values


def plot_individual_mi_results(data_path, output_folder):
    """
    Plots individual Mutual Information (MI) results with normalized values.

    Parameters:
    - data_path (str): Path to the CSV file containing MI results.
    - output_folder (str): Directory to save the generated plots.
    """
    # Load the data
    data = pd.read_csv(data_path)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract x and y values
    num_taps = range(1, len(data) + 1)
    individual_MI_values = data['Mutual_Information_Result']

    # Calculate mean, min, max, and normalized MI values
    individual_mean_MI_values = data['Mutual_Information_Result'].mean()
    individual_min_MI_values = data['Mutual_Information_Result'].min()
    individual_max_MI_values = data['Mutual_Information_Result'].max()

    normalized_individual_MI_values = (individual_MI_values - individual_min_MI_values) / (individual_max_MI_values - individual_min_MI_values)

    # Define x-tick labels
    xticks_labels = [0, 2, 4, 6, 8, 9, 10, 11, 13, 15, 17, 19]

    # Plot: Normalized Mutual Information Values
    fig, ax = plt.subplots(figsize=(6.5 / 2, 3))
    ax.bar(num_taps, normalized_individual_MI_values, color='#FFC458', alpha=0.8)
    ax.set_xlabel('Pressure Tap Index (#)', fontsize=8)
    ax.set_ylabel('Normalized Mutual Information Value', fontsize=8)
    ax.set_ylim(min(normalized_individual_MI_values) - 0.1, max(normalized_individual_MI_values) + 0.1)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(np.arange(1, len(num_taps) + 1))
    ax.set_xticklabels(xticks_labels, fontsize=8)

    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(f"{output_folder}/Individual_Normalized_MI_Values_per_Pressure_Tap.svg", format="svg", bbox_inches="tight", dpi=600)
    plt.show()

def plot_combination_mi_vs_num_taps(data_path, output_folder):
    """
    Plots Normalized Mutual Information Statistical Values vs Number of Pressure Taps Used.

    Parameters:
    - data_path (str): Path to the CSV file containing combination MI results.
    - output_folder (str): Directory to save the generated plots.
    """
    # Load the data from your CSV file
    data = pd.read_csv(data_path)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Add a new column that counts the number of taps in each combination
    data['Num_Taps'] = data['Combination'].apply(lambda x: len(x.split(',')))

    # Group by 'Num_Taps' and calculate the maximum, minimum, mean, median, and standard deviation of Mutual Information for each group
    max_mi_per_num_taps = data.groupby('Num_Taps')['Mutual_Information'].max().reset_index()
    min_mi_per_num_taps = data.groupby('Num_Taps')['Mutual_Information'].min().reset_index()
    mean_mi_per_num_taps = data.groupby('Num_Taps')['Mutual_Information'].mean().reset_index()
    std_mi_per_num_taps = data.groupby('Num_Taps')['Mutual_Information'].std().reset_index()  # Standard deviation

    # Extract the number of taps and corresponding Mutual Information values
    num_taps = max_mi_per_num_taps['Num_Taps']

    # Calculate global min and max for normalization
    global_min = min(min_mi_per_num_taps['Mutual_Information'].min(), mean_mi_per_num_taps['Mutual_Information'].min(), max_mi_per_num_taps['Mutual_Information'].min())
    global_max = max(max_mi_per_num_taps['Mutual_Information'].max(), mean_mi_per_num_taps['Mutual_Information'].max(), min_mi_per_num_taps['Mutual_Information'].max())

    # Apply min-max normalization
    max_mi_per_num_taps['Normalized_MI'] = (max_mi_per_num_taps['Mutual_Information'] - global_min) / (global_max - global_min)
    min_mi_per_num_taps['Normalized_MI'] = (min_mi_per_num_taps['Mutual_Information'] - global_min) / (global_max - global_min)
    mean_mi_per_num_taps['Normalized_MI'] = (mean_mi_per_num_taps['Mutual_Information'] - global_min) / (global_max - global_min)

    # Calculating normalized standard deviation
    mean_values_norm = mean_mi_per_num_taps['Normalized_MI']
    std_values = std_mi_per_num_taps['Mutual_Information']
    upper_bound_norm = (mean_values_norm * (global_max - global_min) + global_min + std_values - global_min) / (global_max - global_min)
    lower_bound_norm = (mean_values_norm * (global_max - global_min) + global_min - std_values - global_min) / (global_max - global_min)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6.5 / 2, 3))

    ax.plot(num_taps.to_numpy(), max_mi_per_num_taps['Normalized_MI'].to_numpy(),
            color='black', markerfacecolor='black', marker='o',
            linestyle='-', linewidth=1, markersize=4, label='Max MI')
    ax.plot(num_taps.to_numpy(), min_mi_per_num_taps['Normalized_MI'].to_numpy(),
            color='black', markerfacecolor='black', marker='o',
            linestyle='dashed', linewidth=1, markersize=4, label='Min MI')
    ax.plot(num_taps.to_numpy(), mean_mi_per_num_taps['Normalized_MI'].to_numpy(),
            color='black', linewidth=1, markersize=4,
            linestyle=(0, (1, 1)), label='Mean MI')

    ax.fill_between(num_taps.to_numpy(), lower_bound_norm, upper_bound_norm,
                    color='grey', alpha=0.4, label='± 1 Std Dev')

    # Customize legend, limits, and labels
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(-0.1, 1.1)  # Y-axis limits
    ax.set_xlabel('Number of Taps Used', fontsize=8)
    ax.set_ylabel('Global Normalized Mutual Information Value', fontsize=8)

    # Custom ticks
    ax.set_xticks(range(int(num_taps.min()), int(num_taps.max()) + 1))
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Save figure
    fig.savefig(f"{output_folder}/Normalized_Statical_MI_Value_Number_vs_Num_of_Pressure_taps_used.svg", format="svg", bbox_inches="tight", dpi=1000)
    plt.show()


def plot_combination_mi_local_normalized(data_path, output_folder):
    """
    Plots Local Normalized Summed Mutual Information (MI) Values for Pressure Tap Combinations.

    This function reads combination MI results from a CSV file, processes the data to calculate
    local normalized summed MI values for each pressure tap across different numbers of taps used,
    and generates a scatter plot visualizing these values.

    The plot is saved in the specified output folder.
    """
    # Block 7: MI combination summed MI value local normalized
    #-------------------------------------------------------------------------------------------
    # Load the data
    data = pd.read_csv(data_path)

    output_folder_norm = output_folder
    os.makedirs(output_folder_norm, exist_ok=True)

    # Define x-axis labels for consistent tick labels (pressure tap index locations)
    xticks_labels = [0, 2, 4, 6, 8, 9, 10, 11, 13, 15, 17, 19]

    # Initialize an empty DataFrame to store results for all `num_taps_used`
    all_tap_scores_df = pd.DataFrame(columns=['Pressure Tap Index', 'Normalized Mutual Information Score', 'Num_Taps_Used'])

    # Loop through each `num_taps_used` from 2 to 9
    num_min = 1
    num_max = 9

    for num_taps_used in range(num_min, num_max):
        # Add a new column that counts the number of taps in each combination
        data['Num_Taps'] = data['Combination'].apply(lambda x: len(x.split(',')))

        # Filter and sort based on number of taps and mutual information
        pressure_combinations_with_num_taps = data[data['Num_Taps'] == num_taps_used]
        ranked_pressure_combinations_with_num_taps = pressure_combinations_with_num_taps.sort_values(by='Mutual_Information', ascending=False)
        reshape_ranked_pressure_combinations_with_num_taps = ranked_pressure_combinations_with_num_taps.reset_index(drop=True)

        # Extract tap combinations and mutual information values
        tap_combinations_num = len(reshape_ranked_pressure_combinations_with_num_taps['Combination'])
        tap_combinations = (
            reshape_ranked_pressure_combinations_with_num_taps['Combination']
            .iloc[:int(np.floor(tap_combinations_num))]
            .to_numpy() 
        )

        x_pressure_used = np.zeros((len(tap_combinations), num_taps_used))

        for i in range(len(tap_combinations)):
            # assumes extract_tap_numbers returns a 1D sequence of taps
            x_pressure_used[i] = extract_tap_numbers(tap_combinations[i])

        z_mi_values = (
            reshape_ranked_pressure_combinations_with_num_taps['Mutual_Information']
            .to_numpy()  # <<< CHANGED: ensure NumPy for numeric indexing
        )

        # Create an empty matrix to store the MI values for each tap combination, ensuring float data type
        empty_matrix = np.full((len(tap_combinations), 12), np.nan)

        # Fill the matrix with the MI values for each tap combination
        xticks_index_map = {label: idx for idx, label in enumerate(xticks_labels)}
        for row_index, taps in enumerate(x_pressure_used):
            for tap in taps:
                if tap in xticks_index_map:
                    col_index = xticks_index_map[tap]
                    empty_matrix[row_index, col_index] = z_mi_values[row_index]  # <<< CHANGED: z_mi_values is ndarray

        # Transpose the matrix to flip x and y axes
        flipped_matrix = empty_matrix.T

        # Calculate the total mutual information value (score) for each pressure tap
        tap_scores = np.nansum(flipped_matrix, axis=1)  # Sum along each column, ignoring NaNs

        # Normalize the scores
        min_score, max_score = np.nanmin(tap_scores), np.nanmax(tap_scores)  # <<< CHANGED: np.nanmin/max for safety
        den = (max_score - min_score) 
        normalized_tap_scores = (tap_scores - min_score) / den if den != 0 else np.zeros_like(tap_scores)  # <<< CHANGED

        # Add results to the combined DataFrame
        temp_df = pd.DataFrame({
            'Pressure Tap Index': range(len(xticks_labels)),
            'Normalized Mutual Information Score': normalized_tap_scores,
            'Num_Taps_Used': num_taps_used
        })
        
        all_tap_scores_df = pd.concat([all_tap_scores_df, temp_df], ignore_index=True)

    N = num_max - num_min - 1

    fig, ax = plt.subplots(figsize=(6.5/2, 3))
    # fig, ax = plt.subplots(figsize=(5, 4.25)) # For presentation

    # color_1 = '#3b155e' # Dark Purple
    # color_2 = '#693874'
    # color_3 = '#88546e'
    # color_4 = '#a67068'
    # color_5 = '#c38c63'
    # color_6 = '#e2a85d'
    # color_7 = '#FFC458' # Gold

    # colors = [color_7, color_6, color_5, color_4, color_3, color_2, color_1]

    color_1 = '#3b155e' # Dark Purple
    color_2 = '#623371'
    color_3 = '#7f4c70'
    color_4 = '#99646b'
    color_5 = '#b37c66'
    color_6 = '#cc9461'
    color_7 = '#e6ac5c'
    color_8 = '#FFC458' # Gold

    colors = [color_8, color_7, color_6, color_5, color_4, color_3, color_2, color_1]
    custom_cmap = mcolors.ListedColormap(colors, name = 'PurGold')


    plt.cm.register_cmap(name='PurGold', cmap=custom_cmap)
    colormap = plt.cm.get_cmap('PurGold')
    colormap_code = 'PurGold'

    bounds = list(range(num_min, num_max + 1))
    norm = mcolors.BoundaryNorm(bounds, colormap.N)

    # Plot with combined scatter plot
    plot_until_number = 9
    for i in range(num_min, plot_until_number):
        all_tap_scores_df_filtered = all_tap_scores_df[all_tap_scores_df['Num_Taps_Used'] == i]
        norm_mi_scores = all_tap_scores_df_filtered['Normalized Mutual Information Score'].to_numpy() 
        tap_5_relative_performace_percentage = norm_mi_scores[5] / norm_mi_scores.mean()
        tap_6_relative_performace_percentage = norm_mi_scores[6] / norm_mi_scores.mean()
        nostril_relative_performace_percentage = (norm_mi_scores[5] + norm_mi_scores[6]) / norm_mi_scores.sum()
        print(f"Num Taps Used: {i}, Tap 5 Relative Performance: {tap_5_relative_performace_percentage}, Tap 6 Relative Performance: {tap_6_relative_performace_percentage}, Nostril Relative Performance: {nostril_relative_performace_percentage}")
        ax.plot(
            all_tap_scores_df_filtered['Pressure Tap Index'].to_numpy(),
            norm_mi_scores,
            color=colormap(norm(i)), linewidth=1.2, alpha=0.3, marker=None 
        )
        ax.scatter(
            all_tap_scores_df_filtered['Pressure Tap Index'].to_numpy(),
            norm_mi_scores,
            color=colormap(norm(i)), s=50, marker='o', alpha=1.0  
        )

    # Plot with individual scatter plot
    # for i in range(num_min, num_max):
    #     all_tap_scores_df_filtered = all_tap_scores_df[all_tap_scores_df['Num_Taps_Used'] == i]
    #     ax.scatter(all_tap_scores_df_filtered['Pressure Tap Index'],
    #                 all_tap_scores_df_filtered['Normalized Mutual Information Score'],
    #                 c=[i] * len(all_tap_scores_df_filtered),
    #                 cmap='cividis', norm=norm, s=500)
        
    # Set plot labels and title
    ax.set_xlabel('Pressure Tap Index (#)', fontsize=8)
    ax.set_ylabel('Local Normalized Summed MI Value', fontsize=8)
    ax.set_xticks(range(len(xticks_labels)))
    ax.set_xticklabels(xticks_labels, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)

    cbar_ax = fig.add_axes([0.91, 0.11, 0.03, 0.77]) 

    # Create the colorbar using ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=colormap_code, norm=norm)
    sm.set_array([])

    tick_positions = [b + 0.5 for b in bounds[:-1]]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', ticks=tick_positions)
    cbar.ax.set_yticklabels([str(i) for i in range(num_min, num_max)])
    cbar.set_label('n-Taps \nUsed', fontsize=8, rotation=0, labelpad=0, loc = 'bottom')
    cbar.ax.tick_params(labelsize=8)

    # Remove the right and upper borders of the plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Save the plot
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(f"{output_folder_norm}/Combined_Normalized_MI_Scatter.svg", format="svg", bbox_inches="tight", dpi = 600)
    # plt.savefig(f"{output_folder_norm}/Combined_Normalized_MI_Scatter_{plot_until_number-1}.png", format="png", bbox_inches="tight", dpi = 1000)
    # plt.savefig(f"{output_folder_norm}/Combined_Normalized_MI_Scatter_{plot_until_number-1}.svg", format="svg", bbox_inches="tight", dpi=1000)
    plt.show()


if __name__ == "__main__":

    data_path_individual = "MI_results/Individual_MI_results.csv"
    data_path_combination = "MI_results/Combination_MI_results.csv"
    output_folder = "MI_results/Plots"
    # Plot 1: Individual MI Results
    plot_individual_mi_results(data_path_individual, output_folder)

    # Plot 2: Combination MI Results vs Number of Taps Used
    plot_combination_mi_vs_num_taps(data_path_combination, output_folder)

    # Plot 3: Combination MI Results Local Normalized Summed MI Values
    plot_combination_mi_local_normalized(data_path_combination, output_folder)


