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
    print(f"Saved individual normalized MI values plot to {output_folder}")

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
    fig.savefig(f"{output_folder}/Normalized_Statical_MI_Value_Number_vs_Num_of_Pressure_taps_used.pdf", format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved combination MI vs Num Taps plot to {output_folder}")


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
    xticks_labels = [0, 2, 4, 6, 8, 9, 10, 11, 13, 15, 17, 19] # For grabbing the tap index from metadata, but for the paper, we want to show all the taps with 1-12 as the labels
    num_min = 2
    num_max = 9
    all_tap_scores_records = []

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
            .to_numpy() 
        )

        # Create an empty matrix to store the MI values for each tap combination, ensuring float data type
        empty_matrix = np.full((len(tap_combinations), 12), np.nan)

        # Fill the matrix with the MI values for each tap combination
        xticks_index_map = {label: idx for idx, label in enumerate(xticks_labels)}
        for row_index, taps in enumerate(x_pressure_used):
            for tap in taps:
                if tap in xticks_index_map:
                    col_index = xticks_index_map[tap]
                    empty_matrix[row_index, col_index] = z_mi_values[row_index] 

        # Transpose the matrix to flip x and y axes
        flipped_matrix = empty_matrix.T

        # Calculate the total mutual information value (score) for each pressure tap
        tap_scores = np.nansum(flipped_matrix, axis=1) 

        # Normalize the scores
        min_score, max_score = np.nanmin(tap_scores), np.nanmax(tap_scores) 
        den = (max_score - min_score) 
        normalized_tap_scores = (tap_scores - min_score) / den if den != 0 else np.zeros_like(tap_scores)  

        all_tap_scores_records.append(pd.DataFrame({
            'Pressure Tap Index': range(len(xticks_labels)),
            'Normalized Mutual Information Score': normalized_tap_scores,
            'Num_Taps_Used': num_taps_used
        }))

    all_tap_scores_df = pd.concat(all_tap_scores_records, ignore_index=True)
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


    import matplotlib
    if 'PurGold' not in matplotlib.colormaps:
        matplotlib.colormaps.register(cmap=custom_cmap, name='PurGold')
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
        print(f"Num Taps Used: {i}, Tap 5 Relative Performance: {norm_mi_scores[5]}, Tap 6 Relative Performance: {norm_mi_scores[6]}, Total performance: {norm_mi_scores.sum()}, Nostril Relative Performance: {nostril_relative_performace_percentage}")
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
    xticks_labels_show = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # This is for presentation of the taps, but for the paper, we want to show all the taps with 1-12 as the labels
    ax.set_xlabel('Pressure Tap Index (#)', fontsize=8)
    ax.set_ylabel('Local Normalized Summed MI Value', fontsize=8)
    ax.set_xticks(range(len(xticks_labels_show)))
    ax.set_xticklabels(xticks_labels_show, fontsize=8)
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
    print(f"Saved combined normalized MI scatter plot to {output_folder_norm}")


def plot_individual_mi_results_subplots(bin_range, data_folder, output_folder, file_suffix=""):
    bin_nums = list(bin_range)
    n_bins = len(bin_nums)
    ncols = 3
    nrows = int(np.ceil(n_bins / ncols))
    xticks_labels = [0, 2, 4, 6, 8, 9, 10, 11, 13, 15, 17, 19]

    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8),
                             sharex=True, layout='constrained')
    axes = axes.flatten()

    for idx, bin_num in enumerate(bin_nums):
        data_path = f"{data_folder}/bin_{bin_num}_individual_unranked{file_suffix}.csv"
        data = pd.read_csv(data_path)
        ax = axes[idx]

        num_taps = range(1, len(data) + 1)
        individual_MI_values = data['Mutual_Information_Result']
        individual_min_MI_values = individual_MI_values.min()
        individual_max_MI_values = individual_MI_values.max()
        normalized_individual_MI_values = (individual_MI_values - individual_min_MI_values) / (individual_max_MI_values - individual_min_MI_values)

        ax.bar(num_taps, normalized_individual_MI_values, color='#FFC458', alpha=0.8)
        ax.set_ylim(min(normalized_individual_MI_values) - 0.1, max(normalized_individual_MI_values) + 0.1)
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(np.arange(1, len(num_taps) + 1))
        ax.set_xticklabels(xticks_labels, fontsize=7, rotation=45)
        ax.tick_params(axis='both', labelsize=7)
        ax.set_title(f'Bin {bin_num}', fontsize=8)

    for idx in range(n_bins, len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel('Pressure Tap Index (#)', fontsize=8)
    fig.supylabel('Normalized MI Value', fontsize=8)

    os.makedirs(output_folder, exist_ok=True)
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(f"{output_folder}/Sensitivity_Individual_Normalized_MI_Subplots.pdf", format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved individual normalized MI subplots for bins {bin_nums} to {output_folder}")


def plot_combination_mi_vs_num_taps_subplots(bin_range, data_folder, output_folder, file_suffix=""):
    bin_nums = list(bin_range)
    n_bins = len(bin_nums)
    ncols = 3
    nrows = int(np.ceil(n_bins / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8),
                             sharex=True, sharey=True, layout='constrained')
    axes = axes.flatten()

    for idx, bin_num in enumerate(bin_nums):
        data_path = f"{data_folder}/bin_{bin_num}_combo_unranked{file_suffix}.csv"
        data = pd.read_csv(data_path)
        ax = axes[idx]

        data['Num_Taps'] = data['Combination'].apply(lambda x: len(x.split(',')))
        max_mi_per_num_taps = data.groupby('Num_Taps')['Mutual_Information'].max().reset_index()
        min_mi_per_num_taps = data.groupby('Num_Taps')['Mutual_Information'].min().reset_index()
        mean_mi_per_num_taps = data.groupby('Num_Taps')['Mutual_Information'].mean().reset_index()
        std_mi_per_num_taps = data.groupby('Num_Taps')['Mutual_Information'].std().reset_index()

        num_taps = max_mi_per_num_taps['Num_Taps']
        global_min = min(min_mi_per_num_taps['Mutual_Information'].min(), mean_mi_per_num_taps['Mutual_Information'].min(), max_mi_per_num_taps['Mutual_Information'].min())
        global_max = max(max_mi_per_num_taps['Mutual_Information'].max(), mean_mi_per_num_taps['Mutual_Information'].max(), min_mi_per_num_taps['Mutual_Information'].max())

        max_mi_per_num_taps['Normalized_MI'] = (max_mi_per_num_taps['Mutual_Information'] - global_min) / (global_max - global_min)
        min_mi_per_num_taps['Normalized_MI'] = (min_mi_per_num_taps['Mutual_Information'] - global_min) / (global_max - global_min)
        mean_mi_per_num_taps['Normalized_MI'] = (mean_mi_per_num_taps['Mutual_Information'] - global_min) / (global_max - global_min)

        mean_values_norm = mean_mi_per_num_taps['Normalized_MI']
        std_values = std_mi_per_num_taps['Mutual_Information']
        upper_bound_norm = (mean_values_norm * (global_max - global_min) + global_min + std_values - global_min) / (global_max - global_min)
        lower_bound_norm = (mean_values_norm * (global_max - global_min) + global_min - std_values - global_min) / (global_max - global_min)

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

        plateau_num_taps = int(num_taps.iloc[max_mi_per_num_taps['Normalized_MI'].to_numpy().argmax()])
        ax.axvline(x=plateau_num_taps, color='black', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks(range(int(num_taps.min()), int(num_taps.max()) + 1))
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(f'Bin {bin_num}', fontsize=8)

        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    for idx in range(n_bins, len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel('Number of Taps Used', fontsize=8)
    fig.supylabel('Global Normalized MI Value', fontsize=8)

    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(f"{output_folder}/Sensitivity_Combination_MI_vs_NumTaps_Subplots.pdf", format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved combination MI vs Num Taps subplots for bins {bin_nums} to {output_folder}")



def plot_combination_mi_local_normalized_subplots(bin_range, data_folder, output_folder, file_suffix=""):
    import matplotlib

    bin_nums = list(bin_range)
    n_bins = len(bin_nums)
    ncols = 3
    nrows = int(np.ceil(n_bins / ncols))
    xticks_labels = [0, 2, 4, 6, 8, 9, 10, 11, 13, 15, 17, 19]
    num_min = 1
    num_max = 9
    plot_until_number = 9

    color_1 = '#3b155e'
    color_2 = '#623371'
    color_3 = '#7f4c70'
    color_4 = '#99646b'
    color_5 = '#b37c66'
    color_6 = '#cc9461'
    color_7 = '#e6ac5c'
    color_8 = '#FFC458'
    colors = [color_8, color_7, color_6, color_5, color_4, color_3, color_2, color_1]
    custom_cmap = mcolors.ListedColormap(colors, name='PurGold')
    if 'PurGold' not in matplotlib.colormaps:
        matplotlib.colormaps.register(cmap=custom_cmap, name='PurGold')
    colormap = plt.cm.get_cmap('PurGold')
    colormap_code = 'PurGold'

    bounds = list(range(num_min, num_max + 1))
    norm = mcolors.BoundaryNorm(bounds, colormap.N)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8),
                             sharex=True, sharey=True, layout='constrained')
    axes = axes.flatten()

    for idx, bin_num in enumerate(bin_nums):
        data_path = f"{data_folder}/bin_{bin_num}_combo_unranked{file_suffix}.csv"
        data = pd.read_csv(data_path)
        ax = axes[idx]

        all_tap_scores_records = []

        for num_taps_used in range(num_min, num_max):
            data['Num_Taps'] = data['Combination'].apply(lambda x: len(x.split(',')))
            pressure_combinations_with_num_taps = data[data['Num_Taps'] == num_taps_used]
            ranked = pressure_combinations_with_num_taps.sort_values(by='Mutual_Information', ascending=False).reset_index(drop=True)

            tap_combinations_num = len(ranked['Combination'])
            tap_combinations = ranked['Combination'].iloc[:int(np.floor(tap_combinations_num))].to_numpy()

            x_pressure_used = np.zeros((len(tap_combinations), num_taps_used))
            for i in range(len(tap_combinations)):
                x_pressure_used[i] = extract_tap_numbers(tap_combinations[i])

            z_mi_values = ranked['Mutual_Information'].to_numpy()

            empty_matrix = np.full((len(tap_combinations), 12), np.nan)
            xticks_index_map = {label: col_idx for col_idx, label in enumerate(xticks_labels)}
            for row_index, taps in enumerate(x_pressure_used):
                for tap in taps:
                    if tap in xticks_index_map:
                        col_index = xticks_index_map[tap]
                        empty_matrix[row_index, col_index] = z_mi_values[row_index]

            flipped_matrix = empty_matrix.T
            tap_scores = np.nansum(flipped_matrix, axis=1)
            min_score, max_score = np.nanmin(tap_scores), np.nanmax(tap_scores)
            den = max_score - min_score
            normalized_tap_scores = (tap_scores - min_score) / den if den != 0 else np.zeros_like(tap_scores)

            all_tap_scores_records.append(pd.DataFrame({
                'Pressure Tap Index': range(len(xticks_labels)),
                'Normalized Mutual Information Score': normalized_tap_scores,
                'Num_Taps_Used': num_taps_used
            }))

        all_tap_scores_df = pd.concat(all_tap_scores_records, ignore_index=True)
        for i in range(num_min, plot_until_number):
            filtered = all_tap_scores_df[all_tap_scores_df['Num_Taps_Used'] == i]
            norm_mi_scores = filtered['Normalized Mutual Information Score'].to_numpy()
            ax.plot(filtered['Pressure Tap Index'].to_numpy(), norm_mi_scores,
                    color=colormap(norm(i)), linewidth=1.0, alpha=0.3, marker=None)
            ax.scatter(filtered['Pressure Tap Index'].to_numpy(), norm_mi_scores,
                       color=colormap(norm(i)), s=15, marker='o', alpha=1.0)

        xticks_labels_show = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ax.set_xticks(range(len(xticks_labels_show)))
        ax.set_xticklabels(xticks_labels_show, fontsize=7, rotation=45)
        ax.tick_params(axis='both', labelsize=7)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(f'Bin {bin_num}', fontsize=8)

    for idx in range(n_bins, len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel('Pressure Tap Index (#)', fontsize=8)
    fig.supylabel('Local Normalized Summed MI', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=colormap_code, norm=norm)
    sm.set_array([])
    tick_positions = [b + 0.5 for b in bounds[:-1]]
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', ticks=tick_positions,
                        shrink=0.8, pad=0.02, location='bottom')
    cbar.ax.set_xticklabels([str(i) for i in range(num_min, num_max)])
    cbar.set_label('n-Taps Used', fontsize=8, labelpad=4)
    cbar.ax.tick_params(labelsize=7)

    os.makedirs(output_folder, exist_ok=True)
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(f"{output_folder}/Sensitivity_Combined_Normalized_MI_Scatter_Subplots.pdf", format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved combined normalized MI scatter subplots for bins {bin_nums} to {output_folder}")


if __name__ == "__main__":
    # This data path is when bin size = 28 as our results are based on 28 bins 
    data_path_individual = "MI_results/Individual_MI_results.csv"
    data_path_combination = "MI_results/Combination_MI_results.csv"

    data_path_individual_iter_1 = "MI_results_iter1/Individual_MI_results_iter1.csv"
    data_path_combination_iter_1 = "MI_results_iter1/Combination_MI_results_iter1.csv"

    # Folder to save the generated plots
    output_folder = "MI_results/Plots/"
    output_folder_iter_1 = "MI_results_iter1/Plots"

    # # Plot 1: Individual MI Results with 28 bins
    # plot_individual_mi_results(data_path_individual, output_folder)
    # plot_individual_mi_results(data_path_individual_iter_1, output_folder_iter_1)

    # # Plot 2: Combination MI Results vs Number of Taps Used with 28 bins
    # plot_combination_mi_vs_num_taps(data_path_combination, output_folder)
    # plot_combination_mi_vs_num_taps(data_path_combination_iter_1, output_folder_iter_1)

    # # Plot 3: Combination MI Results Local Normalized Summed MI Values with 28 bins
    plot_combination_mi_local_normalized(data_path_combination, output_folder)
    # plot_combination_mi_local_normalized(data_path_combination_iter_1, output_folder_iter_1)

    # Sensitivity analysis subplots for bins 15 to 31
    bin_range = range(15, 32, 1)
    # plot_individual_mi_results_subplots(bin_range, "MI_results", "MI_results/Plots/Sensitivity_Analysis")
    # plot_combination_mi_vs_num_taps_subplots(bin_range, "MI_results", "MI_results/Plots/Sensitivity_Analysis")
    # plot_combination_mi_local_normalized_subplots(bin_range, "MI_results", "MI_results/Plots/Sensitivity_Analysis")

    # Iter1 sensitivity analysis subplots for bins 15 to 31
    # plot_individual_mi_results_subplots(bin_range, "MI_results_iter1", "MI_results_iter1/Plots/Sensitivity_Analysis", file_suffix="_iter1")
    # plot_combination_mi_vs_num_taps_subplots(bin_range, "MI_results_iter1", "MI_results_iter1/Plots/Sensitivity_Analysis", file_suffix="_iter1")
    # plot_combination_mi_local_normalized_subplots(bin_range, "MI_results_iter1", "MI_results_iter1/Plots/Sensitivity_Analysis", file_suffix="_iter1")




