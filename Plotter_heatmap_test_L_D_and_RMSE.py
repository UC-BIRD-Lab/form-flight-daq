import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import glob
import os
import seaborn as sns
from functools import reduce
import numpy.ma as ma

def extract_L_D_data(xpos, ypos, zpos, AoA, file_path, model,
                     is_interpolated=False, tol=1e-4):

    data_frame = pd.read_csv(file_path)
    
    if is_interpolated:
        subset = data_frame[
            (data_frame["AoA"].sub(AoA).abs() < tol) &
            (data_frame["XPos"].sub(xpos).abs() < tol) &
            (data_frame["YPos"].sub(ypos).abs() < tol) &
            (data_frame["ZPos"].sub(zpos).abs() < tol)
        ]
    else:
        subset = data_frame[
            (data_frame['XPos'] == xpos) &
            (data_frame['YPos'] == ypos) &
            (data_frame['ZPos'] == zpos) &
            (data_frame['AoA'] == AoA)
        ]
    
    if subset.empty:
        return np.nan

    row = subset.iloc[0]

    if model == 'Actual':
        return row['delta_L_D_mean']
    elif model == 'CNN_1':
        return row['Predicted_Delta_L_D']
    elif model == 'GB_1':
        return row['Predicted_Delta_L_D_1']
    elif model == 'GB_2':
        return row['Predicted_Delta_L_D_2']
    elif model == 'GB_3':
        return row['Predicted_Delta_L_D_3']
    elif model == 'GB_4':
        return row['Predicted_Delta_L_D_4']
    elif model == 'GB_5':
        return row['Predicted_Delta_L_D_5']
    
    return row['Predicted_Delta_L_D_2']
def L_D_heatmap_matrix(zpos_abs, ypos_abs, AoA, file_path, Model, is_interpolated=False):
    AoA_0, AoA_2_5, AoA_5 = AoA
    
    # Subplot 1
    L_D_plot_matrix_1_1 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_1_1[i][j] = extract_L_D_data(
                xpos=8.25,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_0,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )
    
    # Subplot 2
    L_D_plot_matrix_1_2 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_1_2[i][j] = extract_L_D_data(
                xpos=16.5,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_0,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )
    
    # Subplot 3
    L_D_plot_matrix_1_3 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_1_3[i][j] = extract_L_D_data(
                xpos=24.75,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_0,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )
    
    # Subplot 4
    L_D_plot_matrix_2_1 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_2_1[i][j] = extract_L_D_data(
                xpos=8.25,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_2_5,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )

    # Subplot 5
    L_D_plot_matrix_2_2 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_2_2[i][j] = extract_L_D_data(
                xpos=16.5,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_2_5,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )

    # Subplot 6
    L_D_plot_matrix_2_3 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_2_3[i][j] = extract_L_D_data(
                xpos=24.75,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_2_5,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )

    # Subplot 7
    L_D_plot_matrix_3_1 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_3_1[i][j] = extract_L_D_data(
                xpos=8.25,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_5,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )

    # Subplot 8
    L_D_plot_matrix_3_2 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_3_2[i][j] = extract_L_D_data(
                xpos=16.5,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_5,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )
    
    # Subplot 9
    L_D_plot_matrix_3_3 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            L_D_plot_matrix_3_3[i][j] = extract_L_D_data(
                xpos=24.75,
                ypos=ypos_abs[j],
                zpos=zpos_abs[i],
                AoA=AoA_5,
                file_path=file_path,
                model=Model,
                is_interpolated=is_interpolated,
                tol=1e-4  # or your chosen tolerance
            )

    return L_D_plot_matrix_1_1, L_D_plot_matrix_1_2, L_D_plot_matrix_1_3, L_D_plot_matrix_2_1, L_D_plot_matrix_2_2, L_D_plot_matrix_2_3, L_D_plot_matrix_3_1, L_D_plot_matrix_3_2, L_D_plot_matrix_3_3

def error_L_D_plotter(Model, file_path, actual_L_D_matrix, save_file_path):
    is_interpolated=False
    xpos_abs = [8.25, 16.5, 24.75]
    ypos_abs = [0, 2.75, 5.5, 8.25, 11, 13.75, 16.5, 19.25]
    zpos_abs = [-2.75, 0, 2.75, 5.5]
    AoA = [0, 2.5, 5]

    # Prepare scaled tick labels
    xpos = [np.floor(x*(100/11)) for x in xpos_abs]
    ypos = [np.floor(y*(100/11)) for y in ypos_abs]
    zpos = [np.round(z*(100/11)) for z in zpos_abs]

    # Prepare scaled tick labels
    (L_D_1_1, L_D_1_2, L_D_1_3, 
     L_D_2_1, L_D_2_2, L_D_2_3, 
     L_D_3_1, L_D_3_2, L_D_3_3) = L_D_heatmap_matrix(
        zpos_abs, ypos_abs, AoA, file_path, Model, is_interpolated=is_interpolated
    )

    # Calculate the error matrix
    error_matrix_1_1 = L_D_1_1 - actual_L_D_matrix[0]
    error_matrix_1_2 = L_D_1_2 - actual_L_D_matrix[1]
    error_matrix_1_3 = L_D_1_3 - actual_L_D_matrix[2]
    error_matrix_2_1 = L_D_2_1 - actual_L_D_matrix[3]
    error_matrix_2_2 = L_D_2_2 - actual_L_D_matrix[4]
    error_matrix_2_3 = L_D_2_3 - actual_L_D_matrix[5]
    error_matrix_3_1 = L_D_3_1 - actual_L_D_matrix[6]
    error_matrix_3_2 = L_D_3_2 - actual_L_D_matrix[7]
    error_matrix_3_3 = L_D_3_3 - actual_L_D_matrix[8]

    cmap = plt.get_cmap('Greys') # PuRd for Sequentual data

    fig = plt.figure(figsize=(6.5, 3.25))
    gs = fig.add_gridspec(3, 3, hspace=0, wspace=-0.02)
    axes = gs.subplots(sharex='col', sharey='row')

    error_data = [[error_matrix_1_1, error_matrix_1_2, error_matrix_1_3],
                  [error_matrix_2_1, error_matrix_2_2, error_matrix_2_3],
                  [error_matrix_3_1, error_matrix_3_2, error_matrix_3_3]]
    
    abs_error_data = [[abs(error_matrix_1_1), abs(error_matrix_1_2), abs(error_matrix_1_3)],
                      [abs(error_matrix_2_1), abs(error_matrix_2_2), abs(error_matrix_2_3)],
                      [abs(error_matrix_3_1), abs(error_matrix_3_2), abs(error_matrix_3_3)]]
    
    # Set this for the colorbar limits
    # vmax = np.nanmax(abs_error_data)
    # vmin = np.nanmin(abs_error_data)
    vmax = 4
    vmin = 0
    print(f'Error colorbar Upper:{vmax}')
    print(f'Error colorbar Lower:{vmin}')
    for i in range(3):
        for j in range(3):
            axes[i, j].imshow(error_data[i][j], cmap=cmap, interpolation='nearest', aspect = 1, vmin = vmin, vmax = vmax) 

            # Set x-ticks only for the bottom row (i == 2)
            if i == 2:
                axes[i, j].set_xticks(np.arange(0, 8, step=1))
                axes[i, j].set_xticklabels([int(y) for y in ypos], fontsize=7)
            else:
                axes[i, j].set_xticks([])  # Remove ticks for other rows

            # Set y-ticks only for the first column (j == 0)
            axes[i, j].set_yticks(np.arange(0, 4, step=1))
            axes[i, j].set_yticklabels([int(z) for z in zpos], fontsize=7)

            # Add follower aircraft markers and lines
            axes[i, j].plot(0, 1, 'o', color='black', markersize=8)
            axes[i, j].plot([-0.5, 2], [1, 1], 'k-', linewidth=2)

            # Make ticks thinner
            axes[i, j].tick_params(axis='both', which='both', width=0.25)

            # Make subplot borders thinner
            for spine in axes[i, j].spines.values():
                spine.set_linewidth(0.25)

            # Add y-axis label for the first column
            if j == 0:
                axes[i, j].set_ylabel(r'$\alpha = $' + f'{AoA[i]}' + r'$^{\circ}$', fontsize=10)
    # Add colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(axes[0, 0].imshow(abs_error_data[0][0], cmap=cmap, interpolation='nearest', aspect=1, vmin=vmin, vmax=vmax), cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\Delta L_D Error$', fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    fig.supxlabel(r'dz ($\%$b)', fontsize=10)
    fig.supylabel(r'dy ($\%$b)', fontsize=10)

    plt.savefig(save_file_path, dpi=1000, bbox_inches='tight', transparent=True, format='svg')


def evaluation_force_ploter(Model, file_path, save_file_path):
    is_interpolated=False
    xpos_abs = [8.25, 16.5, 24.75]
    ypos_abs = [0, 2.75, 5.5, 8.25, 11, 13.75, 16.5, 19.25]
    zpos_abs = [-2.75, 0, 2.75, 5.5]
    AoA = [0, 2.5, 5]

    # Prepare scaled tick labels
    xpos = [np.floor(x*(100/11)) for x in xpos_abs]
    ypos = [np.floor(y*(100/11)) for y in ypos_abs]
    zpos = [np.round(z*(100/11)) for z in zpos_abs]


    (L_D_1_1, L_D_1_2, L_D_1_3, 
     L_D_2_1, L_D_2_2, L_D_2_3, 
     L_D_3_1, L_D_3_2, L_D_3_3) = L_D_heatmap_matrix(
        zpos_abs, ypos_abs, AoA, file_path, Model, is_interpolated=is_interpolated
    )

    cmap = plt.get_cmap('coolwarm') 
    vmin = -6
    vmax = 6

    print(f'L_D colorbar Upper:{vmax}')
    print(f'L_D colorbar Lower:{vmin}')
    
    fig = plt.figure(figsize=(6.5, 3.25))
    gs = fig.add_gridspec(3, 3, hspace=0, wspace=-0.02)
    axes = gs.subplots(sharex='col', sharey='row')

    data = [[L_D_1_1, L_D_1_2, L_D_1_3],
            [L_D_2_1, L_D_2_2, L_D_2_3],
            [L_D_3_1, L_D_3_2, L_D_3_3]]
    
    for i in range(3):
        for j in range(3):
            axes[i, j].imshow(data[i][j], cmap=cmap, interpolation='nearest', aspect = 1, vmin=vmin, vmax=vmax)

            # Set x-ticks only for the bottom row (i == 2)
            if i == 2:
                axes[i, j].set_xticks(np.arange(0, 8, step=1))
                axes[i, j].set_xticklabels([int(y) for y in ypos], fontsize=7)
            else:
                axes[i, j].set_xticks([])  # Remove ticks for other rows

            # Set y-ticks only for the first column (j == 0)
            axes[i, j].set_yticks(np.arange(0, 4, step=1))
            axes[i, j].set_yticklabels([int(z) for z in zpos], fontsize=7)

            # Add follower aircraft markers and lines
            axes[i, j].plot(0, 1, 'o', color='black', markersize=8)
            axes[i, j].plot([-0.5, 2], [1, 1], 'k-', linewidth=2)

            # Make ticks thinner
            axes[i, j].tick_params(axis='both', which='both', width=0.25)

            # Make subplot borders thinner
            for spine in axes[i, j].spines.values():
                spine.set_linewidth(0.25)

            # Add y-axis label for the first column
            if j == 0:
                axes[i, j].set_ylabel(r'$\alpha = $' + f'{AoA[i]}' + r'$^{\circ}$', fontsize=10)
    fig.supxlabel(r'dz ($\%$b)', fontsize=10)
    fig.supylabel(r'dy ($\%$b)', fontsize=10)

    plt.savefig(save_file_path, dpi=1000, bbox_inches='tight', transparent=True, format='svg')

def determine_best_model(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    rmse_columns = [f"RMSE_{i}" for i in range(1, 6)]
    mean_rmse = df[rmse_columns].mean()

    best_rmse_column = mean_rmse.idxmin() 
    best_model_number = best_rmse_column.split('_')[1]
    
    return f"GB_{best_model_number}"

def plotting_model_evaluation_heatmap(matrix, folder_path, file_name, plotting_type):
    plotting_matrix = ma.masked_where(matrix == 0., matrix)
    vmax = plotting_matrix.max()

    plt.figure(figsize=(6.5/2, 3.25))
    if plotting_type == "RMSE":
        cbar_name = 'RMSE Value'
        cmap_name = 'Blues'
        annotation_color = 'black'
    elif plotting_type == "MI":
        cbar_name = 'Mutual Information Value'
        cmap_name = 'GnBu'
        annotation_color = 'black'
        
    vmin = 0
    vmax = 1.7 # Set maximum value for color scale for RMSE heatmap, the max is 1.67, so I set to 1.7
    heatmap = plt.imshow(plotting_matrix, cmap=cmap_name, aspect=1, vmax=vmax, vmin=vmin)
    print(f'Heatmap colorbar Upper:{vmax}')
    print(f'Heatmap colorbar Lower:{vmin}')
    cbar = plt.colorbar(heatmap, orientation='horizontal', pad=0.2)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_name, fontsize=8)

    plt.xticks(ticks=np.arange(5), labels=["None", "dx", "dy", "dz", "dx+dy+dz"], fontsize=8)
    plt.yticks(ticks=np.arange(4), labels=["All\n12", "6\noptimal", "2\nnostril", "None"], rotation=90, fontsize=8, va='center', ha='center')

    plt.xlabel('Relative position predictor', fontsize=8)
    plt.ylabel('Pressure taps predictor', fontsize=8)

    # remove border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().tick_params(left=False, bottom=False)

    # for i in range(plotting_matrix.shape[0]):
    #     for j in range(plotting_matrix.shape[1]):
    #         plt.text(j, i, f'{plotting_matrix[i, j]:.2f}', 
    #                 ha='center', va='center', color = annotation_color, fontsize=7)
    plt.tight_layout()
    
    save_path = os.path.join(folder_path, file_name)
    plt.savefig(save_path, format='svg', dpi=1000, transparent=True)

if __name__ == "__main__":
    # From model evaluation
    # #---------------------------------
    folder_path_evaluation = "Predictive_model_data/XGBoost_Model_evaluation"

    save_folder_path = "Predictive_model_data/XGBoost_Model_evaluation/Heatmaps/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    actual_L_D_file_path = "Post_processed_stats_summarized/L_D_True_Stats_Iter_1.csv"
    
    # Plot the actual data
    evaluation_force_ploter("Actual", actual_L_D_file_path, os.path.join(save_folder_path, "Heatmap_Actual.svg"))
    actual_L_D_matrix = L_D_heatmap_matrix(
        [-2.75, 0, 2.75, 5.5], [0, 2.75, 5.5, 8.25, 11, 13.75, 16.5, 19.25], [0, 2.5, 5], actual_L_D_file_path, "Actual", is_interpolated=False)

    # Get list of all CSV files in the evaluation folder
    csv_files = glob.glob(os.path.join(folder_path_evaluation, "*.csv"))

    # Loop over each CSV file
    for file_path_evaluation in csv_files:
        if not file_path_evaluation.endswith("matrix.csv"):
            file_name = os.path.basename(file_path_evaluation)
            file_name_without_ext = os.path.splitext(file_name)[0]
            
            # Best model name
            Model = determine_best_model(file_path_evaluation)
            
            # Create a new file name for the heatmap
            save_file_path = os.path.join(save_folder_path, f"Heatmap_{Model}_{file_name_without_ext}.svg")
            save_file_path_error = os.path.join(save_folder_path, f"Error_Heatmap_Test_{Model}_{file_name_without_ext}.svg")
            
            # Plot the heatmap
            # error_L_D_plotter(Model, file_path_evaluation, actual_L_D_matrix, save_file_path_error)
            # evaluation_force_ploter(Model, file_path_evaluation, save_file_path)

    # Plot the RMSE heatmap
    # -----------------------------------------------------------------
    folder_path_error = "Predictive_model_data/XGBoost_Model_evaluation/Summary"
    file_name = "RMSE_matrix.csv"
    file_path = os.path.join(folder_path_error, file_name)
    error_df = pd.read_csv(file_path)
    plotting_matrix = error_df.values[:, 1:]
    print(plotting_matrix)
    save_name = "Models_RMSE_matrix_without_number.svg"
    plotting_type = "RMSE"
    plotting_model_evaluation_heatmap(plotting_matrix, folder_path_error, save_name, plotting_type)