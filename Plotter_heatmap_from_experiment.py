import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_L_D_data(Iter, xpos, ypos, zpos, AoA):
    folder_path = 'Post_processed_stats_summarized/'
    filename = 'L_D_True_Stats_Iter_' + str(Iter) + '.csv'
    file_path = folder_path + filename
    data_frame = pd.read_csv(file_path)
    expected_sorted_data_row = data_frame.loc[(data_frame['XPos'] == xpos) & (data_frame['YPos'] == ypos) & (data_frame['ZPos'] == zpos) & (data_frame['AoA'] == AoA)]
    expected_delta_L_D = expected_sorted_data_row['delta_L_D_mean'].values[0]
    return expected_delta_L_D

def extract_pressure_data(Iter, xpos, ypos, zpos, AoA):
    folder_path = 'Post_processed_stats_summarized/'
    filename = 'Pressure_True_Stats_Iter_' + str(Iter) + '.csv'
    file_path = folder_path + filename
    data_frame = pd.read_csv(file_path)
    data_frame['AoA'] = data_frame['AoA'].str.replace('_', '.')
    if data_frame['AoA'].dtype == 'object':
        data_frame['AoA'] = data_frame['AoA'].astype(float)
    expected_sorted_data_row = data_frame.loc[(data_frame['XPos'] == xpos) & (data_frame['YPos'] == ypos) & (data_frame['ZPos'] == zpos) & (data_frame['AoA'] == AoA)]

    expected_delta_tap_0 = expected_sorted_data_row['tap0_delta_pressure_mean'].values[0]
    expected_delta_tap_2 = expected_sorted_data_row['tap2_delta_pressure_mean'].values[0]
    expected_delta_tap_4 = expected_sorted_data_row['tap4_delta_pressure_mean'].values[0]
    expected_delta_tap_6 = expected_sorted_data_row['tap6_delta_pressure_mean'].values[0]
    expected_delta_tap_8 = expected_sorted_data_row['tap8_delta_pressure_mean'].values[0]
    expected_delta_tap_9 = expected_sorted_data_row['tap9_delta_pressure_mean'].values[0]
    expected_delta_tap_10 = expected_sorted_data_row['tap10_delta_pressure_mean'].values[0]
    expected_delta_tap_11 = expected_sorted_data_row['tap11_delta_pressure_mean'].values[0]
    expected_delta_tap_13 = expected_sorted_data_row['tap13_delta_pressure_mean'].values[0]
    expected_delta_tap_15 = expected_sorted_data_row['tap15_delta_pressure_mean'].values[0]
    expected_delta_tap_17 = expected_sorted_data_row['tap17_delta_pressure_mean'].values[0]
    expected_delta_tap_19 = expected_sorted_data_row['tap19_delta_pressure_mean'].values[0]

    section_i = (expected_delta_tap_0 + expected_delta_tap_2)/2
    section_ii = (expected_delta_tap_4 + expected_delta_tap_6 + expected_delta_tap_8)/3
    section_iii = (expected_delta_tap_9 + expected_delta_tap_10)/2
    section_iv = (expected_delta_tap_11 + expected_delta_tap_13 + expected_delta_tap_15)/3
    section_v = (expected_delta_tap_17 + expected_delta_tap_19)/2

    result = {
        'section_i': section_i,
        'section_ii': section_ii,
        'section_iii': section_iii,
        'section_iv': section_iv,
        'section_v': section_v
        }

    return result

 
def L_D_heatmap_matrix(zpos_abs, ypos_abs, Iter, AoA):
    AoA_0 = AoA[0]
    AoA_2_5 = AoA[1]
    AoA_5 = AoA[2]

    # For subplot 1
    L_D_plot_matrix_1_1 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 75% x_pos AoA = 0

    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_1_1[i][j] = extract_L_D_data(Iter, 8.25, ypos_abs[j], zpos_abs[i], AoA_0)
            # [BACK-UP] L_D_plot_matrix_1_1[len(zpos_abs)-1-i][len(ypos_abs)-1-j] 

    # For subplot 2
    L_D_plot_matrix_1_2 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 150% x_pos AoA = 0
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_1_2[i][j] = extract_L_D_data(Iter, 16.5, ypos_abs[j], zpos_abs[i], AoA_0)

    # For subplot 3
    L_D_plot_matrix_1_3 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 225% x_pos AoA = 0
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_1_3[i][j] = extract_L_D_data(Iter, 24.75, ypos_abs[j], zpos_abs[i], AoA_0)

    # For subplot 4
    L_D_plot_matrix_2_1 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 75% x_pos AoA = 2.5
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_2_1[i][j] = extract_L_D_data(Iter, 8.25, ypos_abs[j], zpos_abs[i], AoA_2_5)

    # For subplot 5
    L_D_plot_matrix_2_2 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 150% x_pos AoA = 2.5
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_2_2[i][j] = extract_L_D_data(Iter, 16.5, ypos_abs[j], zpos_abs[i], AoA_2_5)

    # For subplot 6
    L_D_plot_matrix_2_3 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 225% x_pos AoA = 2.5
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_2_3[i][j] = extract_L_D_data(Iter, 24.75, ypos_abs[j], zpos_abs[i], AoA_2_5)

    # For subplot 7
    L_D_plot_matrix_3_1 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 75% x_pos AoA = 5
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_3_1[i][j] = extract_L_D_data(Iter, 8.25, ypos_abs[j], zpos_abs[i], AoA_5)
    
    # For subplot 8
    L_D_plot_matrix_3_2 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 150% x_pos AoA = 5
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_3_2[i][j] = extract_L_D_data(Iter, 16.5, ypos_abs[j], zpos_abs[i], AoA_5)
    
    # For subplot 9
    L_D_plot_matrix_3_3 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 225% x_pos AoA = 5
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            L_D_plot_matrix_3_3[i][j] = extract_L_D_data(Iter, 24.75, ypos_abs[j], zpos_abs[i], AoA_5)

    return L_D_plot_matrix_1_1, L_D_plot_matrix_1_2, L_D_plot_matrix_1_3, L_D_plot_matrix_2_1, L_D_plot_matrix_2_2, L_D_plot_matrix_2_3, L_D_plot_matrix_3_1, L_D_plot_matrix_3_2, L_D_plot_matrix_3_3


def pressure_heatmap_matrix(zpos_abs, ypos_abs, Iter, AoA):
    AoA_0 = AoA
    # For subplot 1
    pressure_plot_matrix_1_1 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 75% x_pos section i

    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_1_1[i][j] = extract_pressure_data(Iter, 8.25, ypos_abs[j], zpos_abs[i], AoA_0)['section_i']

    # For subplot 2
    pressure_plot_matrix_1_2 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 150% x_pos section i
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_1_2[i][j] = extract_pressure_data(Iter, 16.5, ypos_abs[j], zpos_abs[i], AoA_0)['section_i']

    # For subplot 3
    pressure_plot_matrix_1_3 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 225% x_pos section i
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_1_3[i][j] = extract_pressure_data(Iter, 24.75, ypos_abs[j], zpos_abs[i], AoA_0)['section_i']

    # For subplot 4
    pressure_plot_matrix_2_1 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 75% x_pos section ii
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_2_1[i][j] = extract_pressure_data(Iter, 8.25, ypos_abs[j], zpos_abs[i], AoA_0)['section_ii']

    # For subplot 5
    pressure_plot_matrix_2_2 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 150% x_pos section ii
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_2_2[i][j] = extract_pressure_data(Iter, 16.5, ypos_abs[j], zpos_abs[i], AoA_0)['section_ii']


    # For subplot 6
    pressure_plot_matrix_2_3 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 225% x_pos section ii
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_2_3[i][j] = extract_pressure_data(Iter, 24.75, ypos_abs[j], zpos_abs[i], AoA_0)['section_ii']

    # For subplot 7
    pressure_plot_matrix_3_1 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 75% x_pos section iii
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_3_1[i][j] = extract_pressure_data(Iter, 8.25, ypos_abs[j], zpos_abs[i], AoA_0)['section_iii']
    
    # For subplot 8
    pressure_plot_matrix_3_2 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 150% x_pos section iii
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_3_2[i][j] = extract_pressure_data(Iter, 16.5, ypos_abs[j], zpos_abs[i], AoA_0)['section_iii']

    # For subplot 9
    pressure_plot_matrix_3_3 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 225% x_pos section iii
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_3_3[i][j] = extract_pressure_data(Iter, 24.75, ypos_abs[j], zpos_abs[i], AoA_0)['section_iii']

    # For subplot 10
    pressure_plot_matrix_4_1 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 75% x_pos section iv
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_4_1[i][j] = extract_pressure_data(Iter, 8.25, ypos_abs[j], zpos_abs[i], AoA_0)['section_iv']

    # For subplot 11
    pressure_plot_matrix_4_2 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 150% x_pos section iv
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_4_2[i][j] = extract_pressure_data(Iter, 16.5, ypos_abs[j], zpos_abs[i], AoA_0)['section_iv']

    # For subplot 12
    pressure_plot_matrix_4_3 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 225% x_pos section iv
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_4_3[i][j] = extract_pressure_data(Iter, 24.75, ypos_abs[j], zpos_abs[i], AoA_0)['section_iv']

    # For subplot 13
    pressure_plot_matrix_5_1 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 75% x_pos section v
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_5_1[i][j] = extract_pressure_data(Iter, 8.25, ypos_abs[j], zpos_abs[i], AoA_0)['section_v']

    # For subplot 14
    pressure_plot_matrix_5_2 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 150% x_pos section v
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_5_2[i][j] = extract_pressure_data(Iter, 16.5, ypos_abs[j], zpos_abs[i], AoA_0)['section_v']

    # For subplot 15
    pressure_plot_matrix_5_3 = np.zeros((len(zpos_abs), len(ypos_abs))) # 4x8 225% x_pos section v
    for i in range(0, len(zpos_abs)):
        for j in range(0, len(ypos_abs)):
            pressure_plot_matrix_5_3[i][j] = extract_pressure_data(Iter, 24.75, ypos_abs[j], zpos_abs[i], AoA_0)['section_v']
            
    return pressure_plot_matrix_1_1, pressure_plot_matrix_1_2, pressure_plot_matrix_1_3, pressure_plot_matrix_2_1, pressure_plot_matrix_2_2, pressure_plot_matrix_2_3, pressure_plot_matrix_3_1, pressure_plot_matrix_3_2, pressure_plot_matrix_3_3, pressure_plot_matrix_4_1, pressure_plot_matrix_4_2, pressure_plot_matrix_4_3, pressure_plot_matrix_5_1, pressure_plot_matrix_5_2, pressure_plot_matrix_5_3


def main():
    xpos_abs = [8.25, 16.5, 24.75]
    ypos_abs = [0, 2.75, 5.5, 8.25, 11, 13.75, 16.5, 19.25]
    zpos_abs = [-2.75, 0, 2.75, 5.5]
    AoA = [0, 2.5, 5]

    xpos = [0]*len(xpos_abs)
    ypos = [0]*len(ypos_abs)
    zpos = [0]*len(zpos_abs)
    
    for i in range(0, len(xpos_abs)):
        xpos[i] = np.floor(xpos_abs[i]*(100/11))
    for i in range(0, len(ypos_abs)):
        ypos[i] = np.floor(ypos_abs[i]*(100/11))
    for i in range(0, len(zpos_abs)):
        zpos[i] = np.round(zpos_abs[i]*(100/11))

    for Iter in [0, 1]:
        # Plotting L_D Heatmap
        #--------------------------------
        L_D_1_1, L_D_1_2, L_D_1_3, L_D_2_1, L_D_2_2, L_D_2_3, L_D_3_1, L_D_3_2, L_D_3_3 = L_D_heatmap_matrix(zpos_abs, ypos_abs, Iter, AoA)

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

        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(f"{out_folder_path}HeatMap_L_D_Iter_{Iter}.svg", format="svg", dpi = 600)


    # Plotting Pressure Heatmap
    #--------------------------------
    for Iter in [0, 1]:
        for alpha in AoA:
            P_1_1, P_1_2, P_1_3, P_2_1, P_2_2, P_2_3, P_3_1, P_3_2, P_3_3, P_4_1, P_4_2, P_4_3, P_5_1, P_5_2, P_5_3 = pressure_heatmap_matrix(zpos_abs, ypos_abs, Iter, alpha)

            # find the max and min values for the colorbar from the data
            v_max = np.max([P_1_1, P_1_2, P_1_3, P_2_1, P_2_2, P_2_3, P_3_1, P_3_2, P_3_3, P_4_1, P_4_2, P_4_3, P_5_1, P_5_2, P_5_3])
            v_min = np.min([P_1_1, P_1_2, P_1_3, P_2_1, P_2_2, P_2_3, P_3_1, P_3_2, P_3_3, P_4_1, P_4_2, P_4_3, P_5_1, P_5_2, P_5_3])

            range_max = np.max([np.abs(v_max), np.abs(v_min)])
            range_max = np.ceil(range_max)

            # vmin = -range_max
            # vmax = range_max
            vmin = -248.84
            vmax = 248.84

            print(f'Pressure colorbar Upper:{vmax}')
            print(f'Pressure colorbar Lower:{vmin}')

            cmap = plt.get_cmap('PuOr_r')

            fig = plt.figure(figsize=(6.5, 5.416))
            gs = fig.add_gridspec(5, 3, hspace=0, wspace=-0.02)
            axes = gs.subplots(sharex='col', sharey='row')

            data = [[P_1_1, P_1_2, P_1_3],
                    [P_2_1, P_2_2, P_2_3],
                    [P_3_1, P_3_2, P_3_3],
                    [P_4_1, P_4_2, P_4_3],
                    [P_5_1, P_5_2, P_5_3]]

            section_labels = ['i', 'ii', 'iii', 'iv', 'v']
            for i in range(5):
                for j in range(3):
                    axes[i, j].imshow(data[i][j], cmap=cmap, interpolation='nearest', aspect = 1, vmin=vmin, vmax=vmax)
                    # Set x-ticks only for the bottom row (i == 4)
                    if i == 4:
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

                    # Add y-axis label for the first column
                    if j == 0:
                        axes[i, j].set_ylabel(section_labels[i], fontsize=10)
                    
                    # Make ticks thinner
                    axes[i, j].tick_params(axis='both', which='both', width=0.25)

                    # Make subplot borders thinner
                    for spine in axes[i, j].spines.values():
                        spine.set_linewidth(0.25)

            fig.supxlabel(r'dz $(\%b)$', fontsize=10)
            fig.supylabel(r'dy $(\%b)$', fontsize=10)

            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(f"{out_folder_path}HeatMap_Pressure_Iter_{Iter}_AoA_{alpha}.svg", format="svg", dpi = 600)
        

if __name__ == '__main__':
    out_folder_path = 'Post_processed_stats_summarized/Heatmaps_from_Experiment/'
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    main()