import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def extract_data(xpos, ypos, zpos, AoA, df, value_col, tol=1e-4):
    subset = df[
        (df['XPos'].sub(xpos).abs() < tol) &
        (df['YPos'].sub(ypos).abs() < tol) &
        (df['ZPos'].sub(zpos).abs() < tol) &
        (df['AoA'].sub(AoA).abs() < tol)
    ]
    if subset.empty:
        return np.nan
    return subset.iloc[0][value_col]


def build_heatmap_matrices(zpos_abs, ypos_abs, AoA, df, value_col):
    AoA_0, AoA_2_5, AoA_5 = AoA

    matrix_1_1 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_1_1[i][j] = extract_data(8.25, ypos_abs[j], zpos_abs[i], AoA_0, df, value_col)

    matrix_1_2 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_1_2[i][j] = extract_data(16.5, ypos_abs[j], zpos_abs[i], AoA_0, df, value_col)

    matrix_1_3 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_1_3[i][j] = extract_data(24.75, ypos_abs[j], zpos_abs[i], AoA_0, df, value_col)

    matrix_2_1 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_2_1[i][j] = extract_data(8.25, ypos_abs[j], zpos_abs[i], AoA_2_5, df, value_col)

    matrix_2_2 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_2_2[i][j] = extract_data(16.5, ypos_abs[j], zpos_abs[i], AoA_2_5, df, value_col)

    matrix_2_3 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_2_3[i][j] = extract_data(24.75, ypos_abs[j], zpos_abs[i], AoA_2_5, df, value_col)

    matrix_3_1 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_3_1[i][j] = extract_data(8.25, ypos_abs[j], zpos_abs[i], AoA_5, df, value_col)

    matrix_3_2 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_3_2[i][j] = extract_data(16.5, ypos_abs[j], zpos_abs[i], AoA_5, df, value_col)

    matrix_3_3 = np.zeros((len(zpos_abs), len(ypos_abs)))
    for i in range(len(zpos_abs)):
        for j in range(len(ypos_abs)):
            matrix_3_3[i][j] = extract_data(24.75, ypos_abs[j], zpos_abs[i], AoA_5, df, value_col)

    return (matrix_1_1, matrix_1_2, matrix_1_3,
            matrix_2_1, matrix_2_2, matrix_2_3,
            matrix_3_1, matrix_3_2, matrix_3_3)


def plot_xgb_predicted_heatmap(df, save_file_path):
    ypos_abs = [0, 2.75, 5.5, 8.25, 11, 13.75, 16.5, 19.25]
    zpos_abs = [-2.75, 0, 2.75, 5.5]
    AoA = [0, 2.5, 5]

    ypos = [np.floor(y * (100 / 11)) for y in ypos_abs]
    zpos = [np.round(z * (100 / 11)) for z in zpos_abs]

    (m_1_1, m_1_2, m_1_3,
     m_2_1, m_2_2, m_2_3,
     m_3_1, m_3_2, m_3_3) = build_heatmap_matrices(
        zpos_abs, ypos_abs, AoA, df, 'xgboost_prediction_delta_L_D_mean'
    )

    cmap = plt.get_cmap('coolwarm')
    vmin = -6
    vmax = 6
    print(f'Predicted heatmap colorbar Upper:{vmax}')
    print(f'Predicted heatmap colorbar Lower:{vmin}')

    fig = plt.figure(figsize=(6.5, 3.25))
    gs = fig.add_gridspec(3, 3, hspace=0, wspace=-0.02)
    axes = gs.subplots(sharex='col', sharey='row')

    data = [[m_1_1, m_1_2, m_1_3],
            [m_2_1, m_2_2, m_2_3],
            [m_3_1, m_3_2, m_3_3]]

    for i in range(3):
        for j in range(3):
            axes[i, j].imshow(data[i][j], cmap=cmap, interpolation='nearest', aspect=1, vmin=vmin, vmax=vmax)

            if i == 2:
                axes[i, j].set_xticks(np.arange(0, 8, step=1))
                axes[i, j].set_xticklabels([int(y) for y in ypos], fontsize=7)
            else:
                axes[i, j].set_xticks([])

            axes[i, j].set_yticks(np.arange(0, 4, step=1))
            axes[i, j].set_yticklabels([int(z) for z in zpos], fontsize=7)

            axes[i, j].plot(0, 1, 'o', color='black', markersize=8)
            axes[i, j].plot([-0.5, 2], [1, 1], 'k-', linewidth=2)

            axes[i, j].tick_params(axis='both', which='both', width=0.25)
            for spine in axes[i, j].spines.values():
                spine.set_linewidth(0.25)

            if j == 0:
                axes[i, j].set_ylabel(r'$\alpha = $' + f'{AoA[i]}' + r'$^{\circ}$', fontsize=10)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(
        axes[0, 0].imshow(data[0][0], cmap=cmap, interpolation='nearest', aspect=1, vmin=vmin, vmax=vmax),
        cax=cbar_ax, orientation='horizontal'
    )
    cbar.set_label(r'Predicted $\Delta L/D$', fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    fig.supxlabel(r'dy ($\%$b)', fontsize=10)
    fig.supylabel(r'dz ($\%$b)', fontsize=10)

    plt.savefig(save_file_path, dpi=600, bbox_inches='tight', transparent=True, format='svg')
    print(f'Saved: {save_file_path}')


def plot_xgb_error_heatmap(df, save_file_path):
    ypos_abs = [0, 2.75, 5.5, 8.25, 11, 13.75, 16.5, 19.25]
    zpos_abs = [-2.75, 0, 2.75, 5.5]
    AoA = [0, 2.5, 5]

    ypos = [np.floor(y * (100 / 11)) for y in ypos_abs]
    zpos = [np.round(z * (100 / 11)) for z in zpos_abs]

    (pred_1_1, pred_1_2, pred_1_3,
     pred_2_1, pred_2_2, pred_2_3,
     pred_3_1, pred_3_2, pred_3_3) = build_heatmap_matrices(
        zpos_abs, ypos_abs, AoA, df, 'xgboost_prediction_delta_L_D_mean'
    )

    (actual_1_1, actual_1_2, actual_1_3,
     actual_2_1, actual_2_2, actual_2_3,
     actual_3_1, actual_3_2, actual_3_3) = build_heatmap_matrices(
        zpos_abs, ypos_abs, AoA, df, 'delta_L_D_mean'
    )

    error_1_1 = pred_1_1 - actual_1_1
    error_1_2 = pred_1_2 - actual_1_2
    error_1_3 = pred_1_3 - actual_1_3
    error_2_1 = pred_2_1 - actual_2_1
    error_2_2 = pred_2_2 - actual_2_2
    error_2_3 = pred_2_3 - actual_2_3
    error_3_1 = pred_3_1 - actual_3_1
    error_3_2 = pred_3_2 - actual_3_2
    error_3_3 = pred_3_3 - actual_3_3

    cmap = LinearSegmentedColormap.from_list(
        'teal_white_coral', ['#0a7e6e', '#ffffff', '#c84b2f']
    )
    vmin = -4
    vmax = 4
    print(f'Error heatmap colorbar Upper:{vmax}')
    print(f'Error heatmap colorbar Lower:{vmin}')

    fig = plt.figure(figsize=(6.5, 3.25))
    gs = fig.add_gridspec(3, 3, hspace=0, wspace=-0.02)
    axes = gs.subplots(sharex='col', sharey='row')

    error_data = [[error_1_1, error_1_2, error_1_3],
                  [error_2_1, error_2_2, error_2_3],
                  [error_3_1, error_3_2, error_3_3]]

    for i in range(3):
        for j in range(3):
            axes[i, j].imshow(error_data[i][j], cmap=cmap, interpolation='nearest', aspect=1, vmin=vmin, vmax=vmax)

            if i == 2:
                axes[i, j].set_xticks(np.arange(0, 8, step=1))
                axes[i, j].set_xticklabels([int(y) for y in ypos], fontsize=7)
            else:
                axes[i, j].set_xticks([])

            axes[i, j].set_yticks(np.arange(0, 4, step=1))
            axes[i, j].set_yticklabels([int(z) for z in zpos], fontsize=7)

            axes[i, j].plot(0, 1, 'o', color='black', markersize=8)
            axes[i, j].plot([-0.5, 2], [1, 1], 'k-', linewidth=2)

            axes[i, j].tick_params(axis='both', which='both', width=0.25)
            for spine in axes[i, j].spines.values():
                spine.set_linewidth(0.25)

            if j == 0:
                axes[i, j].set_ylabel(r'$\alpha = $' + f'{AoA[i]}' + r'$^{\circ}$', fontsize=10)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(
        axes[0, 0].imshow(error_data[0][0], cmap=cmap, interpolation='nearest', aspect=1, vmin=vmin, vmax=vmax),
        cax=cbar_ax, orientation='horizontal'
    )
    cbar.set_label(r'Prediction error ($\hat{y} - y$)', fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    fig.supxlabel(r'dy ($\%$b)', fontsize=10)
    fig.supylabel(r'dz ($\%$b)', fontsize=10)

    plt.savefig(save_file_path, dpi=600, bbox_inches='tight', transparent=True, format='svg')
    print(f'Saved: {save_file_path}')


def plot_assembled_grid(raw_pred_dir, save_file_path, value_col, cmap, vmin, vmax, cbar_label):
    pressure_order   = [12, 6, 2, 0]
    position_order   = ['None', 'XPos', 'YPos', 'ZPos', 'XPos_YPos_ZPos']
    pressure_labels  = ['All 12', '6 selected', 'Naris-tap\npair', 'None']
    position_labels  = ['None', 'dx', 'dy', 'dz', 'dx+dy+dz']

    ypos_abs = [0, 2.75, 5.5, 8.25, 11, 13.75, 16.5, 19.25]
    zpos_abs = [-2.75, 0, 2.75, 5.5]
    AoA      = [0, 2.5, 5]

    fig = plt.figure(figsize=(22, 12))
    outer = GridSpec(4, 5, figure=fig, hspace=0.08, wspace=0.08,
                     left=0.07, right=0.95, top=0.92, bottom=0.10)

    for r, pressure in enumerate(pressure_order):
        for c, pos in enumerate(position_order):

            # Invalid cell: no predictors at all
            if pressure == 0 and pos == 'None':
                ax = fig.add_subplot(outer[r, c])
                ax.set_facecolor('black')
                ax.plot([0, 1], [0, 1], 'w-', linewidth=2, transform=ax.transAxes)
                ax.plot([0, 1], [1, 0], 'w-', linewidth=2, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(1.0)
                continue

            predictor_set = f'Pressure_{pressure}_Position_{pos}'
            csv_path = os.path.join(raw_pred_dir, f'Raw_predictions_{predictor_set}.csv')
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            if value_col == 'error':
                (p_1_1, p_1_2, p_1_3,
                 p_2_1, p_2_2, p_2_3,
                 p_3_1, p_3_2, p_3_3) = build_heatmap_matrices(
                    zpos_abs, ypos_abs, AoA, df, 'xgboost_prediction_delta_L_D_mean'
                )
                (a_1_1, a_1_2, a_1_3,
                 a_2_1, a_2_2, a_2_3,
                 a_3_1, a_3_2, a_3_3) = build_heatmap_matrices(
                    zpos_abs, ypos_abs, AoA, df, 'delta_L_D_mean'
                )
                data = [[p_1_1 - a_1_1, p_1_2 - a_1_2, p_1_3 - a_1_3],
                        [p_2_1 - a_2_1, p_2_2 - a_2_2, p_2_3 - a_2_3],
                        [p_3_1 - a_3_1, p_3_2 - a_3_2, p_3_3 - a_3_3]]
            else:
                (m_1_1, m_1_2, m_1_3,
                 m_2_1, m_2_2, m_2_3,
                 m_3_1, m_3_2, m_3_3) = build_heatmap_matrices(
                    zpos_abs, ypos_abs, AoA, df, value_col
                )
                data = [[m_1_1, m_1_2, m_1_3],
                        [m_2_1, m_2_2, m_2_3],
                        [m_3_1, m_3_2, m_3_3]]

            inner = GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[r, c], hspace=0, wspace=0)

            for i in range(3):
                for j in range(3):
                    ax = fig.add_subplot(inner[i, j])
                    ax.imshow(data[i][j], cmap=cmap, interpolation='nearest',
                              aspect='auto', vmin=vmin, vmax=vmax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.tick_params(axis='both', which='both', width=0.25)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.25)
                    ax.plot(0, 1, 'o', color='black', markersize=4)
                    ax.plot([-0.5, 2], [1, 1], 'k-', linewidth=1)

    # Row labels (pressure) on the left
    for r, label in enumerate(pressure_labels):
        pos = outer[r, 0].get_position(fig)
        fig.text(0.01, (pos.y0 + pos.y1) / 2, label,
                 ha='center', va='center', fontsize=9, rotation=90)

    # Column labels (position) on top
    for c, label in enumerate(position_labels):
        pos = outer[0, c].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, 0.95, label,
                 ha='center', va='center', fontsize=9)

    fig.text(0.50, 0.96, 'Positional predictor', ha='center', va='center', fontsize=11)
    fig.text(0.03, 0.51, 'Pressure predictor',   ha='center', va='center', fontsize=11, rotation=90)

    # Shared colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.04, 0.70, 0.015])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(save_file_path, dpi=600, bbox_inches='tight', transparent=True, format='svg')
    print(f'Saved: {save_file_path}')


if __name__ == '__main__':
    raw_pred_dir  = 'Predictive_model_data/Latency_Benchmark/Raw_prediction_tables'
    predicted_dir = 'Predictive_model_data/Plots/XGBoost_predicted_heatmap'
    error_dir     = 'Predictive_model_data/Plots/XGBoost_Error_map'

    if not os.path.exists(predicted_dir):
        os.makedirs(predicted_dir)
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    csv_files = sorted([f for f in os.listdir(raw_pred_dir) if f.startswith('Raw_predictions_') and f.endswith('.csv')])

    # for csv_file in csv_files:
    #     predictor_set = csv_file[len('Raw_predictions_'):-len('.csv')]
    #     csv_path = os.path.join(raw_pred_dir, csv_file)
    #     df = pd.read_csv(csv_path)
    #     print(f'Loaded {len(df)} rows | predictor set: {predictor_set}')

    #     plot_xgb_predicted_heatmap(df, os.path.join(predicted_dir, f'Heatmap_XGB_Predicted_{predictor_set}.svg'))
    #     plot_xgb_error_heatmap(df, os.path.join(error_dir, f'Heatmap_XGB_Error_{predictor_set}.svg'))

    # Assembled overview grids
    assembled_dir = 'Predictive_model_data/Plots/XGBoost_results_assembled_grids'
    if not os.path.exists(assembled_dir):
        os.makedirs(assembled_dir)
    error_cmap = LinearSegmentedColormap.from_list(
        'teal_white_coral', ['#0a7e6e', '#ffffff', '#c84b2f']
    )
    plot_assembled_grid(
        raw_pred_dir,
        os.path.join(assembled_dir, 'Assembled_XGB_Predicted_All.svg'),
        value_col='xgboost_prediction_delta_L_D_mean',
        cmap=plt.get_cmap('coolwarm'), vmin=-6, vmax=6,
        cbar_label=r'Predicted $\Delta L/D$'
    )
    plot_assembled_grid(
        raw_pred_dir,
        os.path.join(assembled_dir, 'Assembled_XGB_Error_All.svg'),
        value_col='error',
        cmap=error_cmap, vmin=-4, vmax=4,
        cbar_label=r'Prediction error ($\hat{y} - y$)'
    )
