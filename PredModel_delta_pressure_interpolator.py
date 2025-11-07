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

def interpolate_pressure_dataset(input_dataset, AoA, XPos, tap_number, interpolate_resolutions):
    group_df = input_dataset[
    (input_dataset["AoA"] == AoA) & 
    (input_dataset["XPos"] == XPos)]

    pivot_df = group_df.pivot(
        index="YPos", 
        columns="ZPos", 
        values=f"tap{tap_number}_delta_pressure_mean"
    )

    # Sort the pivoted index and columns by ascending order
    pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)

    # Extract the sorted arrays
    Y_unique = pivot_df.index.to_numpy()  
    Z_unique = pivot_df.columns.to_numpy()  

    pressure_grid = pivot_df.values  # shape (#unique_Y, #unique_Z)
    interpolator = RegularGridInterpolator((Y_unique, Z_unique), pressure_grid, method='linear', bounds_error=False, fill_value=None)

    y_dense = np.linspace(Y_unique.min(), Y_unique.max(), interpolate_resolutions)
    z_dense = np.linspace(Z_unique.min(), Z_unique.max(), interpolate_resolutions)
    Y_dense, Z_dense = np.meshgrid(y_dense, z_dense, indexing='ij')

    points = np.column_stack([Y_dense.ravel(), Z_dense.ravel()])  # shape (N, 2)

    delta_pressure_tap_dense = interpolator(points).reshape(Y_dense.shape)

    interpolated_data = pd.DataFrame({
        "AoA": AoA,
        "XPos": XPos,
        "YPos": Y_dense.ravel(),
        "ZPos": Z_dense.ravel(),
        f"tap{tap_number}_delta_pressure_mean": delta_pressure_tap_dense.ravel()
    })
    print(f"Interpolated tap {tap_number} data for AoA {AoA} and XPos {XPos}")
    return interpolated_data

def plotter(dataset, AoA, XPos, tap_number):
    pivot_df = dataset.pivot(index="ZPos", columns="YPos", values=f"tap{tap_number}_delta_pressure_mean")
    
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    
    # Extract grid boundaries for proper axis scaling.
    y_min, y_max = dataset["YPos"].min(), dataset["YPos"].max()
    z_min, z_max = dataset["ZPos"].min(), dataset["ZPos"].max()
    
    plt.figure(figsize=(8,6))
    plt.imshow(pivot_df.values, extent=(y_min, y_max, z_min, z_max),
               origin='lower', aspect='auto', cmap='viridis', vmax=250, vmin=-250)
    plt.xlabel("YPos")
    plt.ylabel("ZPos")
    plt.title(f"Pressure Tap {tap_number} Delta Pressure at AoA {AoA} and XPos {XPos}")
    plt.colorbar(label="Delta Pressure")
    plt.show()


if __name__ == "__main__":
    AoA = [0, 2.5, 5]
    XPos = [8.25, 16.5, 24.75]
    interpolate_resolutions = 200
    
    folder_path = "Post_processed_stats_summarized/"
    pressure_iter_0 = pd.read_csv(os.path.join(folder_path, "Pressure_True_Stats_Iter_0.csv"))
    pressure_iter_1 = pd.read_csv(os.path.join(folder_path, "Pressure_True_Stats_Iter_1.csv"))
    pressure_iter_0["AoA"] = pressure_iter_0["AoA"].astype(str).str.replace("_", ".").astype(float)
    pressure_iter_1["AoA"] = pressure_iter_1["AoA"].astype(str).str.replace("_", ".").astype(float)

    pressure_input_data = pressure_iter_1
    pressure_output_list = []

    for i in range(len(AoA)):
        for j in range(len(XPos)):
            interpolated_tap_0 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 0, interpolate_resolutions)
            interpolated_tap_2 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 2, interpolate_resolutions)
            interpolated_tap_4 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 4, interpolate_resolutions)
            interpolated_tap_6 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 6, interpolate_resolutions)
            interpolated_tap_8 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 8, interpolate_resolutions)
            interpolated_tap_9 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 9, interpolate_resolutions)
            interpolated_tap_10 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 10, interpolate_resolutions)
            interpolated_tap_11 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 11, interpolate_resolutions)
            interpolated_tap_13 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 13, interpolate_resolutions)
            interpolated_tap_15 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 15, interpolate_resolutions)
            interpolated_tap_17 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 17, interpolate_resolutions)
            interpolated_tap_19 = interpolate_pressure_dataset(pressure_input_data, AoA[i], XPos[j], 19, interpolate_resolutions)

            ## Plotting the interpolated data (if needed)
            # plotter(interpolated_tap_0, AoA[i], XPos[j], 0)
            # plotter(interpolated_tap_2, AoA[i], XPos[j], 2)
            # plotter(interpolated_tap_4, AoA[i], XPos[j], 4)
            # plotter(interpolated_tap_6, AoA[i], XPos[j], 6)
            # plotter(interpolated_tap_8, AoA[i], XPos[j], 8)
            # plotter(interpolated_tap_9, AoA[i], XPos[j], 9)
            # plotter(interpolated_tap_10, AoA[i], XPos[j], 10)
            # plotter(interpolated_tap_11, AoA[i], XPos[j], 11)
            # plotter(interpolated_tap_13, AoA[i], XPos[j], 13)
            # plotter(interpolated_tap_15, AoA[i], XPos[j], 15)
            # plotter(interpolated_tap_17, AoA[i], XPos[j], 17)
            # plotter(interpolated_tap_19, AoA[i], XPos[j], 19)

            tap_dfs = [interpolated_tap_0, interpolated_tap_2, interpolated_tap_4,
                        interpolated_tap_6, interpolated_tap_8, interpolated_tap_9,
                        interpolated_tap_10, interpolated_tap_11, interpolated_tap_13,
                        interpolated_tap_15, interpolated_tap_17, interpolated_tap_19]

            pressure_output_in_AoA_XPos_df = reduce(
                lambda left, right: pd.merge(left, right, on=['AoA', 'XPos', 'YPos', 'ZPos'], how='outer'),
                tap_dfs
            )

            pressure_output_list.append(pressure_output_in_AoA_XPos_df)
        
    pressure_output_master_df = pd.concat(pressure_output_list, ignore_index=True)
    
    output_folder_path = "Predictive_model_data/Interpolated_pressure_data/"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    pressure_output_master_df.to_csv(os.path.join(output_folder_path, "Pressure_True_Stats_Iter_1_Interpolated.csv"), index=False)
