import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRESSURE_ORDER = [12, 6, 2, 0]
POSITION_ORDER = ["None", "XPos", "YPos", "ZPos", "XPos_YPos_ZPos"]
PRESSURE_LABELS = ["All\n12", "6\noptimal", "2\nnostril", "None"]
POSITION_LABELS = ["None", "dx", "dy", "dz", "dx+dy+dz"]
MODEL_ORDER = ["LinearRegression", "RidgeRegression", "LassoRegression", "PLSRegression"]
MODEL_LABELS = ["OLS", "Ridge", "Lasso", "PLS"]


def parse_predictor_set_name(name):
    parts = name.split("_")
    pressure_idx = parts.index("Pressure")
    position_idx = parts.index("Position")
    pressure = int(parts[pressure_idx + 1])
    position = "_".join(parts[position_idx + 1:])
    return pressure, position


def build_rmse_matrix(model_df):
    matrix = np.full((len(PRESSURE_ORDER), len(POSITION_ORDER)), np.nan)
    for _, row in model_df.iterrows():
        pressure, position = parse_predictor_set_name(row["predictor_set_name"])
        if pressure in PRESSURE_ORDER and position in POSITION_ORDER:
            r = PRESSURE_ORDER.index(pressure)
            c = POSITION_ORDER.index(position)
            matrix[r, c] = row["rmse_mean"]
    return matrix


def plot_simple_model_heatmaps(
    combined_summary_df,
    save_folder_path,
    file_name="Heatmap_simple_models_rmse.svg"
):
    vmin = combined_summary_df["rmse_mean"].min()
    vmax = combined_summary_df["rmse_mean"].max()

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.0))

    for ax_idx, (model_name, model_label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
        ax = axes.flat[ax_idx]
        model_df = combined_summary_df[combined_summary_df["model_name"] == model_name]
        matrix = build_rmse_matrix(model_df)
        masked_matrix = np.ma.masked_invalid(matrix)

        im = ax.imshow(masked_matrix, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(POSITION_ORDER)))
        ax.set_xticklabels(POSITION_LABELS, fontsize=8)
        ax.set_yticks(np.arange(len(PRESSURE_ORDER)))
        ax.set_yticklabels(PRESSURE_LABELS, fontsize=8, va="center")
        ax.set_xlabel("Position predictor", fontsize=8)
        ax.set_ylabel("Pressure taps predictor", fontsize=8)
        ax.set_title(model_label, fontsize=9)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", pad=0.08, fraction=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Mean CV RMSE", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_folder_path, exist_ok=True)
    save_path = os.path.join(save_folder_path, file_name)
    plt.savefig(save_path, format="svg", dpi=1000, transparent=True)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    folder_path = "Predictive_model_data/Grouped_CV_Simple_Models"
    combined_summary_df = pd.read_csv(os.path.join(folder_path, "Combined_summary_metrics.csv"))
    plot_simple_model_heatmaps(combined_summary_df, folder_path)
