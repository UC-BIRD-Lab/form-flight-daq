import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRESSURE_ORDER = [12, 6, 2, 0]
POSITION_ORDER = ["None", "XPos", "YPos", "ZPos", "XPos_YPos_ZPos"]
PRESSURE_LABELS = ["All\n12", "6\noptimal", "2\nnostril", "None"]
POSITION_LABELS = ["None", "dx", "dy", "dz", "dx+dy+dz"]


def collect_grouped_nested_cv_summary_info(folder_path):
    summary_files = sorted(glob.glob(os.path.join(folder_path, "Summary_metrics_*.csv")))
    rows = []

    for file_path in summary_files:
        file_name = os.path.basename(file_path)
        stem = os.path.splitext(file_name)[0]
        parts = stem.split("_")

        pressure_idx = parts.index("Pressure")
        position_idx = parts.index("Position")
        summary_df = pd.read_csv(file_path)
        rmse_row = summary_df[summary_df["metric"].str.lower() == "rmse"]

        if rmse_row.empty:
            mean_rmse = None
            std_rmse = None
        else:
            mean_rmse = rmse_row["mean"].iloc[0]
            std_rmse = rmse_row["std"].iloc[0]

        rows.append({
            "file_name": file_name,
            "file_path": file_path,
            "pressure_set": int(parts[pressure_idx + 1]),
            "position_set": "_".join(parts[position_idx + 1:]),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse
        })

    return pd.DataFrame(rows)


def build_rmse_matrix(summary_df):
    matrix = np.full((len(PRESSURE_ORDER), len(POSITION_ORDER)), np.nan)
    for _, row in summary_df.iterrows():
        if row["pressure_set"] in PRESSURE_ORDER and row["position_set"] in POSITION_ORDER:
            r = PRESSURE_ORDER.index(row["pressure_set"])
            c = POSITION_ORDER.index(row["position_set"])
            matrix[r, c] = row["mean_rmse"]
    return matrix


def plot_grouped_nested_cv_heatmap(summary_df, save_folder_path, file_name="Heatmap_grouped_nested_cv_rmse.svg"):
    matrix = build_rmse_matrix(summary_df)
    masked_matrix = np.ma.masked_invalid(matrix)

    vmin = 0
    vmax = np.nanmax(matrix)
    print(f"Heatmap colorbar upper: {vmax:.4f}")
    print(f"Heatmap colorbar lower: {vmin}")

    fig, ax = plt.subplots(figsize=(6.5 / 2, 3.25))

    im = ax.imshow(masked_matrix, cmap="Blues", aspect=1, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.2)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Mean CV RMSE", fontsize=8)

    ax.set_xticks(np.arange(len(POSITION_ORDER)))
    ax.set_xticklabels(POSITION_LABELS, fontsize=8)
    ax.set_yticks(np.arange(len(PRESSURE_ORDER)))
    ax.set_yticklabels(PRESSURE_LABELS, rotation=90, fontsize=8, va="center", ha="center")

    ax.set_xlabel("Relative position predictor", fontsize=8)
    ax.set_ylabel("Pressure taps predictor", fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.show()
    save_path = os.path.join(save_folder_path, file_name)
    # plt.savefig(save_path, format="svg", dpi=1000, transparent=True)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    folder_path = "Predictive_model_data/Resulting_plots_tables/Grouped_Nested_CV_XGB"
    summary_df = collect_grouped_nested_cv_summary_info(folder_path)
    print(summary_df)
    plot_grouped_nested_cv_heatmap(summary_df, folder_path)
