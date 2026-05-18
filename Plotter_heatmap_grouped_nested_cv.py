import glob
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRESSURE_ORDER = [12, 6, 2, 0]
POSITION_ORDER = ["None", "XPos", "YPos", "ZPos", "XPos_YPos_ZPos"]
PRESSURE_LABELS = ["All\n12", "6\nselected", "Naris-tap\npair", "None"]
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

    matrix_df = pd.DataFrame(
        matrix,
        index=PRESSURE_LABELS,
        columns=POSITION_LABELS
    ).round(4)
    print("RMSE matrix:\n", matrix_df.to_string(), "\n")

    vmin = 0
    vmax = np.nanmax(matrix)
    print(f"Heatmap colorbar upper: {vmax:.4f}")
    print(f"Heatmap colorbar lower: {vmin}")

    fig, ax = plt.subplots(figsize=(6.5 / 2, 3.25))

    im = ax.imshow(masked_matrix, cmap="Blues", aspect=1, vmin=vmin, vmax=vmax)

    # Mark the invalid cell (0 pressure taps + no position = no predictors)
    inv_r = PRESSURE_ORDER.index(0)
    inv_c = POSITION_ORDER.index("None")
    ax.add_patch(mpatches.Rectangle(
        (inv_c - 0.5, inv_r - 0.5), 1, 1,
        linewidth=2.0, edgecolor="black", facecolor="black", zorder=3
    ))
    for dx, dy in [(1, 1), (1, -1)]:
        ax.plot([inv_c - 0.4 * dx, inv_c + 0.4 * dx],
                [inv_r - 0.4 * dy, inv_r + 0.4 * dy],
                color="white", linewidth=1.5, zorder=4)

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
    ax.tick_params(left=False, bottom=False, pad=6)

    plt.tight_layout()
    os.makedirs(save_folder_path, exist_ok=True)
    save_path = os.path.join(save_folder_path, file_name)
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(save_path, format="svg", bbox_inches="tight", dpi=600)
    print(f"Saved to {save_path}")
    plt.show()
    plt.close()


PRESSURE_TEX_LABELS = {
    12: r"All 12\,$\Delta p$",
    6:  r"6 selected\,$\Delta p$",
    2:  r"Naris-tap pair",
    0:  r"None",
}
POSITION_TEX_LABELS = {
    "None":           "None",
    "XPos":           r"$\mathrm{d}x$",
    "YPos":           r"$\mathrm{d}y$",
    "ZPos":           r"$\mathrm{d}z$",
    "XPos_YPos_ZPos": r"$\mathrm{d}x$+$\mathrm{d}y$+$\mathrm{d}z$",
}


def write_heatmap_table_latex(summary_df, save_folder_path,
                               file_name="Heatmap_RMSE_Table.tex"):
    col_headers = [POSITION_TEX_LABELS[p] for p in POSITION_ORDER]
    n_cols = len(POSITION_ORDER)
    col_spec = "l|" + "c" * n_cols

    lines = [
        r"        \begin{table}[htbp]",
        r"        \centering",
        r"        \captionsetup{font=footnotesize}",
        r"        \caption{XGBoost grouped nested CV RMSE (mean\,$\pm$\,std across outer folds) "
        r"for all pressure-tap and position-predictor combinations. "
        r"The crossed cell (\textbf{N/A}) has no predictors.}",
        r"        \footnotesize",
        r"        \setlength{\tabcolsep}{5pt}",
        r"        \renewcommand{\arraystretch}{1.2}",
        f"        \\begin{{tabular}}{{{col_spec}}}",
        r"        \hline\hline",
        "        \\textbf{Pressure taps} & " +
        " & ".join(f"\\textbf{{{h}}}" for h in col_headers) + r" \\",
        r"        \hline",
    ]

    for p in PRESSURE_ORDER:
        row_label = PRESSURE_TEX_LABELS[p]
        cells = []
        for pos in POSITION_ORDER:
            if p == 0 and pos == "None":
                cells.append(r"\textbf{N/A}")
                continue
            match = summary_df[
                (summary_df["pressure_set"] == p) &
                (summary_df["position_set"] == pos)
            ]
            if match.empty or pd.isna(match["mean_rmse"].iloc[0]):
                cells.append("---")
            else:
                mean = match["mean_rmse"].iloc[0]
                std  = match["std_rmse"].iloc[0]
                cells.append(f"{mean:.2f}\\,$\\pm$\\,{std:.2f}")
        lines.append(f"        {row_label} & " + " & ".join(cells) + r" \\")

    lines += [
        r"        \hline",
        r"        \end{tabular}",
        r"        \label{tab:heatmap_rmse}",
        r"        \end{table}",
    ]

    os.makedirs(save_folder_path, exist_ok=True)
    out_path = os.path.join(save_folder_path, file_name)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    data_folder   = "Predictive_model_data/Grouped_Nested_CV_XGB"
    plot_folder   = "Predictive_model_data/Plots"
    table_folder  = "Predictive_model_data/Table"
    summary_df = collect_grouped_nested_cv_summary_info(data_folder)
    print(summary_df)
    plot_grouped_nested_cv_heatmap(summary_df, plot_folder)
    write_heatmap_table_latex(summary_df, table_folder)
