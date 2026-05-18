import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nested_cv_dir    = "Predictive_model_data/Grouped_Nested_CV_XGB"
simple_model_dir = "Predictive_model_data/Grouped_CV_Simple_Models"
output_dir       = "Predictive_model_data/Plots"

simple_model_names = ["LinearRegression", "RidgeRegression", "LassoRegression", "PLSRegression"]

model_order = ["LinearRegression", "RidgeRegression", "LassoRegression", "PLSRegression", "XGBoost"]

model_display_name = {
    "LinearRegression": "Linear Reg.",
    "RidgeRegression":  "Ridge Reg.",
    "LassoRegression":  "Lasso Reg.",
    "PLSRegression":    "PLS Reg.",
    "XGBoost":          "XGBoost",
}

model_colors = {
    "LinearRegression": "#4C9BE8",
    "RidgeRegression":  "#F28C38",
    "LassoRegression":  "#6DBF8A",
    "PLSRegression":    "#E8747C",
    "XGBoost":          "#9B72CF",
}

model_markers = {
    "LinearRegression": "o",
    "RidgeRegression":  "s",
    "LassoRegression":  "^",
    "PLSRegression":    "D",
    "XGBoost":          "P",
}

# ---------------------------------------------------------------------------
# Pressure tap trend: dx,dy,dz → 2Δp+XYZ → 6Δp+XYZ → 12Δp+XYZ
# ---------------------------------------------------------------------------

pressure_set_order = [
    "Pressure_0_Position_XPos_YPos_ZPos",
    "Pressure_2_Position_XPos_YPos_ZPos",
    "Pressure_6_Position_XPos_YPos_ZPos",
    "Pressure_12_Position_XPos_YPos_ZPos",
]

pressure_set_labels = {
    "Pressure_0_Position_XPos_YPos_ZPos":  "dx,dy,dz",
    "Pressure_2_Position_XPos_YPos_ZPos":  r"2$\Delta p$+XYZ",
    "Pressure_6_Position_XPos_YPos_ZPos":  r"6$\Delta p$+XYZ",
    "Pressure_12_Position_XPos_YPos_ZPos": r"12$\Delta p$+XYZ",
}

# ---------------------------------------------------------------------------
# Position sensitivity: 12Δp only → +dx → +dy → +dz → +dx,dy,dz
# ---------------------------------------------------------------------------

position_set_order = [
    "Pressure_12_Position_None",
    "Pressure_12_Position_XPos",
    "Pressure_12_Position_YPos",
    "Pressure_12_Position_ZPos",
    "Pressure_12_Position_XPos_YPos_ZPos",
]

position_set_labels = {
    "Pressure_12_Position_None":           r"12$\Delta p$",
    "Pressure_12_Position_XPos":           r"12$\Delta p$+dx",
    "Pressure_12_Position_YPos":           r"12$\Delta p$+dy",
    "Pressure_12_Position_ZPos":           r"12$\Delta p$+dz",
    "Pressure_12_Position_XPos_YPos_ZPos": r"12$\Delta p$+dx,dy,dz",
}


def sanitize_name(name):
    return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in name)


def load_data(set_order):
    """Return {model_name: {raw_predictor_set: (mean_rmse, std_rmse)}}."""
    data = {m: {} for m in model_order}

    if os.path.exists(simple_model_dir):
        for fname in os.listdir(simple_model_dir):
            if not (fname.startswith("Outer_fold_results_") and fname.endswith(".csv")):
                continue
            stem = fname[len("Outer_fold_results_"):-4]
            for model_name in simple_model_names:
                safe = sanitize_name(model_name)
                if stem.startswith(safe + "_"):
                    raw_ps = stem[len(safe) + 1:]
                    if raw_ps in set_order:
                        df = pd.read_csv(os.path.join(simple_model_dir, fname))
                        vals = df["rmse"].dropna()
                        data[model_name][raw_ps] = (vals.mean(), vals.std(ddof=1))
                    break

    if os.path.exists(nested_cv_dir):
        for raw_ps in set_order:
            summary_csv = os.path.join(nested_cv_dir, f"Summary_metrics_{raw_ps}.csv")
            if os.path.exists(summary_csv):
                df = pd.read_csv(summary_csv)
                row = df[df["metric"] == "rmse"]
                if not row.empty:
                    data["XGBoost"][raw_ps] = (float(row["mean"].iloc[0]), float(row["std"].iloc[0]))
                    continue
            outer_csv = os.path.join(nested_cv_dir, f"Outer_fold_results_{raw_ps}.csv")
            if os.path.exists(outer_csv):
                df = pd.read_csv(outer_csv)
                vals = df["rmse"].dropna()
                data["XGBoost"][raw_ps] = (vals.mean(), vals.std(ddof=1))

    return data


def plot_model_trend(data, set_order, set_labels, out_filename, output_dir):
    x_pos        = np.arange(len(set_order))
    x_labels     = [set_labels[ps] for ps in set_order]
    baseline_key = set_order[0]
    band_color   = "#888888"

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # shaded envelope across linear models
    linear_means = np.array([
        [data[m][ps][0] if ps in data[m] else np.nan for ps in set_order]
        for m in simple_model_names
    ])
    linear_stds = np.array([
        [data[m][ps][1] if ps in data[m] else np.nan for ps in set_order]
        for m in simple_model_names
    ])
    band_upper   = np.nanmax(linear_means + linear_stds, axis=0)
    band_lower   = np.nanmin(linear_means - linear_stds, axis=0)
    band_mean    = np.nanmean(linear_means, axis=0)
    baseline_band = band_mean[0]

    ax.fill_between(x_pos, band_lower, band_upper,
                    color=band_color, alpha=0.25, zorder=1,
                    label=r"Linear models (mean$\pm$std envelope)")

    # individual linear model lines (jittered, transparent)
    jitter = np.linspace(-0.06, 0.06, len(simple_model_names))
    for j, model_name in enumerate(simple_model_names):
        ps_data = data[model_name]
        rmses = [ps_data[ps][0] if ps in ps_data else np.nan for ps in set_order]
        ax.plot(x_pos + jitter[j], rmses,
                color=model_colors[model_name], marker=model_markers[model_name],
                linewidth=0.9, markersize=4, alpha=0.45, linestyle="--", zorder=2,
                label=model_display_name[model_name])

    # % annotation for band mean below lower edge
    for i, (rmse, raw_ps) in enumerate(zip(band_mean, set_order)):
        if np.isnan(rmse) or raw_ps == baseline_key:
            continue
        pct = (baseline_band - rmse) / baseline_band * 100
        ax.annotate(f"{pct:.1f}%",
                    xy=(x_pos[i], band_lower[i]),
                    xytext=(0, -10), textcoords="offset points",
                    ha="center", va="top", fontsize=6, color=band_color)

    # XGBoost with error bars
    xgb_data     = data["XGBoost"]
    xgb_rmses    = [xgb_data[ps][0] if ps in xgb_data else np.nan for ps in set_order]
    xgb_stds     = [xgb_data[ps][1] if ps in xgb_data else np.nan for ps in set_order]
    baseline_xgb = xgb_data[baseline_key][0] if baseline_key in xgb_data else np.nan

    ax.errorbar(x_pos, xgb_rmses, yerr=xgb_stds,
                color=model_colors["XGBoost"], marker=model_markers["XGBoost"],
                linewidth=1.8, markersize=5, capsize=3, capthick=0.8, elinewidth=0.8,
                label="XGBoost", zorder=4)

    for i, (rmse, raw_ps) in enumerate(zip(xgb_rmses, set_order)):
        if np.isnan(rmse) or np.isnan(baseline_xgb) or raw_ps == baseline_key:
            continue
        pct = (baseline_xgb - rmse) / baseline_xgb * 100
        ax.annotate(f"{pct:.1f}%",
                    xy=(x_pos[i], rmse), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color=model_colors["XGBoost"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("CV-RMSE", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, framealpha=0.9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_filename)
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=600)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    pressure_data = load_data(pressure_set_order)
    plot_model_trend(pressure_data, pressure_set_order, pressure_set_labels,
                     "Pressure_Tap_Trend.pdf", output_dir)

    position_data = load_data(position_set_order)
    plot_model_trend(position_data, position_set_order, position_set_labels,
                     "Position_Component_Trend.pdf", output_dir)
