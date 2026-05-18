import os
import numpy as np
import pandas as pd

output_dir = "Predictive_model_data/Table"
nested_cv_dir = "Predictive_model_data/Grouped_Nested_CV_XGB"
robustness_dir = "Predictive_model_data/Iter0_Iter1_Robustness_XGB"
latency_dir = "Predictive_model_data/Latency_Benchmark"
simple_model_dir = "Predictive_model_data/Grouped_CV_Simple_Models"

simple_model_names = ["LinearRegression", "RidgeRegression", "LassoRegression", "PLSRegression"]

predictor_set_label = {
    "Pressure_2_Position_None":           r"2$\Delta p$",
    "Pressure_6_Position_None":           r"6$\Delta p$",
    "Pressure_12_Position_None":          r"12$\Delta p$",
    "Pressure_0_Position_XPos":           "dx",
    "Pressure_0_Position_YPos":           "dy",
    "Pressure_0_Position_ZPos":           "dz",
    "Pressure_0_Position_XPos_YPos_ZPos": "dx,dy,dz",
    "Pressure_2_Position_XPos":           r"2$\Delta p$+dx",
    "Pressure_2_Position_YPos":           r"2$\Delta p$+dy",
    "Pressure_2_Position_ZPos":           r"2$\Delta p$+dz",
    "Pressure_2_Position_XPos_YPos_ZPos": r"2$\Delta p$+dx,dy,dz",
    "Pressure_6_Position_XPos":           r"6$\Delta p$+dx",
    "Pressure_6_Position_YPos":           r"6$\Delta p$+dy",
    "Pressure_6_Position_ZPos":           r"6$\Delta p$+dz",
    "Pressure_6_Position_XPos_YPos_ZPos": r"6$\Delta p$+dx,dy,dz",
    "Pressure_12_Position_XPos":          r"12$\Delta p$+dx",
    "Pressure_12_Position_YPos":          r"12$\Delta p$+dy",
    "Pressure_12_Position_ZPos":          r"12$\Delta p$+dz",
    "Pressure_12_Position_XPos_YPos_ZPos":r"12$\Delta p$+dx,dy,dz",
}

predictor_set_order = [
    r"2$\Delta p$", r"6$\Delta p$", r"12$\Delta p$",
    "dx", "dy", "dz", "dx,dy,dz",
    r"2$\Delta p$+dx", r"2$\Delta p$+dy", r"2$\Delta p$+dz", r"2$\Delta p$+dx,dy,dz",
    r"6$\Delta p$+dx", r"6$\Delta p$+dy", r"6$\Delta p$+dz", r"6$\Delta p$+dx,dy,dz",
    r"12$\Delta p$+dx", r"12$\Delta p$+dy", r"12$\Delta p$+dz", r"12$\Delta p$+dx,dy,dz",
]

model_display_name = {
    "LinearRegression":  "Linear Reg.",
    "RidgeRegression":   "Ridge Reg.",
    "LassoRegression":   "Lasso Reg.",
    "PLSRegression":     "PLS Reg.",
    "XGBoost":           "XGBoost",
}


def sanitize_name(name):
    return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in name)


def fmt(v, d=4):
    return f"{v:.{d}f}" if not (v is None or (isinstance(v, float) and np.isnan(v))) else "--"


def write_tex(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_xgb_outer_fold_rmse():
    """Load per-fold RMSE from Outer_fold_results_*.csv for each predictor set."""
    rows = []
    if not os.path.exists(nested_cv_dir):
        return None
    label_to_raw = {v: k for k, v in predictor_set_label.items()}
    for label in predictor_set_order:
        raw_name = label_to_raw.get(label)
        if raw_name is None:
            continue
        csv = os.path.join(nested_cv_dir, f"Outer_fold_results_{raw_name}.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv).sort_values("outer_fold")
        fold_rmses = df["rmse"].tolist()
        row = {"label": label}
        for i, v in enumerate(fold_rmses):
            row[f"fold_{i+1}"] = v
        row["avg_rmse"] = np.mean(fold_rmses)
        rows.append(row)
    return pd.DataFrame(rows) if rows else None

# ---------------------------------------------------------------------------
# Table S2: XGBoost grouped nested CV performance (per predictor set)
# ---------------------------------------------------------------------------

def build_xgb_cv_rows(cv_df):
    if cv_df is None:
        return [], 0
    fold_cols = [c for c in cv_df.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)
    rows = []
    for _, r in cv_df.iterrows():
        vals = [r[c] for c in fold_cols]
        min_val = min(v for v in vals if not np.isnan(v))
        cells = []
        for v in vals:
            cell = fmt(v, 3)
            if not np.isnan(v) and np.isclose(v, min_val, atol=1e-12):
                cell = r"\cellcolor{gray!20}{" + cell + "}"
            cells.append(cell)
        avg = fmt(r["avg_rmse"], 3)
        std = fmt(np.std([v for v in vals if not np.isnan(v)], ddof=1), 3)
        rows.append([r["label"]] + cells + [avg, std])
    return rows, n_folds


def write_xgb_cv_latex(rows, n_folds):
    fold_labels = " & ".join(
        [r"\textit{Fold: 1}"] + [rf"\textit{{{i}}}" for i in range(2, n_folds + 1)]
    )
    col_spec = r"l|" + "c" * n_folds + r"|cc"
    header1 = (
        rf"        \textbf{{Predictor set}} & \multicolumn{{{n_folds}}}{{c|}}{{\textbf{{CV-RMSE (outer folds)}}}} & \textbf{{Avg.}} & \textbf{{Std}} \\"
    )
    header2 = f"         & {fold_labels} & & \\\\"

    def header_block():
        return [header1, header2, r"        \hline"]

    lines = [
        r"        \begin{table}[htbp]",
        r"        \centering",
        r"        \captionsetup{font=footnotesize}",
        r"        \caption{XGBoost grouped nested cross-validation performance across all"
        r" predictor set configurations. Each column shows the RMSE on the held-out outer"
        r" fold; the minimum per row is shaded.}",
        r"        \footnotesize",
        r"        \setlength{\tabcolsep}{5pt}",
        r"        \renewcommand{\arraystretch}{1.2}",
        r"        \begin{tabular}{" + col_spec + "}",
        r"        \hline\hline",
    ] + header_block()

    for row in rows:
        lines.append("        " + " & ".join(row) + r" \\")
    lines += [
        r"        \hline",
        r"        \end{tabular}",
        r"        \label{tab:xgb_cv_performance}",
        r"        \end{table}",
    ]
    write_tex(lines, os.path.join(output_dir, "XGBoost_CV_Performance_Table.tex"))


# ---------------------------------------------------------------------------
# Table S3: XGBoost robustness (per predictor set, mean ± std across 5 seeds)
# ---------------------------------------------------------------------------

def load_cv_rmse_mean(raw_name):
    """Return CV-RMSE mean from Summary_metrics CSV, or NaN if unavailable."""
    summary_csv = os.path.join(nested_cv_dir, f"Summary_metrics_{raw_name}.csv")
    if not os.path.exists(summary_csv):
        return np.nan
    df = pd.read_csv(summary_csv)
    row = df[df["metric"] == "rmse"]
    if row.empty:
        return np.nan
    return float(row["mean"].iloc[0])


def build_robustness_rows():
    rows = []
    if not os.path.exists(robustness_dir):
        return rows

    label_to_dir = {v: k for k, v in predictor_set_label.items()}
    for label in predictor_set_order:
        raw_name = label_to_dir.get(label)
        if raw_name is None:
            continue
        safe = sanitize_name(raw_name)
        csv = os.path.join(robustness_dir, safe, f"Robustness_results_{safe}.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv).sort_values("seed")
        seed_vals = df["rmse"].tolist()
        seeds = df["seed"].tolist()
        min_val = min(seed_vals)
        cells = []
        for v in seed_vals:
            cell = fmt(v, 3)
            if np.isclose(v, min_val, atol=1e-12):
                cell = r"\cellcolor{gray!20}{" + cell + "}"
            cells.append(cell)
        avg_val = np.mean(seed_vals)
        avg = fmt(avg_val, 3)
        std = fmt(np.std(seed_vals, ddof=1), 3)
        cv_rmse = load_cv_rmse_mean(raw_name)
        cv_rmse_str = fmt(cv_rmse, 3)
        delta = abs(avg_val - cv_rmse) if not np.isnan(cv_rmse) else np.nan
        delta_str = fmt(delta, 3)
        rows.append({
            "label": label, "cells": cells, "avg": avg, "std": std, "seeds": seeds,
            "cv_rmse": cv_rmse_str, "delta": delta_str,
        })
    return rows


def write_robustness_latex(rows):
    if not rows:
        write_tex([], os.path.join(output_dir, "Robustness_Table.tex"))
        return
    seeds = rows[0]["seeds"]
    n_seeds = len(seeds)
    seed_labels = " & ".join(
        [rf"\textit{{Seed: {seeds[0]}}}"] + [rf"\textit{{{s}}}" for s in seeds[1:]]
    )
    col_spec = r"l|" + "c" * n_seeds + r"|cc|cc"
    header1 = (
        rf"        \textbf{{Predictor set}} & \multicolumn{{{n_seeds}}}{{c|}}{{\textbf{{RMSE (seeds)}}}} &"
        r" \textbf{Avg.} & \textbf{Std} & \textbf{CV-RMSE} & \textbf{$|\Delta|$} \\"
    )
    header2 = f"         & {seed_labels} & & & & \\\\"

    def header_block():
        return [header1, header2, r"        \hline"]

    lines = [
        r"        \begin{table}[htbp]",
        r"        \centering",
        r"        \captionsetup{font=footnotesize}",
        r"        \caption{XGBoost robustness evaluation: models trained on Iter.~0 and tested"
        r" on Iter.~1 using five independent random seeds. Small \textit{Std} across seeds"
        r" indicates stable, reproducible predictions regardless of random initialization."
        r" \textit{CV-RMSE} is the mean outer-fold RMSE from grouped nested CV on the combined"
        r" dataset; $|\Delta| = |\text{Avg.} - \text{CV-RMSE}|$ measures the cross-iteration"
        r" generalization gap --- small $|\Delta|$ confirms the model generalizes to an unseen"
        r" iteration as well as it does within cross-validation. The minimum RMSE per row is shaded.}",
        r"        \footnotesize",
        r"        \setlength{\tabcolsep}{5pt}",
        r"        \renewcommand{\arraystretch}{1.2}",
        r"        \begin{tabular}{" + col_spec + "}",
        r"        \hline\hline",
    ] + header_block()

    for r in rows:
        lines.append(
            "        " + " & ".join(
                [r["label"]] + r["cells"] + [r["avg"], r["std"], r["cv_rmse"], r["delta"]]
            ) + r" \\"
        )
    lines += [
        r"        \hline",
        r"        \end{tabular}",
        r"        \label{tab:robustness}",
        r"        \end{table}",
    ]
    write_tex(lines, os.path.join(output_dir, "Robustness_Table.tex"))


# ---------------------------------------------------------------------------
# Table S4: Latency (per model type, aggregated across predictor sets)
# ---------------------------------------------------------------------------

def build_latency_rows():
    csv = os.path.join(latency_dir, "Latency_difference_by_model.csv")
    if not os.path.exists(csv):
        return []
    df = pd.read_csv(csv).sort_values("mean_latency_ms")
    rows = []
    for _, r in df.iterrows():
        diff = r.get("difference_vs_xgboost_ms", np.nan)
        diff_str = fmt(diff) if r["model_name"] != "XGBoost" else "---"
        rows.append([
            model_display_name.get(r["model_name"], r["model_name"]),
            fmt(r["mean_latency_ms"]),
            fmt(r["std_latency_ms"]),
            diff_str,
        ])
    return rows


def write_latency_latex(rows):
    col_spec = r"l|ccc"
    header1 = (
        r"        \textbf{Model} & \textbf{Mean (ms)} & \textbf{Std (ms)} &"
        r" \textbf{$\Delta$ vs XGBoost (ms)} \\"
    )

    def header_block():
        return [header1, r"        \hline"]

    lines = [
        r"        \begin{table}[htbp]",
        r"        \centering",
        r"        \captionsetup{font=footnotesize}",
        r"        \caption{Single-sample inference latency aggregated across all predictor"
        r" set configurations, benchmarked on an Intel Core i9-10850K CPU."
        r" $\Delta$ vs XGBoost shows the mean latency difference relative to XGBoost"
        r" (negative = faster than XGBoost).}",
        r"        \footnotesize",
        r"        \setlength{\tabcolsep}{5pt}",
        r"        \renewcommand{\arraystretch}{1.2}",
        r"        \begin{tabular}{" + col_spec + "}",
        r"        \hline\hline",
    ] + header_block()

    for row in rows:
        lines.append("        " + " & ".join(row) + r" \\")
    lines += [
        r"        \hline",
        r"        \end{tabular}",
        r"        \label{tab:latency}",
        r"        \end{table}",
    ]
    write_tex(lines, os.path.join(output_dir, "Latency_Table.tex"))


# ---------------------------------------------------------------------------
# Table S5: Model comparison across predictor sets (pressure trend across models)
# ---------------------------------------------------------------------------

comparison_set_order = [
    "Pressure_0_Position_XPos_YPos_ZPos",
    "Pressure_2_Position_XPos_YPos_ZPos",
    "Pressure_6_Position_XPos_YPos_ZPos",
    "Pressure_12_Position_XPos_YPos_ZPos",
]

comparison_set_labels = {
    "Pressure_0_Position_XPos_YPos_ZPos":  "dx,dy,dz only",
    "Pressure_2_Position_XPos_YPos_ZPos":  r"2$\Delta p$+dx,dy,dz",
    "Pressure_6_Position_XPos_YPos_ZPos":  r"6$\Delta p$+dx,dy,dz",
    "Pressure_12_Position_XPos_YPos_ZPos": r"12$\Delta p$+dx,dy,dz",
}

model_order_comparison = [
    "LinearRegression", "RidgeRegression", "LassoRegression", "PLSRegression", "XGBoost"
]


def load_comparison_data():
    """Return {model_name: {raw_predictor_set: (mean_rmse, std_rmse)}}."""
    data = {m: {} for m in model_order_comparison}

    # Simple models
    if os.path.exists(simple_model_dir):
        for fname in os.listdir(simple_model_dir):
            if not (fname.startswith("Outer_fold_results_") and fname.endswith(".csv")):
                continue
            stem = fname[len("Outer_fold_results_"):-4]
            for model_name in simple_model_names:
                safe = sanitize_name(model_name)
                if stem.startswith(safe + "_"):
                    raw_ps = stem[len(safe) + 1:]
                    if raw_ps in comparison_set_order:
                        df = pd.read_csv(os.path.join(simple_model_dir, fname))
                        vals = df["rmse"].dropna().tolist()
                        data[model_name][raw_ps] = (np.mean(vals), np.std(vals, ddof=1))
                    break

    # XGBoost: prefer Summary_metrics, fall back to Outer_fold_results
    if os.path.exists(nested_cv_dir):
        for raw_ps in comparison_set_order:
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
                vals = df["rmse"].dropna().tolist()
                data["XGBoost"][raw_ps] = (np.mean(vals), np.std(vals, ddof=1))

    return data


def fmt_mean_std(mean, std, d=3):
    return rf"{fmt(mean, d)} $\pm$ {fmt(std, d)}"


def build_model_comparison_rows(data):
    rows = []
    for model_name in model_order_comparison:
        ps_data = data[model_name]
        means, cells = [], []
        for raw_ps in comparison_set_order:
            if raw_ps in ps_data:
                mean, std = ps_data[raw_ps]
                means.append(mean)
                cells.append(fmt_mean_std(mean, std))
            else:
                means.append(np.nan)
                cells.append("--")

        valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if valid:
            min_idx = min(valid, key=lambda x: x[1])[0]
            cells[min_idx] = r"\textbf{" + cells[min_idx] + r"}"

        rows.append([model_display_name.get(model_name, model_name)] + cells)
    return rows


def write_model_comparison_latex(rows):
    n_sets = len(comparison_set_order)
    col_labels = " & ".join(
        rf"\textit{{{comparison_set_labels[ps]}}}" for ps in comparison_set_order
    )
    col_spec = "l|" + "c" * n_sets
    header1 = (
        rf"        \textbf{{Model}} & \multicolumn{{{n_sets}}}{{c}}{{\textbf{{CV-RMSE (mean $\pm$ std)}}}} \\"
    )
    header2 = f"         & {col_labels} \\\\"

    def header_block():
        return [header1, header2, r"        \hline"]

    lines = [
        r"        \begin{table}[htbp]",
        r"        \centering",
        r"        \captionsetup{font=footnotesize}",
        r"        \caption{Cross-validation RMSE comparison across model classes and predictor"
        r" sets (mean~$\pm$~std across outer folds). The lowest RMSE per row is \textbf{bolded}."
        r" XGBoost uses grouped nested CV; linear models use grouped CV. The table examines"
        r" whether pressure-informed predictor sets consistently outperform the position-only"
        r" (dx,dy,dz) baseline across model classes.}",
        r"        \footnotesize",
        r"        \setlength{\tabcolsep}{5pt}",
        r"        \renewcommand{\arraystretch}{1.2}",
        r"        \begin{tabular}{" + col_spec + "}",
        r"        \hline\hline",
    ] + header_block()

    for row in rows:
        lines.append("        " + " & ".join(row) + r" \\")
    lines += [
        r"        \hline",
        r"        \end{tabular}",
        r"        \label{tab:model_comparison}",
        r"        \end{table}",
    ]
    write_tex(lines, os.path.join(output_dir, "Model_Comparison_Table.tex"))





# ---------------------------------------------------------------------------
# Table S6: Position-component sensitivity within 12Δp (which axis matters?)
# ---------------------------------------------------------------------------

position_sens_set_order = [
    "Pressure_12_Position_None",
    "Pressure_12_Position_XPos",
    "Pressure_12_Position_YPos",
    "Pressure_12_Position_ZPos",
    "Pressure_12_Position_XPos_YPos_ZPos",
]

position_sens_set_labels = {
    "Pressure_12_Position_None":           r"12$\Delta p$",
    "Pressure_12_Position_XPos":           r"12$\Delta p$+dx",
    "Pressure_12_Position_YPos":           r"12$\Delta p$+dy",
    "Pressure_12_Position_ZPos":           r"12$\Delta p$+dz",
    "Pressure_12_Position_XPos_YPos_ZPos": r"12$\Delta p$+dx,dy,dz",
}


def load_position_sensitivity_data():
    """Return {model_name: {raw_predictor_set: (mean_rmse, std_rmse)}} for 12Δp variants."""
    data = {m: {} for m in model_order_comparison}

    # Simple models
    if os.path.exists(simple_model_dir):
        for fname in os.listdir(simple_model_dir):
            if not (fname.startswith("Outer_fold_results_") and fname.endswith(".csv")):
                continue
            stem = fname[len("Outer_fold_results_"):-4]
            for model_name in simple_model_names:
                safe = sanitize_name(model_name)
                if stem.startswith(safe + "_"):
                    raw_ps = stem[len(safe) + 1:]
                    if raw_ps in position_sens_set_order:
                        df = pd.read_csv(os.path.join(simple_model_dir, fname))
                        vals = df["rmse"].dropna().tolist()
                        data[model_name][raw_ps] = (np.mean(vals), np.std(vals, ddof=1))
                    break

    # XGBoost
    if os.path.exists(nested_cv_dir):
        for raw_ps in position_sens_set_order:
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
                vals = df["rmse"].dropna().tolist()
                data["XGBoost"][raw_ps] = (np.mean(vals), np.std(vals, ddof=1))

    return data


def build_position_sensitivity_rows(data):
    rows = []
    for model_name in model_order_comparison:
        ps_data = data[model_name]
        means, cells = [], []
        for raw_ps in position_sens_set_order:
            if raw_ps in ps_data:
                mean, std = ps_data[raw_ps]
                means.append(mean)
                cells.append(fmt_mean_std(mean, std))
            else:
                means.append(np.nan)
                cells.append("--")

        valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if valid:
            min_idx = min(valid, key=lambda x: x[1])[0]
            cells[min_idx] = r"\textbf{" + cells[min_idx] + r"}"

        rows.append([model_display_name.get(model_name, model_name)] + cells)
    return rows


def write_position_sensitivity_latex(rows):
    n_sets = len(position_sens_set_order)
    col_labels = " & ".join(
        rf"\textit{{{position_sens_set_labels[ps]}}}" for ps in position_sens_set_order
    )
    col_spec = "l|" + "c" * n_sets
    header1 = (
        rf"        \textbf{{Model}} & \multicolumn{{{n_sets}}}{{c}}{{\textbf{{CV-RMSE (mean $\pm$ std)}}}} \\"
    )
    header2 = f"         & {col_labels} \\\\"

    def header_block():
        return [header1, header2, r"        \hline"]

    lines = [
        r"        \begin{table}[htbp]",
        r"        \centering",
        r"        \captionsetup{font=footnotesize}",
        r"        \caption{CV-RMSE across model classes for the 12$\Delta p$ predictor set"
        r" augmented with individual position components. The lowest RMSE per row is"
        r" \textbf{bolded}. Comparing columns isolates the contribution of each position"
        r" axis (dx, dy, dz) beyond the pressure-only baseline, and assesses whether"
        r" dx alone captures most of the positional information.}",
        r"        \footnotesize",
        r"        \setlength{\tabcolsep}{5pt}",
        r"        \renewcommand{\arraystretch}{1.2}",
        r"        \begin{tabular}{" + col_spec + "}",
        r"        \hline\hline",
    ] + header_block()

    for row in rows:
        lines.append("        " + " & ".join(row) + r" \\")
    lines += [
        r"        \hline",
        r"        \end{tabular}",
        r"        \label{tab:position_sensitivity}",
        r"        \end{table}",
    ]
    write_tex(lines, os.path.join(output_dir, "Position_Sensitivity_Table.tex"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    cv_df              = load_xgb_outer_fold_rmse()
    robustness_rows    = build_robustness_rows()
    latency_rows       = build_latency_rows()
    comparison_data    = load_comparison_data()
    pos_sens_data      = load_position_sensitivity_data()

    cv_rows, n_folds = build_xgb_cv_rows(cv_df)
    write_xgb_cv_latex(cv_rows, n_folds)
    write_robustness_latex(robustness_rows)
    write_latency_latex(latency_rows)

    comparison_rows = build_model_comparison_rows(comparison_data)
    write_model_comparison_latex(comparison_rows)

    pos_sens_rows = build_position_sensitivity_rows(pos_sens_data)
    write_position_sensitivity_latex(pos_sens_rows)
