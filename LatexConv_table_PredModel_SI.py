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
    "Pressure_2_Position_None":           "2P",
    "Pressure_6_Position_None":           "6P",
    "Pressure_12_Position_None":          "12P",
    "Pressure_0_Position_XPos":           "dx",
    "Pressure_0_Position_YPos":           "dy",
    "Pressure_0_Position_ZPos":           "dz",
    "Pressure_0_Position_XPos_YPos_ZPos": "dx,dy,dz",
    "Pressure_2_Position_XPos":           "2P+dx",
    "Pressure_2_Position_YPos":           "2P+dy",
    "Pressure_2_Position_ZPos":           "2P+dz",
    "Pressure_2_Position_XPos_YPos_ZPos": "2P+dx,dy,dz",
    "Pressure_6_Position_XPos":           "6P+dx",
    "Pressure_6_Position_YPos":           "6P+dy",
    "Pressure_6_Position_ZPos":           "6P+dz",
    "Pressure_6_Position_XPos_YPos_ZPos": "6P+dx,dy,dz",
    "Pressure_12_Position_XPos":          "12P+dx",
    "Pressure_12_Position_YPos":          "12P+dy",
    "Pressure_12_Position_ZPos":          "12P+dz",
    "Pressure_12_Position_XPos_YPos_ZPos":"12P+dx,dy,dz",
}

predictor_set_order = [
    "2P", "6P", "12P",
    "dx", "dy", "dz", "dx,dy,dz",
    "2P+dx", "2P+dy", "2P+dz", "2P+dx,dy,dz",
    "6P+dx", "6P+dy", "6P+dz", "6P+dx,dy,dz",
    "12P+dx", "12P+dy", "12P+dz", "12P+dx,dy,dz",
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


def load_simple_model_all():
    dfs = []
    if not os.path.exists(simple_model_dir):
        return None
    for fname in os.listdir(simple_model_dir):
        if not (fname.startswith("Outer_fold_results_") and fname.endswith(".csv")):
            continue
        stem = fname[len("Outer_fold_results_"):-4]
        for model_name in simple_model_names:
            safe = sanitize_name(model_name)
            if stem.startswith(safe + "_"):
                predictor_set = stem[len(safe) + 1:]
                df = pd.read_csv(os.path.join(simple_model_dir, fname))
                df["model_name"] = model_name
                df["predictor_set_name"] = predictor_set
                dfs.append(df)
                break
    return pd.concat(dfs, ignore_index=True) if dfs else None


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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    cv_df           = load_xgb_outer_fold_rmse()
    robustness_rows = build_robustness_rows()
    latency_rows    = build_latency_rows()

    cv_rows, n_folds = build_xgb_cv_rows(cv_df)
    write_xgb_cv_latex(cv_rows, n_folds)
    write_robustness_latex(robustness_rows)
    write_latency_latex(latency_rows)
