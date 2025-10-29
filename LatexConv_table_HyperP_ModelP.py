from __future__ import annotations
import numpy as np
import pandas as pd
import os
import re


def optimal_hyperparameter_identification(df: pd.DataFrame, target_column: str) -> dict:
    """
    Identify the hyperparameters corresponding to the highest value in the target column.

    Parameters:
    df (pd.DataFrame): DataFrame containing hyperparameter tuning results.
    target_column (str): The column name to find the maximum value.

    Returns:
    dict: Dictionary of hyperparameters for the optimal row.
    """
    max_idx = df[target_column].idxmax()
    optimal_row = df.loc[max_idx]
    hyperparameters = optimal_row.to_dict()
    return hyperparameters

def model_performance_data_extraction(df: pd.DataFrame) -> dict:
    """
    Extract model performance metrics from the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing model performance results.

    Returns:
    dict: Dictionary of performance metrics for the best model.
    """
    rmse_1 = df['RMSE_1'].min()
    rmse_2 = df['RMSE_2'].min()
    rmse_3 = df['RMSE_3'].min()
    rmse_4 = df['RMSE_4'].min()
    rmse_5 = df['RMSE_5'].min()
    ave_rmse = np.mean([rmse_1, rmse_2, rmse_3, rmse_4, rmse_5])
    return {"RMSE_1": rmse_1, "RMSE_2": rmse_2, "RMSE_3": rmse_3, "RMSE_4": rmse_4, "RMSE_5": rmse_5, "Ave_RMSE": ave_rmse}

def hyperparm_df_to_latex(df: pd.DataFrame,
                         outfile: str | None = None,
                         digits: int = 3,
                         na_str: str = "--",
                         order: list[str] | None = None) -> str:
    """
    Build LaTeX rows for an XGBoost hyperparam table.

    Expected columns (any prefix ok): 
      special_id,
      params.max_depth, params.eta, params.subsample, params.colsample_bytree,
      params.lambda_ (or lambda), params.alpha

    Args:
      df: pandas DataFrame
      outfile: optional path to write the latex rows (txt)
      digits: decimals for numeric formatting
      na_str: placeholder for missing values
      order: explicit ordering of special_id (if None, a sensible order is inferred)

    Returns:
      LaTeX string with header + rows + \\hline
    """
    def find_col(suffix_options):
        # find first column whose name ends with one of suffixes
        for suf in suffix_options:
            for c in df.columns:
                if c.endswith(suf):
                    return c
        return None

    # Locate columns flexibly
    col_sid   = "special_id" if "special_id" in df.columns else None
    col_max   = find_col(["max_depth"])
    col_eta   = find_col(["eta"])
    col_sub   = find_col(["subsample"])
    col_col   = find_col(["colsample_bytree"])
    col_lam   = find_col(["lambda_", "lambda"])
    col_alpha = find_col(["alpha"])

    needed = [col_sid, col_max, col_eta, col_sub, col_col, col_lam, col_alpha]
    if any(c is None for c in needed):
        missing = ["special_id","max_depth","eta","subsample","colsample_bytree","lambda_","alpha"]
        raise ValueError(f"Missing required columns. Found: {dict(zip(missing, needed))}")

    # Determine row order
    if order is None:
        # Heuristic: order by sensor count (2P/6P/12P), then by suffix complexity
        def key(s):
            # extract 2/6/12
            m = re.match(r"(\d+)P\b", s)
            base = int(m.group(1)) if m else 999
            rest = s.split("P",1)[1] if "P" in s else ""
            # rank feature set
            ranks = {
                "": 0, "+dx":1, "+dy":1, "+dz":1,
                "+dx,dy":2, "+dx,dz":2, "+dy,dz":2,
                "+dx,dy,dz":3, "+dxyz":3
            }
            r = ranks.get(rest, 99)
            return (base, r, rest)
        unique_ids = list(dict.fromkeys(df[col_sid].astype(str)))
        order = sorted(unique_ids, key=key)

    df2 = df.copy()
    df2["_sid"] = pd.Categorical(df2[col_sid].astype(str), categories=order, ordered=True)
    df2 = df2.sort_values("_sid")

    # Format helper
    def fmt(v):
        return na_str if pd.isna(v) else f"{float(v):.{digits}f}"

    # Header
    lines = [
        r"        \textbf{Model configuration} & \textbf{max.depth} & \textbf{$\eta$} & \textbf{subsample} & ",
        r"        \textbf{colsample.bytree} & $\boldsymbol{\lambda}$ & $\boldsymbol{\alpha}$\\",
        r"        \hline"
    ]

    # Rows
    for _, r in df2.iterrows():
        sid = str(r[col_sid]).replace("_", r"\_")  # escape underscores if any
        row = (f"        {sid} & {fmt(r[col_max])} & {fmt(r[col_eta])} & "
               f"{fmt(r[col_sub])} & {fmt(r[col_col])} & {fmt(r[col_lam])} & {fmt(r[col_alpha])} \\\\")
        lines.append(row)

    lines.append(r"        \hline")
    latex = "\n".join(lines)

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(latex)
    return latex

def model_performance_to_latex(df: pd.DataFrame,
                             outfile: str | None = None,
                             digits: int = 3,
                             na_str: str = "--",
                             order: list[str] | None = None,
                             average_col: str | None = None,
                             highlight_color: str = "gray!20",
                             tol: float = 1e-12) -> str:
    """
    Build LaTeX rows for the 'RMSE by seed' table and highlight the smallest RMSE per row.

    Expected columns:
      - 'special_id'
      - RMSE_* columns like RMSE_0..RMSE_4 (any mix of '_' or space before the index is ok)
      - Optional average column (Ave_RMSE / Avg_RMSE), otherwise mean is computed.

    Args:
      outfile: optional path to write the LaTeX block
      digits : decimals for numeric formatting
      na_str : placeholder for missing values
      order  : explicit ordering of special_id (if None, a default sensible order is used)
      average_col: name of the average column if not 'Ave_RMSE' or 'Avg_RMSE'
      highlight_color: LaTeX xcolor value for the min cell(s), e.g. "gray!20"
      tol: tolerance to treat ties as equal

    Returns:
      LaTeX string (header + rows + \\hline)
    """

    # --- find RMSE_* columns and sort by index ---
    pat = re.compile(r"(?i)^rmse[ _]?(\d+)$")
    rmse_cols = []
    for c in df.columns:
        m = pat.match(str(c))
        if m:
            rmse_cols.append((c, int(m.group(1))))
    if not rmse_cols:
        raise ValueError("No RMSE_* columns found (expected e.g. RMSE_0..RMSE_4).")
    rmse_cols = [c for c, _ in sorted(rmse_cols, key=lambda x: x[1])]
    n_seeds = len(rmse_cols)

    # average column
    avg_col = (average_col if average_col is not None else
               next((c for c in df.columns
                     if str(c).lower() in ("ave_rmse", "avg_rmse")), None))

    # default order if none provided
    if order is None:
        order = [
            "2P","6P","12P",
            "2P+dx","6P+dx","12P+dx",
            "2P+dy","6P+dy","12P+dy",
            "2P+dz","6P+dz","12P+dz",
            "2P+dx,dy","6P+dx,dy","12P+dx,dy",
            "2P+dx,dz","6P+dx,dz","12P+dx,dz",
            "2P+dxyz"
        ]

    ids_in_df = df["special_id"].astype(str).tolist()
    ordered_ids = [s for s in order if s in ids_in_df] + [s for s in ids_in_df if s not in order]

    df2 = df.copy()
    df2["_sid"] = pd.Categorical(df2["special_id"].astype(str), categories=ordered_ids, ordered=True)
    df2 = df2.sort_values("_sid")

    def fmt(v):
        return na_str if pd.isna(v) else f"{float(v):.{digits}f}"

    # header (# seeds may not be 5; adapt multicolumn and labels)
    seed_labels = list(range(n_seeds))
    header = [
        rf"        \textbf{{Model configuration}} & \multicolumn{{{n_seeds}}}{{c|}}{{\textbf{{RMSE}}}} & \textbf{{Average RMSE}} \\",
        "         " + " & " +
        " & ".join([f"\\textit{{Seed: {seed_labels[0]}}}"] +
                  [f"\\textit{{{s}}}" for s in seed_labels[1:]]) + " & \\\\",
        r"        \hline"
    ]
    lines = header

    # rows with highlighting of per-row min
    for _, r in df2.iterrows():
        sid = str(r["special_id"]).replace("_", r"\_")
        vals = pd.to_numeric(r[rmse_cols], errors="coerce").to_numpy(dtype=float)
        # Determine minima (ignore NaNs)
        if np.all(np.isnan(vals)):
            min_mask = np.zeros_like(vals, dtype=bool)
        else:
            m = np.nanmin(vals)
            min_mask = np.isclose(vals, m, atol=tol, rtol=0.0)

        # build seed cells with optional \cellcolor
        seed_cells = []
        for v, is_min in zip(vals, min_mask):
            cell = fmt(v)
            if (not pd.isna(v)) and is_min:
                cell = rf"\cellcolor{{{highlight_color}}}" + "{" + cell + "}"
            seed_cells.append(cell)

        # average: use provided column or computed mean
        if avg_col and avg_col in df2.columns:
            avg_val = fmt(r[avg_col])
        else:
            avg_val = fmt(np.nanmean(vals))

        lines.append(f"        {sid} & " + " & ".join(seed_cells) + f" & {avg_val} \\\\")

    lines.append(r"        \hline")
    latex = "\n".join(lines)

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(latex)
    return latex

if __name__ == "__main__":
    data_folder = "Predictive_model_data/Bayes_results_XGB_HyperP_using_dP_for_delta_L_D"
    # Identify optimal hyperparameters for each file and collect them
    optimal_rows = []
    for file in os.listdir(data_folder):
        if file.endswith(".csv") and file.startswith("Bayes_history_for_hyperparameter_tuning_"):
            df_temp = pd.read_csv(os.path.join(data_folder, file))
            optimal_row = optimal_hyperparameter_identification(df_temp, target_column="target")
            optimal_row["source_file"] = file
            special_id_map = {
                "Bayes_history_for_hyperparameter_tuning_Pressure_0_Position_XPos_YPos_ZPos.csv": "dx,dy,dz",
                "Bayes_history_for_hyperparameter_tuning_Pressure_2_Position_XPos_YPos_ZPos.csv": "2P+dx,dy,dz",
                "Bayes_history_for_hyperparameter_tuning_Pressure_6_Position_XPos_YPos_ZPos.csv": "6P+dx,dy,dz",
                "Bayes_history_for_hyperparameter_tuning_Pressure_12_Position_XPos_YPos_ZPos.csv": "12P+dx,dy,dz",
                "Bayes_history_for_hyperparameter_tuning_Pressure_0_Position_XPos.csv": "dx",
                "Bayes_history_for_hyperparameter_tuning_Pressure_2_Position_XPos.csv": "2P+dx",
                "Bayes_history_for_hyperparameter_tuning_Pressure_6_Position_XPos.csv": "6P+dx",
                "Bayes_history_for_hyperparameter_tuning_Pressure_12_Position_XPos.csv": "12P+dx",
                "Bayes_history_for_hyperparameter_tuning_Pressure_0_Position_YPos.csv": "dy",
                "Bayes_history_for_hyperparameter_tuning_Pressure_2_Position_YPos.csv": "2P+dy",
                "Bayes_history_for_hyperparameter_tuning_Pressure_6_Position_YPos.csv": "6P+dy",
                "Bayes_history_for_hyperparameter_tuning_Pressure_12_Position_YPos.csv": "12P+dy",
                "Bayes_history_for_hyperparameter_tuning_Pressure_0_Position_ZPos.csv": "dz",
                "Bayes_history_for_hyperparameter_tuning_Pressure_2_Position_ZPos.csv": "2P+dz",
                "Bayes_history_for_hyperparameter_tuning_Pressure_6_Position_ZPos.csv": "6P+dz",
                "Bayes_history_for_hyperparameter_tuning_Pressure_12_Position_ZPos.csv": "12P+dz",
                "Bayes_history_for_hyperparameter_tuning_Pressure_2_Position_None.csv": "2P",
                "Bayes_history_for_hyperparameter_tuning_Pressure_6_Position_None.csv": "6P",
                "Bayes_history_for_hyperparameter_tuning_Pressure_12_Position_None.csv": "12P"
            }
            optimal_row["special_id"] = special_id_map.get(optimal_row["source_file"], "")
            optimal_rows.append(optimal_row)
    df = pd.DataFrame(optimal_rows)
    hyperparm_latex = hyperparm_df_to_latex(df)
    print(hyperparm_latex)
    print(df)

    # Extract model performance metrics from evaluation files
    model_performance_data_folder = "Predictive_model_data"
    model_evaluation_folder = model_performance_data_folder + "/XGBoost_Model_evaluation"
    result_df = []
    for file in os.listdir(model_evaluation_folder):
        if file.endswith(".csv") and file.startswith("Predicted_dataset_"):
            df_temp = pd.read_csv(os.path.join(model_evaluation_folder, file))
            performance_metrics = model_performance_data_extraction(df_temp)
            performance_metrics["source_file"] = file
            special_id_map = {
                "Predicted_dataset_Pressure_0_Position_XPos_YPos_ZPos.csv": "dx,dy,dz",
                "Predicted_dataset_Pressure_2_Position_XPos_YPos_ZPos.csv": "2P+dx,dy,dz",
                "Predicted_dataset_Pressure_6_Position_XPos_YPos_ZPos.csv": "6P+dx,dy,dz",
                "Predicted_dataset_Pressure_12_Position_XPos_YPos_ZPos.csv": "12P+dx,dy,dz",
                "Predicted_dataset_Pressure_0_Position_XPos.csv": "dx",
                "Predicted_dataset_Pressure_2_Position_XPos.csv": "2P+dx",
                "Predicted_dataset_Pressure_6_Position_XPos.csv": "6P+dx",
                "Predicted_dataset_Pressure_12_Position_XPos.csv": "12P+dx",
                "Predicted_dataset_Pressure_0_Position_YPos.csv": "dy",
                "Predicted_dataset_Pressure_2_Position_YPos.csv": "2P+dy",
                "Predicted_dataset_Pressure_6_Position_YPos.csv": "6P+dy",
                "Predicted_dataset_Pressure_12_Position_YPos.csv": "12P+dy",
                "Predicted_dataset_Pressure_0_Position_ZPos.csv": "dz",
                "Predicted_dataset_Pressure_2_Position_ZPos.csv": "2P+dz",
                "Predicted_dataset_Pressure_6_Position_ZPos.csv": "6P+dz",
                "Predicted_dataset_Pressure_12_Position_ZPos.csv": "12P+dz",
                "Predicted_dataset_Pressure_2_Position_None.csv": "2P",
                "Predicted_dataset_Pressure_6_Position_None.csv": "6P",
                "Predicted_dataset_Pressure_12_Position_None.csv": "12P"
            }
            performance_metrics["special_id"] = special_id_map.get(file, "")
            result_df.append(performance_metrics)
    result_df = pd.DataFrame(result_df)
    print(result_df)
    latex_output = model_performance_to_latex(result_df)
    print(latex_output)
