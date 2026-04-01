import json
import os

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


RANDOM_SEED = 42
INNER_N_SPLITS = 5
N_ESTIMATORS_MAX = 2000
EARLY_STOPPING_ROUNDS = 50
INIT_POINTS = 20
N_ITER = 100
ROBUSTNESS_SEED_LIST = [0, 7, 21, 42, 84]


# Data Import and Preprocessing
# ------------------------------

# Data Alignment function
# ------------------------------
def check_alignment(df1, df2):
    key_cols = ["XPos", "YPos", "ZPos", "AoA"]
    if df1[key_cols].equals(df2[key_cols]):
        print("Datasets are correctly aligned!")
        return True
    else:
        print("Mismatch found! Differences:")
        print(df1[key_cols].compare(df2[key_cols]))
        return False


def create_config_group_id(dataframe, config_cols):
    missing_cols = [col for col in config_cols if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing configuration columns for grouping: {missing_cols}")

    group_keys = dataframe[config_cols].astype(str).agg("_".join, axis=1)
    dataframe = dataframe.copy()
    dataframe["config_group_id"] = pd.factorize(group_keys)[0]

    group_size_summary = dataframe.groupby("config_group_id").size()
    print(
        "Configuration groups created:",
        f"{group_size_summary.shape[0]} unique groups;",
        f"min repeats = {group_size_summary.min()},",
        f"max repeats = {group_size_summary.max()}"
    )
    return dataframe


# ------------------------------
# Read CSV files and prepare pooled data function
def data_prep():
    l_d_iter_0 = pd.read_csv("Post_processed_stats_summarized/L_D_True_Stats_Iter_0.csv")
    l_d_iter_1 = pd.read_csv("Post_processed_stats_summarized/L_D_True_Stats_Iter_1.csv")
    pressure_iter_0 = pd.read_csv("Post_processed_stats_summarized/Pressure_True_Stats_Iter_0.csv")
    pressure_iter_1 = pd.read_csv("Post_processed_stats_summarized/Pressure_True_Stats_Iter_1.csv")

    # Convert AoA to numeric; for pressure data, replace "_" with "."
    l_d_iter_0["AoA"] = pd.to_numeric(l_d_iter_0["AoA"], errors="coerce")
    l_d_iter_1["AoA"] = pd.to_numeric(l_d_iter_1["AoA"], errors="coerce")

    pressure_iter_0["AoA"] = pressure_iter_0["AoA"].astype(str).str.replace("_", ".").astype(float)
    pressure_iter_1["AoA"] = pressure_iter_1["AoA"].astype(str).str.replace("_", ".").astype(float)

    # Define list of pressure tap columns (upper taps)
    upper_taps = [
        "tap0_delta_pressure_mean",
        "tap2_delta_pressure_mean",
        "tap4_delta_pressure_mean",
        "tap6_delta_pressure_mean",
        "tap8_delta_pressure_mean",
        "tap9_delta_pressure_mean",
        "tap10_delta_pressure_mean",
        "tap11_delta_pressure_mean",
        "tap13_delta_pressure_mean",
        "tap15_delta_pressure_mean",
        "tap17_delta_pressure_mean",
        "tap19_delta_pressure_mean"
    ]

    # Select only necessary force columns
    force_cols = ["XPos", "YPos", "ZPos", "AoA", "delta_L_D_mean"]
    force_selected_iter_0 = l_d_iter_0[force_cols].copy()
    force_selected_iter_1 = l_d_iter_1[force_cols].copy()

    # Select only necessary pressure columns (keys + pressure taps)
    pressure_columns = ["XPos", "YPos", "ZPos", "AoA"] + upper_taps
    pressure_selected_iter_0 = pressure_iter_0[pressure_columns].copy()
    pressure_selected_iter_1 = pressure_iter_1[pressure_columns].copy()

    # Run alignment checks
    check_alignment(force_selected_iter_0, pressure_selected_iter_0)
    check_alignment(force_selected_iter_1, pressure_selected_iter_1)

    # Merge force and pressure data on key columns
    merge_keys = ["XPos", "YPos", "ZPos", "AoA"]
    data_model_iter_0 = pd.merge(force_selected_iter_0, pressure_selected_iter_0, on=merge_keys)
    data_model_iter_1 = pd.merge(force_selected_iter_1, pressure_selected_iter_1, on=merge_keys)

    data_model_iter_0["Iteration"] = 0
    data_model_iter_1["Iteration"] = 1

    data_model_full = pd.concat(
        [data_model_iter_0, data_model_iter_1],
        ignore_index=True
    )
    data_model_full = create_config_group_id(data_model_full, merge_keys)

    return data_model_full


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def sanitize_name(input_name):
    return "".join(char if char.isalnum() or char in ["_", "-"] else "_" for char in input_name)


def build_predictor_columns_from_combo(pressure_combo, position_combo):
    common_cols_to_drop = ["AoA", "delta_L_D_mean", "Iteration", "config_group_id"]

    if pressure_combo == 12:
        pressure_col_to_drop = []
    elif pressure_combo == 6:
        pressure_col_to_drop = [
            "tap2_delta_pressure_mean", "tap6_delta_pressure_mean",
            "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap17_delta_pressure_mean"
        ]
    elif pressure_combo == 2:
        pressure_col_to_drop = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap15_delta_pressure_mean", "tap17_delta_pressure_mean",
            "tap19_delta_pressure_mean"
        ]
    elif pressure_combo == 0:
        pressure_col_to_drop = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap15_delta_pressure_mean", "tap17_delta_pressure_mean",
            "tap19_delta_pressure_mean", "tap9_delta_pressure_mean", "tap10_delta_pressure_mean"
        ]
    else:
        raise ValueError(f"Unsupported pressure_combo: {pressure_combo}")

    if position_combo == "None":
        position_cols_to_drop = ["XPos", "YPos", "ZPos"]
    elif position_combo == "XPos":
        position_cols_to_drop = ["YPos", "ZPos"]
    elif position_combo == "YPos":
        position_cols_to_drop = ["XPos", "ZPos"]
    elif position_combo == "ZPos":
        position_cols_to_drop = ["XPos", "YPos"]
    elif position_combo == "XPos_YPos":
        position_cols_to_drop = ["ZPos"]
    elif position_combo == "XPos_ZPos":
        position_cols_to_drop = ["YPos"]
    elif position_combo == "YPos_ZPos":
        position_cols_to_drop = ["XPos"]
    elif position_combo == "XPos_YPos_ZPos":
        position_cols_to_drop = []
    else:
        raise ValueError(f"Unsupported position_combo: {position_combo}")

    all_predictor_cols = [
        "XPos", "YPos", "ZPos",
        "tap0_delta_pressure_mean", "tap2_delta_pressure_mean",
        "tap4_delta_pressure_mean", "tap6_delta_pressure_mean",
        "tap8_delta_pressure_mean", "tap9_delta_pressure_mean",
        "tap10_delta_pressure_mean", "tap11_delta_pressure_mean",
        "tap13_delta_pressure_mean", "tap15_delta_pressure_mean",
        "tap17_delta_pressure_mean", "tap19_delta_pressure_mean"
    ]
    cols_to_drop = set(position_cols_to_drop + pressure_col_to_drop + common_cols_to_drop)
    predictor_cols = [col for col in all_predictor_cols if col not in cols_to_drop]
    return predictor_cols


def build_predictor_sets_from_pressure_position_combos():
    pressure_combos = [0, 2, 6, 12]
    position_combos = ["None", "XPos", "YPos", "ZPos", "XPos_YPos_ZPos"]

    predictor_sets = {}
    for pressure_combo in pressure_combos:
        for position_combo in position_combos:
            if pressure_combo == 0 and position_combo == "None":
                continue

            predictor_set_name = f"Pressure_{pressure_combo}_Position_{position_combo}"
            predictor_sets[predictor_set_name] = build_predictor_columns_from_combo(
                pressure_combo,
                position_combo
            )

    return predictor_sets


def ensure_grouping_column(dataframe, config_cols):
    dataframe = dataframe.copy()
    if "config_group_id" not in dataframe.columns:
        if config_cols is None:
            raise ValueError("config_cols must be provided when config_group_id is not already in the dataframe.")
        dataframe = create_config_group_id(dataframe, config_cols)
    return dataframe


def split_iter0_iter1_data(
    dataframe,
    iteration_col="Iteration",
    iter0_value=0,
    iter1_value=1
):
    iter0_df = dataframe[dataframe[iteration_col] == iter0_value].copy()
    iter1_df = dataframe[dataframe[iteration_col] == iter1_value].copy()

    if iter0_df.empty:
        raise ValueError(f"No rows found for iter_0 value {iter0_value} in column {iteration_col}.")
    if iter1_df.empty:
        raise ValueError(f"No rows found for iter_1 value {iter1_value} in column {iteration_col}.")

    return iter0_df, iter1_df


def create_xgb_model(model_params, random_seed):
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(model_params["n_estimators"]),
        max_depth=int(model_params["max_depth"]),
        learning_rate=float(model_params["learning_rate"]),
        subsample=float(model_params["subsample"]),
        colsample_bytree=float(model_params["colsample_bytree"]),
        min_child_weight=float(model_params["min_child_weight"]),
        gamma=float(model_params["gamma"]),
        reg_lambda=float(model_params["reg_lambda"]),
        reg_alpha=float(model_params["reg_alpha"]),
        eval_metric="rmse",
        random_state=random_seed,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )


def evaluate_params_with_iter0_cv(X, y, groups, model_params, inner_n_splits=INNER_N_SPLITS):
    inner_cv = GroupKFold(n_splits=inner_n_splits)
    inner_fold_metrics = []
    best_iteration_list = []

    for inner_fold, (train_idx, valid_idx) in enumerate(inner_cv.split(X, y, groups), start=1):
        X_inner_train = X[train_idx]
        X_inner_valid = X[valid_idx]
        y_inner_train = y[train_idx]
        y_inner_valid = y[valid_idx]

        model = create_xgb_model(model_params, random_seed=RANDOM_SEED)
        model.fit(
            X_inner_train,
            y_inner_train,
            eval_set=[(X_inner_valid, y_inner_valid)],
            verbose=False
        )

        y_inner_pred = model.predict(X_inner_valid)
        fold_metric = regression_metrics(y_inner_valid, y_inner_pred)
        fold_metric["inner_fold"] = inner_fold
        inner_fold_metrics.append(fold_metric)

        if model.best_iteration is not None:
            best_iteration_list.append(model.best_iteration + 1)
        else:
            best_iteration_list.append(model_params["n_estimators"])

    inner_fold_metrics_df = pd.DataFrame(inner_fold_metrics)
    mean_best_n_estimators = int(np.round(np.mean(best_iteration_list)))
    return inner_fold_metrics_df, mean_best_n_estimators


def bayes_optimization_iter0_only(
    iter0_df,
    predictor_cols,
    target_col,
    group_col="config_group_id",
    init_points=INIT_POINTS,
    n_iter=N_ITER,
    inner_n_splits=INNER_N_SPLITS
):
    X_iter0 = iter0_df[predictor_cols].to_numpy()
    y_iter0 = iter0_df[target_col].to_numpy()
    groups_iter0 = iter0_df[group_col].to_numpy()

    pbounds = {
        "max_depth": (2, 6),
        "learning_rate": (0.01, 0.15),
        "subsample": (0.60, 1.00),
        "colsample_bytree": (0.60, 1.00),
        "min_child_weight": (1, 8),
        "gamma": (0, 3),
        "reg_lambda": (0, 8),
        "reg_alpha": (0, 4)
    }

    search_history = []

    def xgb_iter0_cv(
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        min_child_weight,
        gamma,
        reg_lambda,
        reg_alpha
    ):
        candidate_params = {
            "n_estimators": N_ESTIMATORS_MAX,
            "max_depth": int(round(max_depth)),
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha
        }

        inner_fold_metrics_df, mean_best_n_estimators = evaluate_params_with_iter0_cv(
            X_iter0,
            y_iter0,
            groups_iter0,
            candidate_params,
            inner_n_splits=inner_n_splits
        )
        mean_rmse = inner_fold_metrics_df["rmse"].mean()

        search_history.append({
            "max_depth": candidate_params["max_depth"],
            "learning_rate": candidate_params["learning_rate"],
            "subsample": candidate_params["subsample"],
            "colsample_bytree": candidate_params["colsample_bytree"],
            "min_child_weight": candidate_params["min_child_weight"],
            "gamma": candidate_params["gamma"],
            "reg_lambda": candidate_params["reg_lambda"],
            "reg_alpha": candidate_params["reg_alpha"],
            "mean_inner_rmse": mean_rmse,
            "mean_inner_mae": inner_fold_metrics_df["mae"].mean(),
            "mean_inner_r2": inner_fold_metrics_df["r2"].mean(),
            "mean_best_n_estimators": mean_best_n_estimators
        })

        return -mean_rmse

    optimizer = BayesianOptimization(
        f=xgb_iter0_cv,
        pbounds=pbounds,
        random_state=RANDOM_SEED,
        verbose=2
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params_raw = optimizer.max["params"]
    best_params = {
        "n_estimators": N_ESTIMATORS_MAX,
        "max_depth": int(round(best_params_raw["max_depth"])),
        "learning_rate": best_params_raw["learning_rate"],
        "subsample": best_params_raw["subsample"],
        "colsample_bytree": best_params_raw["colsample_bytree"],
        "min_child_weight": best_params_raw["min_child_weight"],
        "gamma": best_params_raw["gamma"],
        "reg_lambda": best_params_raw["reg_lambda"],
        "reg_alpha": best_params_raw["reg_alpha"]
    }

    best_inner_metrics_df, mean_best_n_estimators = evaluate_params_with_iter0_cv(
        X_iter0,
        y_iter0,
        groups_iter0,
        best_params,
        inner_n_splits=inner_n_splits
    )
    best_params["n_estimators"] = mean_best_n_estimators

    search_history_df = pd.DataFrame(search_history)
    return best_params, best_inner_metrics_df, search_history_df


def summarize_robustness_results(results_df):
    summary_rows = []
    for metric_name in ["rmse", "mae", "r2"]:
        summary_rows.append({
            "metric": metric_name,
            "mean": results_df[metric_name].mean(),
            "std": results_df[metric_name].std(ddof=1)
        })
    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def run_iter0_iter1_robustness_xgb(
    df,
    predictor_cols,
    target_col,
    iteration_col="Iteration",
    iter0_value=0,
    iter1_value=1,
    config_cols=None,
    save_name="default",
    output_dir="Predictive_model_data/Iter0_Iter1_Robustness_XGB",
    seed_list=None
):
    if seed_list is None:
        seed_list = ROBUSTNESS_SEED_LIST

    os.makedirs(output_dir, exist_ok=True)
    df = ensure_grouping_column(df, config_cols)

    missing_cols = [col for col in predictor_cols + [target_col, iteration_col, "config_group_id"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for robustness workflow: {missing_cols}")

    iter0_df, iter1_df = split_iter0_iter1_data(
        dataframe=df,
        iteration_col=iteration_col,
        iter0_value=iter0_value,
        iter1_value=iter1_value
    )

    # Hyperparameter tuning is performed using iter_0 only.
    best_params, best_inner_metrics_df, search_history_df = bayes_optimization_iter0_only(
        iter0_df=iter0_df,
        predictor_cols=predictor_cols,
        target_col=target_col,
        group_col="config_group_id"
    )

    X_iter0 = iter0_df[predictor_cols].to_numpy()
    y_iter0 = iter0_df[target_col].to_numpy()
    X_iter1 = iter1_df[predictor_cols].to_numpy()
    y_iter1 = iter1_df[target_col].to_numpy()

    robustness_rows = []

    for seed in seed_list:
        final_model = create_xgb_model(best_params, random_seed=seed)
        final_model.set_params(early_stopping_rounds=None)
        final_model.fit(X_iter0, y_iter0, verbose=False)

        y_pred = final_model.predict(X_iter1)
        fold_metrics = regression_metrics(y_iter1, y_pred)

        model_path = os.path.join(
            output_dir,
            f"XGB_iter0train_iter1test_seed_{seed}_{save_name}.json"
        )
        final_model.save_model(model_path)

        robustness_row = {
            "seed": seed,
            "n_train_rows": iter0_df.shape[0],
            "n_test_rows": iter1_df.shape[0],
            "rmse": fold_metrics["rmse"],
            "mae": fold_metrics["mae"],
            "r2": fold_metrics["r2"],
            "best_n_estimators": best_params["n_estimators"],
            "best_max_depth": best_params["max_depth"],
            "best_learning_rate": best_params["learning_rate"],
            "best_subsample": best_params["subsample"],
            "best_colsample_bytree": best_params["colsample_bytree"],
            "best_min_child_weight": best_params["min_child_weight"],
            "best_gamma": best_params["gamma"],
            "best_reg_lambda": best_params["reg_lambda"],
            "best_reg_alpha": best_params["reg_alpha"],
            "best_params_json": json.dumps(best_params),
            "saved_model_path": model_path
        }
        robustness_rows.append(robustness_row)

    robustness_results_df = pd.DataFrame(robustness_rows)
    robustness_results_df["predictor_set_name"] = save_name
    robustness_summary_df = summarize_robustness_results(robustness_results_df)
    robustness_summary_df["predictor_set_name"] = save_name
    tuned_hyperparameters_df = pd.DataFrame([best_params])
    tuned_hyperparameters_df["predictor_set_name"] = save_name

    robustness_results_df.to_csv(
        os.path.join(output_dir, f"Robustness_results_{save_name}.csv"),
        index=False
    )
    robustness_summary_df.to_csv(
        os.path.join(output_dir, f"Robustness_summary_{save_name}.csv"),
        index=False
    )
    tuned_hyperparameters_df.to_csv(
        os.path.join(output_dir, f"Tuned_hyperparameters_{save_name}.csv"),
        index=False
    )
    if not search_history_df.empty:
        search_history_df.to_csv(
            os.path.join(output_dir, f"Bayes_search_history_{save_name}.csv"),
            index=False
        )

    print("Best iter_0-only hyperparameters:", best_params)
    print("Best iter_0-only mean CV RMSE:", f"{best_inner_metrics_df['rmse'].mean():.5f}")
    print(robustness_results_df)
    print(robustness_summary_df)

    return {
        "best_params": best_params,
        "robustness_results_df": robustness_results_df,
        "robustness_summary_df": robustness_summary_df,
        "tuned_hyperparameters_df": tuned_hyperparameters_df,
        "search_history_df": search_history_df
    }


def run_all_predictor_set_robustness_xgb(
    df,
    predictor_sets,
    target_col,
    iteration_col="Iteration",
    iter0_value=0,
    iter1_value=1,
    config_cols=None,
    output_dir="Predictive_model_data/Iter0_Iter1_Robustness_XGB",
    seed_list=None
):
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    all_summaries = []
    all_tuned_params = []
    per_predictor_set_results = {}

    for save_name, predictor_cols in predictor_sets.items():
        print(f"Running robustness workflow for predictor set: {save_name}")

        safe_save_name = sanitize_name(save_name)
        predictor_set_output_dir = os.path.join(output_dir, safe_save_name)

        predictor_set_results = run_iter0_iter1_robustness_xgb(
            df=df,
            predictor_cols=predictor_cols,
            target_col=target_col,
            iteration_col=iteration_col,
            iter0_value=iter0_value,
            iter1_value=iter1_value,
            config_cols=config_cols,
            save_name=safe_save_name,
            output_dir=predictor_set_output_dir,
            seed_list=seed_list
        )

        all_results.append(predictor_set_results["robustness_results_df"])
        all_summaries.append(predictor_set_results["robustness_summary_df"])
        all_tuned_params.append(predictor_set_results["tuned_hyperparameters_df"])
        per_predictor_set_results[safe_save_name] = predictor_set_results

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_summary_df = pd.concat(all_summaries, ignore_index=True)
    all_tuned_params_df = pd.concat(all_tuned_params, ignore_index=True)

    combined_summary_df = (
        all_summary_df
        .pivot_table(
            index="predictor_set_name",
            columns="metric",
            values=["mean", "std"]
        )
        .reset_index()
    )
    combined_summary_df.columns = [
        "_".join([str(level) for level in column if str(level) != ""]).strip("_")
        if isinstance(column, tuple) else column
        for column in combined_summary_df.columns
    ]
    combined_summary_df = combined_summary_df.rename(
        columns={
            "mean_rmse": "rmse_mean",
            "std_rmse": "rmse_std",
            "mean_mae": "mae_mean",
            "std_mae": "mae_std",
            "mean_r2": "r2_mean",
            "std_r2": "r2_std"
        }
    )

    all_results_df.to_csv(
        os.path.join(output_dir, "All_predictor_set_robustness_results.csv"),
        index=False
    )
    all_summary_df.to_csv(
        os.path.join(output_dir, "All_predictor_set_robustness_summaries.csv"),
        index=False
    )
    all_tuned_params_df.to_csv(
        os.path.join(output_dir, "All_predictor_set_tuned_hyperparameters.csv"),
        index=False
    )
    combined_summary_df.to_csv(
        os.path.join(output_dir, "Combined_predictor_set_robustness_summary.csv"),
        index=False
    )

    return {
        "all_results_df": all_results_df,
        "all_summary_df": all_summary_df,
        "all_tuned_params_df": all_tuned_params_df,
        "combined_summary_df": combined_summary_df,
        "per_predictor_set_results": per_predictor_set_results
    }


if __name__ == "__main__":
    output_dir = "Predictive_model_data/Iter0_Iter1_Robustness_XGB"
    os.makedirs(output_dir, exist_ok=True)

    data_model_full = data_prep()
    print("Data preparation completed successfully!")

    predictor_sets = build_predictor_sets_from_pressure_position_combos()

    results = run_all_predictor_set_robustness_xgb(
        df=data_model_full,
        predictor_sets=predictor_sets,
        target_col="delta_L_D_mean",
        iteration_col="Iteration",
        iter0_value=0,
        iter1_value=1,
        config_cols=["XPos", "YPos", "ZPos", "AoA"],
        output_dir=output_dir,
        seed_list=ROBUSTNESS_SEED_LIST
    )

    print(results["combined_summary_df"])
