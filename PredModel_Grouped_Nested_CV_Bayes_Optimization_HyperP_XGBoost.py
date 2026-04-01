import json
import os

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# Global Grouped Nested CV and Bayesian Optimization Settings
RANDOM_SEED = 42
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 5
N_ESTIMATORS_MAX = 2000
EARLY_STOPPING_ROUNDS = 50
INIT_POINTS = 20
N_ITER = 100


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

    # Pool both experiment iterations. Grouped CV will keep repeated
    # measurements from the same aerodynamic configuration together.
    data_model_full = pd.concat(
        [data_model_iter_0, data_model_iter_1],
        ignore_index=True
    )
    data_model_full = create_config_group_id(data_model_full, merge_keys)

    return data_model_full


# Prepare Data for XGBoost (Full Dataset)
# ------------------------------
# For full dataset, use all selected predictor columns and keep grouping columns.
def xgbdata_prep(data_model_full, pressure_combo, position_combo, target_col="delta_L_D_mean"):
    common_cols_to_drop = ["AoA", target_col, "Iteration", "config_group_id"]

    if pressure_combo == 12:
        if position_combo == "None":
            col_to_drop = ["XPos", "YPos", "ZPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos":
            col_to_drop = ["YPos", "ZPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "YPos":
            col_to_drop = ["XPos", "ZPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "ZPos":
            col_to_drop = ["XPos", "YPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos":
            col_to_drop = ["ZPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_ZPos":
            col_to_drop = ["YPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "YPos_ZPos":
            col_to_drop = ["XPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos_ZPos":
            col_to_drop = common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        else:
            raise ValueError(f"Unsupported position_combo: {position_combo}")

    elif pressure_combo == 6:
        pressure_col_to_drop = [
            "tap2_delta_pressure_mean", "tap6_delta_pressure_mean",
            "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap17_delta_pressure_mean"
        ]
        if position_combo == "None":
            col_to_drop = ["XPos", "YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos":
            col_to_drop = ["YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "YPos":
            col_to_drop = ["XPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "ZPos":
            col_to_drop = ["XPos", "YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos":
            col_to_drop = ["ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_ZPos":
            col_to_drop = ["YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "YPos_ZPos":
            col_to_drop = ["XPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos_ZPos":
            col_to_drop = pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        else:
            raise ValueError(f"Unsupported position_combo: {position_combo}")

    elif pressure_combo == 2:
        pressure_col_to_drop = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap15_delta_pressure_mean", "tap17_delta_pressure_mean",
            "tap19_delta_pressure_mean"
        ]
        if position_combo == "None":
            col_to_drop = ["XPos", "YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos":
            col_to_drop = ["YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "YPos":
            col_to_drop = ["XPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "ZPos":
            col_to_drop = ["XPos", "YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos":
            col_to_drop = ["ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_ZPos":
            col_to_drop = ["YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "YPos_ZPos":
            col_to_drop = ["XPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos_ZPos":
            col_to_drop = pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        else:
            raise ValueError(f"Unsupported position_combo: {position_combo}")

    elif pressure_combo == 0:
        pressure_col_to_drop = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap15_delta_pressure_mean", "tap17_delta_pressure_mean",
            "tap19_delta_pressure_mean", "tap9_delta_pressure_mean", "tap10_delta_pressure_mean"
        ]
        if position_combo == "None":
            col_to_drop = ["XPos", "YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos":
            col_to_drop = ["YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "YPos":
            col_to_drop = ["XPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "ZPos":
            col_to_drop = ["XPos", "YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos":
            col_to_drop = ["ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_ZPos":
            col_to_drop = ["YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "YPos_ZPos":
            col_to_drop = ["XPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos_ZPos":
            col_to_drop = pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_full.columns if col not in col_to_drop]
        else:
            raise ValueError(f"Unsupported position_combo: {position_combo}")

    else:
        raise ValueError(f"Unsupported pressure_combo: {pressure_combo}")

    model_dataframe = data_model_full[feature_cols_full + [target_col, "Iteration", "config_group_id"]].copy()
    return model_dataframe, feature_cols_full


def create_xgb_model(model_params, random_seed=RANDOM_SEED):
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


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def evaluate_params_with_grouped_cv(X, y, groups, model_params, inner_n_splits=INNER_N_SPLITS):
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


def bayes_optimization_grouped_cv(
    X,
    y,
    groups,
    init_points=INIT_POINTS,
    n_iter=N_ITER,
    inner_n_splits=INNER_N_SPLITS
):
    # Bayesian optimization is kept here because it already exists in your
    # pipeline, but the search budget is reduced for nested CV practicality.
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

    def xgb_grouped_cv(
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

        inner_fold_metrics_df, mean_best_n_estimators = evaluate_params_with_grouped_cv(
            X,
            y,
            groups,
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
        f=xgb_grouped_cv,
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

    best_inner_metrics_df, mean_best_n_estimators = evaluate_params_with_grouped_cv(
        X,
        y,
        groups,
        best_params,
        inner_n_splits=inner_n_splits
    )
    best_params["n_estimators"] = mean_best_n_estimators

    search_history_df = pd.DataFrame(search_history)
    return best_params, best_inner_metrics_df, search_history_df


def summarize_nested_cv_results(results_df):
    summary_rows = []
    for metric_name in ["rmse", "mae", "r2"]:
        summary_rows.append({
            "metric": metric_name,
            "mean": results_df[metric_name].mean(),
            "std": results_df[metric_name].std(ddof=1)
        })
    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def run_grouped_nested_cv_xgboost(
    model_dataframe,
    feature_cols,
    target_col,
    save_folder_path=None,
    save_name="grouped_nested_cv_xgboost"
):
    X = model_dataframe[feature_cols].to_numpy()
    y = model_dataframe[target_col].to_numpy()
    groups = model_dataframe["config_group_id"].to_numpy()

    outer_cv = GroupKFold(n_splits=OUTER_N_SPLITS)
    outer_fold_results = []
    outer_fold_best_params = []
    search_history_frames = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        print(f"Running outer fold {outer_fold}/{OUTER_N_SPLITS}...")

        X_outer_train = X[train_idx]
        X_outer_test = X[test_idx]
        y_outer_train = y[train_idx]
        y_outer_test = y[test_idx]
        groups_outer_train = groups[train_idx]
        groups_outer_test = groups[test_idx]

        best_params, best_inner_metrics_df, search_history_df = bayes_optimization_grouped_cv(
            X_outer_train,
            y_outer_train,
            groups_outer_train,
            init_points=INIT_POINTS,
            n_iter=N_ITER,
            inner_n_splits=INNER_N_SPLITS
        )

        final_model = create_xgb_model(best_params, random_seed=RANDOM_SEED)
        final_model.set_params(early_stopping_rounds=None)
        final_model.fit(X_outer_train, y_outer_train, verbose=False)
        y_outer_pred = final_model.predict(X_outer_test)

        fold_metrics = regression_metrics(y_outer_test, y_outer_pred)
        fold_result = {
            "outer_fold": outer_fold,
            "n_train_rows": len(train_idx),
            "n_test_rows": len(test_idx),
            "n_train_groups": len(np.unique(groups_outer_train)),
            "n_test_groups": len(np.unique(groups_outer_test)),
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
            "best_params_json": json.dumps(best_params)
        }
        outer_fold_results.append(fold_result)

        best_params_row = {"outer_fold": outer_fold, **best_params}
        outer_fold_best_params.append(best_params_row)

        if not search_history_df.empty:
            search_history_df = search_history_df.copy()
            search_history_df["outer_fold"] = outer_fold
            search_history_frames.append(search_history_df)

        print(
            f"Outer fold {outer_fold} complete:",
            f"RMSE = {fold_result['rmse']:.5f},",
            f"MAE = {fold_result['mae']:.5f},",
            f"R2 = {fold_result['r2']:.5f}"
        )
        print("Best hyperparameters:", best_params)
        print("Best inner-fold mean RMSE:", f"{best_inner_metrics_df['rmse'].mean():.5f}")

    results_df = pd.DataFrame(outer_fold_results)
    best_params_df = pd.DataFrame(outer_fold_best_params)
    summary_df = summarize_nested_cv_results(results_df)

    if search_history_frames:
        search_history_df = pd.concat(search_history_frames, ignore_index=True)
    else:
        search_history_df = pd.DataFrame()

    if save_folder_path is not None:
        os.makedirs(save_folder_path, exist_ok=True)
        results_df.to_csv(
            os.path.join(save_folder_path, f"Outer_fold_results_{save_name}.csv"),
            index=False
        )
        summary_df.to_csv(
            os.path.join(save_folder_path, f"Summary_metrics_{save_name}.csv"),
            index=False
        )
        best_params_df.to_csv(
            os.path.join(save_folder_path, f"Best_hyperparameters_{save_name}.csv"),
            index=False
        )
        if not search_history_df.empty:
            search_history_df.to_csv(
                os.path.join(save_folder_path, f"Bayes_search_history_{save_name}.csv"),
                index=False
            )

    return results_df, summary_df, best_params_df, search_history_df


def grouped_nested_cv_workflow(
    data_model_full,
    pressure_combo,
    position_combo,
    target_col="delta_L_D_mean",
    save_folder_path=None,
    save_name=None
):
    if save_name is None:
        save_name = f"Pressure_{pressure_combo}_Position_{position_combo}"

    model_dataframe, feature_cols = xgbdata_prep(
        data_model_full,
        pressure_combo,
        position_combo,
        target_col=target_col
    )

    print("Grouped nested CV uses configuration groups so repeated")
    print("iter 0 / iter 1 measurements never split across folds.")
    print("Predictor columns:", feature_cols)

    results_df, summary_df, best_params_df, search_history_df = run_grouped_nested_cv_xgboost(
        model_dataframe=model_dataframe,
        feature_cols=feature_cols,
        target_col=target_col,
        save_folder_path=save_folder_path,
        save_name=save_name
    )
    return results_df, summary_df, best_params_df, search_history_df


if __name__ == "__main__":
    folder_path = "Predictive_model_data/Grouped_Nested_CV_XGB"
    os.makedirs(folder_path, exist_ok=True)

    data_model_full = data_prep()
    print("Data preparation completed successfully!")

    pressure_combos = [0, 2, 6, 12]
    position_combos = ["None", "XPos", "YPos", "ZPos", "XPos_YPos_ZPos"]

    for pressure_combo in pressure_combos:
        for position_combo in position_combos:
            if pressure_combo == 0 and position_combo == "None":
                continue

            save_name = f"Pressure_{pressure_combo}_Position_{position_combo}"
            print(f"Running grouped nested CV for {save_name}...")

            results_df, summary_df, best_params_df, _ = grouped_nested_cv_workflow(
                data_model_full=data_model_full,
                pressure_combo=pressure_combo,
                position_combo=position_combo,
                target_col="delta_L_D_mean",
                save_folder_path=folder_path,
                save_name=save_name
            )

            print(f"Grouped nested CV completed successfully for {save_name}!")
            print(results_df)
            print(summary_df)
            print(best_params_df)
