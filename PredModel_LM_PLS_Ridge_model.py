import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 5
RIDGE_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
LASSO_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
PLS_MAX_COMPONENTS = 8


# Data Import and Preprocessing
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


def ensure_grouping_columns(dataframe, config_cols):
    dataframe = dataframe.copy()
    if "config_group_id" not in dataframe.columns:
        dataframe = create_config_group_id(dataframe, config_cols)
    return dataframe


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


def build_model_dictionary(n_features):
    pls_max_components = min(n_features, PLS_MAX_COMPONENTS)
    if pls_max_components < 1:
        raise ValueError("PLSRegression requires at least one predictor column.")

    model_dict = {
        "LinearRegression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            "param_grid": None
        },
        "RidgeRegression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ]),
            "param_grid": {
                "model__alpha": RIDGE_ALPHA_GRID
            }
        },
        "LassoRegression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=100000, random_state=RANDOM_SEED))
            ]),
            "param_grid": {
                "model__alpha": LASSO_ALPHA_GRID
            }
        },
        "PLSRegression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", PLSRegression())
            ]),
            "param_grid": {
                "model__n_components": list(range(1, pls_max_components + 1))
            }
        }
    }
    return model_dict


def fit_model_with_optional_inner_tuning(
    pipeline,
    param_grid,
    X_train,
    y_train,
    groups_train
):
    if param_grid is None:
        fitted_model = clone(pipeline)
        fitted_model.fit(X_train, y_train)
        return fitted_model, {}, np.nan

    inner_cv = GroupKFold(n_splits=INNER_N_SPLITS)
    grid_search = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train, groups=groups_train)

    best_inner_rmse = -grid_search.best_score_
    return grid_search.best_estimator_, grid_search.best_params_, best_inner_rmse


def summarize_outer_fold_results(fold_results_df):
    summary_rows = []
    for metric_name in ["rmse", "mae", "r2"]:
        summary_rows.append({
            "model_name": fold_results_df["model_name"].iloc[0],
            "predictor_set_name": fold_results_df["predictor_set_name"].iloc[0],
            "metric": metric_name,
            "mean": fold_results_df[metric_name].mean(),
            "std": fold_results_df[metric_name].std(ddof=1)
        })
    return pd.DataFrame(summary_rows)


def run_grouped_cv_for_model(
    dataframe,
    predictor_cols,
    target_col,
    model_name,
    pipeline,
    param_grid
):
    missing_cols = [col for col in predictor_cols + [target_col, "config_group_id"] if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for grouped CV: {missing_cols}")

    X = dataframe[predictor_cols].to_numpy()
    y = dataframe[target_col].to_numpy()
    groups = dataframe["config_group_id"].to_numpy()

    outer_cv = GroupKFold(n_splits=OUTER_N_SPLITS)
    outer_fold_results = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        best_model, best_params, best_inner_rmse = fit_model_with_optional_inner_tuning(
            pipeline,
            param_grid,
            X_train,
            y_train,
            groups_train
        )

        y_pred = best_model.predict(X_test)
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        fold_metrics = regression_metrics(y_test, y_pred)
        fold_result = {
            "model_name": model_name,
            "outer_fold": outer_fold,
            "n_train_rows": len(train_idx),
            "n_test_rows": len(test_idx),
            "n_train_groups": len(np.unique(groups_train)),
            "n_test_groups": len(np.unique(groups_test)),
            "rmse": fold_metrics["rmse"],
            "mae": fold_metrics["mae"],
            "r2": fold_metrics["r2"],
            "best_params_json": json.dumps(best_params) if best_params else "",
            "best_inner_rmse": best_inner_rmse
        }

        for param_name, param_value in best_params.items():
            clean_param_name = param_name.replace("model__", "")
            fold_result[f"best_{clean_param_name}"] = param_value

        outer_fold_results.append(fold_result)

    fold_results_df = pd.DataFrame(outer_fold_results)
    return fold_results_df


def run_grouped_cv_for_model_and_predictor_set(
    dataframe,
    predictor_set_name,
    predictor_cols,
    target_col,
    model_name,
    pipeline,
    param_grid,
    save_folder_path=None
):
    print(f"Running {model_name} for predictor set {predictor_set_name}...")

    fold_results_df = run_grouped_cv_for_model(
        dataframe=dataframe,
        predictor_cols=predictor_cols,
        target_col=target_col,
        model_name=model_name,
        pipeline=pipeline,
        param_grid=param_grid
    )
    fold_results_df["predictor_set_name"] = predictor_set_name

    summary_df = summarize_outer_fold_results(fold_results_df)

    if save_folder_path is not None:
        os.makedirs(save_folder_path, exist_ok=True)
        safe_model_name = sanitize_name(model_name)
        safe_predictor_set_name = sanitize_name(predictor_set_name)

        fold_results_df.to_csv(
            os.path.join(
                save_folder_path,
                f"Outer_fold_results_{safe_model_name}_{safe_predictor_set_name}.csv"
            ),
            index=False
        )
        summary_df.to_csv(
            os.path.join(
                save_folder_path,
                f"Summary_metrics_{safe_model_name}_{safe_predictor_set_name}.csv"
            ),
            index=False
        )

    return fold_results_df, summary_df


def build_combined_summary(summary_metrics_df):
    combined_summary_df = (
        summary_metrics_df
        .pivot_table(
            index=["model_name", "predictor_set_name"],
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
    return combined_summary_df


def build_ranking_dataframe(combined_summary_df):
    ranking_df = combined_summary_df.copy()
    ranking_df["rmse_rank_within_model"] = (
        ranking_df.groupby("model_name")["rmse_mean"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    ranking_df = ranking_df.sort_values(
        by=["model_name", "rmse_rank_within_model", "predictor_set_name"]
    ).reset_index(drop=True)
    return ranking_df


def build_pivot_ranking_table(combined_summary_df):
    pivot_summary_df = combined_summary_df.pivot_table(
        index="predictor_set_name",
        columns="model_name",
        values="rmse_mean"
    ).reset_index()
    return pivot_summary_df


def run_single_model_across_predictor_sets(
    data_model_full,
    predictor_sets,
    target_col,
    config_cols,
    model_name,
    save_folder_path=None
):
    data_model_full = ensure_grouping_columns(data_model_full, config_cols)

    all_fold_results = []
    all_summary_results = []

    print(f"Starting model: {model_name}")

    for predictor_set_name, predictor_cols in predictor_sets.items():
        if len(predictor_cols) == 0:
            raise ValueError(f"Predictor set {predictor_set_name} is empty.")

        model_dict_local = build_model_dictionary(len(predictor_cols))
        model_info = model_dict_local[model_name]

        fold_results_df, summary_df = run_grouped_cv_for_model_and_predictor_set(
            dataframe=data_model_full,
            predictor_set_name=predictor_set_name,
            predictor_cols=predictor_cols,
            target_col=target_col,
            model_name=model_name,
            pipeline=model_info["pipeline"],
            param_grid=model_info["param_grid"],
            save_folder_path=save_folder_path
        )
        all_fold_results.append(fold_results_df)
        all_summary_results.append(summary_df)

    model_fold_results_df = pd.concat(all_fold_results, ignore_index=True)
    model_summary_results_df = pd.concat(all_summary_results, ignore_index=True)
    model_combined_summary_df = build_combined_summary(model_summary_results_df)
    model_ranking_df = build_ranking_dataframe(model_combined_summary_df)

    if save_folder_path is not None:
        os.makedirs(save_folder_path, exist_ok=True)
        safe_model_name = sanitize_name(model_name)
        model_fold_results_df.to_csv(
            os.path.join(save_folder_path, f"All_outer_fold_results_{safe_model_name}.csv"),
            index=False
        )
        model_summary_results_df.to_csv(
            os.path.join(save_folder_path, f"All_summary_metrics_{safe_model_name}.csv"),
            index=False
        )
        model_combined_summary_df.to_csv(
            os.path.join(save_folder_path, f"Combined_summary_metrics_{safe_model_name}.csv"),
            index=False
        )
        model_ranking_df.to_csv(
            os.path.join(save_folder_path, f"Predictor_set_ranking_{safe_model_name}.csv"),
            index=False
        )

    print(f"Completed model: {model_name}")
    return {
        "outer_fold_results_df": model_fold_results_df,
        "summary_metrics_df": model_summary_results_df,
        "combined_summary_df": model_combined_summary_df,
        "ranking_df": model_ranking_df
    }


def linear_regression_model(
    data_model_full,
    predictor_sets,
    target_col,
    config_cols,
    save_folder_path=None
):
    return run_single_model_across_predictor_sets(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        model_name="LinearRegression",
        save_folder_path=save_folder_path
    )


def ridge_regression_model(
    data_model_full,
    predictor_sets,
    target_col,
    config_cols,
    save_folder_path=None
):
    return run_single_model_across_predictor_sets(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        model_name="RidgeRegression",
        save_folder_path=save_folder_path
    )


def lasso_regression_model(
    data_model_full,
    predictor_sets,
    target_col,
    config_cols,
    save_folder_path=None
):
    return run_single_model_across_predictor_sets(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        model_name="LassoRegression",
        save_folder_path=save_folder_path
    )


def pls_regression_model(
    data_model_full,
    predictor_sets,
    target_col,
    config_cols,
    save_folder_path=None
):
    return run_single_model_across_predictor_sets(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        model_name="PLSRegression",
        save_folder_path=save_folder_path
    )


def run_grouped_model_comparison(
    data_model_full,
    predictor_sets,
    target_col,
    config_cols,
    save_folder_path=None,
    model_dict=None
):
    data_model_full = ensure_grouping_columns(data_model_full, config_cols)

    all_fold_results = []
    all_summary_results = []
    per_model_results = {}

    linear_results = linear_regression_model(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        save_folder_path=save_folder_path
    )
    per_model_results["LinearRegression"] = linear_results
    all_fold_results.append(linear_results["outer_fold_results_df"])
    all_summary_results.append(linear_results["summary_metrics_df"])

    ridge_results = ridge_regression_model(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        save_folder_path=save_folder_path
    )
    per_model_results["RidgeRegression"] = ridge_results
    all_fold_results.append(ridge_results["outer_fold_results_df"])
    all_summary_results.append(ridge_results["summary_metrics_df"])

    lasso_results = lasso_regression_model(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        save_folder_path=save_folder_path
    )
    per_model_results["LassoRegression"] = lasso_results
    all_fold_results.append(lasso_results["outer_fold_results_df"])
    all_summary_results.append(lasso_results["summary_metrics_df"])

    pls_results = pls_regression_model(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        save_folder_path=save_folder_path
    )
    per_model_results["PLSRegression"] = pls_results
    all_fold_results.append(pls_results["outer_fold_results_df"])
    all_summary_results.append(pls_results["summary_metrics_df"])

    all_fold_results_df = pd.concat(all_fold_results, ignore_index=True)
    all_summary_results_df = pd.concat(all_summary_results, ignore_index=True)
    combined_summary_df = build_combined_summary(all_summary_results_df)
    ranking_df = build_ranking_dataframe(combined_summary_df)
    pivot_summary_df = build_pivot_ranking_table(combined_summary_df)

    if save_folder_path is not None:
        os.makedirs(save_folder_path, exist_ok=True)
        all_fold_results_df.to_csv(
            os.path.join(save_folder_path, "All_outer_fold_results.csv"),
            index=False
        )
        all_summary_results_df.to_csv(
            os.path.join(save_folder_path, "All_summary_metrics.csv"),
            index=False
        )
        combined_summary_df.to_csv(
            os.path.join(save_folder_path, "Combined_summary_metrics.csv"),
            index=False
        )
        ranking_df.to_csv(
            os.path.join(save_folder_path, "Predictor_set_ranking_by_model.csv"),
            index=False
        )
        pivot_summary_df.to_csv(
            os.path.join(save_folder_path, "Predictor_set_rmse_pivot_table.csv"),
            index=False
        )

    return {
        "outer_fold_results_df": all_fold_results_df,
        "summary_metrics_df": all_summary_results_df,
        "combined_summary_df": combined_summary_df,
        "ranking_df": ranking_df,
        "pivot_summary_df": pivot_summary_df,
        "per_model_results": per_model_results
    }


if __name__ == "__main__":
    folder_path = "Predictive_model_data/Grouped_CV_Simple_Models"
    os.makedirs(folder_path, exist_ok=True)

    data_model_full = data_prep()
    print("Data preparation completed successfully!")

    target_col = "delta_L_D_mean"
    config_cols = ["XPos", "YPos", "ZPos", "AoA"]

    # Use the same 19 pressure/position predictor-set combinations as the
    # grouped XGBoost workflow so model-class comparisons stay aligned.
    predictor_sets = build_predictor_sets_from_pressure_position_combos()

    results = run_grouped_model_comparison(
        data_model_full=data_model_full,
        predictor_sets=predictor_sets,
        target_col=target_col,
        config_cols=config_cols,
        save_folder_path=folder_path
    )

    print(results["combined_summary_df"])
    print(results["ranking_df"])
    print(results["pivot_summary_df"])
