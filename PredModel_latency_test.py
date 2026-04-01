import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


RANDOM_SEED = 42
WARMUP_SAMPLES = 20


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
    return dataframe


def load_prepared_data():
    l_d_iter_0 = pd.read_csv("Post_processed_stats_summarized/L_D_True_Stats_Iter_0.csv")
    l_d_iter_1 = pd.read_csv("Post_processed_stats_summarized/L_D_True_Stats_Iter_1.csv")
    pressure_iter_0 = pd.read_csv("Post_processed_stats_summarized/Pressure_True_Stats_Iter_0.csv")
    pressure_iter_1 = pd.read_csv("Post_processed_stats_summarized/Pressure_True_Stats_Iter_1.csv")

    l_d_iter_0["AoA"] = pd.to_numeric(l_d_iter_0["AoA"], errors="coerce")
    l_d_iter_1["AoA"] = pd.to_numeric(l_d_iter_1["AoA"], errors="coerce")

    pressure_iter_0["AoA"] = pressure_iter_0["AoA"].astype(str).str.replace("_", ".").astype(float)
    pressure_iter_1["AoA"] = pressure_iter_1["AoA"].astype(str).str.replace("_", ".").astype(float)

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

    force_cols = ["XPos", "YPos", "ZPos", "AoA", "delta_L_D_mean"]
    force_selected_iter_0 = l_d_iter_0[force_cols].copy()
    force_selected_iter_1 = l_d_iter_1[force_cols].copy()

    pressure_columns = ["XPos", "YPos", "ZPos", "AoA"] + upper_taps
    pressure_selected_iter_0 = pressure_iter_0[pressure_columns].copy()
    pressure_selected_iter_1 = pressure_iter_1[pressure_columns].copy()

    check_alignment(force_selected_iter_0, pressure_selected_iter_0)
    check_alignment(force_selected_iter_1, pressure_selected_iter_1)

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


def split_iter0_iter1_data(dataframe, iteration_col="Iteration", iter0_value=0, iter1_value=1):
    iter0_df = dataframe[dataframe[iteration_col] == iter0_value].copy()
    iter1_df = dataframe[dataframe[iteration_col] == iter1_value].copy()
    return iter0_df, iter1_df


def select_median_xgb_model(xgb_results_csv):
    results_df = pd.read_csv(xgb_results_csv)
    median_rmse = results_df["rmse"].median()
    median_idx = (results_df["rmse"] - median_rmse).abs().idxmin()
    selected_row = results_df.loc[median_idx]
    return {
        "selected_seed": selected_row["seed"],
        "selected_rmse": selected_row["rmse"],
        "saved_model_path": selected_row["saved_model_path"]
    }


def load_xgb_model(model_path):
    xgb_model = XGBRegressor()
    xgb_model.load_model(model_path)
    return xgb_model


def parse_best_params_json(best_params_json):
    if pd.isna(best_params_json) or best_params_json == "":
        return {}
    return json.loads(best_params_json)


def select_median_simple_model_settings(simple_model_results_csv):
    results_df = pd.read_csv(simple_model_results_csv)
    median_rmse = results_df["rmse"].median()
    median_idx = (results_df["rmse"] - median_rmse).abs().idxmin()
    selected_row = results_df.loc[median_idx]
    best_params = parse_best_params_json(selected_row.get("best_params_json", ""))
    return {
        "selected_rmse": selected_row["rmse"],
        "best_params": best_params
    }


def build_simple_model(model_name, tuned_params):
    if model_name == "LinearRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

    if model_name == "RidgeRegression":
        alpha_value = tuned_params.get("model__alpha", tuned_params.get("alpha", 1.0))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(alpha_value)))
        ])

    if model_name == "LassoRegression":
        alpha_value = tuned_params.get("model__alpha", tuned_params.get("alpha", 1.0))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=float(alpha_value), max_iter=100000, random_state=RANDOM_SEED))
        ])

    if model_name == "PLSRegression":
        n_components = tuned_params.get("model__n_components", tuned_params.get("n_components", 2))
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", PLSRegression(n_components=int(n_components)))
        ])

    raise ValueError(f"Unsupported simple model: {model_name}")


def maybe_load_saved_simple_model(saved_simple_model_dir, model_name, predictor_set_name):
    if saved_simple_model_dir is None:
        return None

    model_filename = f"{sanitize_name(model_name)}_{sanitize_name(predictor_set_name)}.pkl"
    model_path = os.path.join(saved_simple_model_dir, model_filename)
    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model


def predict_single_sample(model, single_row_2d):
    prediction = model.predict(single_row_2d)
    if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
        prediction = prediction.ravel()
    return prediction


def benchmark_single_sample_latency(model, X_test, warmup_samples=WARMUP_SAMPLES):
    warmup_count = min(warmup_samples, X_test.shape[0])
    for row_idx in range(warmup_count):
        predict_single_sample(model, X_test[row_idx:row_idx + 1])

    latency_seconds = []
    predictions = []
    for row_idx in range(X_test.shape[0]):
        single_row = X_test[row_idx:row_idx + 1]
        start_time = time.perf_counter()
        prediction = predict_single_sample(model, single_row)
        end_time = time.perf_counter()
        latency_seconds.append(end_time - start_time)
        predictions.append(float(np.asarray(prediction).ravel()[0]))

    latency_ms = np.array(latency_seconds) * 1000.0

    batch_start_time = time.perf_counter()
    model.predict(X_test)
    batch_end_time = time.perf_counter()
    total_batch_time_ms = (batch_end_time - batch_start_time) * 1000.0

    latency_summary = {
        "mean_latency_ms": float(np.mean(latency_ms)),
        "median_latency_ms": float(np.median(latency_ms)),
        "std_latency_ms": float(np.std(latency_ms, ddof=1)) if latency_ms.shape[0] > 1 else 0.0,
        "min_latency_ms": float(np.min(latency_ms)),
        "max_latency_ms": float(np.max(latency_ms)),
        "total_batch_time_ms": float(total_batch_time_ms),
        "total_predictions": int(X_test.shape[0]),
        "predictions_per_second": float(1000.0 / np.mean(latency_ms)) if np.mean(latency_ms) > 0 else np.nan,
        "per_row_latency_ms": latency_ms,
        "predictions": np.array(predictions)
    }
    return latency_summary


def benchmark_xgb_predictor_set(
    iter1_df,
    predictor_set_name,
    predictor_cols,
    xgb_saved_model_root
):
    predictor_set_dir = os.path.join(xgb_saved_model_root, sanitize_name(predictor_set_name))
    robustness_results_csv = os.path.join(
        predictor_set_dir,
        f"Robustness_results_{sanitize_name(predictor_set_name)}.csv"
    )
    if not os.path.exists(robustness_results_csv):
        return None

    selection_info = select_median_xgb_model(robustness_results_csv)
    xgb_model = load_xgb_model(selection_info["saved_model_path"])

    X_test = iter1_df[predictor_cols].to_numpy()
    latency_summary = benchmark_single_sample_latency(xgb_model, X_test)

    return {
        "model_name": "XGBoost",
        "predictor_set_name": predictor_set_name,
        "selected_seed": selection_info["selected_seed"],
        "n_test_rows": iter1_df.shape[0],
        "mean_latency_ms": latency_summary["mean_latency_ms"],
        "median_latency_ms": latency_summary["median_latency_ms"],
        "std_latency_ms": latency_summary["std_latency_ms"],
        "min_latency_ms": latency_summary["min_latency_ms"],
        "max_latency_ms": latency_summary["max_latency_ms"],
        "predictions_per_second": latency_summary["predictions_per_second"],
        "total_batch_time_ms": latency_summary["total_batch_time_ms"],
        "model_source": "loaded_saved_model",
        "selected_reference_rmse": selection_info["selected_rmse"],
        "per_row_latency_ms": latency_summary["per_row_latency_ms"],
        "predictions": latency_summary["predictions"]
    }


def benchmark_simple_model_predictor_set(
    iter0_df,
    iter1_df,
    predictor_set_name,
    predictor_cols,
    target_col,
    model_name,
    simple_model_results_dir,
    saved_simple_model_dir=None
):
    saved_model = maybe_load_saved_simple_model(saved_simple_model_dir, model_name, predictor_set_name)
    model_source = "loaded_saved_model" if saved_model is not None else "retrained_from_settings"

    if saved_model is None:
        outer_fold_results_csv = os.path.join(
            simple_model_results_dir,
            f"Outer_fold_results_{sanitize_name(model_name)}_{sanitize_name(predictor_set_name)}.csv"
        )
        if not os.path.exists(outer_fold_results_csv):
            return None

        settings_info = select_median_simple_model_settings(outer_fold_results_csv)
        simple_model = build_simple_model(model_name, settings_info["best_params"])
        X_train = iter0_df[predictor_cols].to_numpy()
        y_train = iter0_df[target_col].to_numpy()
        simple_model.fit(X_train, y_train)
        selected_reference_rmse = settings_info["selected_rmse"]
    else:
        simple_model = saved_model
        selected_reference_rmse = np.nan

    X_test = iter1_df[predictor_cols].to_numpy()
    latency_summary = benchmark_single_sample_latency(simple_model, X_test)

    return {
        "model_name": model_name,
        "predictor_set_name": predictor_set_name,
        "selected_seed": np.nan,
        "n_test_rows": iter1_df.shape[0],
        "mean_latency_ms": latency_summary["mean_latency_ms"],
        "median_latency_ms": latency_summary["median_latency_ms"],
        "std_latency_ms": latency_summary["std_latency_ms"],
        "min_latency_ms": latency_summary["min_latency_ms"],
        "max_latency_ms": latency_summary["max_latency_ms"],
        "predictions_per_second": latency_summary["predictions_per_second"],
        "total_batch_time_ms": latency_summary["total_batch_time_ms"],
        "model_source": model_source,
        "selected_reference_rmse": selected_reference_rmse,
        "per_row_latency_ms": latency_summary["per_row_latency_ms"],
        "predictions": latency_summary["predictions"]
    }


def model_column_prefix(model_name):
    return sanitize_name(model_name).lower()


def build_latency_difference_by_model(detailed_results_df):
    model_summary_df = (
        detailed_results_df
        .groupby("model_name", as_index=False)
        .agg(
            mean_latency_ms=("mean_latency_ms", "mean"),
            median_latency_ms=("mean_latency_ms", "median"),
            std_latency_ms=("mean_latency_ms", "std"),
            min_latency_ms=("mean_latency_ms", "min"),
            max_latency_ms=("mean_latency_ms", "max"),
            predictor_set_count=("predictor_set_name", "count")
        )
    )
    fastest_model_latency = model_summary_df["mean_latency_ms"].min()
    xgboost_latency = model_summary_df.loc[
        model_summary_df["model_name"] == "XGBoost",
        "mean_latency_ms"
    ].iloc[0]

    model_summary_df["difference_vs_fastest_model_ms"] = (
        model_summary_df["mean_latency_ms"] - fastest_model_latency
    )
    model_summary_df["difference_vs_xgboost_ms"] = (
        model_summary_df["mean_latency_ms"] - xgboost_latency
    )
    model_summary_df = model_summary_df.sort_values("mean_latency_ms").reset_index(drop=True)
    return model_summary_df


def build_latency_difference_by_predictor_set(detailed_results_df):
    pivot_df = (
        detailed_results_df
        .pivot_table(
            index="predictor_set_name",
            columns="model_name",
            values="mean_latency_ms"
        )
        .reset_index()
    )
    pivot_df.columns.name = None

    latency_model_cols = [col for col in pivot_df.columns if col != "predictor_set_name"]
    if "XGBoost" in pivot_df.columns:
        for model_name in latency_model_cols:
            if model_name == "XGBoost":
                continue
            pivot_df[f"{sanitize_name(model_name)}_minus_XGBoost_ms"] = (
                pivot_df[model_name] - pivot_df["XGBoost"]
            )

    pivot_df["fastest_model_in_set"] = pivot_df[latency_model_cols].idxmin(axis=1)
    pivot_df["fastest_model_latency_ms"] = pivot_df[latency_model_cols].min(axis=1)
    return pivot_df


def run_latency_benchmark_all_models(
    df,
    predictor_sets,
    target_col,
    iteration_col="Iteration",
    iter0_value=0,
    iter1_value=1,
    xgb_saved_model_root="Predictive_model_data/Iter0_Iter1_Robustness_XGB",
    simple_model_results_dir="Predictive_model_data/Grouped_CV_Simple_Models",
    saved_simple_model_dir=None,
    output_dir="Predictive_model_data/Latency_Benchmark",
    save_per_row_latencies=False
):
    os.makedirs(output_dir, exist_ok=True)
    raw_output_dir = os.path.join(output_dir, "Raw_prediction_tables")
    os.makedirs(raw_output_dir, exist_ok=True)

    iter0_df, iter1_df = split_iter0_iter1_data(
        dataframe=df,
        iteration_col=iteration_col,
        iter0_value=iter0_value,
        iter1_value=iter1_value
    )

    detailed_rows = []
    per_row_latency_records = []
    simple_model_names = ["LinearRegression", "RidgeRegression", "LassoRegression", "PLSRegression"]

    for predictor_set_name, predictor_cols in predictor_sets.items():
        print(f"Benchmarking predictor set: {predictor_set_name}")
        raw_predictor_set_df = iter1_df.copy()

        xgb_result = benchmark_xgb_predictor_set(
            iter1_df=iter1_df,
            predictor_set_name=predictor_set_name,
            predictor_cols=predictor_cols,
            xgb_saved_model_root=xgb_saved_model_root
        )
        if xgb_result is not None:
            detailed_rows.append({key: value for key, value in xgb_result.items() if key != "per_row_latency_ms"})
            raw_predictor_set_df[f"{model_column_prefix('XGBoost')}_prediction_{target_col}"] = xgb_result["predictions"]
            raw_predictor_set_df[f"{model_column_prefix('XGBoost')}_latency_ms"] = xgb_result["per_row_latency_ms"]
            if save_per_row_latencies:
                for latency_value in xgb_result["per_row_latency_ms"]:
                    per_row_latency_records.append({
                        "model_name": xgb_result["model_name"],
                        "predictor_set_name": predictor_set_name,
                        "latency_ms": latency_value
                    })

        for model_name in simple_model_names:
            simple_result = benchmark_simple_model_predictor_set(
                iter0_df=iter0_df,
                iter1_df=iter1_df,
                predictor_set_name=predictor_set_name,
                predictor_cols=predictor_cols,
                target_col=target_col,
                model_name=model_name,
                simple_model_results_dir=simple_model_results_dir,
                saved_simple_model_dir=saved_simple_model_dir
            )
            if simple_result is None:
                continue

            detailed_rows.append({key: value for key, value in simple_result.items() if key != "per_row_latency_ms"})
            raw_predictor_set_df[f"{model_column_prefix(model_name)}_prediction_{target_col}"] = simple_result["predictions"]
            raw_predictor_set_df[f"{model_column_prefix(model_name)}_latency_ms"] = simple_result["per_row_latency_ms"]
            if save_per_row_latencies:
                for latency_value in simple_result["per_row_latency_ms"]:
                    per_row_latency_records.append({
                        "model_name": simple_result["model_name"],
                        "predictor_set_name": predictor_set_name,
                        "latency_ms": latency_value
                    })

        raw_predictor_set_df.to_csv(
            os.path.join(
                raw_output_dir,
                f"Raw_predictions_{sanitize_name(predictor_set_name)}.csv"
            ),
            index=False
        )

    detailed_results_df = pd.DataFrame(detailed_rows)
    ranked_results_df = detailed_results_df.sort_values(
        by=["mean_latency_ms", "model_name", "predictor_set_name"]
    ).reset_index(drop=True)
    latency_difference_by_model_df = build_latency_difference_by_model(detailed_results_df)
    latency_difference_by_predictor_set_df = build_latency_difference_by_predictor_set(detailed_results_df)

    detailed_results_df.to_csv(
        os.path.join(output_dir, "Latency_results_detailed.csv"),
        index=False
    )
    ranked_results_df.to_csv(
        os.path.join(output_dir, "Latency_results_ranked.csv"),
        index=False
    )
    latency_difference_by_model_df.to_csv(
        os.path.join(output_dir, "Latency_difference_by_model.csv"),
        index=False
    )
    latency_difference_by_predictor_set_df.to_csv(
        os.path.join(output_dir, "Latency_difference_by_predictor_set.csv"),
        index=False
    )

    if save_per_row_latencies and per_row_latency_records:
        per_row_latency_df = pd.DataFrame(per_row_latency_records)
        per_row_latency_df.to_csv(
            os.path.join(output_dir, "Latency_results_per_row.csv"),
            index=False
        )

    return {
        "detailed_results_df": detailed_results_df,
        "ranked_results_df": ranked_results_df,
        "latency_difference_by_model_df": latency_difference_by_model_df,
        "latency_difference_by_predictor_set_df": latency_difference_by_predictor_set_df
    }


if __name__ == "__main__":
    output_dir = "Predictive_model_data/Latency_Benchmark"
    xgb_saved_model_root = "Predictive_model_data/Iter0_Iter1_Robustness_XGB"
    simple_model_results_dir = "Predictive_model_data/Grouped_CV_Simple_Models"
    saved_simple_model_dir = None

    data_model_full = load_prepared_data()
    predictor_sets = build_predictor_sets_from_pressure_position_combos()

    results = run_latency_benchmark_all_models(
        df=data_model_full,
        predictor_sets=predictor_sets,
        target_col="delta_L_D_mean",
        iteration_col="Iteration",
        iter0_value=0,
        iter1_value=1,
        xgb_saved_model_root=xgb_saved_model_root,
        simple_model_results_dir=simple_model_results_dir,
        saved_simple_model_dir=saved_simple_model_dir,
        output_dir=output_dir,
        save_per_row_latencies=False
    )

    print(results["ranked_results_df"])
