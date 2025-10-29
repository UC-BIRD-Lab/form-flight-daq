import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import os

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

# ------------------------------
# Read CSV files and prepare data function
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

    # upper_taps = [
    #     "tap0_delta_pressure_mean", "tap0_delta_pressure_error",
    #     "tap2_delta_pressure_mean", "tap2_delta_pressure_error",
    #     "tap4_delta_pressure_mean", "tap4_delta_pressure_error",
    #     "tap6_delta_pressure_mean", "tap6_delta_pressure_error",
    #     "tap8_delta_pressure_mean", "tap8_delta_pressure_error",
    #     "tap9_delta_pressure_mean", "tap9_delta_pressure_error",
    #     "tap10_delta_pressure_mean", "tap10_delta_pressure_error",
    #     "tap11_delta_pressure_mean", "tap11_delta_pressure_error",
    #     "tap13_delta_pressure_mean", "tap13_delta_pressure_error",
    #     "tap15_delta_pressure_mean", "tap15_delta_pressure_error",
    #     "tap17_delta_pressure_mean", "tap17_delta_pressure_error",
    #     "tap19_delta_pressure_mean", "tap19_delta_pressure_error"
    # ] # Include error columns for pressure taps

    # Select only necessary force columns
    force_cols = ["XPos", "YPos", "ZPos", "AoA", "delta_L_D_mean"]
    force_selected_train = l_d_iter_0[force_cols].copy()
    force_selected_test = l_d_iter_1[force_cols].copy()

    # Select only necessary pressure columns (keys + pressure taps)
    pressure_columns = ["XPos", "YPos", "ZPos", "AoA"] + upper_taps
    pressure_selected_train = pressure_iter_0[pressure_columns].copy()
    pressure_selected_test = pressure_iter_1[pressure_columns].copy()

    # Run alignment checks
    check_alignment(force_selected_train, pressure_selected_train)
    check_alignment(force_selected_test, pressure_selected_test)

    # Merge force and pressure data on key columns
    merge_keys = ["XPos", "YPos", "ZPos", "AoA"]
    data_model_train_full = pd.merge(force_selected_train, pressure_selected_train, on=merge_keys)
    data_model_test_full = pd.merge(force_selected_test, pressure_selected_test, on=merge_keys)

    return data_model_test_full, data_model_train_full

# Prepare Data for XGBoost (Full Dataset)
# ------------------------------
# For full dataset, use all columns except the target and AoA as features
def xgbdata_prep(data_model_test_full, data_model_train_full, pressure_combo, position_combo):
    common_cols_to_drop = ["AoA", "delta_L_D_mean"]
    if pressure_combo == 12:
        if position_combo == "None":
            col_to_drop = ["XPos", "YPos", "ZPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos":
            col_to_drop = ["YPos", "ZPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "YPos":
            col_to_drop = ["XPos", "ZPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "ZPos":
            col_to_drop = ["XPos", "YPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos":
            col_to_drop = ["ZPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_ZPos":
            col_to_drop = ["YPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "YPos_ZPos":
            col_to_drop = ["XPos"] + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos_ZPos":
            col_to_drop = common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
            
    elif pressure_combo == 6:
        pressure_col_to_drop = [
        "tap2_delta_pressure_mean", "tap6_delta_pressure_mean",
        "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
        "tap13_delta_pressure_mean", "tap17_delta_pressure_mean"]
        if position_combo == "None":
            col_to_drop = ["XPos", "YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos":
            col_to_drop = ["YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "YPos":
            col_to_drop = ["XPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "ZPos":
            col_to_drop = ["XPos", "YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos":
            col_to_drop = ["ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_ZPos":
            col_to_drop = ["YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "YPos_ZPos":
            col_to_drop = ["XPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos_ZPos":
            col_to_drop = pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
    
    elif pressure_combo == 2:
        pressure_col_to_drop = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap15_delta_pressure_mean", "tap17_delta_pressure_mean",
            "tap19_delta_pressure_mean"
        ]
        if position_combo == "None":
            col_to_drop = ["XPos", "YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos":
            col_to_drop = ["YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "YPos":
            col_to_drop = ["XPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "ZPos":
            col_to_drop = ["XPos", "YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos":
            col_to_drop = ["ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_ZPos":
            col_to_drop = ["YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "YPos_ZPos":
            col_to_drop = ["XPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos_ZPos":
            col_to_drop = pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
    
    elif pressure_combo == 0:
        pressure_col_to_drop = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap15_delta_pressure_mean", "tap17_delta_pressure_mean",
            "tap19_delta_pressure_mean", "tap9_delta_pressure_mean", "tap10_delta_pressure_mean"
        ]
        if position_combo == "None":
            col_to_drop = ["XPos", "YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos":
            col_to_drop = ["YPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "YPos":
            col_to_drop = ["XPos", "ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "ZPos":
            col_to_drop = ["XPos", "YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos":
            col_to_drop = ["ZPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_ZPos":
            col_to_drop = ["YPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "YPos_ZPos":
            col_to_drop = ["XPos"] + pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]
        elif position_combo == "XPos_YPos_ZPos":
            col_to_drop = pressure_col_to_drop + common_cols_to_drop
            feature_cols_full = [col for col in data_model_train_full.columns if col not in col_to_drop]

    train_matrix_full = data_model_train_full[feature_cols_full].values # Convert to NumPy arrays
    test_matrix_full = data_model_test_full[feature_cols_full].values

    train_labels_full = data_model_train_full["delta_L_D_mean"].values # delta_L_D_mean is the target column
    test_labels_full = data_model_test_full["delta_L_D_mean"].values

    dtrain = xgb.DMatrix(train_matrix_full, label=train_labels_full) # Create DMatrix objects for xgboost
    dtest = xgb.DMatrix(test_matrix_full, label=test_labels_full)
    return dtrain, dtest, train_labels_full, test_labels_full, data_model_test_full, feature_cols_full

def test_structure_baye(max_depth, eta, subsample, colsample_bytree, lambda_, alpha, dtrain):
    params = {
        "objective": "reg:squarederror",
        "max_depth": int(max_depth),
        "eta": eta,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": lambda_,
        "alpha": alpha,
        "eval_metric": "rmse",
    }
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=10000,
        nfold=6,
        early_stopping_rounds=100,
        verbose_eval=False,
        seed=42
    )
    best_rmse = cv_results["test-rmse-mean"].min()
    best_iteration = int(cv_results["test-rmse-mean"].idxmin())
    print(f"Best RMSE: {best_rmse} for {best_iteration} rounds")

    # Return negative RMSE for maximization
    return -best_rmse, best_iteration


def bayes_optimization(pbounds, init_points, n_iter, folder_path, save_name, col_names, dtrain):
    """
    bayes_optimization now expects 'dtrain' so each call is tied to the correct data.
    """

    # Define an inner function that uses the local 'dtrain'
    def xgb_cv(max_depth, eta, subsample, colsample_bytree, lambda_, alpha):
        score, _ = test_structure_baye(
            max_depth, eta, subsample, 
            colsample_bytree, lambda_, alpha, 
            dtrain
        )
        return score

    # Create the BayesianOptimizer with our new local xgb_cv
    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds=pbounds,
        random_state=3,
        verbose=2
    )

    # Run the optimization: initial points and iterations
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # Save optimization history
    bayes_history = pd.json_normalize(optimizer.res)
    bayes_history["params.max_depth"] = bayes_history["params.max_depth"].astype(int)
    bayes_history["input_names"] = [col_names] * len(bayes_history)
    bayes_history.to_csv(os.path.join(folder_path, f"Bayes_history_for_hyperparameter_tuning_{save_name}.csv"), index=False)

    # Retrieve the best hyperparameters
    best_params_raw = optimizer.max["params"]
    best_params = {
        "objective": "reg:squarederror",
        "max_depth": int(best_params_raw["max_depth"]),
        "eta": best_params_raw["eta"],
        "subsample": best_params_raw["subsample"],
        "colsample_bytree": best_params_raw["colsample_bytree"],
        "lambda": best_params_raw["lambda_"],
        "alpha": best_params_raw["alpha"],
        "eval_metric": "rmse",
        "seed": 42
    }

    print("Best parameters found:", best_params)
    return best_params


if __name__ == "__main__":
    folder_path = "Predictive_model_data/Bayes_results_XGB_HyperP_using_dP_for_delta_L_D"
    os.makedirs(folder_path, exist_ok=True)

    data_model_test_full, data_model_train_full = data_prep()
    print("Data preparation completed successfully!")

    pressure_combos = [0, 2, 6, 12]
    position_combos = ["None", "XPos", "YPos", "ZPos",
                       "XPos_YPos_ZPos"]

    for pressure_combo in pressure_combos:
        for position_combo in position_combos:
            if pressure_combo == 0 and position_combo == "None":
                continue
            save_name = f"Pressure_{pressure_combo}_Position_{position_combo}"
            print(f"Running Bayesian Optimization for {save_name}...")

            dtrain, dtest, train_labels_full, test_labels_full, data_model_test_full, col_names = xgbdata_prep(
                data_model_test_full, 
                data_model_train_full, 
                pressure_combo, 
                position_combo
            )

            pbounds = {
                "max_depth": (2, 8),
                "eta": (0.0005, 0.1),
                "subsample": (0.4, 1),
                "colsample_bytree": (0.4, 1),
                "lambda_": (0, 10),
                "alpha": (0, 10)
            }

            init_points = 20
            n_iter = 1000
            best_params = bayes_optimization(
                pbounds, 
                init_points, 
                n_iter, 
                folder_path, 
                save_name, 
                col_names, 
                dtrain
            )
            print(f"Hyperparameter tuning completed successfully for {save_name}!")
