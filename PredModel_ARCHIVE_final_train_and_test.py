import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Data Import and Preprocessing
# ------------------------------

# Data Alignment function
# ------------------------------
def check_alignment(df1, df2):
    """
    Checks alignment by comparing key columns after sorting.
    Avoids false mismatches from row-order differences.
    """
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
    """
    Reads four CSVs, coerces AoA, selects columns, checks alignment,
    merges force & pressure by keys, and returns merged train/test.
    """
    folder_path = "Post_processed_stats_summarized/"
    l_d_iter_0 = pd.read_csv(folder_path + "L_D_True_Stats_Iter_0.csv")
    l_d_iter_1 = pd.read_csv(folder_path + "L_D_True_Stats_Iter_1.csv")
    pressure_iter_0 = pd.read_csv(folder_path + "Pressure_True_Stats_Iter_0.csv")
    pressure_iter_1 = pd.read_csv(folder_path + "Pressure_True_Stats_Iter_1.csv")

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
def xgbdata_prep(data_model_test_full, data_model_train_full):
    """
    Baseline builder: features = 12 Δp only (no AoA, no XYZ).
    Returns DMatrix train/test and label arrays.
    """
    feature_cols_full = [col for col in data_model_train_full.columns if col not in ["delta_L_D_mean", "AoA", "XPos", "YPos", "ZPos"]] # AoA, delta_L_D_mean are removed for the feature columns, only pressure data in the feature columns
    
    train_matrix_full = data_model_train_full[feature_cols_full].values # Convert to NumPy arrays
    test_matrix_full = data_model_test_full[feature_cols_full].values

    train_labels_full = data_model_train_full["delta_L_D_mean"].values # delta_L_D_mean is the target column
    test_labels_full = data_model_test_full["delta_L_D_mean"].values

    dtrain = xgb.DMatrix(train_matrix_full, label=train_labels_full) # Create DMatrix objects for xgboost
    dtest = xgb.DMatrix(test_matrix_full, label=test_labels_full)
    return dtrain, dtest, train_labels_full, test_labels_full, data_model_test_full

def best_params_from_csv(folder_path, file_name):
    best_params_df = pd.read_csv(os.path.join(folder_path, file_name))
    best_row = best_params_df.iloc[best_params_df["target"].idxmax()] # best row is the one with the highest target value because the target is negative value (for bayesian optimization purpose)
    best_params = {
        "objective": "reg:squarederror",
        "max_depth": int(best_row["params.max_depth"]), # Convert to integer
        "eta": best_row["params.eta"],
        "subsample": best_row["params.subsample"],
        "colsample_bytree": best_row["params.colsample_bytree"],
        "lambda": best_row["params.lambda_"],
        "alpha": best_row["params.alpha"],
        "eval_metric": "rmse",
        # GPU acceleration parameters
        #"tree_method": "hist",
        #"device": "cuda",
        "seed": 42
    }
    return best_params

# Model Training Function
# ------------------------------
def train_model(
    dtrain,
    dtest,
    best_params,
    test_labels,
    is_interpolated,
    interpolated_test_Dmatrix_df,
    *,
    nfold=6,
    max_rounds=10000,
    early_stopping_rounds=100,
    verbose_eval=False,
    seed=42
):
    """
    Cross-validated training on dtrain (Iter_0 only), NO peeking at dtest.
    1) xgb.cv picks best boosting rounds via early stopping on 6-fold CV.
    2) Train final model on ALL dtrain with best_rounds.
    3) Predict on dtest (Iter_1) or interpolated grid as requested.
    """

    # Ensure eval metric is set
    params = dict(best_params)
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("eval_metric", "rmse")
    params.setdefault("seed", seed)

    # Cross-validation on TRAIN ONLY
    cv_hist = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=max_rounds,
        nfold=nfold,
        early_stopping_rounds=early_stopping_rounds,
        metrics=("rmse",),
        verbose_eval=verbose_eval,
        seed=seed
    )

    best_rounds = len(cv_hist)  # early-stopped rounds
    best_cv_rmse = float(cv_hist["test-rmse-mean"].iloc[-1])

    # Train final model on ALL dtrain with best_rounds
    final_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=best_rounds,
        verbose_eval=False
    )

    # Predict
    if is_interpolated:
        preds = final_model.predict(interpolated_test_Dmatrix_df)
        return {
            "model": final_model,
            "prediction": preds,
            "best_rounds": best_rounds,
            "cv_rmse": best_cv_rmse
        }
    else:
        preds = final_model.predict(dtest)
        test_rmse = float(np.sqrt(mean_squared_error(test_labels, preds)))
        # Guard against zero range
        denom = (np.max(test_labels) - np.min(test_labels))
        test_nrmse = float(test_rmse / denom) if denom != 0 else np.nan

        if verbose_eval:
            print(f"Best rounds from CV: {best_rounds}")
            print(f"CV RMSE (6-fold):   {best_cv_rmse:.6f}")
            print(f"Final Test RMSE:    {test_rmse:.6f}")
            print(f"Final Test NRMSE:   {test_nrmse:.6f}")

        return {
            "model": final_model,
            "prediction": preds,
            "rmse": test_rmse,
            "nrmse": test_nrmse,
            "best_rounds": best_rounds,
            "cv_rmse": best_cv_rmse
        }


# Model prediction function with training and testing
# ------------------------------
def model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated, train_labels_full, test_labels_full, data_model_test_full, tap_number, is_interpolated, interpolated_test_dataset, 
                  XPos, YPos, ZPos):
    """
    Builds feature matrices based on tap_number and which position keys to include,
    loads best params, trains 5 seeds with early stopping on a validation split
    carved from TRAIN, evaluates on TEST (Iter_1), saves outputs if not interpolated.
    """
    # Position codes:
    if XPos and YPos and ZPos:
        position_code = "XPos_YPos_ZPos"
    elif XPos and YPos:
        position_code = "XPos_YPos"
    elif XPos and ZPos:
        position_code = "XPos_ZPos"
    elif YPos and ZPos:
        position_code = "YPos_ZPos"
    elif XPos:
        position_code = "XPos"
    elif YPos:
        position_code = "YPos"
    elif ZPos:
        position_code = "ZPos"
    else:
        position_code = "None"
    
    # Common columns to drop
    common_cols_to_drop = ["AoA", "delta_L_D_mean"]
    common_cols_to_drop_interpolated = ["AoA"]

    if not XPos:
        common_cols_to_drop.append("XPos")
        common_cols_to_drop_interpolated.append("XPos")
    if not YPos:
        common_cols_to_drop.append("YPos")
        common_cols_to_drop_interpolated.append("YPos")
    if not ZPos:
        common_cols_to_drop.append("ZPos")
        common_cols_to_drop_interpolated.append("ZPos")

    if tap_number == 12:
        cols_to_drop_twelve = common_cols_to_drop
        train_matrix_twelve = data_model_train_full.drop(columns=cols_to_drop_twelve).values
        test_matrix_twelve = data_model_test_full.drop(columns=cols_to_drop_twelve).values

        cols_to_drop_twelve_interpolated = common_cols_to_drop_interpolated
        interpolated_test_dataset_twelve = interpolated_test_dataset.drop(columns=cols_to_drop_twelve_interpolated).values

        dtrain = xgb.DMatrix(train_matrix_twelve, label=train_labels_full)
        dtest = xgb.DMatrix(test_matrix_twelve, label=test_labels_full)
        interpolated_test_Dmatrix_df = xgb.DMatrix(interpolated_test_dataset_twelve)

    elif tap_number == 6:
        cols_to_drop_six = [
            "tap2_delta_pressure_mean", "tap6_delta_pressure_mean",
            "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap17_delta_pressure_mean"
        ] + common_cols_to_drop
        train_matrix_six = data_model_train_full.drop(columns=cols_to_drop_six).values
        test_matrix_six = data_model_test_full.drop(columns=cols_to_drop_six).values

        cols_to_drop_six_interpolated = [
            "tap2_delta_pressure_mean", "tap6_delta_pressure_mean",
            "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap17_delta_pressure_mean"
        ] + common_cols_to_drop_interpolated
        interpolated_test_dataset_six = interpolated_test_dataset.drop(columns=cols_to_drop_six_interpolated).values
            
        dtrain = xgb.DMatrix(train_matrix_six, label=train_labels_full)
        dtest = xgb.DMatrix(test_matrix_six, label=test_labels_full)
        interpolated_test_Dmatrix_df = xgb.DMatrix(interpolated_test_dataset_six)

    elif tap_number == 2:
        cols_to_drop_two = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap15_delta_pressure_mean", "tap17_delta_pressure_mean",
            "tap19_delta_pressure_mean"
        ] + common_cols_to_drop
        train_matrix_two = data_model_train_full.drop(columns=cols_to_drop_two).values
        test_matrix_two = data_model_test_full.drop(columns=cols_to_drop_two).values

        cols_to_drop_two_interpolated = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap11_delta_pressure_mean",
            "tap13_delta_pressure_mean", "tap15_delta_pressure_mean", "tap17_delta_pressure_mean",
            "tap19_delta_pressure_mean"
        ] + common_cols_to_drop_interpolated
        interpolated_test_dataset_two = interpolated_test_dataset.drop(columns=cols_to_drop_two_interpolated).values

        dtrain = xgb.DMatrix(train_matrix_two, label=train_labels_full)
        dtest = xgb.DMatrix(test_matrix_two, label=test_labels_full)
        interpolated_test_Dmatrix_df = xgb.DMatrix(interpolated_test_dataset_two)

    elif tap_number == 0:
        cols_to_drop_zero = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap9_delta_pressure_mean",
            "tap10_delta_pressure_mean", "tap11_delta_pressure_mean", "tap13_delta_pressure_mean",
            "tap15_delta_pressure_mean", "tap17_delta_pressure_mean", "tap19_delta_pressure_mean"
        ] + common_cols_to_drop

        train_matrix_zero = data_model_train_full.drop(columns=cols_to_drop_zero).values
        test_matrix_zero = data_model_test_full.drop(columns=cols_to_drop_zero).values

        cols_to_drop_zero_interpolated = [
            "tap0_delta_pressure_mean", "tap2_delta_pressure_mean", "tap4_delta_pressure_mean",
            "tap6_delta_pressure_mean", "tap8_delta_pressure_mean", "tap9_delta_pressure_mean",
            "tap10_delta_pressure_mean", "tap11_delta_pressure_mean", "tap13_delta_pressure_mean",
            "tap15_delta_pressure_mean", "tap17_delta_pressure_mean", "tap19_delta_pressure_mean"
        ] + common_cols_to_drop_interpolated
        interpolated_test_dataset_zero = interpolated_test_dataset.drop(columns=cols_to_drop_zero_interpolated).values

        dtrain = xgb.DMatrix(train_matrix_zero, label=train_labels_full)
        dtest = xgb.DMatrix(test_matrix_zero, label=test_labels_full)
        interpolated_test_Dmatrix_df = xgb.DMatrix(interpolated_test_dataset_zero)

    # Load best parameters from CSV
    best_params = load_best_params(folder_path, tap_number, position_code)

    if is_interpolated == True:
        folder_path_to_save = folder_path_interpolated
        print(f"\n--- Training models with {tap_number} pressure taps ---")
        # Train 5 models with the same hyperparameters but different random seeds
        best_params["seed"] = 0
        model_1 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        best_params["seed"] = 1
        model_2 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        best_params["seed"] = 2
        model_3 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        best_params["seed"] = 3
        model_4 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        best_params["seed"] = 4
        model_5 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)
        
        # Create predictions DataFrame
        predicted_dataset = interpolated_test_dataset[[ "AoA"]].copy()
        predicted_dataset["Predicted_Delta_L_D_1"] = model_1["prediction"]
        predicted_dataset["Predicted_Delta_L_D_2"] = model_2["prediction"]
        predicted_dataset["Predicted_Delta_L_D_3"] = model_3["prediction"]
        predicted_dataset["Predicted_Delta_L_D_4"] = model_4["prediction"]
        predicted_dataset["Predicted_Delta_L_D_5"] = model_5["prediction"]
        print(f"Predictions for {tap_number} taps(interpolated); Xpos: {XPos} Ypos: {YPos}, Zpos: {ZPos} saved successfully!")

        return predicted_dataset
    else:
        folder_path_to_save = folder_path_evaluation
        print(f"\n--- Training models with {tap_number} pressure taps ---")
        # Train 5 models with the same hyperparameters but different random seeds
        best_params["seed"] = 0
        model_1 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        best_params["seed"] = 1
        model_2 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        best_params["seed"] = 2
        model_3 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        best_params["seed"] = 3
        model_4 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        best_params["seed"] = 4
        model_5 = train_model(dtrain, dtest, best_params, test_labels_full, is_interpolated, interpolated_test_Dmatrix_df)

        # Create predictions DataFrame
        predicted_dataset = data_model_test_full[["XPos", "YPos", "ZPos", "AoA"]].copy()
        predicted_dataset["Actual_Delta_L_D"] = test_labels_full
        predicted_dataset["Predicted_Delta_L_D_1"] = model_1["prediction"]
        predicted_dataset["Predicted_Delta_L_D_2"] = model_2["prediction"]
        predicted_dataset["Predicted_Delta_L_D_3"] = model_3["prediction"]
        predicted_dataset["Predicted_Delta_L_D_4"] = model_4["prediction"]
        predicted_dataset["Predicted_Delta_L_D_5"] = model_5["prediction"]
        predicted_dataset["Error_1"] = test_labels_full - model_1["prediction"]
        predicted_dataset["Error_2"] = test_labels_full - model_2["prediction"]
        predicted_dataset["Error_3"] = test_labels_full - model_3["prediction"]
        predicted_dataset["Error_4"] = test_labels_full - model_4["prediction"]
        predicted_dataset["Error_5"] = test_labels_full - model_5["prediction"]
        predicted_dataset["RMSE_1"] = model_1["rmse"]
        predicted_dataset["RMSE_2"] = model_2["rmse"]
        predicted_dataset["RMSE_3"] = model_3["rmse"]
        predicted_dataset["RMSE_4"] = model_4["rmse"]
        predicted_dataset["RMSE_5"] = model_5["rmse"]

        # Save to CSV
        predicted_dataset.to_csv(os.path.join(folder_path_to_save, f"Predicted_dataset_Pressure_{tap_number}_Position_{position_code}.csv"), index=False)

        print(f"Predictions for {tap_number} taps; Xpos: {XPos} Ypos: {YPos}, Zpos: {ZPos} saved successfully!")
        print(f"Predicted_dataset_Pressure_{tap_number}_Position_{position_code}.csv - saved successfully!")
        
        return predicted_dataset

def load_best_params(folder_path, tap_number, position_code):    
    """Builds the filename and returns parsed params."""
    # Construct file name
    if tap_number in [0, 2, 6, 12]:
        file_name = f"Bayes_results_XGB_HyperP_using_dP_for_delta_L_D/Bayes_history_for_hyperparameter_tuning_Pressure_{tap_number}_Position_{position_code}.csv"
    else:
        raise ValueError("tap_number must be one of [0, 2, 6, 12]")
    
    print(f"Loading: {file_name}")
    
    # Load best params from file
    return best_params_from_csv(folder_path, file_name)


if __name__ == "__main__":
    # Global:
    folder_path = "Predictive_model_data"
    folder_path_evaluation = f"{folder_path}/XGBoost_Model_evaluation"
    folder_path_interpolated = f"{folder_path}/XGBoost_model_predictions_on_interpolated_data"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(folder_path_evaluation):
        os.makedirs(folder_path_evaluation)

    # Optional folder for interpolated data predictions
    # if not os.path.exists(folder_path_interpolated):
    #     os.makedirs(folder_path_interpolated)

    # Data Preparation
    # ------------------------------
    data_model_test_full, data_model_train_full = data_prep()
    print("Data preparation completed successfully!")

    # Prepare Data for XGBoost 12 pressure taps
    # ------------------------------
    dtrain, dtest, train_labels_full, test_labels_full, data_model_test_full = xgbdata_prep(data_model_test_full, data_model_train_full)
    print("Data preparation for XGBoost completed successfully!")

    # load interpolated data
    pressure_interpolated_file_path = "Predictive_model_data/Interpolated_pressure_data/Pressure_True_Stats_Iter_1_Interpolated.csv"
    data_model_test_full_interpolated = pd.read_csv(pressure_interpolated_file_path)

    ## Model Training with Pressure Taps Only
    # ------------------------------
    # 12 Pressure Taps
    predicted_dataset_12_taps = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=False, 
                                              interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=False)

    # Interpolated (Optiaonal - Uncomment if needed)
    # predicted_dataset_12_taps_interpolated = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated, train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=True, 
    #                                                       interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=False)

    # 6 Pressure Taps
    predicted_dataset_6_taps = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=False, 
                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_6_taps_interpolated = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated, train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=True, 
    #                                                       interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=False)

    # 2 Pressure Taps
    predicted_dataset_2_taps = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=False, 
                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_2_taps_interpolated = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated, train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=True, 
    #                                                       interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=False)

    ## Model Training with Position only (zero pressure taps), Xpos = True, Ypos = True, Zpos = True
    # ------------------------------
    # 0 Pressure Taps
    predicted_dataset_0_taps = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=0, is_interpolated=False, 
                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=True, ZPos=True)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_0_taps_interpolated = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated, train_labels_full, test_labels_full, data_model_test_full, tap_number=0, is_interpolated=True, 
    #                                                       interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=True, ZPos=True)

    ## Model Training with preesures taps and XPos only, XPos = True, Ypos = False, Zpos = False
    # ------------------------------
    # 12 Pressure Taps
    predicted_dataset_12_taps_XPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=False, 
                                                   interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=False, ZPos=False)
    
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_12_taps_interpolated_XPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=True, 
    #                                                              interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=False, ZPos=False)

    # 6 Pressure Taps
    predicted_dataset_6_taps_XPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=False, 
                                                  interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=False, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_6_taps_interpolated_XPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=True, 
    #                                                              interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=False, ZPos=False)  

    # 2 Pressure Taps
    predicted_dataset_2_taps_XPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=False, 
                                                  interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=False, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_2_taps_interpolated_XPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=False, ZPos=False)

    # 0 Pressure Taps
    predicted_dataset_0_taps_XPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=0, is_interpolated=False,
                                                   interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=False, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    #predicted_dataset_0_taps_interpolated_XPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=0, is_interpolated=True,
    #                                                         interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=False, ZPos=False)

    ## Model Training with preesures taps and YPos only, XPos = False, Ypos = True, Zpos = False
    # ------------------------------
    # 12 Pressure Taps
    predicted_dataset_12_taps_YPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=False, 
                                                   interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=True, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_12_taps_interpolated_YPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=True, ZPos=False)

    # 6 Pressure Taps
    predicted_dataset_6_taps_YPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=False, 
                                                  interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=True, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_6_taps_interpolated_YPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=True, ZPos=False)

    # 2 Pressure Taps
    predicted_dataset_2_taps_YPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=False, 
                                                  interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=True, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_2_taps_interpolated_YPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=True, ZPos=False)

    # 0 Pressure Taps
    predicted_dataset_0_taps_YPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=0, is_interpolated=False,
                                                    interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=True, ZPos=False)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_0_taps_interpolated_YPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=0, is_interpolated=True,
    #                                                              interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=True, ZPos=False)

    ## Model Training with preesures taps and ZPos only, XPos = False, Ypos = False, Zpos = True
    # ------------------------------
    # 12 Pressure Taps
    predicted_dataset_12_taps_ZPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=False, 
                                                   interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=True)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_12_taps_interpolated_ZPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=True, 
    #                                                            interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=True)

    # 6 Pressure Taps
    predicted_dataset_6_taps_ZPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=False, 
                                                  interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=True)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_6_taps_interpolated_ZPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=True)

    # 2 Pressure Taps
    predicted_dataset_2_taps_ZPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=False, 
                                                  interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=True)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_2_taps_interpolated_ZPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=True)

    # 0 Pressure Taps
    predicted_dataset_0_taps_ZPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=0, is_interpolated=False,
                                                    interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=True)

    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_0_taps_interpolated_ZPos = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=0, is_interpolated=True,
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=False, YPos=False, ZPos=True)

    ## Model Training with preesures taps and XPos, YPos, ZPos
    # ------------------------------
    # 12 Pressure Taps
    predicted_dataset_12_taps_XYZ = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=False, 
                                                  interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=True, ZPos=True)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_12_taps_interpolated_XYZ = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=12, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=True, ZPos=True)

    # 6 Pressure Taps
    predicted_dataset_6_taps_XYZ = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=False, 
                                                 interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=True, ZPos=True)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_6_taps_interpolated_XYZ = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=6, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=True, ZPos=True)

    # 2 Pressure Taps
    predicted_dataset_2_taps_XYZ = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=False, 
                                                 interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=True, ZPos=True)
    # Interpolated (Optional - Uncomment if needed)
    # predicted_dataset_2_taps_interpolated_XYZ = model_summary(dtrain, dtest, folder_path, folder_path_evaluation, folder_path_interpolated,  train_labels_full, test_labels_full, data_model_test_full, tap_number=2, is_interpolated=True, 
    #                                                             interpolated_test_dataset=data_model_test_full_interpolated, XPos=True, YPos=True, ZPos=True)

    ## Model Training with pressures taps and XPos, YPos, ZPos
    print("All models trained and predictions saved successfully!")