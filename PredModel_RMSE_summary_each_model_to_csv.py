import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
def overall_best_RMSE(folder_path, tap_number, XPos, YPos, ZPos):
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

    # Load the dataset
    predicted_dataset = pd.read_csv(os.path.join(folder_path, f"Predicted_dataset_Pressure_{tap_number}_Position_{position_code}.csv"))

    rmse_cols = ["RMSE_1", "RMSE_2", "RMSE_3", "RMSE_4", "RMSE_5"]
    first_row_rmse = predicted_dataset.loc[predicted_dataset.index[0], rmse_cols]
    best_rmse_value = first_row_rmse.min()

    best_model = first_row_rmse.idxmin()
    return best_rmse_value, best_model

def create_plotting_matrix(folder_path):
    # Plotting the minimum RSME of each model evaluated in previous
    # ------------------------------
    # Start make matrix for heatmap
    plotting_matrix = np.zeros((4, 5))
    # Fourth row: None pressure taps : None, XPos, YPos, ZPos, XYZ
    plotting_matrix[3, 0] = 0
    plotting_matrix[3, 1] = overall_best_RMSE(folder_path, 0, XPos= True, YPos= False, ZPos= False)[0]
    plotting_matrix[3, 2] = overall_best_RMSE(folder_path, 0, XPos= False, YPos= True, ZPos= False)[0]
    plotting_matrix[3, 3] = overall_best_RMSE(folder_path, 0, XPos= False, YPos= False, ZPos= True)[0]
    plotting_matrix[3, 4] = overall_best_RMSE(folder_path, 0, XPos= True, YPos= True, ZPos= True)[0]

    # Third row: 2 pressure taps : None, XPos, YPos, ZPos, XYZ
    plotting_matrix[2, 0] = overall_best_RMSE(folder_path, 2, XPos= False, YPos= False, ZPos= False)[0]
    plotting_matrix[2, 1] = overall_best_RMSE(folder_path, 2, XPos= True, YPos= False, ZPos= False)[0]
    plotting_matrix[2, 2] = overall_best_RMSE(folder_path, 2, XPos= False, YPos= True, ZPos= False)[0]
    plotting_matrix[2, 3] = overall_best_RMSE(folder_path, 2, XPos= False, YPos= False, ZPos= True)[0]
    plotting_matrix[2, 4] = overall_best_RMSE(folder_path, 2, XPos= True, YPos= True, ZPos= True)[0]

    # Second row: 6 pressure taps : None, XPos, YPos, ZPos, XYZ
    plotting_matrix[1, 0] = overall_best_RMSE(folder_path, 6, XPos= False, YPos= False, ZPos= False)[0]
    plotting_matrix[1, 1] = overall_best_RMSE(folder_path, 6, XPos= True, YPos= False, ZPos= False)[0]
    plotting_matrix[1, 2] = overall_best_RMSE(folder_path, 6, XPos= False, YPos= True, ZPos= False)[0]
    plotting_matrix[1, 3] = overall_best_RMSE(folder_path, 6, XPos= False, YPos= False, ZPos= True)[0]
    plotting_matrix[1, 4] = overall_best_RMSE(folder_path, 6, XPos= True, YPos= True, ZPos= True)[0]

    # First row: 12 pressure taps : None, XPos, YPos, ZPos, XYZ
    plotting_matrix[0, 0] = overall_best_RMSE(folder_path, 12, XPos= False, YPos= False, ZPos= False)[0]
    plotting_matrix[0, 1] = overall_best_RMSE(folder_path, 12, XPos= True, YPos= False, ZPos= False)[0]
    plotting_matrix[0, 2] = overall_best_RMSE(folder_path, 12, XPos= False, YPos= True, ZPos= False)[0]
    plotting_matrix[0, 3] = overall_best_RMSE(folder_path, 12, XPos= False, YPos= False, ZPos= True)[0]
    plotting_matrix[0, 4] = overall_best_RMSE(folder_path, 12, XPos= True, YPos= True, ZPos= True)[0]
    
    plotting_matrix_df = pd.DataFrame(plotting_matrix)
    plotting_matrix_df.columns = ["None", "XPos", "YPos", "ZPos", "XPos_YPos_ZPos"]
    plotting_matrix_df.index = ["12", "6", "2", "None"]
    return plotting_matrix_df

if __name__ == "__main__":
    folder_path = "Predictive_model_data/XGBoost_Model_evaluation"
    plotting_matrix_df = create_plotting_matrix(folder_path)
    # save the plotting matrix to a CSV file
    summary_folder = os.path.join(folder_path, "Summary")
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)
    plotting_matrix_df.to_csv(os.path.join(summary_folder, "RMSE_matrix.csv"), index=True)
    print("RMSE matrix saved to CSV.")



    
