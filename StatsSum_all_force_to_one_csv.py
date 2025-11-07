import numpy as np
import pandas as pd
import os

def get_info_from_csvname(csv_file):
    # Extract information from file name
    file_name = os.path.basename(csv_file)
    file_info = file_name.split('_')  # Split file name by underscores

    ypos = file_info[2]  
    zpos = file_info[4]  

    if file_info[-8] == '2' and file_info[-7] == '5':
        AoA = '2.5'
    else:
        AoA = file_info[-7]

    return ypos, zpos, AoA

def process_csv_files(testdate, folder_path, AoA, xpos):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(f'{testdate}_truestats.csv')]
    data = []
    columns = ['XPos', 'YPos', 'ZPos', 'AoA']

    chord_length = 11 # inches
    xpos = (int(xpos)+1) * chord_length * 0.75  # Convert to inches
    
    for csv_file in csv_files:
        ypos, zpos, AoA_value = get_info_from_csvname(csv_file)
        ypos = int(ypos) * chord_length * 0.25  # Convert to inches
        zpos = int(zpos) * chord_length * 0.25 
        
        truestat_df = pd.read_csv(folder_path + csv_file)
        # Combine static columns with CSV columns
        if len(columns) == 4:  # Only update once
            columns.extend(truestat_df.columns)
        
        for row in truestat_df.itertuples(index=False):
            data.append([xpos, ypos, zpos, AoA_value, *row])
    
    new_df = pd.DataFrame(data, columns=columns)
    new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')]
    
    return new_df

if __name__ == "__main__":
    AoA = ['0', '2_5', '5']
    for Iter in ['0', '1']:
        for AoA_str in AoA:
            # Create a new folder called 
            truestats_folder = 'Post_processed_stats_summarized'
            os.makedirs(truestats_folder, exist_ok=True)

            for i in range(len(AoA)):
                AoA_str = str(AoA[i])
                if AoA_str == '0':
                    X_0_Test_date_1 = '2024_04_04'
                    X_0_Test_date_2 = '2024_04_04'
                    X_1_Test_date_1 = '2024_04_03'
                    X_1_Test_date_2 = '2024_04_04'
                    X_2_Test_date_1 = '2024_04_03'
                    X_2_Test_date_2 = '2024_04_04'
                    if Iter == '0':
                        X_0_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_0/Iter_0_{X_0_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_1, X_0_folder_path_Iter_0, AoA_str, xpos = "0")

                        X_1_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_1/Iter_0_{X_1_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_1, X_1_folder_path_Iter_0, AoA_str, xpos = "1")

                        X_2_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_2/Iter_0_{X_2_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_1, X_2_folder_path_Iter_0, AoA_str, xpos = "2")

                        new_df_X_0.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)

                    elif Iter == '1':
                        X_0_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_0/Iter_1_{X_0_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_2, X_0_folder_path_Iter_1, AoA_str, xpos = "0")

                        X_1_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_1/Iter_1_{X_1_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_2, X_1_folder_path_Iter_1, AoA_str, xpos = "1")

                        X_2_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_2/Iter_1_{X_2_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_2, X_2_folder_path_Iter_1, AoA_str, xpos = "2")

                        new_df_X_0.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        
                elif AoA_str == '2_5':
                    X_0_Test_date_1 = '2024_04_05'
                    X_0_Test_date_2 = '2024_04_08'
                    X_1_Test_date_1 = '2024_04_05'
                    X_1_Test_date_2 = '2024_04_08'
                    X_2_Test_date_1 = '2024_04_05'
                    X_2_Test_date_2 = '2024_04_09'
                    if Iter == '0':
                        X_0_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_0/Iter_0_{X_0_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_1, X_0_folder_path_Iter_0, AoA_str, xpos = "0")

                        X_1_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_1/Iter_0_{X_1_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_1, X_1_folder_path_Iter_0, AoA_str, xpos = "1")

                        X_2_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_2/Iter_0_{X_2_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_1, X_2_folder_path_Iter_0, AoA_str, xpos = "2")

                        new_df_X_0.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)

                    elif Iter == '1':
                        X_0_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_0/Iter_1_{X_0_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_2, X_0_folder_path_Iter_1, AoA_str, xpos = "0")

                        X_1_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_1/Iter_1_{X_1_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_2, X_1_folder_path_Iter_1, AoA_str, xpos = "1")

                        X_2_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_2/Iter_1_{X_2_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_2, X_2_folder_path_Iter_1, AoA_str, xpos = "2")

                        new_df_X_0.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)

                elif AoA_str == '5':
                    X_0_Test_date_1 = '2024_04_10'
                    X_0_Test_date_2 = '2024_04_10'
                    X_1_Test_date_1 = '2024_04_10'
                    X_1_Test_date_2 = '2024_04_10'
                    X_2_Test_date_1 = '2024_04_09'
                    X_2_Test_date_2 = '2024_04_10'
                    if Iter == '0':
                        X_0_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_0/Iter_0_{X_0_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_1, X_0_folder_path_Iter_0, AoA_str, xpos = "0")

                        X_1_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_1/Iter_0_{X_1_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_1, X_1_folder_path_Iter_0, AoA_str, xpos = "1")

                        X_2_folder_path_Iter_0 = f'Post_Data/Position/Tared/Force/X_2/Iter_0_{X_2_Test_date_1}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_1, X_2_folder_path_Iter_0, AoA_str, xpos = "2")

                        new_df_X_0.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)

                    elif Iter == '1':
                        X_0_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_0/Iter_1_{X_0_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_0 = process_csv_files(X_0_Test_date_2, X_0_folder_path_Iter_1, AoA_str, xpos = "0")

                        X_1_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_1/Iter_1_{X_1_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_1 = process_csv_files(X_1_Test_date_2, X_1_folder_path_Iter_1, AoA_str, xpos = "1")

                        X_2_folder_path_Iter_1 = f'Post_Data/Position/Tared/Force/X_2/Iter_1_{X_2_Test_date_2}/AoA_{AoA_str}/'
                        new_df_X_2 = process_csv_files(X_2_Test_date_2, X_2_folder_path_Iter_1, AoA_str, xpos = "2")

                        new_df_X_0.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_1.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                        new_df_X_2.to_csv(f'{truestats_folder}/L_D_True_Stats_Iter_{Iter}.csv', mode='a', header=False, index=False)
                
                print(f'Completed processing for AoA {AoA_str} in Iteration {Iter}.')
        
    print('All CSV files have been processed and combined successfully into L_D_True_Stats files.')







