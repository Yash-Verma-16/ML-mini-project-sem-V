# merge_data.py (v3 - Handles both .xls and .xlsx)
#
# This script automates the process of merging historical tennis match data
# from two different sources:
# 1. Jeff Sackmann's detailed match stats (e.g., atp_matches_2001.csv)
# 2. Tennis-Data.co.uk's betting odds data (e.g., 2001.xls or 2013.xlsx)
#
# The script performs the following key steps:
# - Intelligently finds the correct betting data file (.xls or .xlsx).
# - Loops through a defined range of years for both training and testing sets.
# - Loads the corresponding data files for each year.
# - Standardizes the player names and date formats to create a common merge key.
# - Merges the two datasets for each year based on this key.
# - Concatenates all the yearly data into two final master files:
#   - training_data.csv
#   - testing_data.csv

import pandas as pd
import os
from pandas.api.types import is_numeric_dtype

def standardize_player_name(name):
    """
    Converts a full name (e.g., "Lleyton Hewitt") to the format used in
    the betting data (e.g., "Hewitt L."). This is crucial for merging.
    Handles single-word names gracefully.
    """
    if not isinstance(name, str):
        return name
    parts = name.split()
    if len(parts) > 1:
        last_name = parts[-1]
        first_initial = parts[0][0]
        return f"{last_name} {first_initial}."
    return name

def merge_year_data(sackmann_path, betting_path):
    """
    Loads, cleans, and merges the data for a single year from both sources.
    """
    try:
        df_sackmann = pd.read_csv(sackmann_path)
        df_sackmann['tourney_date'] = pd.to_datetime(df_sackmann['tourney_date'], format='%Y%m%d')
        df_sackmann['winner_name_std'] = df_sackmann['winner_name'].apply(standardize_player_name)
        df_sackmann['loser_name_std'] = df_sackmann['loser_name'].apply(standardize_player_name)
        df_sackmann['merge_key'] = list(zip(df_sackmann['tourney_date'], df_sackmann['winner_name_std'], df_sackmann['loser_name_std']))

        df_betting = pd.read_excel(betting_path)
        
        if is_numeric_dtype(df_betting['Date']):
            df_betting['Date'] = pd.to_datetime(df_betting['Date'], unit='D', origin='1899-12-30')
        else:
            df_betting['Date'] = pd.to_datetime(df_betting['Date'])
        
        df_betting['Date'] = df_betting['Date'].dt.normalize()
        df_betting['Winner'] = df_betting['Winner'].str.strip()
        df_betting['Loser'] = df_betting['Loser'].str.strip()
        df_betting['merge_key'] = list(zip(df_betting['Date'], df_betting['Winner'], df_betting['Loser']))

        df_sackmann['merge_key_alt'] = list(zip(df_sackmann['tourney_date'] - pd.Timedelta(days=1), df_sackmann['winner_name_std'], df_sackmann['loser_name_std']))

        df_merged = pd.merge(df_sackmann, df_betting, on='merge_key', how='inner')
        
        if len(df_merged) < len(df_sackmann) * 0.7:
            unmatched_sackmann = df_sackmann[~df_sackmann['merge_key'].isin(df_merged['merge_key'])]
            alt_merged = pd.merge(unmatched_sackmann, df_betting, left_on='merge_key_alt', right_on='merge_key', how='inner')
            df_merged = pd.concat([df_merged, alt_merged], ignore_index=True)

        df_merged = df_merged.drop(columns=['merge_key', 'winner_name_std', 'loser_name_std', 'merge_key_alt'], errors='ignore')
        
        print(f"  - Successfully loaded and merged. Found {len(df_merged)} matches.")
        return df_merged

    except FileNotFoundError:
        print(f"  - Warning: Could not find one or both data files. Skipping.")
        return None
    except Exception as e:
        print(f"  - Error processing files: {e}. Skipping.")
        return None

if __name__ == '__main__':
    SACKMANN_DATA_DIR = 'sackmann_data/'
    BETTING_DATA_DIR = 'betting_data/'
    os.makedirs(SACKMANN_DATA_DIR, exist_ok=True)
    os.makedirs(BETTING_DATA_DIR, exist_ok=True)

    TRAINING_YEARS = range(2001, 2016)
    TESTING_YEARS = range(2016, 2018)

    all_training_data = []
    all_testing_data = []

    print("--- Starting Data Merging Process (v3) ---")

    print(f"\nProcessing training years: {TRAINING_YEARS.start}-{TRAINING_YEARS.stop-1}...")
    for year in TRAINING_YEARS:
        print(f"- Processing {year}...")
        sackmann_file = os.path.join(SACKMANN_DATA_DIR, f'atp_matches_{year}.csv')
        
        # --- NEW LOGIC: Check for .xlsx first, then fall back to .xls ---
        betting_file_xlsx = os.path.join(BETTING_DATA_DIR, f'{year}.xlsx')
        betting_file_xls = os.path.join(BETTING_DATA_DIR, f'{year}.xls')
        
        if os.path.exists(betting_file_xlsx):
            betting_file = betting_file_xlsx
        else:
            betting_file = betting_file_xls
        
        merged_df = merge_year_data(sackmann_file, betting_file)
        if merged_df is not None and not merged_df.empty:
            all_training_data.append(merged_df)

    print(f"\nProcessing testing years: {TESTING_YEARS.start}-{TESTING_YEARS.stop-1}...")
    for year in TESTING_YEARS:
        print(f"- Processing {year}...")
        sackmann_file = os.path.join(SACKMANN_DATA_DIR, f'atp_matches_{year}.csv')
        
        # --- NEW LOGIC: Check for .xlsx first, then fall back to .xls ---
        betting_file_xlsx = os.path.join(BETTING_DATA_DIR, f'{year}.xlsx')
        betting_file_xls = os.path.join(BETTING_DATA_DIR, f'{year}.xls')

        if os.path.exists(betting_file_xlsx):
            betting_file = betting_file_xlsx
        else:
            betting_file = betting_file_xls
            
        merged_df = merge_year_data(sackmann_file, betting_file)
        if merged_df is not None and not merged_df.empty:
            all_testing_data.append(merged_df)

    if all_training_data:
        final_training_df = pd.concat(all_training_data, ignore_index=True)
        final_training_df.to_csv('training_data.csv', index=False)
        print(f"\nSuccessfully created 'training_data.csv' with {len(final_training_df)} merged matches.")
    else:
        print("\nNo training data was merged. Please check your file paths and data.")

    if all_testing_data:
        final_testing_df = pd.concat(all_testing_data, ignore_index=True)
        final_testing_df.to_csv('testing_data.csv', index=False)
        print(f"Successfully created 'testing_data.csv' with {len(final_testing_df)} merged matches.")
    else:
        print("No testing data was merged. Please check your file paths and data.")

    print("\n--- Data Merging Process Complete ---")

