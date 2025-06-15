import pandas as pd
import os

# IMPORTANT: Set this path to your processed training data CSV file
PROCESSED_TRAIN_DATA_PATH ='processed_train_data.csv'
# For example: 'C:/Users/User/OneDrive/Desktop/Terra net Training/processed_train_data.csv'

try:
    processed_df = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)
    
    # Exclude the 'class' column as it's the target, not a feature
    feature_names = processed_df.drop('class', axis=1).columns.tolist()

    print("Copy and paste the following list EXACTLY into IDS_FEATURE_NAMES in network_dashboard.py:")
    print("-" * 80)
    print(feature_names)
    print("-" * 80)
    print(f"\nThis list contains {len(feature_names)} features.")

except FileNotFoundError:
    print(f"Error: '{PROCESSED_TRAIN_DATA_PATH}' not found.")
    print("Please ensure the path is correct and the file exists.")
except Exception as e:
    print(f"An error occurred: {e}")

