import pandas as pd
import numpy as np
import os

CLEANED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_cleaned.csv'

def verify_data():
    print(f"Verifying {CLEANED_FILE}...")
    
    if not os.path.exists(CLEANED_FILE):
        print("ERROR: Cleaned file not found!")
        return
        
    try:
        df = pd.read_csv(CLEANED_FILE)
        print(f"Loaded cleaned data. Shape: {df.shape}")
        
        # 1. Check Columns
        cols = df.columns.tolist()
        print(f"Number of columns: {len(cols)}")
        print(f"First column: {cols[0]}, Last column: {cols[-1]}")
        
        # Verify Q1 to Q290 are present (or at least the range looks right)
        # We expect 290 columns if none were dropped as empty
        if len(cols) != 290:
            print(f"WARNING: Expected 290 columns, found {len(cols)}")
        
        # 2. Check for Negative Values
        numeric_cols = df.select_dtypes(include=['number']).columns
        n_negatives = (df[numeric_cols] < 0).sum().sum()
        if n_negatives == 0:
            print("PASS: No negative values found.")
        else:
            print(f"FAIL: Found {n_negatives} negative values!")
            
        # 3. Check for Duplicates
        n_dupes = df.duplicated().sum()
        if n_dupes == 0:
            print("PASS: No duplicate rows found.")
        else:
            print(f"FAIL: Found {n_dupes} duplicate rows!")
            
        # 4. Check Missing Values
        n_missing = df.isna().sum().sum()
        print(f"Total missing values (NaN): {n_missing}")
        print(f"Percentage of data missing: {n_missing / (df.shape[0] * df.shape[1]) * 100:.2f}%")
        
        print("\nVerification Complete.")
        
    except Exception as e:
        print(f"Verification failed with error: {e}")

if __name__ == "__main__":
    verify_data()
