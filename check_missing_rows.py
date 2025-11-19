import pandas as pd

FILE_PATH = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_cleaned.csv'

try:
    print(f"Loading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    total_rows = len(df)
    rows_with_na = df.isna().any(axis=1).sum()
    proportion = rows_with_na / total_rows
    
    print(f"Total Rows: {total_rows}")
    print(f"Rows with at least one NA: {rows_with_na}")
    print(f"Proportion: {proportion:.4f} ({proportion*100:.2f}%)")

except Exception as e:
    print(f"Error: {e}")
