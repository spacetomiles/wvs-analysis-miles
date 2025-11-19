import pandas as pd

FILE_PATH = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_cleaned.csv'

try:
    print(f"Loading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH, usecols=['Q49'])
    
    print("\n--- Q49 Analysis ---")
    print("Unique values:", sorted(df['Q49'].dropna().unique()))
    print("\nValue Counts:")
    print(df['Q49'].value_counts().sort_index())
    print(f"\nMissing values in Q49: {df['Q49'].isna().sum()}")

except Exception as e:
    print(f"Error: {e}")
