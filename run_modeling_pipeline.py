import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Configuration
INPUT_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_cleaned.csv'
IMPUTED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_imputed.csv'
TOP_N_FEATURES = 20 # Number of features to select from AdaBoost for the final model

def run_pipeline():
    print("--- Starting Modeling Pipeline ---")
    
    # 1. Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # Drop rows where target (Q49) is missing
    initial_len = len(df)
    df = df.dropna(subset=['Q49'])
    print(f"Dropped {initial_len - len(df)} rows with missing target Q49.")
    
    # Separate X and y
    y = df['Q49']
    X = df.drop(columns=['Q49'])
    
    # 2. Imputation
    print("\n--- Step 1: Imputation (Median) ---")
    print("Imputing missing values in predictors...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Save imputed data
    print(f"Saving imputed data to {IMPUTED_FILE}...")
    # Recombine with y for saving
    df_imputed = pd.concat([X_imputed, y.reset_index(drop=True)], axis=1)
    df_imputed.to_csv(IMPUTED_FILE, index=False)
    
    # 3. Feature Selection (AdaBoost)
    print("\n--- Step 2: Feature Selection (AdaBoost) ---")
    print("Training AdaBoostClassifier to rank features...")
    
    # Use a simple decision tree as base estimator
    base_estimator = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    
    # Fit AdaBoost
    ada.fit(X_imputed, y)
    
    # Get feature importances
    importances = ada.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Features from AdaBoost:")
    print(feature_imp_df.head(10))
    
    # Select top N features
    top_features = feature_imp_df.head(TOP_N_FEATURES)['Feature'].tolist()
    print(f"\nSelected top {TOP_N_FEATURES} features for final model: {top_features}")
    
    # 4. Ordinal Logistic Regression
    print("\n--- Step 3: Ordinal Logistic Regression ---")
    print("Running Ordinal Logistic Regression on selected features...")
    
    X_final = X_imputed[top_features]
    
    # Standardize features for better convergence/interpretation
    scaler = StandardScaler()
    X_final_scaled = pd.DataFrame(scaler.fit_transform(X_final), columns=X_final.columns)
    
    # Ensure y is integer type for statsmodels
    y_final = y.reset_index(drop=True).astype(int)
    
    try:
        # Fit Ordered Model (Proportional Odds)
        # dist='logit' makes it Ordinal Logistic Regression
        mod = OrderedModel(y_final, X_final_scaled, distr='logit')
        res = mod.fit(method='bfgs')
        
        print("\n--- Model Summary ---")
        print(res.summary())
        
        # Extract significant features (p < 0.05)
        print("\n--- Significant Predictors (p < 0.05) ---")
        p_values = res.pvalues
        significant_features = p_values[p_values < 0.05].index.tolist()
        # Filter out the threshold parameters (which are named like 1.0/2.0)
        significant_predictors = [f for f in significant_features if f in top_features]
        
        if significant_predictors:
            print(significant_predictors)
        else:
            print("No predictors found to be statistically significant at p < 0.05.")
            
    except Exception as e:
        print(f"Error running Ordinal Model: {e}")
        print("Attempting fallback to standard Logistic Regression (Multinomial)...")
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        lr.fit(X_final_scaled, y_final)
        print("Logistic Regression (Multinomial) trained successfully.")
        print("Coefficients shape:", lr.coef_.shape)

if __name__ == "__main__":
    run_pipeline()
