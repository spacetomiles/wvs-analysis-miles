import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from statsmodels.miscmodels.ordinal_model import OrderedModel
import os

# --- Configuration ---
RAW_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_csv_v6_0.xlsx'
CLEANED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_Cross-National_Wave_7_cleaned.csv'
IMPUTED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_imputed.csv'
ROC_PLOT_BINARY = '/Users/a11/Downloads/HKU/7005/7005 group project/roc_curve_ordinal_binary.svg'
ROC_PLOT_MULTI = '/Users/a11/Downloads/HKU/7005/7005 group project/roc_curve_ordinal_multiclass.svg'
FEATURE_PLOT_ADA = '/Users/a11/Downloads/HKU/7005/7005 group project/feature_importance_adaboost.svg'
FEATURE_PLOT_ORDINAL = '/Users/a11/Downloads/HKU/7005/7005 group project/feature_importance_ordinal.svg'
REGRESSION_TABLE_CSV = '/Users/a11/Downloads/HKU/7005/7005 group project/regression_coefficients.csv'

# Selected Features for Final Model
SELECTED_FEATURES = [
    'Q50', 'Q48', 'Q46', 'Q164', 'Q112', 'Q110', 'Q120', 'Q55', 'Q47', 'Q213'
]

def main():
    print("========================================================")
    print("   WVS ANALYSIS MASTER PIPELINE")
    print("========================================================")

    # ---------------------------------------------------------
    # 1. Data Cleaning
    # ---------------------------------------------------------
    print("\n[Step 1] Data Cleaning...")
    if os.path.exists(CLEANED_FILE):
        print(f"Cleaned file found at {CLEANED_FILE}. Loading...")
        df = pd.read_csv(CLEANED_FILE)
    else:
        print(f"Loading raw data from {RAW_FILE} (This may take a moment)...")
        try:
            df = pd.read_excel(RAW_FILE)
        except FileNotFoundError:
            print("Error: Raw file not found!")
            return

        print(f"Original shape: {df.shape}")
        
        # Filter Columns Q1-Q290
        cols_to_keep = [c for c in df.columns if c.startswith('Q') and c[1:].isdigit() and 1 <= int(c[1:]) <= 290]
        df = df[cols_to_keep]
        print(f"Filtered to {len(df.columns)} columns (Q1-Q290).")
        
        # Handle Negative Values
        print("Replacing negative values with NaN...")
        num_neg = (df.select_dtypes(include=np.number) < 0).sum().sum()
        print(f"Found {num_neg} negative values.")
        df[df < 0] = np.nan
        
        # Deduplicate
        print("Removing duplicates...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_rows - len(df)} duplicate rows.")
        
        # Save
        df.to_csv(CLEANED_FILE, index=False)
        print(f"Saved cleaned data to {CLEANED_FILE}")

    # ---------------------------------------------------------
    # 2. Imputation
    # ---------------------------------------------------------
    print("\n[Step 2] Imputation...")
    # Drop rows where target Q49 is missing
    df = df.dropna(subset=['Q49']).reset_index(drop=True)
    y = df['Q49'].astype(int)
    X = df.drop(columns=['Q49'])
    
    print("Imputing missing predictors with Median...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # ---------------------------------------------------------
    # 3. Feature Selection (AdaBoost)
    # ---------------------------------------------------------
    print("\n[Step 3] Feature Selection (AdaBoost)...")
    print("Training AdaBoostClassifier to rank features...")
    base_estimator = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    ada.fit(X_imputed, y)
    
    importances = ada.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    print("Top 10 Features:")
    print(feature_imp_df.head(10).to_string(index=False))
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    top_plot = feature_imp_df.head(20).sort_values(by='Importance', ascending=True)
    plt.barh(top_plot['Feature'], top_plot['Importance'], color='skyblue')
    plt.title('Top 20 Predictors (AdaBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(FEATURE_PLOT_ADA)
    print(f"Saved AdaBoost feature importance plot to {FEATURE_PLOT_ADA}")

    # ---------------------------------------------------------
    # 4. Recoding & Preparation for Final Model
    # ---------------------------------------------------------
    print("\n[Step 4] Recoding Variables...")
    X_final = X_imputed[SELECTED_FEATURES].copy()
    
    # Recode Q46 (Happiness): 1=Very Happy -> 4=Very Happy
    print("Recoding Q46 (Happiness): Reversing scale so Higher = Happier")
    X_final['Q46'] = 5 - X_final['Q46']
    
    # Recode Q47 (Health): 1=Very Good -> 5=Very Good
    print("Recoding Q47 (Health): Reversing scale so Higher = Healthier")
    X_final['Q47'] = 6 - X_final['Q47']
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_final), columns=X_final.columns)

    # ---------------------------------------------------------
    # 5. Ordinal Logistic Regression
    # ---------------------------------------------------------
    print("\n[Step 5] Ordinal Logistic Regression...")
    try:
        mod = OrderedModel(y, X_scaled, distr='logit')
        res = mod.fit(method='bfgs', disp=False)
        
        print(res.summary())
        
        print("\n--- Odds Ratios (Interpretation) ---")
        params = res.params
        conf = res.conf_int()
        # --- Save Regression Table ---
        results_df = pd.DataFrame({
            'Feature': params.index,
            'Coefficient': params.values,
            'Std Error': res.bse.values,
            'z-score': res.tvalues.values,
            'P-value': res.pvalues.values,
            'Odds Ratio': np.exp(params.values),
            'CI Lower (2.5%)': np.exp(conf.iloc[:, 0].values),
            'CI Upper (97.5%)': np.exp(conf.iloc[:, 1].values)
        })
        # Filter only predictors (exclude threshold parameters like 1/2, 2/3)
        predictors_df = results_df[results_df['Feature'].isin(SELECTED_FEATURES)].copy()
        predictors_df.to_csv(REGRESSION_TABLE_CSV, index=False)
        print(f"Saved regression coefficients table to {REGRESSION_TABLE_CSV}")

        # --- Plot Ordinal Regression Feature Importance (based on absolute z-score) ---
        print("\nGenerating Ordinal Regression Feature Importance Plot...")
        predictors_df['Abs_Z_Score'] = predictors_df['z-score'].abs()
        predictors_df = predictors_df.sort_values(by='Abs_Z_Score', ascending=True) # Ascending for barh

        plt.figure(figsize=(10, 6))
        plt.barh(predictors_df['Feature'], predictors_df['Abs_Z_Score'], color='lightgreen')
        plt.title('Feature Importance (Ordinal Regression: Absolute Z-Score)')
        plt.xlabel('Absolute Z-Score (Significance)')
        plt.tight_layout()
        plt.savefig(FEATURE_PLOT_ORDINAL)
        print(f"Saved Ordinal Regression feature importance plot to {FEATURE_PLOT_ORDINAL}")
        
        # Pseudo R-squared
        if hasattr(res, 'llnull'):
            mcfadden_r2 = 1 - (res.llf / res.llnull)
            print(f"\nMcFadden's Pseudo R-squared: {mcfadden_r2:.4f}")
            
    except Exception as e:
        print(f"Regression failed: {e}")

    # ---------------------------------------------------------
    # 6. ROC Curves
    # ---------------------------------------------------------
    print("\n[Step 6] Generating ROC Curves...")
    y_prob = res.predict(X_scaled)
    
    # Binary ROC (High Satisfaction > 5)
    y_binary = (y > 5).astype(int)
    # Sum probs for classes > 5 (indices 5-9)
    # Assuming classes are 1-10 sorted
    high_sat_indices = [i for i, c in enumerate(sorted(y.unique())) if c > 5]
    y_prob_binary = y_prob.iloc[:, high_sat_indices].sum(axis=1)
    
    fpr, tpr, _ = roc_curve(y_binary, y_prob_binary)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Binary ROC: High Life Satisfaction (>5)')
    plt.legend(loc="lower right")
    plt.savefig(ROC_PLOT_BINARY)
    print(f"Binary ROC AUC: {roc_auc:.4f}")
    print(f"Saved Binary ROC plot to {ROC_PLOT_BINARY}")
    
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
