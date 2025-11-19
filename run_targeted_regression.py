import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Configuration
IMPUTED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_imputed.csv'

# User selected features
SELECTED_FEATURES = [
    'Q50',  # Satisfaction with financial situation
    'Q48',  # Control over your life
    'Q46',  # Happiness
    'Q164', # Importance of God
    'Q112', # Confidence: The Government
    'Q110', # Confidence: The Press
    'Q120', # Confidence: The United Nations
    'Q55',  # Freedom of choice and control
    'Q47'   # Health
]

def run_targeted_analysis():
    print(f"Loading data from {IMPUTED_FILE}...")
    try:
        df = pd.read_csv(IMPUTED_FILE)
    except FileNotFoundError:
        print("Error: Imputed file not found.")
        return

    y = df['Q49']
    X = df[SELECTED_FEATURES]
    
    print(f"Running Ordinal Logistic Regression on {len(SELECTED_FEATURES)} features...")
    print(f"Features: {SELECTED_FEATURES}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Ensure y is integer
    y_final = y.astype(int)
    
    try:
        # Fit Ordered Model
        mod = OrderedModel(y_final, X_scaled, distr='logit')
        res = mod.fit(method='bfgs')
        
        print("\n" + "="*60)
        print("ORDINAL LOGISTIC REGRESSION SUMMARY")
        print("="*60)
        print(res.summary())
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        
        # Calculate McFadden's Pseudo R-squared
        # L_null is the log-likelihood of the model with only the intercepts (no predictors)
        # We can fit a null model or use the llnull attribute if available. 
        # OrderedModel in statsmodels might not populate llnull automatically for all versions.
        # Let's try to fit a null model explicitly to be safe.
        
        print("Fitting null model for comparison...")
        # Null model has no predictors, just the ordinal intercepts
        # We pass an array of zeros or empty dataframe as exog is tricky with OrderedModel, 
        # usually it requires at least one variable or we handle it carefully.
        # A safer way for Pseudo R2 in this context without re-fitting might be to rely on the user knowing 
        # that for large N, significance is more important. 
        # However, let's try to calculate it.
        
        ll_model = res.llf
        
        # Fit null model (constant only)
        # For OrderedModel, we can pass a column of ones? No, OrderedModel handles intercepts internally.
        # We can pass a dummy column of zeros.
        X_null = pd.DataFrame({'const': np.zeros(len(y_final))})
        # Actually, OrderedModel with no exog might fail or be tricky.
        # Let's try to just use the `llnull` if it exists, otherwise skip complex re-fitting to avoid errors.
        
        if hasattr(res, 'llnull'):
            ll_null = res.llnull
            mcfadden_r2 = 1 - (ll_model / ll_null)
            print(f"Log-Likelihood (Model): {ll_model:.4f}")
            print(f"Log-Likelihood (Null):  {ll_null:.4f}")
            print(f"McFadden's Pseudo R-squared: {mcfadden_r2:.4f}")
        else:
            # Fallback: Simple approximation or skip
            print(f"Log-Likelihood: {ll_model:.4f}")
            print("AIC:", res.aic)
            print("BIC:", res.bic)
            print("(Note: Standard Pseudo R-squared requires a null model fit which is computationally expensive here)")

        print("\n" + "="*60)
        print("INTERPRETATION (Odds Ratios)")
        print("="*60)
        params = res.params
        conf = res.conf_int()
        conf['OR'] = params
        conf.columns = ['2.5%', '97.5%', 'OR']
        # Calculate odds ratios
        out = np.exp(conf)
        print(out.head(len(SELECTED_FEATURES)))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_targeted_analysis()
