import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Configuration
IMPUTED_FILE = '/Users/a11/Downloads/HKU/7005/7005 group project/WVS_imputed.csv'
OUTPUT_PLOT_MULTI = '/Users/a11/Downloads/HKU/7005/7005 group project/roc_curve_ordinal_multiclass.png'
OUTPUT_PLOT_BINARY = '/Users/a11/Downloads/HKU/7005/7005 group project/roc_curve_ordinal_binary.png'

# User selected features
SELECTED_FEATURES = ['Q50', 'Q48', 'Q46', 'Q164', 'Q112', 'Q110', 'Q120', 'Q55', 'Q47']

def generate_roc():
    print(f"Loading data from {IMPUTED_FILE}...")
    try:
        df = pd.read_csv(IMPUTED_FILE)
    except FileNotFoundError:
        print("Error: Imputed file not found.")
        return

    y = df['Q49'].astype(int)
    X = df[SELECTED_FEATURES]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print("Training Ordinal Logistic Regression...")
    mod = OrderedModel(y, X_scaled, distr='logit')
    res = mod.fit(method='bfgs', disp=False)
    
    # Predict probabilities
    # statsmodels predict returns probabilities for each class
    y_prob = res.predict(X_scaled)
    
    # --- 1. Multi-class ROC (One-vs-Rest) ---
    print("\nCalculating Multi-class ROC (One-vs-Rest)...")
    classes = sorted(y.unique())
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    print(f"Micro-average AUC: {roc_auc['micro']:.4f}")
    
    # Plot Multi-class ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # Plot curves for specific classes (e.g., 1, 5, 10 to avoid clutter)
    for i in [0, 4, 9]: # Classes 1, 5, 10 (indices 0, 4, 9)
        if i < n_classes:
            plt.plot(fpr[i], tpr[i], lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC for Ordinal Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig(OUTPUT_PLOT_MULTI)
    print(f"Multi-class ROC plot saved to {OUTPUT_PLOT_MULTI}")
    
    # --- 2. Simplified Binary ROC (High vs Low Satisfaction) ---
    # Often easier to interpret. Let's split at median (or say > 5 is High)
    print("\nCalculating Binary ROC (High Satisfaction > 5)...")
    y_binary = (y > 5).astype(int)
    
    # Probability of High Satisfaction = Sum of probabilities for classes > 5
    # Assuming classes are 1-10, indices 5-9 correspond to 6-10
    high_sat_indices = [i for i, c in enumerate(classes) if c > 5]
    y_prob_binary = y_prob.iloc[:, high_sat_indices].sum(axis=1)
    
    fpr_bin, tpr_bin, _ = roc_curve(y_binary, y_prob_binary)
    auc_bin = auc(fpr_bin, tpr_bin)
    
    print(f"Binary AUC (High vs Low): {auc_bin:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_bin, tpr_bin, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_bin)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Binary ROC: High Life Satisfaction (>5)')
    plt.legend(loc="lower right")
    plt.savefig(OUTPUT_PLOT_BINARY)
    print(f"Binary ROC plot saved to {OUTPUT_PLOT_BINARY}")

if __name__ == "__main__":
    generate_roc()
