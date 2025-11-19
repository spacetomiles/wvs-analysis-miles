# WVS Data Cleaning Walkthrough

## Overview
This document summarizes the cleaning process performed on the World Values Survey (WVS) dataset `WVS_Cross-National_Wave_7_csv_v6_0.xlsx`.

## Changes Made

### 1. Data Loading
-   Loaded the raw Excel file.
-   Original Shape: (97,220 rows, 613 columns).

### 2. Column Filtering
-   Filtered to include **only** columns `Q1` through `Q290`.
-   Found **290** matching columns.

### 3. Missing Values Handling
-   Identified **1,485,569** negative values (e.g., -1, -2, -4, -5) which represent missing or inapplicable data in WVS.
-   Replaced all negative values with `NaN`.

### 4. Deduplication
-   Removed **86** duplicate rows.

### 5. Saving
-   Saved the cleaned dataset to: `WVS_Cross-National_Wave_7_cleaned.csv`
-   Final Shape: **(97,134 rows, 290 columns)**

## Verification Results
A verification script `verify_wvs_data.py` was run to ensure data quality:

| Check | Result | Details |
| :--- | :--- | :--- |
| **File Exists** | PASS | File created successfully. |
| **Column Count** | PASS | Exactly 290 columns (Q1-Q290). |
| **Negative Values** | PASS | 0 negative values found. |
| **Duplicates** | PASS | 0 duplicate rows found. |
| **Missing Data** | INFO | ~5.27% of data is missing (NaN). |

## Files Created
-   `clean_wvs_data.py`: The script used to clean the data.
-   `verify_wvs_data.py`: The script used to verify the cleaned data.
-   `WVS_Cross-National_Wave_7_cleaned.csv`: The final cleaned dataset.

## Modeling Results

### 1. Imputation
-   Missing values in predictors (Q1-Q290) were imputed using the **median** value.
-   Output: `WVS_imputed.csv`

### 2. Feature Selection (AdaBoost)
Top features identified by AdaBoostClassifier:
1.  **Q50**: Satisfaction with financial situation of household
2.  **Q48**: Control over your life
3.  **Q46**: Happiness
4.  **Q164**: Importance of God
5.  **Q112**: Confidence: The Government
6.  **Q110**: Confidence: The Press
7.  **Q120**: Confidence: The United Nations
8.  **Q55**: Freedom of choice and control
9.  **Q47**: Health
10. **Q213**: I see myself as someone who: Is reserved

![Feature Importance Plot](/Users/a11/Downloads/HKU/7005/7005%20group%20project/feature_importance_plot.png)

### 3. Ordinal Logistic Regression (Targeted Analysis)
The user selected 9 specific features for detailed analysis. All 9 were found to be **statistically significant (p < 0.000)**.

| Feature | Description | Coef | Odds Ratio | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Q50** | Satisfaction with financial situation | 1.06 | 2.90 | Higher financial satisfaction strongly increases life satisfaction. |
| **Q48** | Control over your life | 0.63 | 1.88 | More control leads to higher satisfaction. |
| **Q46** | Happiness | -0.54 | 0.58 | *Note: Lower values in Q46 mean "Very Happy", so negative coef means happier people are more satisfied.* |
| **Q164** | Importance of God | 0.09 | 1.09 | Higher importance is positively associated. |
| **Q112** | Confidence: The Government | 0.07 | 1.07 | Higher confidence is positively associated. |
| **Q110** | Confidence: The Press | -0.04 | 0.96 | Lower confidence is slightly associated with higher satisfaction (or scale direction nuance). |
| **Q120** | Confidence: The UN | 0.06 | 1.06 | Higher confidence is positively associated. |
| **Q55** | Freedom of choice and control | 0.12 | 1.12 | More freedom leads to higher satisfaction. |
| **Q47** | Health | -0.13 | 0.88 | *Note: Lower values in Q47 mean "Very Good Health", so negative coef means healthier people are more satisfied.* |

**Model Fit**:
-   **Log-Likelihood (Model)**: -169,467
-   **Log-Likelihood (Null)**: -200,512
-   **McFadden's Pseudo R-squared**: **0.1548**
    *   *Interpretation*: A Pseudo R-squared between 0.2 and 0.4 is generally considered an excellent fit for logistic regression. A value of **0.155** indicates a **good fit** for social science survey data, meaning the model explains a substantial portion of the variance in Life Satisfaction compared to a null model.

### 4. ROC Curve and AUC Score
To evaluate the model's predictive power, we generated ROC curves.

#### Binary ROC (High vs Low Satisfaction)
We binarized the outcome (High Satisfaction > 5) to calculate a simplified AUC score, which is easier to interpret.

![Binary ROC Curve](/Users/a11/Downloads/HKU/7005/7005%20group%20project/roc_curve_ordinal_binary.png)

-   **AUC Score**: **0.8461**
-   *Interpretation*: An AUC of 0.85 is considered **excellent**. It means there is an 85% chance that the model will correctly distinguish between a person with high life satisfaction and one with low life satisfaction.

#### Multi-class ROC (Micro-average)
-   **Micro-average AUC**: **0.8153**

## Files Created
-   `clean_wvs_data.py`: Data cleaning script.
-   `verify_wvs_data.py`: Data verification script.
-   `run_modeling_pipeline.py`: Modeling pipeline script.
-   `run_targeted_regression.py`: Targeted regression script.
-   `check_missing_rows.py`: Script to check missing rows proportion.
-   `inspect_target.py`: Script to inspect target variable.
-   `visualize_feature_importance.py`: Script to visualize feature importance.
-   `WVS_Cross-National_Wave_7_cleaned.csv`: Cleaned dataset.
-   `WVS_imputed.csv`: Imputed dataset used for modeling.
