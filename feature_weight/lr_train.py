#!/usr/bin/env python3
# Logistic regression for feature weight calculation

import os
import json
import re
from collections import defaultdict
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Compatibility fix for older sklearn versions
np.float = float

# ===== Configurable paths (replace with actual paths during reproduction) =====
# The following paths should be set according to the experimental environment.
# For anonymity, specific directory names have been replaced with placeholders.
OUTPUT_DIR = "./output_dir"          # Directory to store output JSON files
FEATURE_CSV = "./features.csv"  # Input feature CSV file
IC_JSON = f"{OUTPUT_DIR}/lr_scores.json"
MERGED_IC_JSON = f"{OUTPUT_DIR}/merged_lr_scores.json"
WEIGHTS_JSON = f"{OUTPUT_DIR}/merged_weights.json"


def compute_logistic_weights(df: pd.DataFrame, label_col: str = 'label') -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Compute logistic regression coefficients and normalized importance weights.

    Args:
        df: Input DataFrame containing features and label column.
        label_col: Name of the label column (binary target).

    Returns:
        coef_dict: Raw logistic regression coefficients.
        merged_ic_scores: Absolute coefficient values (importance scores).
        merged_weights: Normalized importance weights (sum to 1).
    """
    print("Computing logistic regression weights...")

    df_work = df.copy()
    df_work = df_work.drop(columns=['ip_domain', 'date'], errors='ignore')

    if label_col not in df_work.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    # Remove date prefix from feature names (e.g., '20250101_feature' -> 'feature')
    feature_map = {col: re.sub(r"^\d{8}_", "", col) for col in df_work.columns if col != label_col}
    df_renamed = df_work.rename(columns=feature_map)

    # Check for duplicate column names after renaming
    duplicated_cols = df_renamed.columns[df_renamed.columns.duplicated()].unique().tolist()
    if duplicated_cols:
        print(f"Warning: {len(duplicated_cols)} duplicate column names detected. They will be aggregated by mean.")

    # Convert all feature columns to numeric (coerce errors to NaN)
    df_converted = df_renamed.apply(pd.to_numeric, errors='coerce')

    # Extract label and features
    if label_col not in df_converted.columns:
        raise ValueError("Label column missing after conversion.")
    y = df_converted[label_col].fillna(0).astype(int)
    X = df_converted.drop(columns=[label_col], errors='ignore')

    if X.shape[1] == 0:
        raise ValueError("No feature columns available.")

    # Ensure binary classification is possible
    if len(set(y)) < 2:
        raise ValueError("Label column contains only one class. Logistic regression requires two classes.")

    # Aggregate duplicate feature columns by mean
    X_grouped = X.groupby(by=X.columns, axis=1).mean()

    # Impute missing values with column mean
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_grouped)

    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train logistic regression without regularization
    try:
        model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    except TypeError:
        # Fallback for older sklearn versions
        model = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)

    model.fit(X_scaled, y)

    # Extract coefficients (absolute values for importance)
    coef_dict = {
        X_grouped.columns[i]: float(model.coef_[0][i])
        for i in range(len(X_grouped.columns))
    }

    # Importance score = absolute coefficient
    merged_ic_scores = {k: abs(v) for k, v in coef_dict.items()}

    # Normalize to obtain feature weights (sum to 1)
    total = sum(merged_ic_scores.values())
    if total == 0:
        print("Warning: All feature coefficients are zero. Cannot normalize.")
        merged_weights = {k: 0.0 for k in merged_ic_scores}
    else:
        merged_weights = {k: v / total for k, v in merged_ic_scores.items()}

    return coef_dict, merged_ic_scores, merged_weights


def main() -> None:
    """Main execution function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FEATURE_CSV):
        raise FileNotFoundError(f"Feature file not found: {FEATURE_CSV}")

    print(f"Loading feature data from {FEATURE_CSV} ...")
    df = pd.read_csv(FEATURE_CSV)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in features.csv")

    # Compute logistic regression weights
    ic_scores, merged_ic_scores, merged_weights = compute_logistic_weights(df)

    # Save results to JSON files
    with open(IC_JSON, "w") as f:
        json.dump(ic_scores, f, indent=2)

    with open(MERGED_IC_JSON, "w") as f:
        json.dump(merged_ic_scores, f, indent=2)

    with open(WEIGHTS_JSON, "w") as f:
        json.dump(merged_weights, f, indent=2)

    # Print top-10 features by importance
    print("\nTop-10 merged feature importance scores (absolute coefficients):")
    for k, v in sorted(merged_ic_scores.items(), key=lambda x: -x[1])[:10]:
        print(f"{k:40s} IC (|coef|): {v:.6f}")

    print("\nTop-10 normalized feature weights (importance distribution):")
    for k, v in sorted(merged_weights.items(), key=lambda x: -x[1])[:10]:
        print(f"{k:40s} Weight: {v:.6f}")

    print("\nExecution completed. Results saved to:")
    print(f"  - Raw coefficients: {IC_JSON}")
    print(f"  - Aggregated IC scores: {MERGED_IC_JSON}")
    print(f"  - Normalized weights: {WEIGHTS_JSON}")


if __name__ == "__main__":
    main()