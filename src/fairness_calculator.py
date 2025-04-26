import logging
import numpy as np
from fairlearn.metrics import demographic_parity_difference

# Fix: Ensure reproducibility for NumPy operations
np.random.seed(42)

def evaluate_fairness(df):
    logging.info("Starting fairness evaluation.")

    # Define a set of candidate names for protected attributes
    protected_candidates = {"sex", "gender", "race", "ethnicity"}
    
    # Detect potential protected attribute columns (case-insensitive match)
    protected_cols = [col for col in df.columns if col.lower() in protected_candidates]
    
    # Detect a candidate target column: a binary column not in the protected candidates.
    candidate_target = None
    for col in df.columns:
        # Skip columns that are potential protected attributes
        if col in protected_cols:
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            candidate_target = col
            break

    if candidate_target is None:
        logging.warning("No binary target column found. Skipping fairness evaluation.")
        return {"demographic_parity_difference": None, "mean_difference": None}

    if not protected_cols:
        logging.warning("No protected attribute columns found. Skipping fairness evaluation.")
        return {"demographic_parity_difference": None, "mean_difference": None}

    logging.info(f"Using target column: {candidate_target}")
    logging.info(f"Using protected attribute: {protected_cols[0]}")
    
    y_true = df[candidate_target]
    y_pred = df[candidate_target]  # Dummy predictions

    try:
        dp = demographic_parity_difference(y_true, y_pred, sensitive_features=df[protected_cols[0]])
    except Exception as e:
        logging.error(f"Error computing demographic parity difference: {e}")
        dp = None

    mean_diff = np.abs(np.mean(y_true) - np.mean(y_pred))
    
    fairness_results = {"demographic_parity_difference": dp, "mean_difference": mean_diff}
    logging.info("Fairness evaluation completed.")
    return fairness_results
