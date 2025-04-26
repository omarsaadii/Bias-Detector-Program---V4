import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_accountability(df):
    """
    Dynamically evaluate accountability by scanning the DataFrame's column names.
    It checks for keywords indicating:
      - Auditability (e.g., "audit", "log", "audit_flag")
      - Explainability (e.g., "explain", "reason", "justification", "explanation")
      - Traceability (e.g., "time", "timestamp", "history", "trace")
    
    Returns:
      A dictionary with boolean values for each aspect.
    """
    logging.info("Starting dynamic accountability evaluation.")

    # Convert column names to lowercase for case-insensitive matching.
    cols_lower = [col.lower() for col in df.columns]

    # Define keyword lists 
    audit_keywords = ["audit", "log", "audit_flag"]
    explain_keywords = ["explain", "reason", "justification", "explanation"]
    trace_keywords = ["time", "timestamp", "history", "trace"]

    # Check if any column name contains any of the keywords for each aspect.
    auditability = any(any(keyword in col for keyword in audit_keywords) for col in cols_lower)
    explainability = any(any(keyword in col for keyword in explain_keywords) for col in cols_lower)
    traceability = any(any(keyword in col for keyword in trace_keywords) for col in cols_lower)

    logging.info(f"Auditability detected: {auditability}")
    logging.info(f"Explainability detected: {explainability}")
    logging.info(f"Traceability detected: {traceability}")

    accountability_results = {
        "auditability": auditability,
        "explainability": explainability,
        "traceability": traceability
    }

    logging.info(f"Accountability evaluation completed: {accountability_results}")
    return accountability_results
