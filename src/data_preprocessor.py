import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_data(df):
    """
    Process the DataFrame:
      - Keep numeric columns.
      - Encode categorical columns if they are explicitly listed or contain accountability keywords.
      - Drop irrelevant categorical columns.
    """
    logging.info("Starting data preprocessing.")

    explicit_preserve = {"sex", "gender", "race", "ethnicity", "approved", "income", "credit_score",
                         "transaction_count", "debt_ratio", "account_balance", "loan_amount"}

    accountability_keywords = {"audit", "log", "explain", "reason", "justification", "time", "trace"}

    columns_to_keep = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            columns_to_keep.append(col)
        else:
            col_lower = col.lower()
            if (col_lower in explicit_preserve) or any(keyword in col_lower for keyword in accountability_keywords):
                logging.info(f"Label-encoding column: {col}")
                df[col] = df[col].fillna("Missing")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                columns_to_keep.append(col)
            else:
                logging.info(f"Dropping column: {col}")

    df = df[columns_to_keep].copy()
    df.fillna(0, inplace=True)

    # Debugging: Show which columns are kept
    print("🟢 Remaining columns after preprocessing:", df.columns.tolist())

    logging.info("Data preprocessing completed successfully.")
    return df

def save_processed_data(df, file_path):
    """
    Saves the processed DataFrame to a timestamped CSV file in an 'Output_report' folder.
    """
    try:
        timestamp = datetime.now().strftime("%d%m%y%H%M")
        output_dir = "Output_report"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        processed_file_name = f"processed_data_{base_name}_{timestamp}.csv"
        processed_file_path = os.path.join(output_dir, processed_file_name)

        df.to_csv(processed_file_path, index=False)
        logging.info(f"Processed data saved to '{processed_file_path}'.")
        return processed_file_path

    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise
