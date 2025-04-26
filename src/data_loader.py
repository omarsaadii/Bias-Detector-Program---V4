
import os
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_csv(file_path):
    """
    Loads a CSV file without relying on any external configuration.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path)
        logging.info(f"CSV file '{file_path}' loaded successfully.")
        return df

    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        raise
