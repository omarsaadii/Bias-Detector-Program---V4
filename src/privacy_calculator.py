import logging
import numpy as np
from diffprivlib.models import LogisticRegression as DiffPrivLogReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_privacy(df):
    try:
        logging.info("Starting privacy evaluation.")

        # Use internal default: assume target is the last column
        if df.shape[1] < 2:
            raise ValueError("Not enough columns to separate features and target.")
        
        target_column = df.columns[-1]
        feature_columns = list(df.columns[:-1])

        # Split data into train and test sets (Fix: Set random_state=42)
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a differentially private logistic regression model (Fix: Set random_state=42)
        priv_model = DiffPrivLogReg(epsilon=1.0, max_iter=1000, random_state=42)
        priv_model.fit(X_train, y_train)

        # Evaluate model accuracy on test data
        y_pred = priv_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Privacy-preserving model accuracy: {accuracy}")

        logging.info("Privacy evaluation completed successfully.")
        return {"privacy_accuracy": accuracy}

    except Exception as e:
        logging.error(f"Error during privacy evaluation: {e}")
        raise
