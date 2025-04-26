import logging
from datetime import datetime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_transparency(df):
    try:
        logging.info("Starting transparency evaluation.")

        # Use internal default: assume target is the last column
        if df.shape[1] < 2:
            raise ValueError("Not enough columns to separate features and target.")
        
        target_column = df.columns[-1]
        feature_columns = list(df.columns[:-1])

        # Split data into train and test sets (Fix: Set random_state=42)
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a logistic regression model (Fix: Set random_state=42)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy}")

        # Use LIME for interpretability
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_columns,
            class_names=["Negative", "Positive"],
            mode="classification"
        )
        explanation = explainer.explain_instance(
            data_row=X_test.values[0],
            predict_fn=model.predict_proba
        )
        lime_results = {"feature_importance": explanation.as_list()}

        logging.info("Transparency evaluation completed successfully.")
        return {"model_accuracy": accuracy, "lime_results": lime_results}

    except Exception as e:
        logging.error(f"Error during transparency evaluation: {e}")
        raise
