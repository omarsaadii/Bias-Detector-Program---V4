
import logging
import numpy as np
from datetime import datetime
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ensure reproducibility for NumPy
np.random.seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_robustness(df):
    try:
        logging.info("Starting robustness evaluation.")

        # Use internal default: assume target is the last column
        if df.shape[1] < 2:
            raise ValueError("Not enough columns to separate features and target.")
        
        target_column = df.columns[-1]
        feature_columns = list(df.columns[:-1])

        # Split data into train and test sets (Set random_state=42)
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a logistic regression model (Set random_state=42)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Wrap model with ART classifier
        art_classifier = SklearnClassifier(model=model)

        # Evaluate initial accuracy
        initial_accuracy = model.score(X_test, y_test)
        logging.info(f"Initial model accuracy: {initial_accuracy}")

        # Generate adversarial examples (Ensure deterministic attack if possible)
        attack = FastGradientMethod(estimator=art_classifier, eps=0.2)
        X_test_adv = attack.generate(X_test.to_numpy())

        # Evaluate accuracy on adversarial examples
        adversarial_accuracy = model.score(X_test_adv, y_test)
        logging.info(f"Adversarial model accuracy: {adversarial_accuracy}")

        logging.info("Robustness evaluation completed successfully.")
        return {"initial_accuracy": initial_accuracy, "adversarial_accuracy": adversarial_accuracy}

    except Exception as e:
        logging.error(f"Error during robustness evaluation: {e}")
        raise


