# Bias Detector Tool

The Bias Detector Tool is a modular Python program designed to evaluate datasets and machine learning models from an ethical perspective. It focuses on five key dimensions: **fairness**, **transparency**, **privacy**, **robustness**, and **accountability**. The goal is to help developers, researchers, and policy makers assess how responsibly AI systems behave — and to identify potential risks before deployment.

##  What the Tool Does

- Loads and validates structured CSV files.
- Preprocesses the dataset (encoding, filtering, missing value handling).
- Automatically detects sensitive features and binary targets.
- Evaluates ethical indicators using dedicated modules.
- Outputs the results in human-readable and machine-readable formats.

##  Ethical Indicators

| Indicator     | Method Used                                           | Library         |
|---------------|--------------------------------------------------------|-----------------|
| Fairness      | Demographic Parity Difference                         | `fairlearn`     |
| Transparency  | Logistic Regression + LIME explanation                 | `lime`          |
| Privacy       | Differentially Private Logistic Regression             | `diffprivlib`   |
| Robustness    | FGSM adversarial testing                               | `adversarial-robustness-toolbox` (ART) |
| Accountability| Keyword detection in dataset column names             | Custom heuristic |

Each indicator returns a score from **0 to 1**. Scores closer to 1 indicate stronger ethical compliance. Scores ≥ 0.7 are marked as **"present"**, scores < 0.7 as **"not present"**, and indicators with missing inputs are marked **"not applicable (NA)"**.

## Folder Structure
Bias-Detector-Program
├── data/ # Input datasets (CSV)
├── Output_report/ # All generated reports
├── src/ # Core source code modules
│ ├── data_loader.py
│ ├── data_preprocessor.py
│ ├── fairness_calculator.py
│ ├── privacy_calculator.py
│ ├── robustness_calculator.py
│ ├── transparency_calculator.py
│ ├── accountability_calculator.py
│ └── report_generator.py
├── main.py # the full evaluation
└── README.md

##  How to Run

1. **Install dependencies**  
   Make sure you have Python 3.10+ and install required libraries:

   ```bash
   pip install -r requirements.txt

