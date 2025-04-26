import os
import logging
import pandas as pd
from datetime import datetime

# Import updated modules
from data_loader import load_csv
from data_preprocessor import preprocess_data, save_processed_data
from fairness_calculator import evaluate_fairness
from transparency_calculator import evaluate_transparency
from robustness_calculator import evaluate_robustness
from privacy_calculator import evaluate_privacy
from accountability_calculator import evaluate_accountability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def process_all_files(data_folder):
    results_list = []

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_folder, file_name)
            logging.info(f"Processing file: {file_name}")

            try:
                # 1. Load and preprocess the data.
                df = load_csv(file_path)
                df = preprocess_data(df)
                # Debug: print columns after preprocessing.
                print("Columns after preprocessing:", df.columns.tolist())

                # 2. Initialize indicator variables as None.
                fairness_score = None
                transparency_score = None
                robustness_score = None
                privacy_score = None
                accountability_score = None

                # 3. Evaluate fairness.
                try:
                    fairness_results = evaluate_fairness(df)
                    dp_diff = fairness_results.get("demographic_parity_difference")
                    if dp_diff is not None:
                        fairness_score = max(0.0, min(1.0, 1 - abs(dp_diff)))
                except Exception as e:
                    logging.warning(f"Fairness evaluation failed for {file_name}: {e}")
                    fairness_score = None

                # 4. Evaluate transparency.
                try:
                    transparency_results = evaluate_transparency(df)
                    transparency_score = transparency_results.get("model_accuracy", None)
                except Exception as e:
                    logging.warning(f"Transparency evaluation failed for {file_name}: {e}")
                    transparency_score = None

                # 5. Evaluate robustness.
                try:
                    robustness_results = evaluate_robustness(df)
                    robustness_score = robustness_results.get("adversarial_accuracy", None)
                except Exception as e:
                    logging.warning(f"Robustness evaluation failed for {file_name}: {e}")
                    robustness_score = None

                # 6. Evaluate privacy.
                try:
                    privacy_results = evaluate_privacy(df)
                    privacy_score = privacy_results.get("privacy_accuracy", None)
                except Exception as e:
                    logging.warning(f"Privacy evaluation failed for {file_name}: {e}")
                    privacy_score = None

                # 7. Evaluate accountability.
                try:
                    accountability_results = evaluate_accountability(df)
                    if accountability_results:
                        # Sum the boolean flags (True->1, False->0)
                        a_sum = (float(accountability_results.get("auditability", False)) +
                                 float(accountability_results.get("explainability", False)) +
                                 float(accountability_results.get("traceability", False)))
                        # If all flags are false, set accountability to None.
                        if a_sum == 0:
                            accountability_score = None
                        else:
                            accountability_score = a_sum / 3
                except Exception as e:
                    logging.warning(f"Accountability evaluation failed for {file_name}: {e}")
                    accountability_score = None

                # 8. Build a list of valid scores (skip any that are None).
                valid_scores = []
                if fairness_score is not None:
                    valid_scores.append(fairness_score)
                if transparency_score is not None:
                    valid_scores.append(transparency_score)
                if robustness_score is not None:
                    valid_scores.append(robustness_score)
                if privacy_score is not None:
                    valid_scores.append(privacy_score)
                if accountability_score is not None:
                    valid_scores.append(accountability_score)

                # 9. Compute the final compliance score as the average of valid indicators.
                if valid_scores:
                    final_score = sum(valid_scores) / len(valid_scores)
                else:
                    final_score = None

                # 10. Append results to the list; convert None values to "NA" for output.
                results_list.append({
                    "File Name": file_name,
                    "Fairness Score": fairness_score if fairness_score is not None else "NA",
                    "Transparency Score": transparency_score if transparency_score is not None else "NA",
                    "Robustness Score": robustness_score if robustness_score is not None else "NA",
                    "Privacy Score": privacy_score if privacy_score is not None else "NA",
                    "Accountability Score": accountability_score if accountability_score is not None else "NA",
                    "Final Compliance Score": final_score if final_score is not None else "NA"
            
                })

            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")

    # 11. Save all results to a consolidated CSV file.
    if results_list:
        results_df = pd.DataFrame(results_list)
        output_dir = "Output_report"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "results.csv")
        results_df.to_csv(output_path, index=False)
        logging.info(f"All results have been consolidated in '{output_path}'")
    else:
        logging.warning("No CSV files found or no results generated.")

if __name__ == "__main__":
    data_folder = "data"
    process_all_files(data_folder)