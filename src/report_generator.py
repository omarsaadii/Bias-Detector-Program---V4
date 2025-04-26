import os
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Report generation function
def generate_report(results, config, file_name):
    try:
        logging.info("Starting report generation.")

        # Create output directory
        output_dir = "Output report"
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamped file names
        timestamp = datetime.now().strftime("%d%m%y%H%M")
        detailed_report_path = os.path.join(output_dir, f"report_{file_name}_{timestamp}.json")

        # Interpretation based on thresholds
        thresholds = config.get("indicator_thresholds", {})
        summary = {
            "fairness_present": results.get("fairness", {}).get("demographic_parity_difference", 0) >= thresholds.get("fairness", 0),
            "transparency_present": results.get("transparency", {}).get("model_accuracy", 0) >= thresholds.get("transparency", 0),
            "robustness_present": results.get("robustness", {}).get("adversarial_accuracy", 0) >= thresholds.get("robustness", 0),
            "privacy_present": results.get("privacy", {}).get("privacy_accuracy", 0) >= thresholds.get("privacy", 0),
        }

        # Add interpretation to the report
        final_report = {
            "results": results,
            "summary": {
                "fairness": "Present" if summary["fairness_present"] else "Not Present",
                "transparency": "Present" if summary["transparency_present"] else "Not Present",
                "robustness": "Present" if summary["robustness_present"] else "Not Present",
                "privacy": "Present" if summary["privacy_present"] else "Not Present",
            },
        }

        # Save detailed report
        with open(detailed_report_path, "w") as report_file:
            json.dump(final_report, report_file, indent=4)
        logging.info(f"Detailed report saved to '{detailed_report_path}'.")

        logging.info("Report generation completed successfully.")
        return detailed_report_path

    except Exception as e:
        logging.error(f"Error during report generation: {e}")
        raise

# Conclusion report generation function
def generate_conclusion_report(results):
    try:
        logging.info("Generating conclusion report for normal users.")

        # Compute final note (average compliance score from 0 to 1)
        fairness_score = min(1.0, max(0.0, 1 - abs(results['fairness']['demographic_parity_difference'])))
        transparency_score = min(1.0, max(0.0, results['transparency']['model_accuracy']))
        robustness_score = min(1.0, max(0.0, results['robustness']['adversarial_accuracy']))
        privacy_score = min(1.0, max(0.0, results['privacy']['privacy_accuracy']))
        accountability_score = sum([
            float(results['accountability']['auditability']),
            float(results['accountability']['explainability']),
            float(results['accountability']['traceability'])
        ]) / 3

        final_note = (fairness_score + transparency_score + robustness_score + privacy_score + accountability_score) / 5

        # Explanation strings
        fairness_explanation = f"Fairness Score: {fairness_score:.2f}\n"
        transparency_explanation = f"Transparency Score: {transparency_score:.2f}\n"
        robustness_explanation = f"Robustness Score: {robustness_score:.2f}\n"
        privacy_explanation = f"Privacy Score: {privacy_score:.2f}\n"
        accountability_explanation = (
            f"Accountability:\n"
            f"- Auditability: {'Present' if results['accountability']['auditability'] else 'Not Present'}.\n"
            f"  This checks if decision-making logs are available.\n"
            f"- Explainability: {'Present' if results['accountability']['explainability'] else 'Not Present'}.\n"
            f"  This verifies if justifications for decisions are recorded.\n"
            f"- Traceability: {'Present' if results['accountability']['traceability'] else 'Not Present'}.\n"
            f"  This ensures that model decisions are tracked over time.\n\n"
        )

        final_note_explanation = f"Final Note (Overall Compliance Score): {final_note:.2f} (0-1 scale)\n\n"

        # Write to .txt file
        output_dir = "Output report"
        os.makedirs(output_dir, exist_ok=True)
        conclusion_file_path = os.path.join(output_dir, "conclusion.txt")

        with open(conclusion_file_path, "w") as conclusion_file:
            conclusion_file.write(final_note_explanation + fairness_explanation + transparency_explanation + robustness_explanation + privacy_explanation + accountability_explanation)

        logging.info(f"Conclusion report saved to '{conclusion_file_path}'.")
        return conclusion_file_path

    except Exception as e:
        logging.error(f"Error generating conclusion report: {e}")
        raise

   