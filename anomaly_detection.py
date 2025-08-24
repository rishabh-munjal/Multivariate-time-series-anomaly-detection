"""
Anomaly Detection for Multivariate Time Series Data

This script detects anomalies in multivariate time series data and identifies the top contributing features for each anomaly.

Usage:
    python anomaly_detection.py input_csv_path output_csv_path

Requirements:
    - pandas
    - numpy
    - scikit-learn
"""

import sys
from typing import List
from anomaly_pipeline import AnomalyPipeline

def main(input_csv_path: str, output_csv_path: str) -> None:
    """
    Main function to run the anomaly detection pipeline.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file.
    """
    pipeline = AnomalyPipeline()
    pipeline.run(input_csv_path, output_csv_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python anomaly_detection.py <input_csv_path> <output_csv_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
