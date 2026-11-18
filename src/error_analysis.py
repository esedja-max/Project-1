"""Summarise the misclassified sample and print examples for manual reading.

Usage:
    python src/error_analysis.py
    python src/error_analysis.py "Debt collection" "Credit card"
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
err = pd.read_csv(ROOT / "results" / "baseline_errors_sample.csv")

print("Most common confusions (true -> predicted):")
print(err.groupby(["label", "predicted"]).size().sort_values(ascending=False).head(10))
print("\nPredicted label among errors:\n", err["predicted"].value_counts())
print("\nTrue label among errors:\n", err["label"].value_counts())

if len(sys.argv) == 3:
    true_label, pred_label = sys.argv[1], sys.argv[2]
    sub = err[(err["label"] == true_label) & (err["predicted"] == pred_label)]
    print(f"\nExamples: true = {true_label} | predicted = {pred_label}\n")
    for t in sub["text"].head(5):
        print("-", t[:400].replace("\n", " "), "\n")
