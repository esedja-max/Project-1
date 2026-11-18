"""TF-IDF + Logistic Regression baseline with evaluation and error analysis.

Run after prepare_data.py:
    python src/train_baseline.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42


def main():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    test = pd.read_csv(DATA / "test.csv")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=5, max_df=0.9,
            sublinear_tf=True, stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED,
        )),
    ])
    model.fit(train["text"], train["label"])

    # Validation: use this to tune things
    val_pred = model.predict(val["text"])
    print("Validation macro-F1:", round(f1_score(val["label"], val_pred, average="macro"), 4))

    # Test: look at this only once, at the end
    test_pred = model.predict(test["text"])
    print("\nTest report:\n")
    report = classification_report(test["label"], test_pred, digits=3)
    print(report)
    (RESULTS / "baseline_report.txt").write_text(report)

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        test["label"], test_pred, xticks_rotation=90,
        normalize="true", values_format=".2f", ax=ax, colorbar=False,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "baseline_confusion_matrix.png", dpi=150)

    # Mistakes for manual error analysis
    errors = test.assign(predicted=test_pred)
    errors = errors[errors["label"] != errors["predicted"]]
    errors.sample(min(100, len(errors)), random_state=SEED).to_csv(
        RESULTS / "baseline_errors_sample.csv", index=False
    )

    # Top words per class (explainability)
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = vec.get_feature_names_out()
    lines = []
    for i, cls in enumerate(clf.classes_):
        top = clf.coef_[i].argsort()[-10:][::-1]
        lines.append(f"{cls}: " + ", ".join(names[top]))
    (RESULTS / "top_features.txt").write_text("\n".join(lines))

    joblib.dump(model, RESULTS / "baseline.joblib")


if __name__ == "__main__":
    main()
