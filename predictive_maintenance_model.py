"""Reproducible *synthetic* binary classification demo; not a deployed failure predictor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [f"synthetic_feature_{i}" for i in range(10)]


def make_demo_data(n_samples: int = 1000, seed: int = 42):
    """Generate abstract features; these are NOT actual temperature or vibration sensors."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")
    return make_classification(
        n_samples=n_samples, n_features=len(FEATURE_NAMES), n_informative=5,
        n_redundant=2, weights=[0.85, 0.15], flip_y=0.02, random_state=seed,
    )


def evaluate(y_true, predicted, scores):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predicted)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def run_experiment(n_samples: int = 1000, seed: int = 42) -> dict:
    X, y = make_demo_data(n_samples=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    # Fit the scaler ONLY on training rows, preventing preprocessing leakage.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    baseline = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    return {
        "dataset": "scikit-learn make_classification (synthetic, not sensor data)",
        "seed": seed,
        "samples": n_samples,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "test_positive_count": int(y_test.sum()),
        "feature_names": FEATURE_NAMES,
        "model": evaluate(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1]),
        "majority_baseline": evaluate(
            y_test, baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1]
        ),
        "limitations": [
            "Artificial independent rows are not equipment histories.",
            "Random split does not test generalization to future time periods or unseen machines.",
            "Metrics are not evidence of real-world predictive maintenance performance.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional local JSON report path")
    args = parser.parse_args(argv)
    try:
        result = run_experiment(args.samples, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Saved evaluation to {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
