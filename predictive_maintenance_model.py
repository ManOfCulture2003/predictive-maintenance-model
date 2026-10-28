"""Educational maintenance-risk classifier; synthetic data only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["temperature_c", "vibration_mm_s", "pressure_bar", "equipment_age_years"]

def generate_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(72, 12, n).clip(20, 125)
    vibration = rng.lognormal(1, .42, n)
    pressure = rng.normal(5.2, .9, n).clip(.5, 11)
    age = rng.uniform(0, 15, n)
    risk = (temp-72)/12 + (vibration-3)*.7 + (age-7)*.11
    risk += np.abs(pressure-5.2)*.2 + rng.normal(0, 1.1, n)
    label = rng.binomial(1, 1/(1+np.exp(-risk)))
    return pd.DataFrame({"temperature_c":temp, "vibration_mm_s":vibration,
                         "pressure_bar":pressure, "equipment_age_years":age,
                         "maintenance_needed":label})

def evaluate(df, seed=42):
    if df.empty or df[FEATURES + ["maintenance_needed"]].isna().any().any():
        raise ValueError("Missing values or no samples")
    if not set(df["maintenance_needed"].unique()).issubset({0, 1}) or df["maintenance_needed"].nunique() != 2:
        raise ValueError("Need both binary classes 0 and 1")
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["maintenance_needed"], test_size=.25,
        stratify=df["maintenance_needed"], random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(x_train, y_train)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "dataset": "synthetic or user-supplied", "samples":len(df), "test_samples":len(y_test),
        "baseline_accuracy":float(accuracy_score(y_test, baseline.predict(x_test))),
        "accuracy":float(accuracy_score(y_test, pred)),
        "precision":float(precision_score(y_test, pred, zero_division=0)),
        "recall":float(recall_score(y_test, pred, zero_division=0)),
        "f1":float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix":confusion_matrix(y_test, pred, labels=[0, 1]).tolist()}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional input CSV with sensor columns and maintenance_needed")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    data = pd.read_csv(args.input) if args.input else generate_data(seed=args.seed)
    metrics = evaluate(data, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
