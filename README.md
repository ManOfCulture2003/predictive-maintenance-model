# Predictive Maintenance Model — Synthetic Classification Lab

A **reproducible educational classification experiment**, not a real equipment-failure predictor. The generated data uses abstract `synthetic_feature_0` through `synthetic_feature_9`; it does **not** contain temperature, pressure, vibration, machine IDs, or actual breakdowns.

## Run locally (free)

Python 3.11 or newer:

```bash
git clone https://github.com/ManOfCulture2003/predictive-maintenance-model.git
cd predictive-maintenance-model
python -m pip install -r requirements.txt
python predictive_maintenance_model.py
python predictive_maintenance_model.py --samples 200 --seed 42 --output report.json
python -m unittest discover -s tests -v
```

The optional `report.json` is written locally; avoid committing reports from private data if you later adapt this experiment.

## What the evaluation actually tests

1. Generate imbalanced synthetic binary labels (~15% positive) with scikit-learn's `make_classification`.
2. Make a **stratified** 70/30 train/test split with a fixed seed.
3. Fit `StandardScaler` **inside the training-only pipeline** to avoid leakage from test-set statistics.
4. Compare logistic regression against a majority-class `DummyClassifier` baseline.
5. Print test-set accuracy, precision, recall, F1, ROC-AUC and confusion matrices for both.

The majority baseline can have high accuracy while missing all positive cases; recall and the confusion matrix expose that problem. Scores are reproducible for the same seed and dependency versions, but may change across library versions.

## Architecture

`make_demo_data → stratified split → [training-only scaler + logistic regression | majority baseline] → held-out metrics → optional JSON`

## Limitations and next steps

- Independent random rows are not chronological machine histories. A real project needs properly licensed equipment records, feature provenance, group/time-based splits, failure horizons, leakage checks, calibration, operational costs and external validation.
- **Do not interpret demo scores as real-world accuracy or use them to make maintenance decisions.**
- No real sensors, IoT access, or paid services are required.
- Next: add a documented public equipment dataset and evaluate temporal and equipment-group holdouts, only after verifying license and provenance.

## License

The previous README described this project as MIT licensed, but no LICENSE file was present. Licensing requires an explicit owner decision; this change does not assert that a license has been granted.
