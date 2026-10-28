# Predictive Maintenance Model

**Educational portfolio project — default data is synthetic, not real employer, airline, or machinery data.**

Fits logistic regression using named simulated equipment sensors, with stratified holdout, F1/precision/recall, a confusion matrix, and majority-class baseline.

## Setup (Python 3.11+)

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python predictive_maintenance_model.py
python -m unittest discover -v
```

The program writes generated artifacts under `outputs/` (ignored by Git). To analyze a local CSV instead, pass `--input your_file.csv`.

CSV fields: `temperature_c,vibration_mm_s,pressure_bar,equipment_age_years,maintenance_needed`. Read the input validation code before substituting your data. No company or customer information should be uploaded without permission.

## Technical approach

Fits logistic regression using named simulated equipment sensors, with stratified holdout, F1/precision/recall, a confusion matrix, and majority-class baseline.

The automated tests cover reproducibility and core validation/metrics. GitHub Actions runs the tests on pushes and pull requests.

## Limits

The generated figures and model scores demonstrate code behavior **only**. They are not real-world business improvements, production deployments, or evidence of predictive accuracy on actual organizations. More real-world validation, data quality checks, and monitoring are needed before deployment.

## Portfolio talking points

This project demonstrates Python data processing, an automated test suite, reproducible dependencies, and technical documentation. Be ready to walk through the source code yourself.
