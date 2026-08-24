# Loan Approval Prediction with a Decision Tree

An end-to-end machine-learning project that predicts a loan application's status (`Approved` or `Rejected`) from applicant, financial, and loan information. The project uses a Scikit-learn pipeline so that preprocessing and the Decision Tree classifier are applied together during training and inference.

> The dataset is synthetic and this repository is intended for learning and demonstration. It must not be used as the sole basis for real lending decisions.

## Highlights

- Leakage-aware workflow: data is split before preprocessing is fitted.
- Reusable `ColumnTransformer` preprocessing pipeline.
- Five-fold cross-validation on the training partition.
- Holdout-set evaluation using accuracy, a confusion matrix, and a classification report.
- Saved end-to-end pipeline for single-applicant predictions.

## Model Results

The recorded run in `notebooks/02_model_training.ipynb` reports:

| Metric | Result |
| --- | ---: |
| Mean 5-fold cross-validation accuracy | 93.72% |
| Cross-validation standard deviation | 0.27% |
| Holdout test accuracy | 93.45% |
| Test samples | 4,000 |

The close cross-validation and holdout scores indicate consistent performance on this synthetic dataset. Accuracy alone is not enough for a lending use case, so the notebook also reports precision, recall, F1-score, and the confusion matrix for both classes.

## Project Structure

```text
.
├── data/
│   ├── raw/                         # Source synthetic dataset
│   └── splits/                      # Raw train/test feature and target splits
├── models/
│   └── loan_approval_pipeline.pkl   # Saved preprocessing + classifier pipeline
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  # Data audit, validation, splitting, feature engineering
│   ├── 02_model_training.ipynb      # Cross-validation, training, evaluation, serialization
│   └── 03_model_prediction.ipynb    # Single-applicant inference
└── src/
    └── preprocessing.py             # Reusable preprocessing-pipeline factory
```

## Data and Features

The source dataset contains 20,000 synthetic loan applications. The model receives 33 raw input features after feature selection and engineering, including:

- Applicant profile: age, education, marital status, dependents, residence type, and employment information.
- Financial profile: income, savings, investments, bank balance, credit score, credit-card utilization, defaults, and missed payments.
- Loan details: purpose, amount, tenure, interest rate, collateral, collateral value, and loan-to-value ratio.
- Engineered feature: `Total_Debt_Exposure`, calculated as `Existing_Loan_Amount + Loan_Amount`.

The target is `Loan_Status`, with the classes `Approved` and `Rejected`.

## Preprocessing Design

`src/preprocessing.py` defines the preprocessing pipeline used by the model:

| Feature group | Treatment |
| --- | --- |
| `Loan_to_Value`, `Collateral_Value`, `Bank_Balance`, `Credit_Score`, `Interest_Rate` | Median imputation |
| `Other_Income`, `Investments` | Constant zero-fill |
| `Marital_Status`, `Employment_Type`, `Loan_Purpose`, `Tax_Return_Filed` | Imputation followed by one-hot encoding |
| `Residence_Type`, `Collateral` | Explicit ordinal encoding |
| `Education` | Ordered ordinal encoding |

The preprocessing pipeline is nested inside the model pipeline. Therefore, each cross-validation fold learns its preprocessing values from that fold's training portion only, and new applicants are transformed consistently at prediction time.

## Getting Started

### Prerequisites

- Python 3.12 or later
- Jupyter Notebook or JupyterLab

Create a virtual environment, then install the project's pinned dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the Workflow

Run the notebooks in this order from the `notebooks/` directory:

1. `01_data_preprocessing.ipynb` — inspect data, validate domains, create train/test splits, and engineer `Total_Debt_Exposure`.
2. `02_model_training.ipynb` — perform cross-validation, train the complete pipeline, evaluate the holdout set, and save the artifact.
3. `03_model_prediction.ipynb` — load the saved pipeline and predict one new applicant.

## Make a Prediction

Load only trusted serialized model files. `joblib.load()` can execute malicious code when pointed at an untrusted pickle file.

```python
from pathlib import Path

import joblib
import pandas as pd

model_path = Path("models/loan_approval_pipeline.pkl")
pipeline = joblib.load(model_path)

applicant = pd.DataFrame([{
    # Provide every raw feature used during training, including
    # Total_Debt_Exposure.
}])

prediction = pipeline.predict(applicant)[0]
probabilities = dict(zip(pipeline.classes_, pipeline.predict_proba(applicant)[0]))

print(prediction)
print(probabilities)
```

The input column names must exactly match the pipeline's 33 training features. See `03_model_prediction.ipynb` for a complete example applicant and an input-schema check.

## Responsible Use and Limitations

- This is a synthetic educational dataset; its measured performance does not demonstrate real-world lending suitability.
- A model probability is not a guarantee or a final lending decision.
- Any real deployment requires data-quality monitoring, fairness testing, human review, clear adverse-action processes, security controls, and compliance with applicable laws and regulations.
- The model should be retrained and re-evaluated when the data distribution, policies, or product rules change.

## License

No license has been specified for this repository. Add a license file before redistributing or using this work beyond personal or educational purposes.
