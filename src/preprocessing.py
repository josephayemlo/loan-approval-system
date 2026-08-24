"""
Reusable preprocessing pipeline for the loan approval ML project.

This module contains only reusable preprocessing logic.
Dataset loading, exploratory analysis, train/test splitting, and
data exporting remain in the notebooks.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ============================================================
# COLUMNTRANSFORMER AND PREPROCESSING PIPELINE
# ============================================================
# ============================================================
# FEATURE GROUP DEFINITIONS
# ============================================================
MEDIAN_NUMERIC_FEATURES = [
    "Loan_to_Value",
    "Collateral_Value",
    "Bank_Balance",
    "Credit_Score",
    "Interest_Rate"
]

ZERO_FILL_FEATURES = ["Other_Income", "Investments"]

NOMINAL_FEATURES = [
    "Marital_Status",
    "Employment_Type",
    "Loan_Purpose"
]

BINARY_FEATURES = ["Residence_Type", "Collateral"]

EDUCATION_CATEGORIES = [[
    "No Formal",
    "High School",
    "Diploma",
    "Graduate",
    "Post Graduate",
    "PhD"
]]

# ============================================================
# PIPELINE FACTORY
# ============================================================
def create_preprocessing_pipeline():
    """Builds and returns the unfitted preprocessing pipeline."""
    
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # FIXED: Removed drop="first" to allow handle_unknown="ignore" safely
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    tax_return_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["Rural", "Urban"], ["No", "Yes"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    education_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=EDUCATION_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("median_numeric", numeric_pipeline, MEDIAN_NUMERIC_FEATURES),
            ("zero_fill", SimpleImputer(
                strategy="constant",
                fill_value=0
            ), ZERO_FILL_FEATURES),
            ("nominal", nominal_pipeline, NOMINAL_FEATURES),
            ("tax_return", tax_return_pipeline, ["Tax_Return_Filed"]),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("education", education_pipeline, ["Education"])
        ],
        remainder="passthrough"
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])