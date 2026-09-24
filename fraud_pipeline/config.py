"""Central configuration for the vehicle insurance fraud detection pipeline.

Every phase of ``run_pipeline.py`` imports its constants from here so that the
training run, the reload/validation run and the standalone prediction script
all agree on column names, category handling and random seeds.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_CSV: Path = PROJECT_ROOT / "insurance_claims.csv"
CACHE_DIR: Path = PROJECT_ROOT / os.environ.get("FRAUD_CACHE_DIR", "cache")
EDA_DIR: Path = PROJECT_ROOT / "eda"

# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
TARGET_RAW: str = "fraud_reported"
TARGET: str = "fraud"                      # engineered 0/1 target used everywhere
POS_LABEL: int = 1                         # 1 = FRAUD
NEG_LABEL: int = 0                         # 0 = NOT FRAUD
TARGET_MAPPING: dict[str, int] = {"Y": 1, "N": 0}
CLASS_NAMES: dict[int, str] = {0: "NOT FRAUD", 1: "FRAUD"}

# --------------------------------------------------------------------------
# Reproducibility / experiment design
# --------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_SPLITS: int = 5
N_ITER_SEARCH: int = 25                    # randomized-search iterations per model
PRIMARY_METRIC: str = "pr_auc"             # optimized during tuning
SCORING_METRICS: tuple[str, ...] = (
    "pr_auc", "roc_auc", "fraud_f1", "fraud_recall",
    "fraud_precision", "balanced_accuracy", "mcc", "accuracy",
)

# Phase 17 - threshold grid (evaluated on cross-validated training predictions only)
THRESHOLD_GRID: tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
)

# --------------------------------------------------------------------------
# Raw-data cleaning rules
# --------------------------------------------------------------------------
#: Tokens that mean "no value" in this dataset / in similar CRM exports.
SENTINEL_VALUES: tuple[str, ...] = (
    "*", "?", "??", "NA", "N/A", "n/a", "na", "NULL", "null", "None",
    "none", "nan", "NaN", "NAN", "", " ", "-", "--", "missing", "unknown",
)
#: Columns that hold identifiers / free-text addresses -> never predictive.
ID_COLUMNS: tuple[str, ...] = (
    "policy_number",      # 1000 unique values for 1000 claims -> pure ID
    "incident_location",  # 1000 unique street addresses -> pure ID
    "insured_zip",        # 995 unique values for 1000 claims -> quasi-ID
)
#: Columns that carry no information at all.
BROKEN_COLUMNS: tuple[str, ...] = (
    "_c39",               # 100 % missing (spreadsheet artefact)
)
#: Raw datetime columns (kept for feature engineering, dropped afterwards).
RAW_DATE_COLUMNS: tuple[str, ...] = ("policy_bind_date", "incident_date")
#: Structured money columns that must be numeric.
COERCED_NUMERIC_COLUMNS: tuple[str, ...] = (
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "auto_year",
)

# --------------------------------------------------------------------------
# Dataset-specific domain sanity rules (Phase 5)
# --------------------------------------------------------------------------
#: columns that can never be negative for a legitimate claim
NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "policy_annual_premium", "policy_deductable", "total_claim_amount",
    "injury_claim", "property_claim", "vehicle_claim", "umbrella_limit",
    "months_as_customer", "number_of_vehicles_involved", "bodily_injuries",
    "witnesses", "incident_hour_of_the_day",
)
HOUR_OF_DAY_MAX: int = 23

# --------------------------------------------------------------------------
# Sampling / modelling
# --------------------------------------------------------------------------
CATEGORICAL_MISSING_TOKEN: str = "__MISSING__"
OHE_MIN_FREQUENCY: int = 5                 # rare-category grouping inside CV folds
SMOTE_K_NEIGHBORS: int = 5
SMOTE_TARGET_RATIO: float = 0.50           # minority -> 50 % of majority (never 1:1 forced)
SAMPLING_METHODS: tuple[str, ...] = ("none", "class_weight", "smote")

# --------------------------------------------------------------------------
# Output artefacts (Phase 26)
# --------------------------------------------------------------------------
RAW_AUDIT_CSV = PROJECT_ROOT / "raw_data_audit.csv"
CLEANING_LOG_CSV = PROJECT_ROOT / "cleaning_log.csv"
DUPLICATE_ANALYSIS_CSV = PROJECT_ROOT / "duplicate_analysis.csv"
OUTLIER_ANALYSIS_CSV = PROJECT_ROOT / "outlier_analysis.csv"
DATA_TYPES_CSV = PROJECT_ROOT / "data_types_report.csv"
LEAKAGE_AUDIT_CSV = PROJECT_ROOT / "leakage_audit.csv"
CLEANED_DATASET_CSV = PROJECT_ROOT / "cleaned_dataset.csv"
MODEL_COMPARISON_CSV = PROJECT_ROOT / "model_comparison.csv"
THRESHOLD_TABLE_CSV = PROJECT_ROOT / "threshold_optimization.csv"
FEATURE_IMPORTANCE_CSV = PROJECT_ROOT / "feature_importance.csv"
FEATURE_IMPORTANCE_PNG = PROJECT_ROOT / "feature_importance.png"
FINAL_METRICS_JSON = PROJECT_ROOT / "final_metrics.json"
MODEL_PKL = PROJECT_ROOT / "vehicle_insurance_fraud_model.pkl"
MODEL_CARD_MD = PROJECT_ROOT / "README.md"
CONFUSION_MATRIX_PNG = PROJECT_ROOT / "confusion_matrix.png"
ROC_CURVE_PNG = PROJECT_ROOT / "roc_curve.png"
PR_CURVE_PNG = PROJECT_ROOT / "precision_recall_curve.png"
CV_RESULTS_JSON = PROJECT_ROOT / "cv_results.json"
DATA_DICTIONARY_CSV = PROJECT_ROOT / "data_dictionary.csv"

RUN_MANIFEST_JSON = CACHE_DIR / "run_manifest.json"
TRAIN_CSV = CACHE_DIR / "train.csv"
TEST_CSV = CACHE_DIR / "test.csv"
TUNED_PARAMS_JSON = CACHE_DIR / "tuned_params.json"
COMPARISON_CACHE_CSV = CACHE_DIR / "comparison_cache.csv"
OOF_VALIDATION_CSV = CACHE_DIR / "oof_validation_predictions.csv"
THRESHOLD_CACHE_JSON = CACHE_DIR / "threshold_choice.json"
RELOAD_REFERENCE_JSON = CACHE_DIR / "reload_reference_predictions.json"
UNSEEN_SAMPLE_CSV = CACHE_DIR / "unseen_raw_claims_sample.csv"
