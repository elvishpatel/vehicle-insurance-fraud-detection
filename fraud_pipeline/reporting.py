"""Phase 27 / 28 - README generation and the final console summary.

Everything in the generated README is filled in from the artefacts that were
actually produced by the run (metrics JSON, comparison table, cleaning log,
leakage audit), so the numbers can never drift from the model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as cfg


def _fmt(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if df is None or df.empty:
        return "_(empty)_"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, bool):
                cells.append(str(value))
            elif isinstance(value, float):
                cells.append(floatfmt.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Placeholders use the form <<key>> so that literal { } and % stay untouched.
README_TEMPLATE = """# Vehicle Insurance Fraud Detection - Production Model

**Generated:** <<generated_at>> | **Package version:** <<package_version>> | **Random seed:** 42

---

## 1. Dataset

| Property | Value |
|---|---|
| Source file | `<<raw_file>>` |
| Raw shape | <<raw_rows>> rows x <<raw_columns>> columns |
| Cleaned modelling frame | <<clean_rows>> rows, <<clean_columns>> engineered features |
| Training split | <<train_rows>> rows (<<train_fraud>> fraud / <<train_nonfraud>> not fraud) |
| Test split (untouched) | <<test_rows>> rows (<<test_fraud>> fraud / <<test_nonfraud>> not fraud) |
| Exact duplicate rows removed | <<dup_rows>> |
| Rows removed for a missing label | <<missing_target_rows>> |
| Identifier columns excluded | <<id_columns>> |
| Constant / no-information columns dropped | <<constant_columns>> |

## 2. Target

The target was detected automatically as `<<target_column>>` and **verified before mapping**:

<<target_table>>

Mapping applied: N -> 0 (not fraud), Y -> 1 (fraud). <<missing_target_rows>> row(s) with a genuinely
missing label were removed.

## 3. Cleaning (deterministic, label-free)

Every operation is logged in `cleaning_log.csv`; the full table is reproduced here:

<<cleaning_table>>

No missing value was replaced with `0`, and no statistic of the data (mean / median
/ mode / target rate) was used while cleaning - which is why cleaning before the
train/test split cannot leak information. All imputation and encoding statistics
are fitted *inside* the pipeline, on each training fold.

## 4. Features

* Numeric attributes: median imputation + an explicit "was missing" indicator column.
* Categorical attributes: a dedicated `__MISSING__` level instead of an invented value.
* One-hot encoding with `handle_unknown="ignore"` and `min_frequency=<<ohe_min_frequency>>`, so
  unseen categories in production never raise an error.
* Deterministic, investigation-time-safe engineered features: date parts of
  `policy_bind_date` / `incident_date`, `policy_age_days`, `vehicle_age_years`,
  bodily/property limits parsed out of `policy_csl`, claim-share ratios, `log1p`
  claim amounts, tenure ratios, financial flags and
  `policy_state_matches_incident_state`.

<<n_engineered>> engineered features are used in total. Feature importance is reported at
raw-feature level (`feature_importance.csv`) and at one-hot level
(`feature_importance_encoded.csv`).

## 5. Removed columns and reasons

<<removed_table>>

## 6. Data-leakage prevention

The complete audit is `leakage_audit.csv`; its empirical diagnostics were computed
on the **training split only**.

* The strongest single training-set signal is `<<top_feature>>` with a single-feature
  ROC-AUC of <<top_feature_auc>> - no perfect separator and no duplicate of the target exists.
* Identifiers (`policy_number`, `incident_location`, `insured_zip`) are excluded: they are
  1:1 with the claim row and could only memorise the training set.
* `total_claim_amount` equals `injury_claim + property_claim + vehicle_claim` exactly
  (verified in the audit). This is recorded as **redundancy**, not leakage: all four
  values exist at claim intake, before any investigation. Perfectly collinear pairs
  are listed explicitly in the audit.
* The test set was never used for preprocessing, resampling, feature selection,
  hyper-parameter tuning, threshold selection or model selection. It is read exactly
  once, in Phase 19.

## 7. Class imbalance strategy

Actual imbalance: **<<fraud_pct>>% fraud / <<nonfraud_pct>>% not fraud** - the dataset is never
forced to 50/50. Three options were compared on identical folds:

| Option | Implementation | Where it is applied |
|---|---|---|
| A. Original distribution | no resampling, no weights | - |
| B. Class weighting | `BalancedWeightClassifier` (balanced `sample_weight` from the fold's own labels) | inside `fit` of every fold |
| C. SMOTENC | `SmoteNCSampler` (numeric interpolation + categorical copy, k=<<smote_k>>), minority raised to <<smote_ratio>> of the majority | `fit_transform` of the training fold only |

SMOTE is never applied to the whole dataset, to a validation fold or to the test
set: the sampler's `transform` is a pure pass-through and scikit-learn only calls
`fit_transform` while fitting. Results: `model_comparison.csv` (`Sampling_Method`).

## 8. Models tested and comparison

Baselines: DummyClassifier, LogisticRegression, DecisionTree, RandomForest,
ExtraTrees, HistGradientBoosting, XGBoost, LightGBM, CatBoost - each in the three
imbalance configurations where applicable. All numbers below are
<<cv_splits>>-fold `StratifiedKFold(shuffle=True, random_state=42)` results on the **training
split** (out-of-fold predictions, preprocessing and resampling inside every fold):

<<comparison_table>>

Ranking rule (no test data involved): PR-AUC first, then fraud F1, fraud recall,
balanced accuracy, MCC, fraud precision, ROC-AUC, accuracy.

## 9. Hyper-parameter optimisation

`RandomizedSearchCV`, <<n_iter>> iterations per model, 5 stratified folds, refit on **PR-AUC**,
`random_state=42`. Selected parameters:

<<tuned_params_table>>

## 10. Final model

| Property | Value |
|---|---|
| Model | **<<model_name>>** |
| Sampling method | <<sampling_method>> |
| Decision threshold | **<<threshold>>** |
| Selection evidence (CV, training only) | PR-AUC <<cv_pr_auc>>, fraud F1 <<cv_f1>>, balanced accuracy <<cv_ba>>, ROC-AUC <<cv_roc_auc>> |

### Threshold selection (Phase 17)

The full sweep is in `threshold_optimization.csv`. Rule: maximise fraud F1 on the
**out-of-fold training predictions**; among all thresholds within 0.005 F1 of the
best, take the one with the highest fraud precision. At the chosen threshold
<<threshold>> the validation fraud precision was <<val_precision>> and the validation fraud recall
<<val_recall>>. The test set was not involved in any way.

## 11. Final test performance (one shot, untouched 20% test set)

| Metric | Value |
|---|---|
| Accuracy | <<accuracy>> |
| Precision (fraud) | <<precision>> |
| Recall (fraud) | <<recall>> |
| F1 (fraud) | <<f1>> |
| ROC-AUC | <<roc_auc>> |
| PR-AUC | <<pr_auc>> |
| Balanced accuracy | <<balanced_accuracy>> |
| MCC | <<mcc>> |

Confusion matrix:

| | Predicted not fraud | Predicted fraud |
|---|---|---|
| **Actual not fraud** | TN = <<tn>> | FP = <<fp>> |
| **Actual fraud** | FN = <<fn>> | TP = <<tp>> |

<<report_table>>

### Honest note on generalisation

The dataset contains <<raw_rows>> claims, so the test set holds only <<test_fraud>> fraud cases. A
single test set of that size cannot resolve differences of a few F1 points, and a
lucky/unlucky split can move the numbers by several points. That is exactly why
every decision (model, parameters, sampling, threshold) was taken on
cross-validated training data, and why the cross-validated table in section 8 is
reported next to the held-out result. If the cross-validated PR-AUC is clearly
higher than the test PR-AUC, the honest reading is a small-sample effect plus mild
optimism from repeated model selection - not a broken evaluation.

## 12. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 13. Load the saved model

```python
import joblib
bundle = joblib.load("vehicle_insurance_fraud_model.pkl")
print(bundle["model_name"], bundle["threshold"], bundle["target_mapping"])
```

## 14. Predict a new claim

```bash
python predict_fraud.py          # scores a built-in demo claim
python validate_model.py         # reload test + unseen-claim test (Phases 24-25)
```

```python
from predict_fraud import load_model, predict_fraud
import pandas as pd

model = load_model("vehicle_insurance_fraud_model.pkl")
new_claim = pd.read_csv("new_claims.csv")       # same raw columns as insurance_claims.csv

print(predict_fraud(new_claim, bundle=model))
# {'prediction': 'FRAUD', 'fraud_probability': 0.83, 'threshold': 0.4, 'model': 'XGBoost', ...}
```

Cleaning, feature engineering, imputation and encoding all happen inside the
pipeline - never pre-process a claim by hand. Raw columns that are absent are
treated as unknown and echoed back in `missing_input_columns`.

## 15. Artefacts produced by the run

<<artefacts_table>>

## 16. Reproducing the run

```bash
python run_pipeline.py                 # full end-to-end run (all phases)
python run_pipeline.py --stage audit   # Phases 1-10: audit, cleaning, leakage, EDA, split
python run_pipeline.py --stage baseline
python run_pipeline.py --stage tune    # Phase 15-17 (RandomizedSearchCV)
python run_pipeline.py --stage final   # Phase 18-22 + summary + README
python run_pipeline.py --stage validate
```

Stages communicate through `cache/` (train/test frames, CV tables, tuned params,
threshold choice), so a full run is reproducible and can be resumed.
"""


def render_readme(context: dict) -> str:
    """Fill the template; unknown keys are left as-is so gaps are visible."""
    text = README_TEMPLATE
    for key, value in context.items():
        text = text.replace(f"<<{key}>>", str(value))
    return text


def print_final_summary(context: dict) -> None:
    """Phase 28 - the exact summary block requested."""
    m = context["metrics"]
    print()
    print("=============================================")
    print("VEHICLE INSURANCE FRAUD MODEL COMPLETE")
    print("=============================================")
    print()
    print("RAW DATA")
    print(f"Rows: {context['raw_rows']}")
    print(f"Columns: {context['raw_columns']}")
    print()
    print("CLEANED DATA")
    print(f"Rows: {context['clean_rows']}")
    print(f"Columns: {context['clean_columns']}")
    print()
    print("TARGET")
    print(f"Fraud: {context['fraud_count']}")
    print(f"Not Fraud: {context['nonfraud_count']}")
    print(f"Fraud %: {context['fraud_pct']}")
    print()
    print("REMOVED")
    print(f"Duplicate rows: {context['dup_rows']}")
    print(f"Invalid rows: {context['invalid_rows']}")
    print(f"Leakage columns: {context['leakage_columns']}")
    print(f"Identifier columns: {context['id_columns']}")
    print()
    print("BEST MODEL")
    print(f"Model: {context['model_name']}")
    print(f"Sampling: {context['sampling_method']}")
    print(f"Threshold: {context['threshold']}")
    print()
    print("FINAL TEST PERFORMANCE")
    print(f"Accuracy: {m['Accuracy']:.4f}")
    print(f"Precision: {m['Precision']:.4f}")
    print(f"Recall: {m['Recall']:.4f}")
    print(f"F1: {m['F1']:.4f}")
    print(f"ROC-AUC: {m['ROC_AUC']:.4f}")
    print(f"PR-AUC: {m['PR_AUC']:.4f}")
    print(f"Balanced Accuracy: {m['Balanced_Accuracy']:.4f}")
    print(f"MCC: {m['MCC']:.4f}")
    print()
    print("FRAUD PERFORMANCE")
    print(f"Fraud Precision: {m['Fraud_Precision']:.4f}")
    print(f"Fraud Recall: {m['Fraud_Recall']:.4f}")
    print(f"Fraud F1: {m['Fraud_F1']:.4f}")
    print()
    print("CONFUSION MATRIX")
    print(f"TN: {m['TN']}")
    print(f"FP: {m['FP']}")
    print(f"FN: {m['FN']}")
    print(f"TP: {m['TP']}")
    print()
    print("FINAL FILE")
    print(cfg.MODEL_PKL.name)
    print()
    print("RELOAD TEST")
    print(context.get("reload_test", "PENDING"))
    print()
    print("=============================================")
    print()


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
