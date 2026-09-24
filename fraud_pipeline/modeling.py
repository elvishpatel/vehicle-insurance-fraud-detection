"""Phase 10 - 21 - splitting, preprocessing, models, tuning, selection, final fit.

Leakage discipline enforced here:

* the split happens **once** (stratified, ``random_state=42``) and the test frame
  is only ever touched by :func:`final_test_evaluation`;
* every learned transformation (imputation, encoder categories, scaling) lives in
  the pipeline and is fitted on training folds only;
* resampling happens inside ``fit`` on the training fold only;
* hyper-parameters and the decision threshold are chosen from cross-validated
  *training* predictions.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from . import config as cfg
from .composite import ResamplingPipeline
from .features import FraudFeatureEngineer, split_feature_types
from .metrics import cv_summary_from_scores, make_scorers
from .sampling import SmoteNCSampler

try:  # pragma: no cover - optional dependencies
    from xgboost import XGBClassifier
except Exception:  # noqa: BLE001
    XGBClassifier = None
try:  # pragma: no cover
    from lightgbm import LGBMClassifier
except Exception:  # noqa: BLE001
    LGBMClassifier = None
try:  # pragma: no cover
    from catboost import CatBoostClassifier
except Exception:  # noqa: BLE001
    CatBoostClassifier = None


# --------------------------------------------------------------------------
# Phase 10 - split
# --------------------------------------------------------------------------


def split_data(clean_df: pd.DataFrame, target: str = cfg.TARGET):
    """Stratified 80/20 split on the cleaned raw frame (single source of truth)."""
    X = clean_df.drop(columns=[target])
    y = clean_df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE
    )
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
            y_train.reset_index(drop=True), y_test.reset_index(drop=True))


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=cfg.CV_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE)


# --------------------------------------------------------------------------
# Phase 11 - preprocessing
# --------------------------------------------------------------------------


def build_preprocessor(numeric_features, categorical_features, scale: str | None = None) -> ColumnTransformer:
    """Imputation + encoding built for the *engineered* frame.

    * numeric: median imputation (never 0) + ``add_indicator`` so that "was
      missing" stays visible; optional scaler (``robust`` for the linear model);
    * categorical: an explicit ``__MISSING__`` level instead of inventing a value,
      then one-hot encoding with ``handle_unknown="ignore"`` and rare-category
      grouping (``min_frequency``).
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale == "standard":
        numeric_steps.append(("scale", StandardScaler()))
    elif scale == "robust":
        numeric_steps.append(("scale", RobustScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(numeric_features)),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=cfg.CATEGORICAL_MISSING_TOKEN)),
                ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=cfg.OHE_MIN_FREQUENCY,
                                      sparse_output=False)),
            ]), list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------
# Phase 12 - model-specific class weighting (per training fold, no leakage)
# --------------------------------------------------------------------------


class BalancedWeightClassifier(ClassifierMixin, BaseEstimator):
    """Wrap an estimator and fit it with ``compute_sample_weight('balanced', y)``.

    The weights are derived from the labels the wrapper *receives*, i.e. from the
    current training fold only.  This gives every supported model (linear, tree,
    boosting) the same leakage-free class-weighting behaviour - the "model
    specific class weights" option of Phase 12.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.estimator_ = clone(self.base)
        y = np.asarray(y).ravel()
        weights = compute_sample_weight("balanced", y)
        params = inspect.signature(self.estimator_.fit).parameters
        if "sample_weight" in params:
            self.estimator_.fit(X, y, sample_weight=weights)
            self.weighting_applied_ = True
        else:
            self.estimator_.fit(X, y)
            self.weighting_applied_ = False
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    @property
    def feature_importances_(self):
        return getattr(self.estimator_, "feature_importances_", None)

    @property
    def coef_(self):
        return getattr(self.estimator_, "coef_", None)


# --------------------------------------------------------------------------
# Phase 14 / 15 - model zoo and search spaces
# --------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """A candidate model plus its randomized-search space (empty = no tuning)."""

    name: str
    estimator: object
    param_distributions: dict = field(default_factory=dict)
    tunable: bool = False
    scaling: str | None = None          # None (trees) or 'robust' (linear model)
    notes: str = ""


def model_zoo(random_state: int = cfg.RANDOM_STATE) -> dict[str, ModelSpec]:
    """All Phase-14 baseline models; Phase-15 spaces for the 5 main learners."""
    zoo: dict[str, ModelSpec] = {}

    zoo["DummyClassifier"] = ModelSpec(
        "DummyClassifier", DummyClassifier(strategy="prior", random_state=random_state),
        notes="no-skill reference: always predicts the majority class probability")

    zoo["LogisticRegression"] = ModelSpec(
        "LogisticRegression",
        LogisticRegression(max_iter=5000, solver="liblinear", random_state=random_state),
        param_distributions={
            "classifier__C": np.logspace(-2, 2, 25),
            "classifier__penalty": ["l1", "l2"],
        },
        tunable=True, scaling="robust",
        notes="linear reference model, RobustScaler to resist claim-amount outliers")

    zoo["DecisionTree"] = ModelSpec(
        "DecisionTree", DecisionTreeClassifier(random_state=random_state),
        param_distributions={
            "classifier__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
            "classifier__min_samples_leaf": [1, 2, 3, 5, 8, 12, 20],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__criterion": ["gini", "entropy"],
        },
        tunable=True,
        notes="single tree: interpretable but high variance")

    for name, cls in (("RandomForest", RandomForestClassifier),
                      ("ExtraTrees", ExtraTreesClassifier)):
        zoo[name] = ModelSpec(
            name, cls(n_estimators=400, random_state=random_state, n_jobs=-1),
            param_distributions={
                "classifier__n_estimators": [300, 400, 600, 800],
                "classifier__max_depth": [4, 6, 8, 10, 14, None],
                "classifier__min_samples_leaf": [1, 2, 3, 5, 8],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__max_features": ["sqrt", "log2", 0.3, 0.5],
                "classifier__criterion": ["gini", "entropy"],
            },
            tunable=True,
            notes="bagged trees, robust to outliers and monotone transforms")

    zoo["HistGradientBoosting"] = ModelSpec(
        "HistGradientBoosting",
        HistGradientBoostingClassifier(random_state=random_state),
        param_distributions={
            "classifier__learning_rate": np.logspace(-2.3, -0.5, 20),
            "classifier__max_iter": [200, 300, 500, 800],
            "classifier__max_leaf_nodes": [15, 31, 63],
            "classifier__min_samples_leaf": [10, 20, 30, 50],
            "classifier__l2_regularization": [0.0, 0.1, 1.0, 10.0],
            "classifier__max_bins": [64, 128, 255],
        },
        tunable=False,
        notes="fast histogram gradient boosting reference")

    if XGBClassifier is not None:
        zoo["XGBoost"] = ModelSpec(
            "XGBoost",
            XGBClassifier(random_state=random_state, n_jobs=-1, tree_method="hist",
                          eval_metric="logloss"),
            param_distributions={
                "classifier__n_estimators": [200, 300, 500, 700],
                "classifier__learning_rate": np.logspace(-2.3, -0.5, 20),
                "classifier__max_depth": [2, 3, 4, 5, 6, 8],
                "classifier__min_child_weight": [1, 2, 4, 8],
                "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.4, 0.5, 0.7, 0.9, 1.0],
                "classifier__gamma": [0.0, 0.1, 0.5, 1.0],
                "classifier__reg_lambda": np.logspace(-1, 1, 9),
                "classifier__reg_alpha": [0.0, 0.1, 1.0],
            },
            tunable=True, notes="gradient boosted trees")

    if LGBMClassifier is not None:
        zoo["LightGBM"] = ModelSpec(
            "LightGBM",
            LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1,
                           min_child_samples=20),
            param_distributions={
                "classifier__n_estimators": [200, 300, 500, 700],
                "classifier__learning_rate": np.logspace(-2.3, -0.5, 20),
                "classifier__num_leaves": [7, 15, 31, 63],
                "classifier__min_child_samples": [5, 10, 20, 40],
                "classifier__subsample": [0.6, 0.8, 1.0],
                "classifier__subsample_freq": [0, 1],
                "classifier__colsample_bytree": [0.4, 0.6, 0.8, 1.0],
                "classifier__reg_lambda": np.logspace(-1, 1, 9),
                "classifier__reg_alpha": [0.0, 0.1, 1.0],
            },
            tunable=True, notes="leaf-wise gradient boosting")

    if CatBoostClassifier is not None:
        zoo["CatBoost"] = ModelSpec(
            "CatBoost",
            CatBoostClassifier(random_state=random_state, verbose=0,
                               allow_writing_files=False, thread_count=-1),
            param_distributions={
                "classifier__iterations": [300, 500, 800],
                "classifier__depth": [4, 5, 6, 7],
                "classifier__learning_rate": np.logspace(-2, -0.5, 15),
                "classifier__l2_leaf_reg": [1, 3, 5, 10, 20],
                "classifier__random_strength": [0.5, 1.0, 2.0],
                "classifier__bootstrap_type": ["Bernoulli"],
                "classifier__subsample": [0.7, 0.85, 1.0],
            },
            tunable=True, notes="ordered boosting, strong on small tabular data")

    return zoo


# --------------------------------------------------------------------------
# pipeline assembly
# --------------------------------------------------------------------------


def make_sampler(numeric_features, categorical_features, sampling_ratio: float = cfg.SMOTE_TARGET_RATIO):
    return SmoteNCSampler(
        numeric_features=list(numeric_features),
        categorical_features=list(categorical_features),
        sampling_ratio=sampling_ratio,
        k_neighbors=cfg.SMOTE_K_NEIGHBORS,
        random_state=cfg.RANDOM_STATE,
    )


def feature_lists_from_train(train_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Column lists for the preprocessor/sampler, derived from the training split.

    ``drop_constant=False`` is used deliberately: the set of engineered columns must
    not depend on the fold, otherwise a column could be constant in one training
    fold and absent for the preprocessor built for it.  Constant columns
    (e.g. the explicitly verified ``claim_consistency_diff``) are therefore kept
    and simply carry no information; the leakage audit reports them.
    """
    engineer = FraudFeatureEngineer(drop_constant=False)
    engineered = engineer.fit_transform(train_frame)
    return split_feature_types(engineered)


def build_pipeline(
    estimator,
    numeric_features,
    categorical_features,
    sampling: str = "none",
    scaling: str | None = None,
    params: dict | None = None,
) -> ResamplingPipeline:
    """Assemble the deployable composite pipeline.

    Parameters
    ----------
    estimator:
        The (possibly tuned) base classifier.
    sampling:
        ``none`` | ``class_weight`` | ``smote`` (see Phase 12).
    params:
        Optional overrides already validated by the search (they are applied to a
        cloned estimator so that the search object stays untouched).
    """
    import warnings

    base = clone(estimator)
    if params:
        base.set_params(**{k.replace("classifier__", ""): v
                           for k, v in params.items() if k.startswith("classifier__")})

    if sampling == "class_weight":
        classifier = BalancedWeightClassifier(base=base)
    elif sampling == "smote":
        classifier = base
    elif sampling == "none":
        classifier = base
    else:
        raise ValueError(f"Unknown sampling method {sampling!r}")

    sampler = (make_sampler(numeric_features, categorical_features)
               if sampling == "smote" else None)

    preprocessor = build_preprocessor(numeric_features, categorical_features, scale=scaling)
    with warnings.catch_warnings():
        # the OHE/rare-category machinery can emit benign dtype warnings
        warnings.simplefilter("ignore")
        pipeline = ResamplingPipeline(
            classifier=classifier,
            preprocessor=preprocessor,
            sampler=sampler,
            feature_engineer=FraudFeatureEngineer(drop_constant=False),
        )
    return pipeline


# --------------------------------------------------------------------------
# Phase 13 - cross validation
# --------------------------------------------------------------------------


def cross_validate_pipeline(pipeline, X, y, cv=None, scorers=None) -> dict:
    """5-fold stratified CV with all preprocessing/resampling inside the folds."""
    cv = cv or make_cv()
    scorers = scorers or make_scorers()
    result = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, n_jobs=None,
                            error_score="raise", return_train_score=False)
    return result


def oof_probabilities(pipeline, X, y, cv=None) -> np.ndarray:
    """Out-of-fold fraud probabilities (training data only - used for thresholds)."""
    cv = cv or make_cv()
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=None)
    return np.asarray(proba)[:, 1]


def evaluate_comparison_row(name: str, sampling: str, cv_result: dict) -> dict:
    """One row of ``model_comparison.csv`` (cross-validated, training split only)."""
    summary = cv_summary_from_scores(cv_result)
    row = {
        "Model": name,
        "Sampling_Method": sampling,
        "Accuracy": round(summary.get("accuracy", float("nan")), 4),
        "Precision": round(summary.get("fraud_precision", float("nan")), 4),
        "Recall": round(summary.get("fraud_recall", float("nan")), 4),
        "F1": round(summary.get("fraud_f1", float("nan")), 4),
        "ROC_AUC": round(summary.get("roc_auc", float("nan")), 4),
        "PR_AUC": round(summary.get("pr_auc", float("nan")), 4),
        "Balanced_Accuracy": round(summary.get("balanced_accuracy", float("nan")), 4),
        "MCC": round(summary.get("mcc", float("nan")), 4),
        "PR_AUC_Std": round(summary.get("pr_auc_std", float("nan")), 4),
        "Evaluation": f"{cfg.CV_SPLITS}-fold stratified CV on the training split",
    }
    return row


# --------------------------------------------------------------------------
# Phase 15 - hyper-parameter optimisation
# --------------------------------------------------------------------------


def tune_model(
    spec: ModelSpec,
    numeric_features,
    categorical_features,
    X_train,
    y_train,
    n_iter: int = cfg.N_ITER_SEARCH,
    cv=None,
) -> dict:
    """``RandomizedSearchCV`` optimizing PR-AUC over the 5 stratified folds."""
    cv = cv or make_cv()
    pipeline = build_pipeline(spec.estimator, numeric_features, categorical_features,
                             sampling="none", scaling=spec.scaling)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spec.param_distributions,
        n_iter=int(n_iter),
        scoring=make_scorers(),
        refit=cfg.PRIMARY_METRIC,
        cv=cv,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1,
        pre_dispatch="2*n_jobs",
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    best_index = search.best_index_
    keys = [k for k in search.cv_results_ if k.startswith("mean_test_")]
    best_cv = {
        row.replace("mean_test_", ""): search.cv_results_[row][best_index]
        for row in keys
    }
    return {
        "model": spec.name,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "best_cv_metrics": {k: float(v) for k, v in best_cv.items()},
        "n_iter": int(n_iter),
        "search": search,
    }


# --------------------------------------------------------------------------
# Phase 16 / 18 - model selection (training-side evidence only)
# --------------------------------------------------------------------------


def selection_ranking(comparison: pd.DataFrame) -> pd.DataFrame:
    """Objective ranking used to pick the final model.

    Priority (documented, no test data involved):
    PR-AUC -> fraud F1 -> fraud recall -> balanced accuracy -> MCC -> fraud
    precision -> ROC-AUC -> accuracy.  Metrics are rounded to 2 decimals for the
    sort so that candidates within ~0.005 are effectively tied.
    """
    df = comparison.copy()
    df = df[df["Model"] != "DummyClassifier"]
    rounded = {}
    for col in ["PR_AUC", "F1", "Recall", "Balanced_Accuracy", "MCC", "Precision", "ROC_AUC", "Accuracy"]:
        rounded[f"_{col}"] = df[col].round(2)
    df = pd.concat([df, pd.DataFrame(rounded, index=df.index)], axis=1)
    sort_cols = [f"_{c}" for c in ["PR_AUC", "F1", "Recall", "Balanced_Accuracy",
                                   "MCC", "Precision", "ROC_AUC", "Accuracy"]]
    ranked = (df.sort_values(sort_cols, ascending=False)
                .drop(columns=sort_cols)
                .reset_index(drop=True))
    ranked.insert(0, "Rank", ranked.index + 1)
    return ranked


# --------------------------------------------------------------------------
# Phase 19 - final, single-shot test evaluation
# --------------------------------------------------------------------------


def final_test_evaluation(pipeline, X_test, y_test, threshold: float):
    """Evaluate the frozen pipeline on the untouched test set (called exactly once)."""
    from .metrics import classification_report_dict, compute_metrics

    proba = np.asarray(pipeline.predict_proba(X_test))[:, 1]
    metrics = compute_metrics(y_test, proba, threshold)
    metrics["Classification_Report"] = classification_report_dict(y_test, proba, threshold)
    metrics["Predictions_At_0.5"] = compute_metrics(y_test, proba, 0.5)
    return metrics, proba


# --------------------------------------------------------------------------
# Phase 20 - feature importance
# --------------------------------------------------------------------------


def map_encoded_to_raw(encoded_names, categorical_features, numeric_features) -> dict:
    """Map a one-hot encoded feature name back to the raw/engineered feature it came from."""
    mapping: dict[str, str] = {}
    cats = sorted(categorical_features, key=len, reverse=True)
    nums = sorted(numeric_features, key=len, reverse=True)
    for name in encoded_names:
        target = str(name)
        if target.startswith("num__"):
            rest = target[5:]
            if "missingindicator_" in rest:
                col = rest.split("missingindicator_", 1)[1]
                mapping[target] = f"{col} [missing indicator]"
                continue
            for col in nums:
                if rest == col:
                    mapping[target] = col
                    break
            else:
                mapping[target] = rest
        elif target.startswith("cat__"):
            rest = target[5:]
            for col in cats:
                if rest == col or rest.startswith(col + "__"):
                    mapping[target] = col
                    break
            else:
                mapping[target] = rest
        else:
            mapping[target] = target
    return mapping


def _extract_inner_classifier(pipeline: ResamplingPipeline):
    clf = pipeline.classifier_
    if isinstance(clf, BalancedWeightClassifier):
        return clf.estimator_
    return clf


def feature_importance_table(pipeline: ResamplingPipeline, X_train, y_train, n_repeats: int = 10):
    """Model importance + permutation importance, both aggregated back to raw features.

    Both are computed on the **training split** (never the test set).
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, make_scorer

    encoded_names = [str(n) for n in pipeline.encoded_feature_names()]
    inner = _extract_inner_classifier(pipeline)
    X_engineered = (pipeline.feature_engineer.transform(X_train)
                    if pipeline._use_engineer() else X_train)
    numeric_cols, cat_cols = split_feature_types(X_engineered)
    mapping = map_encoded_to_raw(encoded_names, cat_cols, numeric_cols)

    model_importance = np.zeros(len(encoded_names), dtype=float)
    if getattr(inner, "feature_importances_", None) is not None:
        model_importance = np.asarray(inner.feature_importances_, dtype=float)
    elif getattr(inner, "coef_", None) is not None:
        model_importance = np.abs(np.ravel(inner.coef_))[: len(encoded_names)]

    X_encoded = pipeline.preprocessor_.transform(X_engineered)
    scoring = make_scorer(average_precision_score, response_method="predict_proba")
    perm = permutation_importance(
        inner, X_encoded, np.asarray(y_train), scoring=scoring,
        n_repeats=n_repeats, random_state=cfg.RANDOM_STATE, n_jobs=-1,
    )

    encoded_table = pd.DataFrame({
        "Encoded_Feature": encoded_names,
        "Raw_Feature": [mapping.get(n, n) for n in encoded_names],
        "Model_Importance": model_importance,
        "Permutation_Importance": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std,
    })

    agg = encoded_table.groupby("Raw_Feature", as_index=False).agg(
        Model_Importance=("Model_Importance", "sum"),
        Permutation_Importance=("Permutation_Importance", "sum"),
        Permutation_Importance_Std=("Permutation_Importance_Std", "sum"),
        N_Encoded_Columns=("Encoded_Feature", "count"),
    )
    model_sum = max(float(agg["Model_Importance"].sum()), 1e-12)
    agg["Model_Importance_Pct"] = 100.0 * agg["Model_Importance"] / model_sum
    agg["Rank_Permutation"] = agg["Permutation_Importance"].rank(ascending=False, method="min").astype(int)
    agg["Rank_Model"] = agg["Model_Importance"].rank(ascending=False, method="min").astype(int)
    agg["Source"] = np.where(
        agg["Raw_Feature"].str.contains("missing indicator", regex=False),
        "engineered (missing-value indicator)",
        np.where(agg["Raw_Feature"].isin(cfg.ID_COLUMNS), "raw (identifier probe)",
                 "raw / engineered"),
    )
    agg = agg.sort_values("Permutation_Importance", ascending=False).reset_index(drop=True)
    agg["Rank_Overall"] = agg.index + 1
    agg["Feature"] = agg["Raw_Feature"]
    agg = agg[["Rank_Overall", "Feature", "Raw_Feature", "Model_Importance", "Model_Importance_Pct",
               "Permutation_Importance", "Permutation_Importance_Std",
               "Rank_Permutation", "Rank_Model", "N_Encoded_Columns", "Source"]]
    encoded_table = encoded_table.sort_values("Permutation_Importance", ascending=False).reset_index(drop=True)
    return agg, encoded_table
