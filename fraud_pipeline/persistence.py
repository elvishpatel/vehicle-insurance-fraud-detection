"""Phase 21 / 22 - the deployable model bundle.

The single ``.pkl`` written by :func:`save_bundle` contains *everything* needed for
inference on raw claims:

``pipeline``
    the fitted ``ResamplingPipeline`` (feature engineering -> imputation ->
    one-hot encoding -> trained classifier),
``threshold``
    the operating point selected on cross-validated training predictions,
``target_mapping``
    ``{"N": 0, "Y": 1}`` - verified during Phase 2,
``model_name`` / ``sampling_method``
    provenance of the winning candidate,
``feature_names``
    the raw input columns the pipeline expects,
``engineered_feature_names`` / ``encoded_feature_names``
    the intermediate representations (for traceability and importance mapping),
``training_metadata``
    dataset sizes, class balance, hyper-parameters, CV scores, versions.

Custom classes (``ResamplingPipeline``, ``FraudFeatureEngineer``, ``SmoteNCSampler``,
``BalancedWeightClassifier``) live in the ``fraud_pipeline`` package, which must be
importable when the pickle is loaded - ``predict_fraud.py`` takes care of that
automatically.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__ as package_version
from . import config as cfg
from .composite import ResamplingPipeline


def build_bundle(
    pipeline: ResamplingPipeline,
    threshold: float,
    model_name: str,
    sampling_method: str,
    training_metadata: dict,
) -> dict:
    """Assemble the self-contained inference bundle."""
    feature_names = list(getattr(pipeline.feature_engineer, "input_columns_", []))
    engineered = list(getattr(pipeline.feature_engineer, "output_columns_", []))
    try:
        encoded = [str(n) for n in pipeline.encoded_feature_names()]
    except Exception:  # noqa: BLE001
        encoded = []
    return {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "target_mapping": dict(cfg.TARGET_MAPPING),
        "class_names": dict(cfg.CLASS_NAMES),
        "model_name": str(model_name),
        "sampling_method": str(sampling_method),
        "feature_names": feature_names,
        "engineered_feature_names": engineered,
        "encoded_feature_names": encoded,
        "training_metadata": training_metadata,
        "package_version": package_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "custom_classes": [
            "fraud_pipeline.composite.ResamplingPipeline",
            "fraud_pipeline.features.FraudFeatureEngineer",
            "fraud_pipeline.sampling.SmoteNCSampler",
            "fraud_pipeline.modeling.BalancedWeightClassifier",
        ],
        "usage": {
            "load": "joblib.load('vehicle_insurance_fraud_model.pkl')",
            "predict": "float(bundle['pipeline'].predict_proba(raw_df)[0, 1]) >= bundle['threshold']",
        },
    }


def save_bundle(bundle: dict, path: Path | str = cfg.MODEL_PKL) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: Path | str = cfg.MODEL_PKL) -> dict:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"{path} is not a fraud model bundle produced by save_bundle().")
    return bundle


def predict_with_bundle(bundle: dict, raw_claims) -> list[dict]:
    """Score raw claim records (DataFrame / dict / list of dicts) end to end."""
    if isinstance(raw_claims, dict):
        raw_claims = pd.DataFrame([raw_claims])
    elif isinstance(raw_claims, list):
        raw_claims = pd.DataFrame(raw_claims)
    elif isinstance(raw_claims, pd.Series):
        raw_claims = raw_claims.to_frame().T
    if not isinstance(raw_claims, pd.DataFrame):
        raise TypeError("predict_with_bundle expects a pandas DataFrame, dict or list of dicts.")

    pipeline = bundle["pipeline"]
    feature_names = list(bundle.get("feature_names") or [])
    missing = [c for c in feature_names if c not in raw_claims.columns]
    df = raw_claims.copy()
    for col in missing:                      # missing columns become "unknown", not an error
        df[col] = np.nan

    probability = np.asarray(pipeline.predict_proba(df))[:, 1]
    threshold = float(bundle["threshold"])
    results = []
    for prob in probability:
        results.append({
            "prediction": "FRAUD" if prob >= threshold else "NOT FRAUD",
            "fraud_probability": round(float(prob), 6),
            "threshold": round(threshold, 4),
            "model": bundle.get("model_name", "unknown"),
            "missing_input_columns": missing,
        })
    return results
