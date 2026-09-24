"""Leakage-safe composite estimator: feature engineering -> sampling -> encoding -> model.

scikit-learn's own ``Pipeline`` discards the resampled target that a sampler must
return, so it cannot host SMOTENC-style resampling in front of the encoder.  This
module provides a small, explicit composite estimator instead:

* ``fit(X, y)``  : engineer -> **resample the training fold only** -> fit encoder -> fit model
* ``predict`` / ``predict_proba`` / ``decision_function`` : engineer -> encode -> model
  (the sampler is *never* applied at prediction time)

Everything is exposed through scikit-learn's parameter interface, so
``RandomizedSearchCV`` / ``cross_val_score`` can tune ``classifier__...``,
``preprocessor__...`` and ``sampler__...`` parameters and clone the estimator
safely.  The object is also what gets serialised into the final ``.pkl``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ResamplingPipeline(ClassifierMixin, BaseEstimator):
    """Composite pipeline with optional in-fold resampling.

    Parameters
    ----------
    classifier:
        The final estimator (must expose ``fit`` and ``predict``).
    preprocessor:
        Typically a ``ColumnTransformer`` performing imputation + encoding.
    sampler:
        Optional resampler exposing ``fit_transform(X, y) -> (X_res, y_res)``
        (``SmoteNCSampler``) or ``None``/``"passthrough"`` for no resampling.
    feature_engineer:
        Optional transformer applied to the raw frame first (``FraudFeatureEngineer``).
    """

    def __init__(self, classifier=None, preprocessor=None, sampler=None, feature_engineer=None):
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.feature_engineer = feature_engineer

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _use_sampler(self) -> bool:
        return self.sampler is not None and self.sampler != "passthrough"

    def _use_engineer(self) -> bool:
        return self.feature_engineer is not None and self.feature_engineer != "passthrough"

    def _prepare(self, X, y=None, fit: bool = False):
        Xt = X
        if self._use_engineer():
            if fit:
                Xt = self.feature_engineer.fit_transform(Xt, y)
            else:
                Xt = self.feature_engineer.transform(Xt)
        if fit and self._use_sampler():
            Xt, y = self.sampler.fit_transform(Xt, y)
        return Xt, y

    # ------------------------------------------------------------------ #
    # estimator API
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        Xt, yt = self._prepare(X, y, fit=True)
        self.preprocessor_ = self.preprocessor
        self.preprocessor_.fit(Xt)
        Xe = self.preprocessor_.transform(Xt)
        self.classifier_ = self.classifier
        self.classifier_.fit(Xe, yt)
        self.classes_ = np.asarray(self.classifier_.classes_)
        self.n_features_in_ = Xe.shape[1]
        self.fit_summary_ = {
            "train_rows_input": int(len(pd.DataFrame(X))),
            "train_rows_used": int(len(Xe)),
            "class_counts_used": {
                str(int(c)): int(v) for c, v in zip(*np.unique(np.asarray(yt), return_counts=True))
            },
            "sampler": type(self.sampler).__name__ if self._use_sampler() else "none",
        }
        self.fitted_ = True
        return self

    def _encode(self, X):
        check_is_fitted(self, ["preprocessor_", "classifier_"])
        Xt, _ = self._prepare(X, None, fit=False)
        return self.preprocessor_.transform(Xt)

    def predict_proba(self, X):
        """Probability matrix ``[P(not fraud), P(fraud)]``."""
        return self.classifier_.predict_proba(self._encode(X))

    def predict(self, X):
        """Hard label prediction using the classifier's own 0.5 cut-off."""
        return self.classifier_.predict(self._encode(X))

    def predict_fraud_proba(self, X) -> np.ndarray:
        """Convenience: probability of the fraud class (positive label = 1)."""
        proba = self.predict_proba(X)
        idx = int(np.where(self.classes_ == 1)[0][0]) if 1 in self.classes_ else proba.shape[1] - 1
        return proba[:, idx]

    def decision_function(self, X):
        if hasattr(self.classifier_, "decision_function"):
            return self.classifier_.decision_function(self._encode(X))
        return self.predict_proba(X)[:, -1]

    # ------------------------------------------------------------------ #
    # introspection helpers used by reporting / persistence
    # ------------------------------------------------------------------ #
    def encoded_feature_names(self) -> list[str]:
        check_is_fitted(self, ["preprocessor_"])
        try:
            return [str(n) for n in self.preprocessor_.get_feature_names_out()]
        except Exception:  # noqa: BLE001 - fall back to positional names
            return [f"f{i}" for i in range(self.n_features_in_)]

    def raw_feature_names(self) -> list[str]:
        if self._use_engineer():
            return list(getattr(self.feature_engineer, "output_columns_", []))
        return []

    def __sklearn_is_fitted__(self) -> bool:  # pragma: no cover - sklearn protocol
        return bool(getattr(self, "fitted_", False))
