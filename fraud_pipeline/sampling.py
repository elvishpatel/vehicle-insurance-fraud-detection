"""Self-contained SMOTENC-style resampling for mixed numeric/categorical data.

Why is this implemented by hand?
--------------------------------
``imbalanced-learn`` could not be installed in the target environment (no network
access, Python 3.14 wheels unavailable).  The algorithm below is a faithful,
documented implementation of SMOTENC (Chawla et al., 2002 + "SMOTE-NC"):

* neighbour search uses only the **numeric** features (median-imputed, z-scaled),
* synthetic numeric values are interpolated between a minority sample and one of
  its k nearest minority neighbours: ``x_new = x_i + u * (x_j - x_i)``, ``u ~ U(0,1)``,
* synthetic categorical values are copied from ``x_i`` or ``x_j`` (random choice),
* missing values are preserved as missing (never invented), so the downstream
  imputer inside the pipeline still sees a genuine "unknown".

Critical leakage guarantee
--------------------------
``fit_transform`` performs the resampling and is therefore only ever called on a
**training fold**; ``transform`` (used at inference time and on validation/test
data) is a pure pass-through that returns the input unchanged.  Scikit-learn
pipelines call ``fit_transform`` on intermediate steps during ``fit`` and
``transform`` during ``predict``, so a sampler placed early in a pipeline can
never contaminate a validation or test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class SmoteNCSampler(TransformerMixin, BaseEstimator):
    """Oversample minority classes with SMOTENC-consistent synthetic samples.

    Parameters
    ----------
    numeric_features:
        Column names treated as numeric (interpolated).
    categorical_features:
        Column names treated as categorical (copied from a neighbour).
    sampling_ratio:
        Target ratio ``minority_count / majority_count`` after resampling.
        ``0.5`` means "raise the minority class to half of the majority class"
        - the dataset is never forced to a 1:1 balance.
    k_neighbors:
        Number of minority neighbours used for interpolation.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        numeric_features: list[str] | tuple[str, ...] | None = None,
        categorical_features: list[str] | tuple[str, ...] | None = None,
        sampling_ratio: float = 0.5,
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.sampling_ratio = sampling_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _split_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        numeric = [c for c in (self.numeric_features or []) if c in X.columns]
        categorical = [c for c in (self.categorical_features or []) if c in X.columns]
        if not numeric and not categorical:
            raise ValueError("SmoteNCSampler needs at least one feature column.")
        return numeric, categorical

    def _synthesise(
        self,
        rows: pd.DataFrame,
        n_new: int,
        numeric: list[str],
        categorical: list[str],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Create ``n_new`` synthetic minority rows from ``rows``."""
        n_rows = len(rows)
        if n_new <= 0 or n_rows < 2:
            return rows.iloc[0:0]

        # --- neighbour space: numeric only, imputed + scaled ---------------
        if numeric:
            num_matrix = StandardScaler().fit_transform(
                SimpleImputer(strategy="median").fit_transform(rows[numeric].to_numpy(dtype=float))
            )
        else:
            num_matrix = np.zeros((n_rows, 1))

        k = max(min(int(self.k_neighbors), n_rows - 1), 1)
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(num_matrix)
        neighbours = nn.kneighbors(num_matrix, return_distance=False)[:, 1:]  # drop self

        base_idx = rng.integers(0, n_rows, size=n_new)
        neigh_idx = neighbours[base_idx, rng.integers(0, neighbours.shape[1], size=n_new)]

        synthetic = pd.DataFrame(index=np.arange(n_new), columns=list(rows.columns), dtype=object)
        base = rows.iloc[base_idx].reset_index(drop=True)
        neigh = rows.iloc[neigh_idx].reset_index(drop=True)

        # copy every non-feature column from the base row
        for col in rows.columns:
            if col not in numeric and col not in categorical:
                synthetic[col] = base[col].to_numpy()

        u = rng.random(n_new)
        for col in numeric:
            a = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(neigh[col], errors="coerce").to_numpy(dtype=float)
            both = ~np.isnan(a) & ~np.isnan(b)
            synthetic[col] = np.where(both, a + u * (b - a), np.nan)

        for col in categorical:
            a = base[col].to_numpy(dtype=object)
            b = neigh[col].to_numpy(dtype=object)
            take_b = rng.random(n_new) < 0.5
            a_missing = pd.isna(a)
            b_missing = pd.isna(b)
            synthetic[col] = np.where(
                a_missing & ~b_missing, b,
                np.where(b_missing & ~a_missing, a, np.where(take_b, b, a)),
            )

        for col in rows.columns:
            try:
                synthetic[col] = synthetic[col].astype(rows[col].dtype)
            except (TypeError, ValueError):
                pass
        return synthetic

    # ------------------------------------------------------------------ #
    # sklearn API
    # ------------------------------------------------------------------ #
    def fit(self, X, y) -> "SmoteNCSampler":
        """Record class counts and the resampling targets (no data generated)."""
        if y is None:
            raise ValueError("SmoteNCSampler requires y during fit.")
        X_df = pd.DataFrame(X)
        y_ser = pd.Series(np.asarray(y).ravel())
        numeric, categorical = self._split_columns(X_df)
        self.numeric_features_ = numeric
        self.categorical_features_ = categorical
        self.classes_ = np.unique(y_ser)
        counts = y_ser.value_counts()
        self.class_counts_before_ = {int(k): int(v) for k, v in counts.items()}
        majority = int(counts.max())
        self.target_counts_ = {}
        for cls in self.classes_:
            current = int(counts.get(cls, 0))
            if current == 0:
                continue
            target = int(round(majority * float(self.sampling_ratio)))
            self.target_counts_[int(cls)] = max(current, target)   # never undersample
        return self

    def fit_transform(self, X, y=None, **fit_params):  # noqa: D102
        self.fit(X, y)
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_ser = pd.Series(np.asarray(y).ravel(), name="target")
        rng = np.random.default_rng(self.random_state)

        blocks, labels = [X_df], [y_ser]
        generated: dict[int, int] = {}
        for cls in self.classes_:
            mask = (y_ser == cls).to_numpy()
            n_current = int(mask.sum())
            n_new = self.target_counts_.get(int(cls), n_current) - n_current
            generated[int(cls)] = max(0, int(n_new))
            if n_new <= 0:
                continue
            synth = self._synthesise(
                X_df.loc[mask], int(n_new),
                self.numeric_features_, self.categorical_features_, rng,
            )
            if len(synth) == 0:
                generated[int(cls)] = 0
                continue
            blocks.append(synth.reset_index(drop=True))
            labels.append(pd.Series(np.full(len(synth), cls), name="target"))

        X_res = pd.concat(blocks, axis=0, ignore_index=True)[list(X_df.columns)]
        y_res = pd.concat(labels, axis=0, ignore_index=True)
        self.synthetic_counts_ = generated
        self.class_counts_after_ = {int(c): int((y_res == c).sum()) for c in self.classes_}
        return X_res, y_res

    def transform(self, X):  # noqa: D102
        """Pass-through: resampling is a training-time-only operation."""
        check_is_fitted(self, ["classes_"])
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def fit_resample(self, X, y):
        """``imbalanced-learn`` compatible alias."""
        return self.fit_transform(X, y)


def make_sampler(
    numeric_features,
    categorical_features,
    sampling_ratio: float = 0.5,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> SmoteNCSampler:
    """Convenience factory (keeps ``run_pipeline`` free of estimator details)."""
    sampler = SmoteNCSampler(
        numeric_features=list(numeric_features),
        categorical_features=list(categorical_features),
        sampling_ratio=sampling_ratio,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    clone(sampler)  # validates the sklearn constructor contract early
    return sampler
