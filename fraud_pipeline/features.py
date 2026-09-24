"""Deterministic cleaning + feature engineering used by training **and** inference.

The module contains two layers:

1. **Pure functions** (``sanitize_raw_frame``, ``engineer_features``) that apply
   fully deterministic, rule-based transformations.  They use no statistics of
   any kind, so applying them before the train/test split cannot leak
   information (the audit phases call them to document the cleaning).

2. ``FraudFeatureEngineer`` - a scikit-learn transformer wrapping those rules.
   The only *learned* information is bookkeeping: which columns were constant /
   unusable in the training fold and the exact output column order.  It is the
   first step of the saved production pipeline, which is why a freshly loaded
   ``.pkl`` accepts a raw claim DataFrame with the original column names.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import config as cfg

# --------------------------------------------------------------------------
# deterministic helpers
# --------------------------------------------------------------------------


def parse_dates_safely(series: pd.Series) -> pd.Series:
    """Parse a date column, returning ``NaT`` for malformed values.

    The ISO-8601 fast path is tried first so that normal data never triggers
    pandas' slow element-by-element ``dateutil`` fallback (and its warnings).
    """
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    non_null = int(series.notna().sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
        except (ValueError, TypeError):
            parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if non_null and int(parsed.notna().sum()) == 0:
            # fall back to pandas' flexible inference for non-ISO layouts
            try:
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            except (ValueError, TypeError):
                try:
                    parsed = pd.to_datetime(series, errors="coerce")
                except (ValueError, TypeError):
                    pass
    return parsed


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Robust numeric coercion: strips currency/percent symbols and thousands separators."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[\$,£€\s]", "", regex=True)
        .str.replace("%", "", regex=False)
        .replace({"": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def sanitize_raw_frame(df: pd.DataFrame, log: list[dict] | None = None) -> pd.DataFrame:
    """Apply deterministic raw-data cleaning rules (Phase 3).

    * strip surrounding/repeated whitespace in text columns,
    * replace the sentinel tokens ``?``, ``*``, ``NA``, ``null``, ``""`` ... with a
      genuine missing value (never with 0),
    * coerce the structured money/count columns to numeric, turning invalid
      strings into ``NaN`` (no silent interpolation),
    * replace ``+/-inf`` with ``NaN``,
    * parse the date columns with ``errors="coerce"`` so malformed dates become ``NaT``,
    * the target column is *not* touched here (Phase 2 handles it).
    """
    out = pd.DataFrame(df).copy()
    out.columns = [str(c).strip() for c in out.columns]
    records = log if log is not None else []
    lowered = [v.lower() for v in cfg.SENTINEL_VALUES]

    for col in out.columns:
        if col == cfg.TARGET_RAW:
            continue
        s = out[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            stripped = s.astype(str).str.strip()
            n_ws = int(((s.astype(str) != stripped) & s.notna()).sum())
            sentinel_mask = stripped.str.lower().isin(lowered) | stripped.eq("")
            n_sentinel = int(sentinel_mask.sum())
            new = stripped.mask(sentinel_mask, np.nan)
            if n_ws or n_sentinel:
                issue = "whitespace padding" if n_ws else ""
                if n_sentinel:
                    issue = (issue + " + " if issue else "") + f"sentinel placeholder tokens ({n_sentinel})"
                records.append({
                    "Column": col,
                    "Original_Issue": issue,
                    "Number_Affected": max(n_ws, n_sentinel),
                    "Action_Taken": "stripped whitespace / sentinel tokens -> NaN (kept as genuinely missing)",
                })
            out[col] = new

    for col in cfg.COERCED_NUMERIC_COLUMNS:
        if col not in out.columns:
            continue
        before_na = int(out[col].isna().sum())
        coerced = _to_numeric_series(out[col])
        n_invalid = int(coerced.isna().sum() - before_na)
        n_inf = int(np.isinf(coerced.to_numpy(dtype=float)).sum())
        if n_inf:
            coerced = coerced.replace([np.inf, -np.inf], np.nan)
        if n_invalid > 0:
            records.append({
                "Column": col,
                "Original_Issue": f"invalid numeric string ({n_invalid})",
                "Number_Affected": n_invalid,
                "Action_Taken": "coerced via to_numeric(errors='coerce'); invalid -> NaN",
            })
        if n_inf:
            records.append({
                "Column": col,
                "Original_Issue": f"{n_inf} infinite value(s)",
                "Number_Affected": n_inf,
                "Action_Taken": "inf -> NaN (imputed inside the training pipeline, never with 0)",
            })
        out[col] = coerced.astype(float)

    for col in cfg.RAW_DATE_COLUMNS:
        if col not in out.columns:
            continue
        was_datetime = pd.api.types.is_datetime64_any_dtype(out[col])
        parsed = parse_dates_safely(out[col])
        n_bad = int(parsed.isna().sum() - out[col].isna().sum())
        if not was_datetime:
            records.append({
                "Column": col,
                "Original_Issue": "date stored as text"
                                 + (f" ({n_bad} malformed/unparseable)" if n_bad else ""),
                "Number_Affected": max(n_bad, 0),
                "Action_Taken": "parsed to datetime64[ns] with errors='coerce'",
            })
        out[col] = parsed
    return out


def apply_domain_sanity_rules(df: pd.DataFrame, log: list[dict] | None = None) -> pd.DataFrame:
    """Fix values that are *physically impossible* (Phase 5 domain checks).

    Only impossible values are touched - legitimate extreme claims are kept.
    """
    out = df.copy()
    records = log if log is not None else []

    for col in cfg.NON_NEGATIVE_COLUMNS:
        if col not in out.columns:
            continue
        mask = out[col] < 0
        n = int(mask.sum())
        if n:
            records.append({
                "Column": col,
                "Original_Issue": f"{n} negative value(s) where only >= 0 is physically possible",
                "Number_Affected": n,
                "Action_Taken": "impossible negative value -> NaN (imputed in-fold later, NOT set to 0)",
            })
            out.loc[mask, col] = np.nan

    if "incident_hour_of_the_day" in out.columns:
        mask = out["incident_hour_of_the_day"] > cfg.HOUR_OF_DAY_MAX
        n = int(mask.sum())
        if n:
            records.append({
                "Column": "incident_hour_of_the_day",
                "Original_Issue": f"{n} value(s) outside 0-23",
                "Number_Affected": n,
                "Action_Taken": "impossible hour -> NaN",
            })
            out.loc[mask, "incident_hour_of_the_day"] = np.nan

    if {"months_as_customer", "age"} <= set(out.columns):
        months = out["months_as_customer"]
        plausible_cap = out["age"] * 12
        mask = months > plausible_cap
        n = int(mask.sum())
        if n:
            records.append({
                "Column": "months_as_customer",
                "Original_Issue": f"{n} tenure value(s) older than the customer's whole life",
                "Number_Affected": n,
                "Action_Taken": "capped at age * 12 months (consistency fix, record retained)",
            })
            out.loc[mask, "months_as_customer"] = plausible_cap[mask]

    if {"policy_bind_date", "incident_date"} <= set(out.columns):
        delta = (out["incident_date"] - out["policy_bind_date"]).dt.days
        n = int((delta < 0).sum())
        if n:
            records.append({
                "Column": "policy_bind_date / incident_date",
                "Original_Issue": f"{n} record(s) where the policy was bound AFTER the incident",
                "Number_Affected": n,
                "Action_Taken": "impossible chronology -> policy_age_days = NaN (dates kept for date-part features)",
            })
    return out


# --------------------------------------------------------------------------
# Phase 7 - deterministic feature engineering
# --------------------------------------------------------------------------
#: raw columns that are consumed by the engineering step and dropped afterwards
CONSUMED_RAW_COLUMNS: tuple[str, ...] = ("policy_csl",) + cfg.RAW_DATE_COLUMNS


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN instead of inf/0-division."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the deterministic, investigation-time-safe feature set.

    Nothing created here uses the target or any post-investigation artefact.
    """
    out = df.copy()
    created: list[str] = []

    # ---- policy CSL limits ("250/500" -> 250, 500 thousand) --------------
    if "policy_csl" in out.columns:
        parts = out["policy_csl"].astype(str).str.extract(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
        out["policy_csl_bodily_limit"] = pd.to_numeric(parts[0], errors="coerce") * 1000.0
        out["policy_csl_property_limit"] = pd.to_numeric(parts[1], errors="coerce") * 1000.0
        created += ["policy_csl_bodily_limit", "policy_csl_property_limit"]

    # ---- date parts -------------------------------------------------------
    for col in cfg.RAW_DATE_COLUMNS:
        if col not in out.columns:
            continue
        parsed = parse_dates_safely(out[col])
        out[col] = parsed
        prefix = col.replace("_date", "")
        out[f"{prefix}_year"] = parsed.dt.year.astype(float)
        out[f"{prefix}_month"] = parsed.dt.month.astype(float)
        out[f"{prefix}_day"] = parsed.dt.day.astype(float)
        out[f"{prefix}_dayofweek"] = parsed.dt.dayofweek.astype(float)
        out[f"{prefix}_quarter"] = parsed.dt.quarter.astype(float)
        created += [f"{prefix}_year", f"{prefix}_month", f"{prefix}_day",
                    f"{prefix}_dayofweek", f"{prefix}_quarter"]

    # ---- policy age at the time of the incident (available pre-investigation)
    if {"policy_bind_date", "incident_date"} <= set(out.columns):
        delta = (out["incident_date"] - out["policy_bind_date"]).dt.days
        out["policy_age_days"] = delta.where(delta >= 0, np.nan).astype(float)
        created.append("policy_age_days")

    # ---- vehicle age ------------------------------------------------------
    if {"auto_year", "incident_date"} <= set(out.columns):
        veh_age = out["incident_date"].dt.year - pd.to_numeric(out["auto_year"], errors="coerce")
        out["vehicle_age_years"] = veh_age.where(veh_age >= 0, np.nan).astype(float)
        created.append("vehicle_age_years")

    # ---- customer tenure --------------------------------------------------
    if "months_as_customer" in out.columns:
        out["tenure_years"] = (pd.to_numeric(out["months_as_customer"], errors="coerce") / 12.0).astype(float)
        created.append("tenure_years")
        if "age" in out.columns:
            out["tenure_share_of_life"] = _safe_divide(out["months_as_customer"], out["age"] * 12.0)
            created.append("tenure_share_of_life")

    # ---- claim composition -------------------------------------------------
    claim_parts = [c for c in ("injury_claim", "property_claim", "vehicle_claim") if c in out.columns]
    if len(claim_parts) == 3 and "total_claim_amount" in out.columns:
        parts_sum = out[claim_parts].sum(axis=1)
        out["claim_consistency_diff"] = (out["total_claim_amount"] - parts_sum).astype(float)
        created.append("claim_consistency_diff")
        for col in claim_parts:
            out[f"{col}_share"] = _safe_divide(out[col], out["total_claim_amount"])
            created.append(f"{col}_share")
    for col in ("total_claim_amount", "injury_claim", "property_claim", "vehicle_claim", "policy_annual_premium"):
        if col in out.columns:
            out[f"log1p_{col}"] = np.log1p(pd.to_numeric(out[col], errors="coerce").clip(lower=0))
            created.append(f"log1p_{col}")

    # ---- financial flags ---------------------------------------------------
    if "capital-gains" in out.columns:
        out["has_capital_gains"] = (pd.to_numeric(out["capital-gains"], errors="coerce") > 0).astype(float)
        created.append("has_capital_gains")
    if "capital-loss" in out.columns:
        out["has_capital_loss"] = (pd.to_numeric(out["capital-loss"], errors="coerce") < 0).astype(float)
        created.append("has_capital_loss")

    # ---- claim geometry ----------------------------------------------------
    if {"total_claim_amount", "number_of_vehicles_involved"} <= set(out.columns):
        out["claim_per_vehicle"] = _safe_divide(out["total_claim_amount"], out["number_of_vehicles_involved"])
        created.append("claim_per_vehicle")
    if {"policy_state", "incident_state"} <= set(out.columns):
        out["policy_state_matches_incident_state"] = (
            out["policy_state"].astype(str) == out["incident_state"].astype(str)
        ).astype(float)
        created.append("policy_state_matches_incident_state")

    out.attrs["engineered_columns"] = created
    return out


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numeric_columns, categorical_columns)`` for the engineered frame."""
    numeric, categorical = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


# --------------------------------------------------------------------------
# Deployable transformer (first step of the saved production pipeline)
# --------------------------------------------------------------------------


class FraudFeatureEngineer(TransformerMixin, BaseEstimator):
    """Raw insurance-claim records -> model-ready feature matrix.

    Parameters
    ----------
    drop_identifiers:
        Drop the identifier / quasi-identifier columns (Phase 6 + Phase 8).
    drop_constant:
        Drop columns that are constant (or fully missing) in the training data.
    """

    def __init__(self, drop_identifiers: bool = True, drop_constant: bool = True) -> None:
        self.drop_identifiers = drop_identifiers
        self.drop_constant = drop_constant

    # ------------------------------------------------------------------ #
    def fit(self, X, y=None) -> "FraudFeatureEngineer":
        raw = pd.DataFrame(X).copy()
        if cfg.TARGET_RAW in raw.columns:          # never, ever treat the target as a feature
            raw = raw.drop(columns=[cfg.TARGET_RAW])
        self.input_columns_ = [str(c).strip() for c in raw.columns]
        self.target_column_seen_ = cfg.TARGET_RAW in [str(c).strip() for c in pd.DataFrame(X).columns]

        prepared = engineer_features(apply_domain_sanity_rules(sanitize_raw_frame(raw)))

        drop: list[str] = list(CONSUMED_RAW_COLUMNS)
        if self.drop_identifiers:
            drop += list(cfg.ID_COLUMNS) + list(cfg.BROKEN_COLUMNS)
        self.declared_drop_columns_ = [c for c in dict.fromkeys(drop) if c in prepared.columns]

        candidates = [c for c in prepared.columns if c not in self.declared_drop_columns_]
        if self.drop_constant:
            self.constant_columns_ = [
                c for c in candidates if prepared[c].nunique(dropna=True) <= 1
            ]
        else:
            self.constant_columns_ = []
        self.all_missing_columns_ = [
            c for c in candidates if int(prepared[c].isna().sum()) == len(prepared)
        ]
        self.output_columns_ = [
            c for c in candidates
            if c not in set(self.constant_columns_) and c not in set(self.all_missing_columns_)
        ]
        self.dtype_map_ = {
            c: ("numeric" if pd.api.types.is_numeric_dtype(prepared[c]) else "category")
            for c in self.output_columns_
        }
        return self

    # ------------------------------------------------------------------ #
    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, ["output_columns_", "input_columns_"])
        df = pd.DataFrame(X).copy()
        df.columns = [str(c).strip() for c in df.columns]
        if cfg.TARGET_RAW in df.columns:
            df = df.drop(columns=[cfg.TARGET_RAW])     # defensive: never a feature

        missing_inputs = [c for c in self.input_columns_ if c not in df.columns]
        if missing_inputs:
            for col in missing_inputs:
                df[col] = np.nan
        self.missing_input_columns_ = missing_inputs
        self.unexpected_columns_ = [c for c in df.columns if c not in self.input_columns_]

        prepared = engineer_features(apply_domain_sanity_rules(sanitize_raw_frame(df)))
        prepared = prepared.drop(
            columns=[c for c in self.declared_drop_columns_ if c in prepared.columns],
            errors="ignore",
        )
        prepared = prepared.drop(
            columns=[c for c in (self.constant_columns_ + self.all_missing_columns_)
                     if c in prepared.columns],
            errors="ignore",
        )
        out = prepared.reindex(columns=self.output_columns_)
        for col, kind in self.dtype_map_.items():
            if kind == "numeric":
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
            else:
                out[col] = out[col].astype(object)
        return out

    def fit_transform(self, X, y=None, **fit_params):  # noqa: D102
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        check_is_fitted(self, ["output_columns_"])
        return np.asarray(self.output_columns_, dtype=object)

    def drop_report(self) -> dict:
        """Human-readable summary of what was dropped and why (used by the audit)."""
        check_is_fitted(self, ["declared_drop_columns_"])
        return {
            "identifiers_and_broken": list(self.declared_drop_columns_),
            "constant_columns": list(self.constant_columns_),
            "all_missing_columns": list(self.all_missing_columns_),
            "retained_columns": list(self.output_columns_),
        }
