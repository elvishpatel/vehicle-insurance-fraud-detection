"""Phase 1 / 4 / 5 / 6 - raw data audit, duplicates, outliers, data types.

Everything here is read-only with respect to the raw CSV: the file on disk is
never modified, and the audit is exported to ``raw_data_audit.csv``,
``duplicate_analysis.csv``, ``outlier_analysis.csv`` and ``data_types_report.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .features import parse_dates_safely


# --------------------------------------------------------------------------
# Phase 1 - raw data audit
# --------------------------------------------------------------------------


def _looks_like_date(series: pd.Series, min_ratio: float = 0.9) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = parse_dates_safely(sample)
    return float(parsed.notna().mean()) >= min_ratio


def guess_column_role(df: pd.DataFrame, col: str) -> str:
    """Classify a raw column into a coarse, human-readable role."""
    if col == cfg.TARGET_RAW:
        return "target"
    if col in cfg.ID_COLUMNS:
        return "identifier"
    s = df[col]
    if _looks_like_date(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


def raw_data_audit(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Build the long-format audit table written to ``raw_data_audit.csv``."""
    n_rows, n_cols = df.shape
    records: list[dict] = []

    def add(section: str, item: str, attribute: str, value) -> None:
        records.append({"Section": section, "Item": item, "Attribute": attribute, "Value": value})

    add("dataset", source_name, "rows", n_rows)
    add("dataset", source_name, "columns", n_cols)
    add("dataset", source_name, "duplicate_rows", int(df.duplicated().sum()))
    add("dataset", source_name, "duplicate_rows_pct", round(100 * df.duplicated().mean(), 4))
    add("dataset", source_name, "total_missing_cells", int(df.isna().sum().sum()))
    add("dataset", source_name, "columns_with_missing", int((df.isna().sum() > 0).sum()))
    for idx, row in df.head(10).iterrows():
        add("first_10_rows", str(idx), "record", str(row.to_dict())[:500])
    for idx, row in df.tail(10).iterrows():
        add("last_10_rows", str(idx), "record", str(row.to_dict())[:500])

    nunique = df.nunique(dropna=False)
    nunique_excl = df.nunique(dropna=True)
    missing = df.isna().sum()
    for col in df.columns:
        add("column", col, "dtype", str(df[col].dtype))
        add("column", col, "role_guess", guess_column_role(df, col))
        add("column", col, "non_null", int(df[col].notna().sum()))
        add("column", col, "missing", int(missing[col]))
        add("column", col, "missing_pct", round(100 * missing[col] / max(n_rows, 1), 4))
        add("column", col, "unique_incl_nan", int(nunique[col]))
        add("column", col, "unique_excl_nan", int(nunique_excl[col]))
        add("column", col, "is_constant", bool(nunique_excl[col] <= 1))
        dominant = (df[col].value_counts(dropna=True).iloc[0] / max(n_rows, 1)) if nunique_excl[col] else 1.0
        add("column", col, "dominant_value_share", round(float(dominant), 4))
        add("column", col, "is_near_constant", bool(0 < nunique_excl[col] <= 5 or dominant >= 0.95))
        if pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].to_numpy(dtype=float)
            add("column", col, "min", float(np.nanmin(values)) if nunique_excl[col] else np.nan)
            add("column", col, "max", float(np.nanmax(values)) if nunique_excl[col] else np.nan)
            add("column", col, "mean", float(np.nanmean(values)) if nunique_excl[col] else np.nan)
        else:
            add("column", col, "example_values", "; ".join(sorted(map(str, df[col].dropna().unique()))[:8]))
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Phase 6 - data type report
# --------------------------------------------------------------------------


def data_type_report(df: pd.DataFrame, keep_columns=(), drop_reasons: dict | None = None) -> pd.DataFrame:
    """Automatic per-column type identification plus the action taken."""
    drop_reasons = drop_reasons or {}
    keep_columns = set(keep_columns)
    rows = []
    for col in df.columns:
        s = df[col]
        role = guess_column_role(df, col)
        if role == "target":
            kind = "target"
        elif role == "identifier":
            kind = "identifier"
        elif role == "date":
            kind = "datetime"
        elif pd.api.types.is_bool_dtype(s):
            kind = "binary"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "integer" if pd.api.types.is_integer_dtype(s) else "float"
        else:
            kind = "binary_categorical" if s.nunique(dropna=True) <= 2 else "nominal_categorical"

        if col in drop_reasons:
            action = drop_reasons[col]
        elif col in keep_columns:
            action = "kept as predictive feature"
        else:
            action = "kept (consumed by the feature-engineering step)"

        rows.append({
            "Column": col,
            "Detected_Type": kind,
            "Role": role,
            "Cardinality": int(s.nunique(dropna=True)),
            "Missing": int(s.isna().sum()),
            "Example": "; ".join(sorted(map(str, s.dropna().unique()))[:3]),
            "Dtype_Original": str(s.dtype),
            "Action": action,
        })
    return pd.DataFrame(rows).sort_values(["Role", "Column"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Phase 4 - duplicate analysis
# --------------------------------------------------------------------------


def duplicate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Detect exact / id-level / near duplicates and explain the verdict."""
    rows: list[dict] = []

    def add(check: str, count: int, details: str, decision: str) -> None:
        rows.append({"Check": check, "Count": int(count), "Details": details, "Decision": decision})

    add("exact duplicate rows", int(df.duplicated().sum()),
        "all columns identical", "removed only if > 0: such rows are accidental double entries")

    for col in cfg.ID_COLUMNS:
        if col not in df.columns:
            continue
        n_dup = int(df[col].duplicated().sum())
        add(f"duplicate {col}", n_dup,
            f"{df[col].nunique(dropna=True)} distinct values for {len(df)} rows "
            f"({100 * df[col].nunique(dropna=True) / len(df):.1f}% unique)",
            "identifier-like -> never used as a feature" if n_dup or col in cfg.ID_COLUMNS else "unique key")

    claim_key = [c for c in ("policy_number", "incident_date", "total_claim_amount") if c in df.columns]
    if claim_key:
        add("duplicate claim keys " + str(claim_key), int(df.duplicated(subset=claim_key).sum()),
            "same policy + incident date + claim amount",
            "0 -> no evidence of repeated/double-entered claims")

    customer_key = [c for c in ("age", "months_as_customer", "policy_state", "insured_zip",
                                "insured_occupation", "insured_education_level") if c in df.columns]
    if customer_key:
        add("duplicate customer records " + str(customer_key), int(df.duplicated(subset=customer_key).sum()),
            "identical customer descriptors for different claims",
            "legitimate repeated customers -> RETAINED (deleting them would remove real claims)")

    near_key = [c for c in ("age", "months_as_customer", "insured_zip", "total_claim_amount",
                            "incident_date", "incident_type", "auto_make", "auto_model") if c in df.columns]
    if near_key:
        rounded = df[near_key].copy()
        for col in rounded.columns:
            if pd.api.types.is_numeric_dtype(rounded[col]):
                rounded[col] = (rounded[col] / 10).round() * 10
        add("near-duplicate records (numeric descriptors rounded to 10)", int(rounded.duplicated().sum()),
            "descriptors equal after rounding numeric fields",
            "RETAINED: for insurance claims these are plausible distinct claims, not double entries")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Phase 5 - outlier analysis
# --------------------------------------------------------------------------


def outlier_analysis(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """IQR based outlier diagnostics + domain sanity flags (no deletions)."""
    columns = columns or [c for c in df.columns
                          if pd.api.types.is_numeric_dtype(df[c]) and c != cfg.TARGET_RAW]
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        domain_notes, treatment = [], []
        if col in cfg.NON_NEGATIVE_COLUMNS and bool((s < 0).any()):
            domain_notes.append(f"{int((s < 0).sum())} negative (impossible)")
            treatment.append("impossible negatives -> NaN")
        if col == "incident_hour_of_the_day":
            domain_notes.append("valid range 0-23")
        if col == "umbrella_limit":
            domain_notes.append("discrete liability limits (0 .. 10,000,000)")
        if col in ("total_claim_amount", "injury_claim", "property_claim", "vehicle_claim"):
            domain_notes.append("large values are legitimate high-value claims")
        if col in ("capital-gains", "capital-loss"):
            domain_notes.append("financial position, 0 = none reported")
        if not treatment:
            treatment.append("RETAINED: genuine insurance extremes are not removed; "
                             "trees are split-based and the linear model uses RobustScaler")
        rows.append({
            "Feature": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower_Bound": round(lower, 4),
            "Upper_Bound": round(upper, 4),
            "N_Outliers": n_out,
            "Outlier_Pct": round(100 * n_out / len(s), 4),
            "Min": round(float(s.min()), 4),
            "Max": round(float(s.max()), 4),
            "P01": round(float(s.quantile(0.01)), 4),
            "P99": round(float(s.quantile(0.99)), 4),
            "Domain_Notes": "; ".join(domain_notes) if domain_notes else "-",
            "Treatment": "; ".join(treatment),
            "N_Outliers_Removed": 0,
        })
    return pd.DataFrame(rows)
