"""Phase 2 / 3 / 4 / 7 - target validation, deterministic cleaning, export.

Design rule: cleaning is *deterministic and label-free*.  No statistic of the
data (mean/median/mode, target rate, ...) is ever used here, which is why
applying it before the train/test split cannot leak information.  All learned
quantities (imputation values, encoder categories, scaling) live inside the
scikit-learn pipeline and are fitted per training fold.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg
from .features import apply_domain_sanity_rules, engineer_features, sanitize_raw_frame


# --------------------------------------------------------------------------
# Phase 2 - target validation
# --------------------------------------------------------------------------


def detect_target_column(df: pd.DataFrame) -> str:
    """Locate the fraud label column automatically (falls back to the configured name)."""
    if cfg.TARGET_RAW in df.columns:
        return cfg.TARGET_RAW
    for candidate in ("fraud_reported", "fraud", "is_fraud", "fraudulent", "target", "label", "class"):
        match = [c for c in df.columns if c.lower() == candidate]
        if match:
            return match[0]
    # last resort: a binary object column whose values are a Y/N-like pair
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip().str.lower().unique()
        if len(values) == 2 and set(values) <= {"y", "n", "yes", "no", "true", "false", "0", "1"}:
            return col
    raise KeyError("Could not identify the target column in the dataset.")


def validate_target(df: pd.DataFrame, target_col: str | None = None) -> tuple[pd.Series, dict]:
    """Inspect the raw label, map it to 0/1 *after* verification, keep missing as NaN."""
    target_col = target_col or detect_target_column(df)
    raw = df[target_col]

    # sentinel tokens are treated as a genuinely missing label
    lowered = {v.lower() for v in cfg.SENTINEL_VALUES}
    as_text = raw.astype(str).str.strip()
    raw = raw.mask(as_text.str.lower().isin(lowered), other=pd.NA)

    uniques = sorted(map(str, raw.dropna().unique()))
    report: dict = {
        "target_column": target_col,
        "raw_unique_values": uniques,
        "raw_value_counts": {str(k): int(v) for k, v in raw.value_counts(dropna=False).items()},
        "mapping": {},
        "n_missing_target": int(raw.isna().sum()),
        "verification_note": ("mapping was derived from the observed values before any modelling; "
                             "Y/YES/TRUE/1 -> 1 (fraud), N/NO/FALSE/0 -> 0 (not fraud)"),
    }

    normalised = raw.astype(str).str.strip().str.upper()
    mapping: dict[str, int] = {}
    for value in uniques:
        key = value.strip().upper()
        if key in ("Y", "YES", "TRUE", "1", "FRAUD", "FRAUDULENT"):
            mapping[key] = cfg.POS_LABEL
        elif key in ("N", "NO", "FALSE", "0", "NOT FRAUD", "NON-FRAUD", "LEGITIMATE"):
            mapping[key] = cfg.NEG_LABEL
        else:
            raise ValueError(
                f"Unrecognised target value {value!r}. Inspect it before mapping - "
                "assuming that Y means fraud without verification is unsafe."
            )
    report["mapping"] = mapping
    if set(mapping.values()) != {0, 1}:
        raise ValueError(f"Target mapping {mapping} does not contain both classes.")

    mapped = normalised.map(mapping)
    report["value_table"] = [
        {
            "Raw_Value": value,
            "Mapped": mapping[value.strip().upper()],
            "Count": int((normalised == value.strip().upper()).sum()),
            "Percentage": round(100 * float((normalised == value.strip().upper()).mean()), 4),
        }
        for value in uniques
    ]
    mapped = mapped.astype("float")  # keep NaN for genuinely missing labels
    report["n_rows_before"] = int(len(df))
    report["n_rows_removed_missing_target"] = int(mapped.isna().sum())
    report["n_rows_after"] = int(mapped.notna().sum())
    return mapped, report


# --------------------------------------------------------------------------
# Phase 3 - cleaning orchestration
# --------------------------------------------------------------------------


def clean_dataset(df: pd.DataFrame, target_col: str | None = None):
    """Run the full deterministic cleaning chain.

    Returns
    -------
    cleaned_raw : pandas.DataFrame
        Cleaned frame **with the original raw column names** (sentinel tokens
        removed, impossible values fixed) + the 0/1 ``fraud`` target.  This is
        what gets split into train/test and fed to the pipeline, so the pipeline
        genuinely accepts raw claim records.
    engineered : pandas.DataFrame
        Cleaned frame with the engineered features added - written to
        ``cleaned_dataset.csv`` for inspection only.
    log : pandas.DataFrame
        One row per cleaning operation (column, issue, count, action).
    report : dict
        Target validation details and row counts.
    """
    target_col = target_col or detect_target_column(df)
    records: list[dict] = []

    mapped_target, target_report = validate_target(df, target_col)
    if target_report["n_rows_removed_missing_target"]:
        records.append({
            "Column": target_col,
            "Original_Issue": f"{target_report['n_rows_removed_missing_target']} row(s) with a missing label",
            "Number_Affected": target_report["n_rows_removed_missing_target"],
            "Action_Taken": "rows without a verifiable label removed (they cannot be used for supervised training)",
        })
    records.append({
        "Column": target_col,
        "Original_Issue": f"raw values {target_report['raw_unique_values']}",
        "Number_Affected": target_report["n_rows_after"],
        "Action_Taken": f"verified and mapped to 0/1 via {target_report['mapping']} -> column renamed '{cfg.TARGET}'",
    })

    cleaned = sanitize_raw_frame(df, log=records)
    cleaned = apply_domain_sanity_rules(cleaned, log=records)
    cleaned[cfg.TARGET] = mapped_target.to_numpy()

    # remove the raw label column itself so that it can never appear in X
    if cfg.TARGET_RAW != cfg.TARGET and cfg.TARGET_RAW in cleaned.columns:
        cleaned = cleaned.drop(columns=[cfg.TARGET_RAW])
        records.append({
            "Column": cfg.TARGET_RAW,
            "Original_Issue": "raw fraud label together with its verified 0/1 mapping",
            "Number_Affected": int(target_report["n_rows_after"]),
            "Action_Taken": f"raw label column dropped after mapping to '{cfg.TARGET}' so that the model "
                            "matrix cannot contain the answer (Phase 8 hard rule)",
        })

    # duplicate rows (only exact, full-row duplicates are ever removed)
    n_dup = int(cleaned.duplicated().sum())
    if n_dup:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    records.append({
        "Column": "(all)",
        "Original_Issue": f"{n_dup} exact duplicate row(s)",
        "Number_Affected": n_dup,
        "Action_Taken": "exact duplicate rows removed; duplicate IDs / repeated customers were NOT deleted",
    })

    # rows with a missing target are excluded from the modelling matrix
    n_before = len(cleaned)
    modelling_rows = cleaned[cleaned[cfg.TARGET].notna()].reset_index(drop=True)
    n_dropped = n_before - len(modelling_rows)

    for col in cfg.BROKEN_COLUMNS:
        if col in cleaned.columns:
            records.append({
                "Column": col,
                "Original_Issue": "100% missing / carries no information",
                "Number_Affected": int(cleaned[col].isna().sum()),
                "Action_Taken": "column dropped from the modelling matrix",
            })

    engineered = engineer_features(modelling_rows)
    created = engineered.attrs.get("engineered_columns", [])
    records.append({
        "Column": "(feature engineering)",
        "Original_Issue": "raw attributes only",
        "Number_Affected": len(created),
        "Action_Taken": f"{len(created)} deterministic, investigation-time-safe features created: "
                        + ", ".join(created),
    })

    log = pd.DataFrame(records)
    report = dict(target_report)
    report["n_rows_removed_duplicates"] = n_dup
    report["n_rows_dropped_missing_target"] = n_dropped
    report["n_rows_modelling"] = int(len(engineered))
    report["engineered_columns"] = created
    report["n_columns_modelling"] = int(engineered.shape[1])
    report["n_columns_cleaned_raw"] = int(cleaned.shape[1])
    cleaned_raw = cleaned.copy()
    return cleaned_raw, engineered, log, report
