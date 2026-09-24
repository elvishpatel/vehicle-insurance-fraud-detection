"""Phase 8 - strict data-leakage audit.

Two complementary layers:

1. **Semantic / domain audit** (label-free, runs on the raw column list): does a
   column *mean* something that is only known after the fraud investigation?
   Identifiers, decisions, settlements, investigation output ...
2. **Empirical diagnostics** computed on the **training split only**: single
   feature AUC, mutual information, correlation with the target and perfect
   separation checks, plus a redundancy scan (features that are exact linear
   combinations of each other).

Nothing is removed silently: every column gets an explicit risk level, a reason
and the action that the pipeline takes.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from . import config as cfg
from .features import split_feature_types

# --------------------------------------------------------------------------
# semantic rules
# --------------------------------------------------------------------------
#: attribute groups whose meaning is known at claim intake (no post-investigation content)
CLAIM_INTAKE_GROUPS: dict[str, tuple[str, list[str]]] = {
    "policy / underwriting attribute (recorded at policy inception)": [
        "months_as_customer", "age", "policy_state", "policy_csl", "policy_deductable",
        "policy_annual_premium", "umbrella_limit", "insured_sex", "insured_education_level",
        "insured_occupation", "insured_hobbies", "insured_relationship", "capital-gains",
        "capital-loss", "auto_make", "auto_model", "auto_year",
    ],
    "first-notice-of-loss fact (recorded before any investigation)": [
        "incident_type", "collision_type", "incident_severity", "authorities_contacted",
        "incident_state", "incident_city", "incident_hour_of_the_day",
        "number_of_vehicles_involved", "property_damage", "bodily_injuries", "witnesses",
        "police_report_available",
    ],
    "claimed amount submitted with the claim (before adjudication)": [
        "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    ],
    "policy/incident date (known before the claim is investigated)": [
        "policy_bind_date", "incident_date",
    ],
}

#: empirical evidence strong enough to require an explicit explanation
REVIEWED_STRONG_SIGNALS: dict[str, str] = {
    "incident_severity": ("recorded at first notice of loss, before any investigation. Its strong "
                          "relation to fraud (severity drives payouts) is a legitimate business effect, "
                          "not leakage - verified: no perfect separation, no target duplicate"),
    "insured_hobbies": ("recorded at policy inception. The very high fraud rate of a few hobby "
                        "categories looks like a synthetic-data artefact of this dataset; the "
                        "information is genuinely available before the investigation, so the column "
                        "is kept but flagged for monitoring"),
}

SEMANTIC_RULES: dict[str, tuple[str, str, str]] = {
    "fraud_reported": ("TARGET", "raw fraud label - the prediction target itself",
                       "mapped to 0/1 and used ONLY as y, never as a feature"),
    "policy_number": ("HIGH", "1,000 distinct values for 1,000 claims: a record key, not an attribute",
                      "excluded from the feature matrix (identifier)"),
    "incident_location": ("HIGH", "1,000 distinct street addresses: free-text key with no generalisable signal",
                          "excluded from the feature matrix (identifier)"),
    "insured_zip": ("MEDIUM", "995 distinct values for 1,000 claims: effectively a per-row quasi-identifier",
                    "excluded from the feature matrix (quasi-identifier)"),
    "_c39": ("LOW", "100% missing column (spreadsheet artefact)", "dropped"),
    "total_claim_amount": (
        "MEDIUM",
        "exactly equal to injury_claim + property_claim + vehicle_claim (verified): a derived aggregate "
        "that is known at claim intake - redundancy, not leakage",
        "kept, but the perfect linear dependency is reported below as redundancy",
    ),
}

#: identifiers are only detected through explicit patterns, never through loose substrings
ID_PATTERN = re.compile(
    r"(^|_)(id|ids|uuid|guid|key|zip|postcode|location|address|pin)(_|$)"
    r"|(^|_)(policy|claim|record|customer|vehicle|insured|incident)_?(number|no|code|ref)(_|$)"
)
POST_INVESTIGATION_PATTERN = re.compile(
    r"(^|_)(fraud|frd|investig\w*|decision|approv\w*|settle\w*|payout|paid|reject\w*|deni\w*|"
    r"closure|reserve|adjuster|siu|outcome|status|label|target)(_|$)"
)


def _semantic_row(stage: str, feature: str) -> dict:
    if feature in SEMANTIC_RULES:
        risk, reason, action = SEMANTIC_RULES[feature]
        rule = "explicit semantic rule"
    elif feature in REVIEWED_STRONG_SIGNALS:
        risk = "MEDIUM (reviewed)"
        reason = REVIEWED_STRONG_SIGNALS[feature]
        action = "kept as a feature; flagged for monitoring (regime changes would weaken it)"
        rule = "explicit review of an empirically strong signal"
    else:
        group = next((name for name, members in CLAIM_INTAKE_GROUPS.items() if feature in members), None)
        lowered = feature.lower()
        if group:
            risk, reason = "LOW", f"claim-intake attribute: {group}"
            action, rule = "kept", "attribute-group rule"
        elif POST_INVESTIGATION_PATTERN.search(lowered):
            risk = "HIGH"
            reason = "name matches a post-investigation / decision pattern"
            action = "flagged for manual review before being used as a feature"
            rule = "name pattern"
        elif ID_PATTERN.search(lowered):
            risk = "MEDIUM"
            reason = "name matches an identifier pattern (record key / address / code)"
            action = "flagged; excluded automatically when it is a 1:1 identifier"
            rule = "name pattern"
        else:
            risk, reason = "LOW", "derived claim-intake feature (deterministic transformation of raw inputs)"
            action, rule = "kept", "default (engineered feature)"
    return {"Stage": stage, "Feature": feature, "Leakage_Risk": risk, "Reason": reason,
            "Action": action, "Rule": rule}


def domain_audit(raw_columns, engineered_columns) -> pd.DataFrame:
    """Label-free semantic audit of every raw and engineered column."""
    rows = [_semantic_row("raw", c) for c in raw_columns]
    rows += [_semantic_row("engineered", c) for c in engineered_columns
             if c not in set(raw_columns)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# empirical diagnostics (training split only!)
# --------------------------------------------------------------------------


def _single_feature_auc(x: pd.Series, y: np.ndarray, categorical: bool) -> float:
    """AUC of one feature against the label (``nan`` when it cannot be computed)."""
    if categorical:
        frame = pd.DataFrame({"x": x.astype(str), "y": y}).dropna()
        if frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
            return float("nan")
        rates = frame.groupby("x")["y"].mean()
        if rates.nunique() < 2:
            return float("nan")
        scores = frame["x"].map(rates).to_numpy(dtype=float)
        return float(roc_auc_score(frame["y"].to_numpy(dtype=int), scores))
    values = pd.to_numeric(x, errors="coerce")
    mask = values.notna().to_numpy()
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2 or values[mask].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], values[mask].to_numpy(dtype=float)))


def train_diagnostics(train_features: pd.DataFrame, y_train, exclude=()) -> pd.DataFrame:
    """Single-feature AUC / mutual information / correlation, training split only."""
    y = np.asarray(y_train).astype(int)
    _, cat_cols = split_feature_types(train_features)
    rows = []
    for col in train_features.columns:
        if col in exclude:
            continue
        s = train_features[col]
        is_cat = col in cat_cols
        auc = _single_feature_auc(s, y, is_cat)
        if is_cat:
            codes = pd.factorize(s.astype(str))[0].reshape(-1, 1)
            mi = float(mutual_info_classif(codes, y, discrete_features=True,
                                           random_state=cfg.RANDOM_STATE)[0])
            corr = float("nan")
        else:
            values = pd.to_numeric(s, errors="coerce").astype(float)
            values = values.fillna(values.median())
            mi = float(mutual_info_classif(values.to_numpy().reshape(-1, 1), y,
                                           discrete_features=False, random_state=cfg.RANDOM_STATE)[0])
            corr = (float(np.corrcoef(values.to_numpy(dtype=float), y)[0, 1])
                    if values.nunique() > 1 else float("nan"))
        flags = []
        if np.isfinite(auc):
            separation = max(auc, 1 - auc)
            if separation >= 0.99:
                flags.append("PERFECT_SEPARATION")
            elif separation >= 0.90:
                flags.append("VERY_HIGH_SEPARATION")
        if np.isfinite(corr) and abs(corr) >= 0.95:
            flags.append("NEAR_TARGET_DUPLICATE")
        rows.append({
            "Feature": col,
            "Type": "categorical" if is_cat else "numeric",
            "Single_Feature_ROC_AUC": round(auc, 4) if np.isfinite(auc) else np.nan,
            "AUC_Abs": round(max(auc, 1 - auc), 4) if np.isfinite(auc) else np.nan,
            "Mutual_Information": round(mi, 5),
            "Corr_With_Target": round(corr, 4) if np.isfinite(corr) else np.nan,
            "Flags": ";".join(flags) if flags else "-",
        })
    return pd.DataFrame(rows)


def redundancy_report(train_features: pd.DataFrame, threshold: float = 0.999) -> pd.DataFrame:
    """Exact / near-exact linear redundancy between features (reported, not leakage)."""
    numeric_cols, _ = split_feature_types(train_features)
    numeric_cols = [c for c in numeric_cols if train_features[c].notna().sum() > 10]
    empty = pd.DataFrame(columns=["Stage", "Feature", "Leakage_Risk", "Reason", "Action", "Rule"])
    if len(numeric_cols) < 2:
        return empty
    corr = train_features[numeric_cols].astype(float).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    rows = []
    for col in upper.columns:
        for other, value in upper[col][upper[col] >= threshold].items():
            rows.append({
                "Stage": "engineered",
                "Feature": f"{other} ~ {col}",
                "Leakage_Risk": "LOW (redundancy)",
                "Reason": f"|correlation| = {value:.4f} on the training split: one feature is an exact "
                          "linear function of the other - redundant, but NOT target leakage",
                "Action": "both retained (collinearity does not hurt the tuned models) and flagged so that "
                          "nobody mistakes it for leakage",
                "Rule": "redundancy scan",
            })
    return pd.DataFrame(rows) if rows else empty


def build_leakage_audit(raw_columns, engineered_columns, train_features, y_train) -> pd.DataFrame:
    """Full ``leakage_audit.csv`` table: semantics + training-only diagnostics + redundancy."""
    semantic = domain_audit(raw_columns, engineered_columns)
    diag = train_diagnostics(train_features, y_train)
    merged = semantic.merge(diag, on="Feature", how="left")

    def escalate(row) -> str:
        flags = str(row.get("Flags", "-"))
        risk = str(row["Leakage_Risk"])
        if "PERFECT_SEPARATION" in flags or "NEAR_TARGET_DUPLICATE" in flags:
            return "HIGH (empirical)"
        if "VERY_HIGH_SEPARATION" in flags and risk.startswith("LOW"):
            return "MEDIUM (empirical)"
        return risk

    merged["Leakage_Risk"] = merged.apply(escalate, axis=1)
    merged["Reason"] = merged.apply(
        lambda r: r["Reason"] + (f" | diagnostics: {r['Flags']}" if r.get("Flags", "-") != "-" else ""),
        axis=1,
    )
    merged["Diagnostics_Scope"] = "training split only (no test data used)"
    cols = ["Stage", "Feature", "Leakage_Risk", "Reason", "Action", "Rule",
            "Type", "Single_Feature_ROC_AUC", "AUC_Abs", "Mutual_Information",
            "Corr_With_Target", "Flags", "Diagnostics_Scope"]
    merged = merged.reindex(columns=cols)
    redundant = redundancy_report(train_features)
    if not redundant.empty:
        merged = pd.concat([merged, redundant.reindex(columns=cols)], ignore_index=True)
    return merged
