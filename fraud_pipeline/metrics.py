"""Metric definitions, threshold optimisation and evaluation plots.

All functions here operate on *whatever data they are given*.  The pipeline
guarantees that the final test set is only ever passed in Phase 19.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config as cfg

METRIC_KEYS = (
    "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC",
    "Balanced_Accuracy", "MCC", "Fraud_Precision", "Fraud_Recall", "Fraud_F1",
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Full metric bundle for one decision threshold (positive class = fraud)."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(y_true)) < 2
    return {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, pos_label=cfg.POS_LABEL, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, pos_label=cfg.POS_LABEL, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, pos_label=cfg.POS_LABEL, zero_division=0)),
        "ROC_AUC": float("nan") if single_class else float(roc_auc_score(y_true, y_proba)),
        "PR_AUC": float("nan") if single_class else float(average_precision_score(y_true, y_proba)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if not single_class else float("nan"),
        "Fraud_Precision": float(precision_score(y_true, y_pred, pos_label=cfg.POS_LABEL, zero_division=0)),
        "Fraud_Recall": float(recall_score(y_true, y_pred, pos_label=cfg.POS_LABEL, zero_division=0)),
        "Fraud_F1": float(f1_score(y_true, y_pred, pos_label=cfg.POS_LABEL, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "N": int(len(y_true)), "Fraud_Support": int((y_true == 1).sum()),
    }


def classification_report_dict(y_true, y_proba, threshold: float) -> dict:
    """sklearn classification report (as a dict, JSON friendly)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba, dtype=float) >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=[cfg.CLASS_NAMES[0], cfg.CLASS_NAMES[1]],
        output_dict=True, zero_division=0,
    )


# --------------------------------------------------------------------------
# scorers for cross-validation (never touched by the test set)
# --------------------------------------------------------------------------


def make_scorers() -> dict:
    """Metric dictionary used by ``cross_validate`` / ``RandomizedSearchCV``.

    All probability-based metrics use the positive-class column of
    ``predict_proba`` (class 1 = fraud), which is what scikit-learn selects for a
    binary classifier, so no explicit ``pos_label`` is needed there
    (``roc_auc_score`` does not even accept one).
    """
    return {
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "fraud_f1": make_scorer(f1_score, pos_label=cfg.POS_LABEL, zero_division=0),
        "fraud_recall": make_scorer(recall_score, pos_label=cfg.POS_LABEL, zero_division=0),
        "fraud_precision": make_scorer(precision_score, pos_label=cfg.POS_LABEL, zero_division=0),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "mcc": make_scorer(matthews_corrcoef),
        "accuracy": make_scorer(accuracy_score),
    }


def cv_summary_from_scores(cv_result: dict, prefix: str = "test_") -> dict:
    """Mean/std of every cross-validation metric."""
    out = {}
    for name in cfg.SCORING_METRICS:
        key = f"{prefix}{name}"
        if key in cv_result:
            values = np.asarray(cv_result[key], dtype=float)
            out[name] = float(np.nanmean(values))
            out[f"{name}_std"] = float(np.nanstd(values))
    return out


# --------------------------------------------------------------------------
# Phase 17 - threshold optimisation (validation / OOF data only)
# --------------------------------------------------------------------------


def threshold_table(y_true, y_proba, grid=cfg.THRESHOLD_GRID) -> pd.DataFrame:
    """Metrics for every candidate threshold."""
    rows = []
    for thr in grid:
        m = compute_metrics(y_true, y_proba, thr)
        rows.append({
            "Threshold": round(float(thr), 2),
            "Fraud_Precision": round(m["Fraud_Precision"], 4),
            "Fraud_Recall": round(m["Fraud_Recall"], 4),
            "Fraud_F1": round(m["Fraud_F1"], 4),
            "Accuracy": round(m["Accuracy"], 4),
            "Balanced_Accuracy": round(m["Balanced_Accuracy"], 4),
            "MCC": round(m["MCC"], 4),
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
        })
    return pd.DataFrame(rows)


def select_threshold(
    y_true,
    y_proba,
    grid=cfg.THRESHOLD_GRID,
    tolerance: float = 0.005,
) -> tuple[float, pd.DataFrame, dict]:
    """Pick a threshold from *validation* probabilities only.

    Rule (documented in the README): maximise the fraud-class F1; among all
    thresholds within ``tolerance`` of the best F1, prefer the highest fraud
    precision - a cheap way to avoid trading precision away for a
    statistically meaningless F1 gain.
    """
    table = threshold_table(y_true, y_proba, grid)
    best_f1 = float(table["Fraud_F1"].max())
    candidates = table[table["Fraud_F1"] >= best_f1 - tolerance].copy()
    candidates = candidates.sort_values(
        ["Fraud_Precision", "Balanced_Accuracy", "Threshold"], ascending=[False, False, True]
    )
    chosen = float(candidates.iloc[0]["Threshold"])
    rationale = {
        "rule": "max fraud F1 on out-of-fold training predictions (tolerance +/-0.005), ties -> highest fraud precision",
        "best_f1_on_validation": round(best_f1, 4),
        "n_candidates_within_tolerance": int(len(candidates)),
        "chosen_threshold": chosen,
        "chosen_row": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in candidates.iloc[0].to_dict().items()},
        "test_set_used": False,
    }
    return chosen, table, rationale


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str = "Confusion matrix - test set") -> None:
    cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=[cfg.CLASS_NAMES[0], cfg.CLASS_NAMES[1]])
    ax.set_yticks([0, 1], labels=[cfg.CLASS_NAMES[0], cfg.CLASS_NAMES[1]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, path)


def plot_roc_curve(y_true, y_proba, path: Path, label: str = "model") -> None:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve - fraud class (test set)")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_precision_recall_curve(y_true, y_proba, path: Path, label: str = "model",
                                threshold: float | None = None) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(recall, precision, label=f"{label} (PR-AUC = {ap:.3f})")
    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1,
               label=f"no-skill (fraud rate = {baseline:.3f})")
    if threshold is not None:
        y_pred = (y_proba >= threshold).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        ax.scatter([r], [p], color="red", zorder=5, label=f"operating point t={threshold:.2f}")
    ax.set_xlabel("Recall (fraud)")
    ax.set_ylabel("Precision (fraud)")
    ax.set_title("Precision-Recall curve - fraud class (test set)")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def plot_feature_importance(df: pd.DataFrame, path: Path, top_n: int = 20,
                            value_col: str = "Importance", feature_col: str | None = None) -> None:
    feat_col = feature_col or ("Raw_Feature" if "Raw_Feature" in df.columns else "Feature")
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.32 * len(top) + 1.2)))
    ax.barh(top[feat_col].astype(str), top[value_col].astype(float), color="#2b6cb0")
    ax.set_xlabel(value_col)
    ax.set_title(f"Top {len(top)} features - final model")
    _save(fig, path)


def write_json(path: Path, payload: dict) -> None:
    """JSON dump tolerant to numpy / pandas scalars."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):  # noqa: ANN001
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=default)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
