"""Phase 9 - exploratory data analysis.

All EDA is computed on the **training split only**, so that nothing an analyst
reads off these plots can smuggle test-set information into feature selection.
Plots are written to ``eda/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

#: columns shown in the EDA plots (readability: 58 engineered features is too many for a grid)
EDA_NUMERIC = [
    "age", "months_as_customer", "policy_annual_premium", "policy_deductable",
    "umbrella_limit", "capital-gains", "capital-loss", "total_claim_amount",
    "injury_claim", "property_claim", "vehicle_claim", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "policy_age_days", "vehicle_age_years",
]
EDA_CATEGORICAL = [
    "policy_state", "policy_csl", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "insured_relationship",
    "incident_type", "collision_type", "incident_severity", "authorities_contacted",
    "incident_state", "incident_city", "property_damage",
    "police_report_available", "auto_make", "auto_model",
]

MAX_CATEGORIES_PLOTTED = 8


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _grid_histograms(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    cols = [c for c in columns if c in df.columns and df[c].notna().sum() > 0]
    if not cols:
        return
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        values = pd.to_numeric(df[col], errors="coerce")
        ax.hist(values[values.notna()], bins=30, color="#4a7fb5", alpha=0.9)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Numerical feature distributions (training split)", fontsize=12)
    _save(fig, path)


def _grid_boxplots(df: pd.DataFrame, columns: list[str], y, path: Path, title: str) -> None:
    cols = [c for c in columns if c in df.columns and df[c].notna().sum() > 0]
    if not cols:
        return
    y = pd.Series(np.asarray(y).astype(int), index=df.index)
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        values = pd.to_numeric(df[col], errors="coerce")
        groups = [values[y == 0].dropna(), values[y == 1].dropna()]
        ax.boxplot(groups, tick_labels=["not fraud", "fraud"], showfliers=True,
                   flierprops={"markersize": 2, "alpha": 0.4})
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    _save(fig, path)


def _grid_categorical(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        counts = df[col].astype(str).value_counts().head(MAX_CATEGORIES_PLOTTED)
        ax.barh(counts.index[::-1].astype(str), counts.values[::-1], color="#7aa974")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Categorical feature distributions (training split, top categories)", fontsize=12)
    _save(fig, path)


def _fraud_rate_by_category(df: pd.DataFrame, columns: list[str], y, path: Path) -> pd.DataFrame:
    y = pd.Series(np.asarray(y).astype(int), index=df.index)
    rows = []
    cols = [c for c in columns if c in df.columns]
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        grouped = y.groupby(df[col].astype(str))
        rate, support = grouped.mean(), grouped.size()
        rate = rate[support >= 10].sort_values(ascending=False).head(MAX_CATEGORIES_PLOTTED)
        if not rate.empty:
            ax.barh(rate.index[::-1].astype(str), rate.values[::-1], color="#c07850")
            ax.axvline(float(y.mean()), color="grey", linestyle="--", linewidth=1)
            top = rate.index[0]
            rows.append({"Variable": col, "Category": top,
                         "Fraud_Rate": round(float(rate.iloc[0]), 4),
                         "Support": int(support.get(top, 0))})
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Fraud rate by categorical variable (dashed = overall fraud rate)", fontsize=12)
    _save(fig, path)
    return pd.DataFrame(rows)


def _fraud_rate_by_numeric_bins(df: pd.DataFrame, columns: list[str], y, path: Path) -> pd.DataFrame:
    y = pd.Series(np.asarray(y).astype(int), index=df.index)
    rows = []
    cols = [c for c in columns if c in df.columns and df[c].notna().sum() > 50]
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        values = pd.to_numeric(df[col], errors="coerce")
        try:
            bins = pd.qcut(values, 5, duplicates="drop")
        except ValueError:
            bins = None
        if bins is not None and bins.nunique() > 1:
            rate = y.groupby(bins, observed=True).mean()
            counts = y.groupby(bins, observed=True).size()
            ax.plot(range(len(rate)), rate.values, marker="o", color="#8a5fbf")
            ax.set_xticks(range(len(rate)))
            ax.set_xticklabels([str(i) for i in rate.index], rotation=45, fontsize=6)
            ax.axhline(float(y.mean()), color="grey", linestyle="--", linewidth=1)
            rows.append({
                "Variable": col,
                "Bottom_Quintile_Fraud_Rate": round(float(rate.iloc[0]), 4),
                "Top_Quintile_Fraud_Rate": round(float(rate.iloc[-1]), 4),
                "Support_Top": int(counts.iloc[-1]),
            })
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Fraud rate by quintile of each numerical feature", fontsize=12)
    _save(fig, path)
    return pd.DataFrame(rows)


def _correlation_matrix(df: pd.DataFrame, columns: list[str], y, path: Path) -> None:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    block = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    block["fraud"] = np.asarray(y).astype(int)
    corr = block.corr()
    fig, ax = plt.subplots(figsize=(0.42 * len(corr) + 3, 0.42 * len(corr) + 3))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), labels=corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr)), labels=corr.columns, fontsize=6)
    ax.set_title("Correlation matrix (training split)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7)
    _save(fig, path)


def run_eda(train_features: pd.DataFrame, y_train, out_dir: Path | None = None) -> dict:
    """Generate every Phase-9 artefact and return a small summary dict."""
    out_dir = Path(out_dir or cfg.EDA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = pd.Series(np.asarray(y_train).astype(int), index=train_features.index)

    # 1. target distribution
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    counts = y.map(cfg.CLASS_NAMES).value_counts()
    ax.bar(counts.index, counts.values, color=["#4a7fb5", "#c0504d"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({100 * v / len(y):.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Target distribution (training split)")
    ax.set_ylabel("claims")
    _save(fig, out_dir / "target_distribution.png")

    # 2-4. distributions / boxplots / categoricals
    _grid_histograms(train_features, EDA_NUMERIC, out_dir / "numeric_distributions.png")
    _grid_boxplots(train_features, EDA_NUMERIC, y, out_dir / "boxplots.png",
                   "Numerical features by class (training split)")
    _grid_categorical(train_features, EDA_CATEGORICAL, out_dir / "categorical_distributions.png")
    _grid_boxplots(train_features,
                   ["log1p_total_claim_amount", "log1p_vehicle_claim", "tenure_years",
                    "claim_per_vehicle", "policy_age_days", "vehicle_age_years"],
                   y, out_dir / "fraud_vs_nonfraud_engineered.png",
                   "Engineered features by class (training split)")

    # 5-8. fraud-rate analyses + correlation matrix
    cat_rates = _fraud_rate_by_category(train_features, EDA_CATEGORICAL, y,
                                        out_dir / "fraud_rate_by_categorical.png")
    num_rates = _fraud_rate_by_numeric_bins(train_features, EDA_NUMERIC, y,
                                            out_dir / "fraud_rate_by_numeric_bins.png")
    _correlation_matrix(train_features, EDA_NUMERIC, y, out_dir / "correlation_matrix.png")

    cat_rates.to_csv(out_dir / "fraud_rate_by_categorical.csv", index=False)
    num_rates.to_csv(out_dir / "fraud_rate_by_numeric_bins.csv", index=False)

    numeric_cols = [c for c in EDA_NUMERIC if c in train_features.columns]
    stats = (train_features[numeric_cols].apply(pd.to_numeric, errors="coerce")
             .groupby(y).agg(["mean", "median", "std"]))
    stats.to_csv(out_dir / "numeric_stats_by_class.csv")

    return {
        "eda_dir": str(out_dir),
        "plots": sorted(p.name for p in out_dir.glob("*.png")),
        "strongest_categorical_effects": cat_rates.head(5).to_dict(orient="records")
        if not cat_rates.empty else [],
    }
