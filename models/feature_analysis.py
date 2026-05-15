"""
models/feature_analysis.py
SHAP-based feature importance analysis for the MLS ensemble.

Usage:
    from models.feature_analysis import compute_feature_importance, get_shap_figure

    fig = get_shap_figure(model, X_train, feature_names)   # Plotly figure
    importances = compute_feature_importance(model, feature_names)  # DataFrame
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import shap as _shap  # type: ignore

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ── Core helpers ──────────────────────────────────────────────────────────────


def _get_xgb_estimator(model):
    """Extract the XGBoost sub-estimator from a VotingClassifier."""
    if hasattr(model, "estimators_"):
        for name, est in model.estimators_:
            if "xgb" in name.lower():
                return est
    return None


def compute_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return a DataFrame of the top N features ranked by XGBoost gain importance.
    Falls back to ensemble-averaged importances when XGBoost is not available.
    """
    importances: Optional[np.ndarray] = None

    # Try XGBoost first (most stable importances)
    xgb_est = _get_xgb_estimator(model)
    if xgb_est is not None and hasattr(xgb_est, "feature_importances_"):
        importances = xgb_est.feature_importances_

    # Fall back to any tree estimator with feature_importances_
    if importances is None and hasattr(model, "estimators_"):
        arrays = []
        for _, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                arrays.append(est.feature_importances_)
        if arrays:
            importances = np.mean(arrays, axis=0)

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
    return df


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Top Feature Importances (XGBoost gain)",
):
    """
    Return a Plotly horizontal-bar figure of feature importances.
    Returns None when Plotly is unavailable.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    df = compute_feature_importance(model, feature_names, top_n=top_n)
    if df.empty:
        return None

    df_sorted = df.sort_values("importance")
    fig = px.bar(
        df_sorted,
        x="importance_pct",
        y="feature",
        orientation="h",
        title=title,
        labels={"importance_pct": "Importance (%)", "feature": "Feature"},
        color="importance_pct",
        color_continuous_scale="Blues",
        text="importance_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=max(400, top_n * 24),
        coloraxis_showscale=False,
        margin={"l": 200, "r": 60, "t": 50, "b": 40},
        yaxis_title=None,
    )
    return fig


# ── SHAP analysis ─────────────────────────────────────────────────────────────


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for the XGBoost component of the ensemble.
    Returns a 2-D array (samples × features × classes) or None when SHAP
    is not installed or the model type is unsupported.
    """
    if not _SHAP_AVAILABLE:
        return None

    xgb_est = _get_xgb_estimator(model)
    if xgb_est is None:
        return None

    sample = X.iloc[:max_samples]
    try:
        explainer = _shap.TreeExplainer(xgb_est)
        shap_vals = explainer.shap_values(sample)
        return shap_vals
    except Exception as exc:
        print(f"[feature_analysis] SHAP computation failed: {exc}")
        return None


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 500,
    class_index: int = 0,
    title: str = "SHAP Feature Contributions (Home Win)",
):
    """
    Return a Plotly bar figure summarising mean |SHAP| values per feature.
    Returns None when SHAP/Plotly is unavailable or computation fails.
    """
    if not _PLOTLY_AVAILABLE or not _SHAP_AVAILABLE:
        return None

    shap_vals = compute_shap_values(model, X, max_samples=max_samples)
    if shap_vals is None:
        return None

    # Multi-class SHAP returns a list of arrays; single-class returns an array
    if isinstance(shap_vals, list):
        if class_index >= len(shap_vals):
            class_index = 0
        arr = shap_vals[class_index]
    else:
        arr = shap_vals

    mean_abs = np.abs(arr).mean(axis=0)
    n = min(len(feature_names), len(mean_abs))
    df = pd.DataFrame(
        {"feature": feature_names[:n], "mean_abs_shap": mean_abs[:n]}
    ).sort_values("mean_abs_shap").tail(20)

    fig = px.bar(
        df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=title,
        labels={"mean_abs_shap": "Mean |SHAP| value", "feature": "Feature"},
        color="mean_abs_shap",
        color_continuous_scale="RdBu",
    )
    fig.update_layout(
        height=500,
        coloraxis_showscale=False,
        margin={"l": 200, "r": 60, "t": 50, "b": 40},
        yaxis_title=None,
    )
    return fig
