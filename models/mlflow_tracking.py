"""
models/mlflow_tracking.py
MLflow experiment tracking for MLS model training runs.

If MLflow is not installed or a tracking server is unavailable, all calls
are no-ops and the function returns None.

Usage:
    from models.mlflow_tracking import log_training_run

    run_id = log_training_run(
        model       = ensemble,
        X_train     = X_tr,
        y_train     = y_tr,
        X_test      = X_te,
        y_test      = y_te,
        feature_names = names,
        run_name    = "ensemble_v2",
    )
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ── Optional MLflow import ────────────────────────────────────────────────────

try:
    import mlflow  # type: ignore
    import mlflow.sklearn  # type: ignore

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = "mls-predictions"


def log_training_run(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[list[str]] = None,
    run_name: str = "ensemble",
    extra_params: Optional[dict] = None,
) -> Optional[str]:
    """
    Log a model training run to MLflow.

    Parameters
    ----------
    model       : Trained scikit-learn estimator.
    X_train / X_test : Feature DataFrames used for training and testing.
    y_train / y_test : Target Series (0 = H, 1 = D, 2 = A).
    feature_names    : Optional list of original feature names.
    run_name         : Label for the MLflow run.
    extra_params     : Additional key-value params to log.

    Returns
    -------
    MLflow run ID string, or None when MLflow is unavailable.
    """
    if not _MLFLOW_AVAILABLE:
        print("[mlflow] MLflow not installed — skipping run logging.")
        return None

    try:
        from sklearn.metrics import accuracy_score, log_loss  # noqa: PLC0415

        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=run_name) as run:
            # ── Parameters ──────────────────────────────────────────────────
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("model_type", type(model).__name__)
            if extra_params:
                mlflow.log_params(extra_params)

            # ── Metrics ─────────────────────────────────────────────────────
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("test_accuracy", round(acc, 4))

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                try:
                    ll = log_loss(y_test, y_prob)
                    mlflow.log_metric("log_loss", round(ll, 4))
                except Exception:
                    pass

            # Per-class accuracy
            for cls, label in enumerate(["home_win", "draw", "away_win"]):
                mask = y_test == cls
                if mask.any():
                    cls_acc = accuracy_score(y_test[mask], y_pred[mask])
                    mlflow.log_metric(f"acc_{label}", round(cls_acc, 4))

            # ── Feature importances (as artifact) ──────────────────────────
            if feature_names:
                _log_feature_importance(model, feature_names)

            # ── Model artifact ──────────────────────────────────────────────
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = run.info.run_id
            print(f"[mlflow] Logged run '{run_name}' — id={run_id}")
            return run_id

    except Exception as exc:
        print(f"[mlflow] Logging failed: {exc}")
        return None


def _log_feature_importance(model, feature_names: list[str]) -> None:
    """Log a feature_importances.csv artifact."""
    import io  # noqa: PLC0415

    importances: Optional[np.ndarray] = None
    if hasattr(model, "estimators_"):
        arrays = []
        for _, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                arrays.append(est.feature_importances_)
        if arrays:
            importances = np.mean(arrays, axis=0)

    if importances is None or len(importances) != len(feature_names):
        return

    buf = io.StringIO()
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(buf, index=False)
    buf.seek(0)
    mlflow.log_text(buf.getvalue(), "feature_importances.csv")


def get_recent_runs(n: int = 10) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame of the N most recent MLflow runs for this experiment.
    Returns None when MLflow is unavailable or no runs exist.
    """
    if not _MLFLOW_AVAILABLE:
        return None
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            max_results=n,
        )
        return runs if not runs.empty else None
    except Exception:
        return None
