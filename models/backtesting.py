"""Reproducible MLS baselines, rolling backtests, calibration, and release gates."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.frontier_features import FRONTIER_NUMERIC_FEATURES, STADIUMS
from models.governance import ConformalAbstainer, repository_code_version


RESULT_TO_INDEX = {"H": 0, "D": 1, "A": 2}
INDEX_TO_RESULT = {value: key for key, value in RESULT_TO_INDEX.items()}
DEFAULT_FEATURES = [
    "home_xg_l5",
    "away_xg_l5",
    "home_xg_l10",
    "away_xg_l10",
    "home_xg_against_l5",
    "away_xg_against_l5",
    "home_pts_l5",
    "away_pts_l5",
    "home_rest_days",
    "away_rest_days",
    "away_travel_miles",
    "is_long_haul",
    "home_is_turf",
    "is_cross_conference",
    "home_games_from_playoff",
    "away_games_from_playoff",
    "selection_implied_home_prob",
    "selection_implied_draw_prob",
    "selection_implied_away_prob",
    "odds_data_available",
    *FRONTIER_NUMERIC_FEATURES,
]


@dataclass
class MetricSet:
    samples: int
    accuracy: float
    log_loss: float
    brier: float
    rps: float
    ece: float


@dataclass
class ReleaseGate:
    passed: bool
    untouched_season: bool
    frozen_selections: int
    market_relative_brier: float | None
    market_relative_log_loss: float | None
    median_clv: float | None
    ledger_complete: bool
    reasons: list[str]


class TemperatureCalibrator:
    """One-parameter multiclass calibration fitted on an earlier time slice."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        probs = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
        labels = np.asarray(y, dtype=int)
        if len(labels) < 10 or len(np.unique(labels)) < 2:
            return self
        logits = np.log(probs)

        def objective(value: float) -> float:
            scaled = _softmax(logits / value)
            return float(log_loss(labels, scaled, labels=[0, 1, 2]))

        result = minimize_scalar(objective, bounds=(0.35, 4.0), method="bounded")
        if result.success and np.isfinite(result.x):
            self.temperature = float(result.x)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
        return _softmax(np.log(probs) / self.temperature)


class EloBaseline:
    """Rolling club Elo with an MLS home-advantage term."""

    def __init__(self, k_factor: float = 24.0, home_advantage: float = 65.0) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def _rating(self, team: str) -> float:
        return self.ratings.setdefault(team, 1_500.0)

    def predict_one(self, home: str, away: str) -> np.ndarray:
        home_rating = self._rating(home) + self.home_advantage
        away_rating = self._rating(away)
        expected_home_no_draw = 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))
        # MLS draw mass is centered around rating parity and shrinks for mismatches.
        rating_gap = abs(home_rating - away_rating)
        draw = 0.27 * math.exp(-rating_gap / 650.0)
        home_prob = expected_home_no_draw * (1.0 - draw)
        away_prob = (1.0 - expected_home_no_draw) * (1.0 - draw)
        return np.array([home_prob, draw, away_prob], dtype=float)

    def update(self, home: str, away: str, result: str) -> None:
        if result not in RESULT_TO_INDEX:
            return
        home_rating, away_rating = self._rating(home), self._rating(away)
        expected = 1.0 / (1.0 + 10 ** ((away_rating - (home_rating + self.home_advantage)) / 400.0))
        actual = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
        movement = self.k_factor * (actual - expected)
        self.ratings[home] = home_rating + movement
        self.ratings[away] = away_rating - movement


class DixonColesBaseline:
    """Regularized team scoring strengths with the Dixon-Coles low-score correction."""

    def __init__(self, rho: float = -0.08, shrinkage_games: float = 8.0) -> None:
        self.rho = rho
        self.shrinkage_games = shrinkage_games
        self.league_home = 1.55
        self.league_away = 1.25
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> "DixonColesBaseline":
        if matches.empty:
            return self
        home_goals = pd.to_numeric(matches.get("HomeGoals"), errors="coerce")
        away_goals = pd.to_numeric(matches.get("AwayGoals"), errors="coerce")
        self.league_home = float(home_goals.mean()) if home_goals.notna().any() else self.league_home
        self.league_away = float(away_goals.mean()) if away_goals.notna().any() else self.league_away
        teams = sorted(set(matches["HomeTeam"]).union(matches["AwayTeam"]))
        for team in teams:
            home_mask = matches["HomeTeam"] == team
            away_mask = matches["AwayTeam"] == team
            scored = pd.concat([home_goals[home_mask], away_goals[away_mask]]).dropna()
            conceded = pd.concat([away_goals[home_mask], home_goals[away_mask]]).dropna()
            n = float(len(scored))
            weight = n / (n + self.shrinkage_games)
            league_mean = max((self.league_home + self.league_away) / 2.0, 0.1)
            self.attack[team] = weight * float(scored.mean() / league_mean) + (1.0 - weight)
            self.defence[team] = weight * float(conceded.mean() / league_mean) + (1.0 - weight)
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        home_xg = self.league_home * self.attack.get(home, 1.0) * self.defence.get(away, 1.0)
        away_xg = self.league_away * self.attack.get(away, 1.0) * self.defence.get(home, 1.0)
        return float(np.clip(home_xg, 0.2, 4.5)), float(np.clip(away_xg, 0.2, 4.5))

    def predict_one(self, home: str, away: str, max_goals: int = 9) -> np.ndarray:
        grid = self.score_grid(home, away, max_goals=max_goals)
        return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])

    def score_grid(self, home: str, away: str, max_goals: int = 9) -> np.ndarray:
        """Return the normalized joint home/away score distribution."""
        home_xg, away_xg = self.expected_goals(home, away)
        grid = np.outer(poisson.pmf(np.arange(max_goals + 1), home_xg), poisson.pmf(np.arange(max_goals + 1), away_xg))
        # Dixon-Coles correction for 0-0, 0-1, 1-0, 1-1 cells.
        grid[0, 0] *= max(1.0 - home_xg * away_xg * self.rho, 0.0)
        grid[0, 1] *= max(1.0 + home_xg * self.rho, 0.0)
        grid[1, 0] *= max(1.0 + away_xg * self.rho, 0.0)
        grid[1, 1] *= max(1.0 - self.rho, 0.0)
        grid /= grid.sum()
        return grid

    def red_card_score_grid(
        self,
        home: str,
        away: str,
        home_red_probability: float = 0.09,
        away_red_probability: float = 0.09,
        max_goals: int = 9,
    ) -> np.ndarray:
        """Mix base and red-card score scenarios for totals/BTTS sensitivity."""
        home_xg, away_xg = self.expected_goals(home, away)
        p_home_red = float(np.clip(home_red_probability, 0.0, 0.5))
        p_away_red = float(np.clip(away_red_probability, 0.0, 0.5))

        def poisson_grid(home_rate: float, away_rate: float) -> np.ndarray:
            grid = np.outer(
                poisson.pmf(np.arange(max_goals + 1), max(home_rate, 0.05)),
                poisson.pmf(np.arange(max_goals + 1), max(away_rate, 0.05)),
            )
            return grid / grid.sum()

        base_weight = (1.0 - p_home_red) * (1.0 - p_away_red)
        home_red_weight = p_home_red * (1.0 - p_away_red)
        away_red_weight = (1.0 - p_home_red) * p_away_red
        both_weight = p_home_red * p_away_red
        mixture = (
            base_weight * self.score_grid(home, away, max_goals=max_goals)
            + home_red_weight * poisson_grid(home_xg * 0.65, away_xg * 1.25)
            + away_red_weight * poisson_grid(home_xg * 1.25, away_xg * 0.65)
            + both_weight * poisson_grid(home_xg * 0.82, away_xg * 0.82)
        )
        return mixture / mixture.sum()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _ensure_season(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MatchDate"] = pd.to_datetime(out["MatchDate"], errors="coerce")
    out = out.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam", "Result"])
    out = out[out["Result"].isin(RESULT_TO_INDEX)].copy()
    out["season"] = out["MatchDate"].dt.year
    return out.sort_values("MatchDate").reset_index(drop=True)


def available_features(df: pd.DataFrame, requested: Sequence[str] | None = None) -> list[str]:
    """Return numeric, non-constant, point-in-time model features."""
    candidates = list(dict.fromkeys(requested or DEFAULT_FEATURES))
    output: list[str] = []
    for column in candidates:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            output.append(column)
    return output


def build_logistic_model(features: Sequence[str]) -> Pipeline:
    numeric = list(features)
    preprocessing = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", preprocessing),
            ("model", LogisticRegression(max_iter=2_000, C=0.7, random_state=42)),
        ]
    )


def home_field_probabilities(train: pd.DataFrame, count: int) -> np.ndarray:
    frequencies = train["Result"].value_counts(normalize=True)
    base = np.array([frequencies.get(label, 1 / 3) for label in ("H", "D", "A")], dtype=float)
    base /= base.sum()
    return np.tile(base, (count, 1))


def market_probabilities(frame: pd.DataFrame) -> np.ndarray | None:
    columns = ["odds_implied_home_prob", "odds_implied_draw_prob", "odds_implied_away_prob"]
    if not all(column in frame.columns for column in columns):
        return None
    if "odds_data_available" in frame.columns and not pd.to_numeric(
        frame["odds_data_available"], errors="coerce"
    ).fillna(0).astype(bool).any():
        return None
    probs = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probs).any():
        return None
    row_sum = np.nansum(probs, axis=1, keepdims=True)
    valid = np.isfinite(probs).all(axis=1) & (row_sum[:, 0] > 0)
    probs[valid] /= row_sum[valid]
    probs[~valid] = np.nan
    return probs


def score_metrics(y: np.ndarray, probabilities: np.ndarray) -> MetricSet:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.asarray(y, dtype=int)
    one_hot = np.eye(3)[labels]
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= low) & (confidence < low + 0.1)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return MetricSet(
        samples=int(len(labels)),
        accuracy=float(accuracy_score(labels, probs.argmax(axis=1))),
        log_loss=float(log_loss(labels, probs, labels=[0, 1, 2])),
        brier=float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        rps=float(np.mean(np.sum((np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(one_hot, axis=1)[:, :-1]) ** 2, axis=1) / 2.0)),
        ece=ece,
    )


def rolling_elo_probabilities(df: pd.DataFrame) -> np.ndarray:
    model = EloBaseline()
    probabilities: list[np.ndarray] = []
    for _, row in df.iterrows():
        probabilities.append(model.predict_one(str(row["HomeTeam"]), str(row["AwayTeam"])))
        model.update(str(row["HomeTeam"]), str(row["AwayTeam"]), str(row["Result"]))
    return np.vstack(probabilities)


def _competition_phase(row: pd.Series) -> str:
    if "competition_phase" in row and pd.notna(row["competition_phase"]):
        return str(row["competition_phase"])
    competition = str(row.get("Competition", "")).lower()
    stage = str(row.get("Phase", row.get("stage", row.get("stage_name", "")))).lower()
    if "league" in competition and "cup" in competition:
        return "Leagues Cup"
    if "playoff" in stage or "final" in stage or "round" in stage:
        return "Playoffs"
    if "regular" in stage or "regular" in competition:
        return "Regular Season"
    return "Unknown"


def rolling_backtest(
    matches: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
    first_test_season: int | None = None,
    calibration_share: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, object], Pipeline]:
    """Run season-ordered evaluation with one locked calibration window.

    For a 2025 holdout this trains only through 2024, calibrates on the first
    chronological 20% of 2025, and evaluates on the untouched remainder. Later
    seasons are evaluated in full with those calibration parameters locked while
    the underlying model rolls forward using only earlier seasons.
    """
    df = _ensure_season(matches)
    seasons = sorted(df["season"].unique().tolist())
    if len(seasons) < 2:
        raise ValueError("At least two completed seasons are required for a rolling backtest.")
    if first_test_season is None:
        first_test_season = 2025 if 2025 in seasons else seasons[max(1, len(seasons) - 2)]
    test_seasons = [season for season in seasons if season >= first_test_season]
    features = available_features(df, feature_names)
    if not features:
        raise ValueError("No usable numeric point-in-time features were found.")

    all_rows: list[pd.DataFrame] = []
    fold_reports: list[dict[str, object]] = []
    final_model: Pipeline | None = None
    latest_temperature = 1.0
    latest_conformal_quantile = 1.0
    locked_temperature: float | None = None
    locked_conformal_quantile: float | None = None
    locked_segment_temperatures: dict[str, float] = {}

    elo_all = rolling_elo_probabilities(df)
    for fold_index, season in enumerate(test_seasons):
        train = df[df["season"] < season].copy()
        current = df[df["season"] == season].copy()
        if len(train) < 100 or len(current) < 20:
            continue
        cut = max(int(len(current) * calibration_share), 10) if fold_index == 0 else 0
        if cut and len(current) - cut < 10:
            cut = 0
        calibration = current.iloc[:cut]
        test = current.iloc[cut:].copy()
        if test.empty:
            continue
        model = build_logistic_model(features)
        train_weights = pd.to_numeric(
            train.get("season_weight", pd.Series(1.0, index=train.index)), errors="coerce"
        ).fillna(1.0).clip(lower=0.1)
        model.fit(
            train[features],
            train["Result"].map(RESULT_TO_INDEX),
            model__sample_weight=train_weights.to_numpy(),
        )
        if not calibration.empty:
            raw_cal = model.predict_proba(calibration[features])
            calibration_labels = calibration["Result"].map(RESULT_TO_INDEX).to_numpy()
            calibrator = TemperatureCalibrator().fit(raw_cal, calibration_labels)
            conformal = ConformalAbstainer(alpha=0.1).fit(
                calibrator.transform(raw_cal), calibration_labels
            )
            locked_temperature = calibrator.temperature
            locked_conformal_quantile = conformal.quantile
        else:
            calibrator = TemperatureCalibrator()
            conformal = ConformalAbstainer(alpha=0.1)
            calibrator.temperature = locked_temperature or 1.0
            conformal.quantile = locked_conformal_quantile or 1.0
            raw_cal = np.empty((0, 3))
            calibration_labels = np.empty(0, dtype=int)
        raw_test = model.predict_proba(test[features])
        model_probs = calibrator.transform(raw_test)
        segment_temperatures: dict[str, float] = {}
        segment_calibration_diagnostics: dict[str, dict[str, float | bool | int]] = {}
        # Pool the base calibration with any applicable split-specific estimate
        # for each row. Later folds reuse the locked split temperatures without
        # peeking at the season being evaluated.
        pooled = model_probs.copy()
        pool_count = np.ones(len(test), dtype=float)
        for segment in ("home_is_turf", "is_long_haul", "is_cross_conference"):
            if segment not in test.columns:
                continue
            if not calibration.empty and segment in calibration.columns:
                values = sorted(pd.to_numeric(calibration[segment], errors="coerce").dropna().unique())
            else:
                prefix = f"{segment}="
                values = [float(key.removeprefix(prefix)) for key in locked_segment_temperatures if key.startswith(prefix)]
            for value in values:
                test_mask = pd.to_numeric(test[segment], errors="coerce").to_numpy() == value
                if not test_mask.any():
                    continue
                key = f"{segment}={value:g}"
                group_calibrator = TemperatureCalibrator()
                if not calibration.empty:
                    calibration_mask = pd.to_numeric(calibration[segment], errors="coerce").to_numpy() == value
                    if calibration_mask.sum() < 10:
                        continue
                    group_calibrator.fit(raw_cal[calibration_mask], calibration_labels[calibration_mask])
                    baseline_group = calibrator.transform(raw_cal[calibration_mask])
                    candidate_group = group_calibrator.transform(raw_cal[calibration_mask])
                    baseline_loss = float(log_loss(calibration_labels[calibration_mask], baseline_group, labels=[0, 1, 2]))
                    candidate_loss = float(log_loss(calibration_labels[calibration_mask], candidate_group, labels=[0, 1, 2]))
                    applied = candidate_loss + 0.002 < baseline_loss
                    segment_calibration_diagnostics[key] = {
                        "samples": int(calibration_mask.sum()),
                        "base_log_loss": baseline_loss,
                        "candidate_log_loss": candidate_loss,
                        "applied": applied,
                    }
                    if not applied:
                        continue
                    locked_segment_temperatures[key] = group_calibrator.temperature
                elif key in locked_segment_temperatures:
                    group_calibrator.temperature = locked_segment_temperatures[key]
                else:
                    continue
                pooled[test_mask] += group_calibrator.transform(raw_test[test_mask])
                pool_count[test_mask] += 1.0
                segment_temperatures[key] = group_calibrator.temperature
        model_probs = pooled / pool_count[:, None]
        model_probs /= model_probs.sum(axis=1, keepdims=True)
        dc = DixonColesBaseline().fit(train)
        dc_probs = np.vstack([dc.predict_one(str(row.HomeTeam), str(row.AwayTeam)) for row in test.itertuples()])
        home_probs = home_field_probabilities(train, len(test))
        market_probs = market_probabilities(test)
        test_indices = test.index.to_numpy()
        elo_probs = elo_all[test_indices]
        y = test["Result"].map(RESULT_TO_INDEX).to_numpy()

        fold = test[["MatchDate", "HomeTeam", "AwayTeam", "Result", "season"]].copy()
        for prefix, probs in {
            "model": model_probs,
            "elo": elo_probs,
            "dixon_coles": dc_probs,
            "home_field": home_probs,
        }.items():
            fold[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]] = probs
        if market_probs is not None:
            fold[["market_home", "market_draw", "market_away"]] = market_probs
        for odds_column in (
            "odds_data_available",
            "odds_home_value", "odds_draw_value", "odds_away_value",
            "closing_home_value", "closing_draw_value", "closing_away_value",
        ):
            if odds_column in test.columns:
                fold[odds_column] = pd.to_numeric(test[odds_column], errors="coerce").values
        for segment in (
            "competition_phase",
            "travel_km",
            "home_is_turf",
            "is_long_haul",
            "is_cross_conference",
            "turf_to_grass_visitor",
            "home_altitude_ft",
            "home_mid_season_roster_changes",
            "home_manager_tenure_days",
        ):
            if segment in test.columns:
                fold[segment] = test[segment].values
        roster_mechanisms = [column for column in ("home_dp_available", "home_u22_available", "home_tam_available") if column in test.columns]
        if roster_mechanisms:
            mechanism_score = test[roster_mechanisms].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            missing_source = (
                test["roster_data_missing"]
                if "roster_data_missing" in test.columns
                else pd.Series(0, index=test.index)
            )
            missing = pd.to_numeric(missing_source, errors="coerce").fillna(0).astype(bool)
            tier = pd.cut(
                mechanism_score, [-np.inf, 0.5, 0.8, np.inf],
                labels=["disrupted", "mixed", "available"],
            ).astype(object)
            tier[missing] = "unknown"
            fold["salary_cap_roster_tier"] = tier.astype(str).values
        fold["competition_phase"] = test.apply(_competition_phase, axis=1).values
        if "prediction_horizon" in test.columns:
            fold["prediction_horizon"] = test["prediction_horizon"].astype(str).values
        elif "prediction_horizon_hours" in test.columns:
            horizon = pd.to_numeric(test["prediction_horizon_hours"], errors="coerce")
            fold["prediction_horizon"] = pd.cut(
                horizon, [-np.inf, 1, 6, 24, 72, np.inf],
                labels=["confirmed-lineup", "late", "matchday", "day-before", "early"],
            ).astype(str).values
        else:
            fold["prediction_horizon"] = "pre-match"
        fold["temperature"] = calibrator.temperature
        all_rows.append(fold)

        last_match = pd.to_datetime(current["MatchDate"], errors="coerce").max()
        season_teams = set(current["HomeTeam"].dropna()).union(current["AwayTeam"].dropna())
        expected_regular_matches = max(len(season_teams) * 34 / 2, 1)
        season_complete = bool(
            len(current) >= 0.8 * expected_regular_matches
            and pd.notna(last_match)
            and int(last_match.month) >= 10
        )
        report: dict[str, object] = {
            "season": int(season),
            "trained_through": int(train["season"].max()),
            "calibration_matches": int(len(calibration)),
            "untouched_matches": int(len(test)),
            "season_complete": season_complete,
            "full_untouched_season": bool(season_complete and calibration.empty),
            "temperature": calibrator.temperature,
            "conformal_quantile": conformal.quantile,
            "segment_temperatures": segment_temperatures,
            "segment_calibration_diagnostics": segment_calibration_diagnostics,
            "model": asdict(score_metrics(y, model_probs)),
            "elo": asdict(score_metrics(y, elo_probs)),
            "dixon_coles": asdict(score_metrics(y, dc_probs)),
            "home_field": asdict(score_metrics(y, home_probs)),
        }
        if market_probs is not None:
            market_valid = np.isfinite(market_probs).all(axis=1)
            if market_valid.any():
                report["market"] = asdict(score_metrics(y[market_valid], market_probs[market_valid]))
        report["betting"] = betting_performance(fold)
        fold_reports.append(report)
        final_model = model
        latest_temperature = calibrator.temperature
        latest_conformal_quantile = conformal.quantile

    if not all_rows or final_model is None:
        raise ValueError("No season contained enough rows for an untouched evaluation fold.")
    predictions = pd.concat(all_rows, ignore_index=True)
    aggregate = summarize_predictions(predictions)
    deployment_selection = select_deployment_candidate(aggregate)
    market_comparison: dict[str, dict[str, float | int]] = {}
    market_columns = _probability_columns("market")
    if all(column in predictions.columns for column in market_columns):
        market_values = predictions[market_columns].to_numpy(dtype=float)
        comparable = np.isfinite(market_values).all(axis=1)
        if comparable.any():
            comparable_labels = predictions.loc[comparable, "Result"].map(RESULT_TO_INDEX).to_numpy()
            market_comparison = {
                "model": asdict(score_metrics(
                    comparable_labels,
                    predictions.loc[comparable, _probability_columns("model")].to_numpy(dtype=float),
                )),
                "market": asdict(score_metrics(comparable_labels, market_values[comparable])),
            }
    betting = betting_performance(predictions)
    segments = segmented_metrics(predictions)
    expansion = expansion_team_holdout(predictions)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": repository_code_version(),
        "data_hash": dataframe_manifest(df),
        "features": features,
        "folds": fold_reports,
        "aggregate": aggregate,
        "market_comparison": market_comparison,
        "betting": betting,
        "segments": segments,
        "expansion_holdout": expansion,
        "deployment": {
            "temperature": latest_temperature,
            "conformal_quantile": latest_conformal_quantile,
            "trained_through": df["MatchDate"].max().isoformat(),
            **deployment_selection,
        },
    }
    # Evaluation remains untouched above. The deployable artifact is then refit on
    # every completed match; it inherits only calibration parameters learned in the
    # latest earlier calibration slice.
    deployment_model = build_logistic_model(features)
    deployment_weights = pd.to_numeric(
        df.get("season_weight", pd.Series(1.0, index=df.index)), errors="coerce"
    ).fillna(1.0).clip(lower=0.1)
    deployment_model.fit(
        df[features],
        df["Result"].map(RESULT_TO_INDEX),
        model__sample_weight=deployment_weights.to_numpy(),
    )
    return predictions, report, deployment_model


def _probability_columns(prefix: str) -> list[str]:
    return [f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    y = predictions["Result"].map(RESULT_TO_INDEX).to_numpy()
    output: dict[str, dict[str, float | int]] = {}
    for prefix in ("model", "elo", "dixon_coles", "home_field", "market"):
        columns = _probability_columns(prefix)
        if all(column in predictions.columns for column in columns):
            probabilities = predictions[columns].to_numpy(dtype=float)
            valid = np.isfinite(probabilities).all(axis=1)
            if valid.any():
                output[prefix] = asdict(score_metrics(y[valid], probabilities[valid]))
    return output


def select_deployment_candidate(aggregate: Mapping[str, Mapping[str, float | int]]) -> dict[str, object]:
    """Fail closed unless the governed model beats rolling Elo on both primary metrics."""
    model, elo = aggregate.get("model", {}), aggregate.get("elo", {})
    if not model or not elo:
        return {"selected_candidate": "none", "eligible": False, "reason": "Model/Elo comparison is unavailable."}
    model_loss, elo_loss = float(model.get("log_loss", float("inf"))), float(elo.get("log_loss", float("inf")))
    model_brier, elo_brier = float(model.get("brier", float("inf"))), float(elo.get("brier", float("inf")))
    if model_loss < elo_loss and model_brier < elo_brier:
        return {"selected_candidate": "model", "eligible": True, "reason": "Model beats rolling Elo on aggregate log loss and Brier score."}
    return {"selected_candidate": "elo", "eligible": False, "reason": "Governed model does not beat rolling Elo on both aggregate log loss and Brier score; model picks remain disabled."}


def _bucket(column: str, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if column == "home_altitude_ft":
        return pd.cut(numeric, [-1, 1_000, 3_500, np.inf], labels=["low", "medium", "high"])
    if column == "travel_km":
        return pd.cut(numeric, [-1, 800, 2_400, 4_000, np.inf], labels=["local", "regional", "long-haul", "extreme"])
    if column == "home_mid_season_roster_changes":
        return pd.cut(numeric, [-1, 0, 2, np.inf], labels=["stable", "moderate", "high"])
    if column == "home_manager_tenure_days":
        return pd.cut(numeric, [-1, 90, 365, np.inf], labels=["new", "established", "long-tenured"])
    return values.fillna("unknown").astype(str)


def segmented_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, dict[str, float | int]]]:
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    segment_columns = [
        "competition_phase",
        "travel_km",
        "home_is_turf",
        "is_long_haul",
        "is_cross_conference",
        "turf_to_grass_visitor",
        "home_altitude_ft",
        "home_mid_season_roster_changes",
        "home_manager_tenure_days",
        "prediction_horizon",
        "salary_cap_roster_tier",
    ]
    for column in segment_columns:
        if column not in predictions.columns:
            continue
        groups = _bucket(column, predictions[column])
        segment: dict[str, dict[str, float | int]] = {}
        for value in sorted(groups.dropna().unique(), key=str):
            mask = groups == value
            if mask.sum() < 5:
                continue
            y = predictions.loc[mask, "Result"].map(RESULT_TO_INDEX).to_numpy()
            probs = predictions.loc[mask, _probability_columns("model")].to_numpy()
            segment[str(value)] = asdict(score_metrics(y, probs))
        if segment:
            output[column] = segment
    return output


def betting_performance(predictions: pd.DataFrame, minimum_edge: float = 0.03) -> dict[str, float | int | None]:
    """Paper-trade a flat-unit 1X2 strategy and report turnover/ROI/drawdown/CLV."""
    odds_columns = ["odds_home_value", "odds_draw_value", "odds_away_value"]
    if (
        not all(column in predictions.columns for column in odds_columns)
        or (
            "odds_data_available" in predictions.columns
            and not pd.to_numeric(predictions["odds_data_available"], errors="coerce").fillna(0).astype(bool).any()
        )
    ):
        return {
            "selections": 0,
            "turnover_units": 0.0,
            "roi": None,
            "max_drawdown_units": None,
            "median_clv": None,
            "status": "odds-unavailable",
        }
    model_probs = predictions[_probability_columns("model")].to_numpy(dtype=float)
    odds = predictions[odds_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    implied = np.divide(1.0, odds, out=np.full_like(odds, np.nan), where=odds > 1.0)
    implied /= np.nansum(implied, axis=1, keepdims=True)
    edge = model_probs - implied
    choices = np.nanargmax(np.where(np.isfinite(edge), edge, -np.inf), axis=1)
    selected_edges = edge[np.arange(len(edge)), choices]
    valid = np.isfinite(selected_edges) & (selected_edges > minimum_edge)
    if not valid.any():
        return {
            "selections": 0,
            "turnover_units": 0.0,
            "roi": 0.0,
            "max_drawdown_units": 0.0,
            "median_clv": None,
            "status": "paper",
        }
    actual = predictions["Result"].map(RESULT_TO_INDEX).to_numpy()
    selected_odds = odds[np.arange(len(odds)), choices]
    profits = np.where(choices == actual, selected_odds - 1.0, -1.0)[valid]
    cumulative = np.cumsum(profits)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    drawdown = peak - cumulative
    clv_values: list[float] = []
    closing_columns = ["closing_home_value", "closing_draw_value", "closing_away_value"]
    if all(column in predictions.columns for column in closing_columns):
        closing = predictions[closing_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        selected_closing = closing[np.arange(len(closing)), choices]
        valid_closing = valid & np.isfinite(selected_closing) & (selected_closing > 1)
        clv_values = ((selected_odds[valid_closing] / selected_closing[valid_closing]) - 1.0).tolist()
    return {
        "selections": int(valid.sum()),
        "turnover_units": float(valid.sum()),
        "roi": float(profits.sum() / valid.sum()),
        "max_drawdown_units": float(drawdown.max()) if len(drawdown) else 0.0,
        "median_clv": float(np.median(clv_values)) if clv_values else None,
        "status": "paper",
    }


def expansion_team_holdout(predictions: pd.DataFrame) -> dict[str, object]:
    masks = []
    for row in predictions.itertuples():
        season = int(row.season)
        expansion_years = [
            STADIUMS.get(str(row.HomeTeam)).expansion_year if STADIUMS.get(str(row.HomeTeam)) else 1900,
            STADIUMS.get(str(row.AwayTeam)).expansion_year if STADIUMS.get(str(row.AwayTeam)) else 1900,
        ]
        masks.append(any(0 <= season - year <= 1 for year in expansion_years))
    mask = np.asarray(masks, dtype=bool)
    if mask.sum() < 5:
        return {"samples": int(mask.sum()), "status": "insufficient"}
    y = predictions.loc[mask, "Result"].map(RESULT_TO_INDEX).to_numpy()
    probs = predictions.loc[mask, _probability_columns("model")].to_numpy()
    return {"status": "measured", **asdict(score_metrics(y, probs))}


def dataframe_manifest(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def evaluate_release_gate(
    report: dict[str, object],
    frozen_selections: pd.DataFrame | None = None,
) -> ReleaseGate:
    selections = frozen_selections.copy() if frozen_selections is not None else pd.DataFrame()
    if "selection_id" in selections.columns:
        selections = selections.drop_duplicates("selection_id", keep="last")
    if "settlement" in selections.columns:
        selections = selections[selections["settlement"].notna()].copy()
    else:
        selections = selections.iloc[0:0].copy()
    reasons: list[str] = []
    folds = report.get("folds", []) if isinstance(report, dict) else []
    untouched = any(bool(fold.get("full_untouched_season", False)) for fold in folds if isinstance(fold, dict))
    if not untouched:
        reasons.append("No full untouched season has been evaluated.")
    if len(selections) < 300:
        reasons.append("Fewer than 300 frozen selections are settled.")

    aggregate = report.get("aggregate", {}) if isinstance(report, dict) else {}
    comparison = report.get("market_comparison", {}) if isinstance(report, dict) else {}
    model_metrics = comparison.get("model", {}) if isinstance(comparison, dict) else {}
    market_metrics = comparison.get("market", {}) if isinstance(comparison, dict) else {}
    rel_brier = None
    rel_log = None
    if market_metrics and model_metrics:
        rel_brier = float(market_metrics["brier"] - model_metrics["brier"])
        rel_log = float(market_metrics["log_loss"] - model_metrics["log_loss"])
        if rel_brier <= 0 or rel_log <= 0:
            reasons.append("Model has not beaten the de-vigged market on both Brier score and log loss.")
    else:
        reasons.append("Market-relative Brier/log-loss evidence is missing.")

    median_clv = None
    if "clv" in selections.columns and pd.to_numeric(selections["clv"], errors="coerce").notna().any():
        median_clv = float(pd.to_numeric(selections["clv"], errors="coerce").median())
        if median_clv <= 0:
            reasons.append("Median closing-line value is not positive.")
    else:
        reasons.append("Closing-line value is missing.")

    required = {"selection_time", "book", "odds", "limits", "result", "settlement", "code_version"}
    ledger_complete = bool(not selections.empty and required.issubset(selections.columns) and selections[list(required)].notna().all().all())
    if not ledger_complete:
        reasons.append("The timestamped selection ledger is incomplete.")
    return ReleaseGate(
        passed=not reasons,
        untouched_season=untouched,
        frozen_selections=int(len(selections)),
        market_relative_brier=rel_brier,
        market_relative_log_loss=rel_log,
        median_clv=median_clv,
        ledger_complete=ledger_complete,
        reasons=reasons,
    )


def save_backtest_artifacts(
    predictions: pd.DataFrame,
    report: dict[str, object],
    model: Pipeline,
    output_dir: str | Path = "data_files/backtests",
    model_path: str | Path = "models/frontier_logistic.pkl",
) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "latest_predictions.csv"
    report_path = output / "latest_report.json"
    predictions.to_csv(predictions_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    artifact_path = Path(model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        deployment = report.get("deployment", {})
        pickle.dump(
            {
                "model": model,
                "features": report.get("features", []),
                "report_hash": report.get("data_hash"),
                "code_version": report.get("code_version"),
                "temperature": deployment.get("temperature", 1.0),
                "conformal_quantile": deployment.get("conformal_quantile", 1.0),
            },
            handle,
        )
    return predictions_path, report_path, artifact_path
