"""Point-in-time feature preparation and artifact loading for upcoming fixtures."""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from models.frontier_features import STADIUMS, add_frontier_features, haversine_km
from team_name_mapping import normalize_team_name


def load_frontier_artifact(path: str | Path = "models/frontier_logistic.pkl") -> dict:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "model" not in artifact or "features" not in artifact:
        raise ValueError("Unsupported model artifact; run scripts/run_backtest.py to rebuild it.")
    return artifact


def _date_column(fixtures: pd.DataFrame) -> str:
    for column in ("MatchDate", "Date", "date"):
        if column in fixtures.columns:
            return column
    raise ValueError("Fixtures require a Date or MatchDate column.")


def _team_latest_value(history: pd.DataFrame, team: str, base: str) -> float:
    normalized = normalize_team_name(team)
    home_rows = history[history["HomeTeam"].astype(str).map(normalize_team_name) == normalized]
    away_rows = history[history["AwayTeam"].astype(str).map(normalize_team_name) == normalized]
    candidates: list[tuple[pd.Timestamp, float]] = []
    for rows, prefix in ((home_rows, "home_"), (away_rows, "away_")):
        column = f"{prefix}{base}"
        if column not in rows.columns or rows.empty:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        valid = rows.loc[values.notna()].copy()
        if not valid.empty:
            value = float(pd.to_numeric(valid.iloc[-1][column], errors="coerce"))
            candidates.append((pd.to_datetime(valid.iloc[-1]["MatchDate"], errors="coerce"), value))
    if not candidates:
        return np.nan
    return max(candidates, key=lambda item: item[0])[1]


def _recent_team_stats(
    history: pd.DataFrame,
    team: str,
    cutoff: pd.Timestamp,
    lookback: int,
) -> dict[str, float]:
    normalized = normalize_team_name(team)
    team_rows = history[
        ((history["HomeTeam"].astype(str).map(normalize_team_name) == normalized)
         | (history["AwayTeam"].astype(str).map(normalize_team_name) == normalized))
        & (history["MatchDate"] < cutoff)
    ].sort_values("MatchDate").tail(lookback)
    if team_rows.empty:
        return {"xg": np.nan, "xg_against": np.nan, "points": np.nan, "rest_days": np.nan}
    observations: list[dict[str, float]] = []
    for row in team_rows.itertuples(index=False):
        is_home = normalize_team_name(str(row.HomeTeam)) == normalized
        goals_for = float(getattr(row, "HomeGoals" if is_home else "AwayGoals", 0.0) or 0.0)
        goals_against = float(getattr(row, "AwayGoals" if is_home else "HomeGoals", 0.0) or 0.0)
        xg_for_raw = getattr(row, "home_xgoals" if is_home else "away_xgoals", goals_for)
        xg_against_raw = getattr(row, "away_xgoals" if is_home else "home_xgoals", goals_against)
        xg_for = goals_for if pd.isna(xg_for_raw) else float(xg_for_raw)
        xg_against = goals_against if pd.isna(xg_against_raw) else float(xg_against_raw)
        result = str(getattr(row, "Result", ""))
        points = 3.0 if result == ("H" if is_home else "A") else (1.0 if result == "D" else 0.0)
        observations.append({"xg": xg_for, "xg_against": xg_against, "points": points})
    weights = np.power(0.85, np.arange(len(observations) - 1, -1, -1))
    return {
        "xg": float(np.average([item["xg"] for item in observations], weights=weights)),
        "xg_against": float(np.average([item["xg_against"] for item in observations], weights=weights)),
        "points": float(np.mean([item["points"] for item in observations])),
        "rest_days": float((cutoff - team_rows["MatchDate"].max()).days),
    }


def prepare_upcoming_features(
    fixtures: pd.DataFrame,
    historical: pd.DataFrame,
    feature_names: list[str],
    sources: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Construct an inference frame without using information after kickoff."""
    if fixtures.empty:
        return pd.DataFrame(columns=feature_names)
    upcoming = fixtures.copy()
    date_col = _date_column(upcoming)
    upcoming["MatchDate"] = pd.to_datetime(upcoming[date_col], errors="coerce")
    if "Time" in upcoming.columns:
        combined_time = upcoming["MatchDate"].dt.strftime("%Y-%m-%d") + " " + upcoming["Time"].fillna("00:00")
        upcoming["Kickoff"] = pd.to_datetime(combined_time, errors="coerce", format="mixed")
    upcoming["HomeTeam"] = upcoming["HomeTeam"].astype(str).map(normalize_team_name)
    upcoming["AwayTeam"] = upcoming["AwayTeam"].astype(str).map(normalize_team_name)
    history = historical.copy()
    history["MatchDate"] = pd.to_datetime(history["MatchDate"], errors="coerce")
    history = history.sort_values("MatchDate")

    # Recompute frontier histories over completed games and append upcoming rows.
    combined = pd.concat([history, upcoming], ignore_index=True, sort=False)
    enriched = add_frontier_features(combined, sources)
    result = enriched.tail(len(upcoming)).copy().reset_index(drop=True)

    for index, fixture in result.iterrows():
        home, away = str(fixture["HomeTeam"]), str(fixture["AwayTeam"])
        cutoff = pd.to_datetime(fixture.get("Kickoff", fixture.get("MatchDate")), errors="coerce")
        if pd.isna(cutoff):
            cutoff = pd.Timestamp.max.normalize()
        for side, team in (("home", home), ("away", away)):
            for lookback in (5, 10):
                recent = _recent_team_stats(history, team, cutoff, lookback)
                derived = {
                    f"{side}_xg_l{lookback}": recent["xg"],
                    f"{side}_xg_against_l{lookback}": recent["xg_against"],
                    f"{side}_pts_l{lookback}": recent["points"],
                }
                for feature, value in derived.items():
                    if feature in feature_names:
                        result.loc[index, feature] = value
                if f"{side}_rest_days" in feature_names:
                    result.loc[index, f"{side}_rest_days"] = recent["rest_days"]
        for feature in feature_names:
            if feature.startswith("home_"):
                result.loc[index, feature] = result.loc[index, feature] if feature in result and pd.notna(result.loc[index, feature]) else _team_latest_value(history, home, feature[5:])
            elif feature.startswith("away_"):
                result.loc[index, feature] = result.loc[index, feature] if feature in result and pd.notna(result.loc[index, feature]) else _team_latest_value(history, away, feature[5:])

        home_meta, away_meta = STADIUMS.get(home), STADIUMS.get(away)
        miles = haversine_km(away, home) * 0.621371
        static_values = {
            "away_travel_miles": miles,
            "is_long_haul": float(miles > 1_500),
            "home_is_turf": float(bool(home_meta and home_meta.surface == "turf")),
            "is_cross_conference": float(bool(home_meta and away_meta and home_meta.conference != away_meta.conference)),
            "season_weight": 1.0 + max(int(fixture["MatchDate"].year) - 2020, 0) * 0.15 if pd.notna(fixture["MatchDate"]) else 1.0,
        }
        odds = []
        for side in ("home", "draw", "away"):
            preferred = fixture.get(f"draftkings_{side}_odds")
            odds.append(preferred if pd.notna(preferred) else fixture.get(f"best_{side}_odds"))
        if all(pd.notna(value) for value in odds):
            values = np.asarray(odds, dtype=float)
            implied = np.where(values < 0, np.abs(values) / (np.abs(values) + 100.0), 100.0 / (values + 100.0))
            implied /= implied.sum()
            static_values.update({
                "selection_implied_home_prob": float(implied[0]),
                "selection_implied_draw_prob": float(implied[1]),
                "selection_implied_away_prob": float(implied[2]),
                "odds_data_available": 1.0,
            })
        else:
            static_values["odds_data_available"] = 0.0
        for feature, value in static_values.items():
            if feature in feature_names:
                result.loc[index, feature] = value
    for feature in feature_names:
        if feature not in result.columns:
            result[feature] = np.nan
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
    return result[feature_names]


def predict_upcoming(
    fixtures: pd.DataFrame,
    historical: pd.DataFrame,
    artifact: dict,
    sources: Mapping[str, pd.DataFrame] | None = None,
) -> np.ndarray:
    features = prepare_upcoming_features(fixtures, historical, list(artifact["features"]), sources)
    return predict_feature_frame(features, artifact)


def predict_feature_frame(features: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Predict and calibrate an already frozen inference feature frame."""
    probabilities = np.clip(
        np.asarray(artifact["model"].predict_proba(features), dtype=float), 1e-9, 1.0
    )
    temperature = float(artifact.get("temperature", 1.0) or 1.0)
    logits = np.log(probabilities) / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)
