"""Generate governed, frozen MLS selections for today's fixture slate."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager
from models.backtesting import DixonColesBaseline
from models.frontier_features import load_optional_sources
from models.governance import (
    cross_market_consistency,
    decide_bet,
    make_snapshot,
    new_selection_id,
    score_distribution_markets,
    repository_code_version,
)
from models.inference import load_frontier_artifact, predict_feature_frame, prepare_upcoming_features

TODAY_PATH = ROOT / "data_files" / "picks_today.csv"
FIXTURES_PATH = ROOT / "data_files" / "upcoming_fixtures.csv"
HISTORICAL_PATH = ROOT / "data_files" / "combined_historical_data.csv"
MODEL_PATH = ROOT / "models" / "frontier_logistic.pkl"
REPORT_PATH = ROOT / "data_files" / "backtests" / "latest_report.json"


def _implied_probability(american_odds: float) -> float:
    value = float(american_odds)
    return abs(value) / (abs(value) + 100.0) if value < 0 else 100.0 / (value + 100.0)


def _coalesce(row: pd.Series, preferred: str, fallback: str) -> object:
    preferred_value = row.get(preferred)
    return preferred_value if pd.notna(preferred_value) else row.get(fallback)


def _slate_fixtures(fixtures: pd.DataFrame, target: date) -> pd.DataFrame:
    date_col = next((column for column in ("MatchDate", "Date", "date") if column in fixtures.columns), None)
    if date_col is None:
        raise ValueError("Fixture file has no date column.")
    parsed = pd.to_datetime(fixtures[date_col], errors="coerce").dt.date
    # Friday delivery covers the full weekend; other runs remain a one-day slate.
    end = target + timedelta(days=2 if target.weekday() == 4 else 0)
    return fixtures.loc[(parsed >= target) & (parsed <= end)].copy()


def _report_state() -> tuple[dict, bool, bool]:
    if not REPORT_PATH.exists():
        return {}, False, False
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    gate_passed = bool(report.get("release_gate", {}).get("passed", False))
    aggregate = report.get("aggregate", {}).get("model", {})
    deployment = report.get("deployment", {})
    calibrated = bool(
        aggregate and aggregate.get("samples", 0) > 0 and "ece" in aggregate
        and deployment.get("selected_candidate") == "model"
        and deployment.get("eligible", False)
    )
    return report, gate_passed, calibrated


def _context_change_explanations(
    sources: dict[str, pd.DataFrame],
    home: str,
    away: str,
    since: pd.Timestamp | None,
) -> list[str]:
    if since is None:
        return ["Initial snapshot from the current point-in-time data manifest."]
    explanations: list[str] = []
    for source_name, frame in sources.items():
        if frame.empty or "available_at" not in frame.columns:
            continue
        available = pd.to_datetime(frame["available_at"], errors="coerce", utc=True)
        mask = available > since
        team_col = next((column for column in ("team", "Team", "team_name") if column in frame.columns), None)
        if team_col:
            mask &= frame[team_col].astype(str).isin({home, away})
        count = int(mask.sum())
        if count:
            explanations.append(f"{count} new {source_name.replace('_', ' ')} record(s) since the prior snapshot.")
    return explanations or ["No newly timestamped context records; probability movement is model/data-refresh only."]


def main() -> None:
    target = date.today()
    if not FIXTURES_PATH.exists() or not HISTORICAL_PATH.exists() or not MODEL_PATH.exists():
        missing = [str(path) for path in (FIXTURES_PATH, HISTORICAL_PATH, MODEL_PATH) if not path.exists()]
        pd.DataFrame().to_csv(TODAY_PATH, index=False)
        print(f"[MLS] Missing required artifacts: {missing}")
        return

    fixtures = _slate_fixtures(pd.read_csv(FIXTURES_PATH), target)
    if fixtures.empty:
        pd.DataFrame().to_csv(TODAY_PATH, index=False)
        print(f"[MLS] No fixtures for {target}")
        return
    history = pd.read_csv(HISTORICAL_PATH)
    artifact = load_frontier_artifact(MODEL_PATH)
    sources = load_optional_sources(ROOT / "data_files" / "raw")
    feature_frame = prepare_upcoming_features(
        fixtures, history, list(artifact["features"]), sources
    )
    probabilities = predict_feature_frame(feature_frame, artifact)
    score_model = DixonColesBaseline().fit(history)
    report, release_passed, calibrated = _report_state()
    code_version = str(artifact.get("code_version") or repository_code_version())
    generated_at = datetime.now(timezone.utc)
    db = DatabaseManager(str(ROOT / "data_files" / "mls.db"))
    rows: list[dict[str, object]] = []

    for position, (_, fixture) in enumerate(fixtures.iterrows()):
        home, away = str(fixture["HomeTeam"]), str(fixture["AwayTeam"])
        home_prob, draw_prob, away_prob = map(float, probabilities[position])
        odds = [
            _coalesce(fixture, "draftkings_home_odds", "best_home_odds"),
            _coalesce(fixture, "draftkings_draw_odds", "best_draw_odds"),
            _coalesce(fixture, "draftkings_away_odds", "best_away_odds"),
        ]
        books = [
            "draftkings" if pd.notna(fixture.get("draftkings_home_odds")) else fixture.get("best_home_book", "best-available"),
            "draftkings" if pd.notna(fixture.get("draftkings_draw_odds")) else fixture.get("best_draw_book", "best-available"),
            "draftkings" if pd.notna(fixture.get("draftkings_away_odds")) else fixture.get("best_away_book", "best-available"),
        ]
        if not all(pd.notna(value) for value in odds):
            market_probs = np.array([0.45, 0.26, 0.29])
        else:
            market_probs = np.array([_implied_probability(float(value)) for value in odds])
            market_probs /= market_probs.sum()
        model_probs = np.array([home_prob, draw_prob, away_prob])
        score_grid = score_model.red_card_score_grid(home, away)
        score_markets = score_distribution_markets(score_grid)
        consistent, consistency_reasons = cross_market_consistency(
            {"home": home_prob, "draw": draw_prob, "away": away_prob}, score_grid
        )
        edges = model_probs - market_probs
        best_index = int(np.argmax(edges))
        labels = ["Home Win", "Draw", "Away Win"]
        selections = ["home", "draw", "away"]
        selected_odds = float(odds[best_index]) if pd.notna(odds[best_index]) else 100.0
        entropy = float(-np.sum(np.clip(model_probs, 1e-12, 1.0) * np.log(np.clip(model_probs, 1e-12, 1.0))) / np.log(3))
        conformal_quantile = float(artifact.get("conformal_quantile", 1.0))
        prediction_set = tuple(int(index) for index in np.flatnonzero(1.0 - model_probs <= conformal_quantile))
        abstain = len(prediction_set) != 1
        decision = decide_bet(
            "1x2",
            model_probability=float(model_probs[best_index]),
            de_vigged_market_probability=float(market_probs[best_index]),
            odds=selected_odds,
            calibrated=calibrated,
            release_gate_passed=release_passed,
            conformal_abstain=abstain,
            consistent=consistent,
            paper_trade=not release_passed,
        )
        fixture_date_raw = fixture.get("Date", fixture.get("MatchDate", target))
        fixture_date = pd.to_datetime(fixture_date_raw, errors="coerce")
        fixture_date_text = str(target) if pd.isna(fixture_date) else fixture_date.date().isoformat()
        raw_fixture_id = fixture.get("ESPN_ID")
        fixture_id = str(raw_fixture_id) if pd.notna(raw_fixture_id) else f"{fixture_date_text}:{home}:{away}"
        previous = db.get_prediction_snapshots(fixture_id)
        previous_time = None
        if not previous.empty:
            parsed_previous = pd.to_datetime(previous.iloc[-1]["prediction_time"], errors="coerce", utc=True)
            previous_time = parsed_previous if pd.notna(parsed_previous) else None
        context_explanations = _context_change_explanations(sources, home, away, previous_time)
        snapshot = make_snapshot(
            fixture_id,
            {"home": home_prob, "draw": draw_prob, "away": away_prob},
            model_version=code_version,
            manifest_payload={
                "fixture": fixture.to_dict(),
                "inference_features": feature_frame.iloc[position].to_dict(),
                "training_data_hash": artifact.get("report_hash"),
                "feature_names": artifact.get("features", []),
            },
            uncertainty=entropy,
            abstain=abstain,
            explanations=(*decision.reasons, *consistency_reasons, *context_explanations),
            prediction_time=generated_at,
        )
        db.store_prediction_snapshot(snapshot)
        # Both paper and released selections are immutable evidence. Paper rows
        # carry zero stake but are essential for the 300-selection release gate.
        if decision.allowed:
            db.freeze_selection({
                "selection_id": new_selection_id(),
                "fixture_id": fixture_id,
                "match_date": fixture_date_text,
                "home_team": home,
                "away_team": away,
                "selection_time": generated_at.isoformat(),
                "prediction_time": snapshot.prediction_time.isoformat(),
                "book": str(books[best_index]),
                "market": "1x2",
                "selection": selections[best_index],
                "line": None,
                "odds": selected_odds,
                "limits": fixture.get("limit_amount"),
                "model_probability": float(model_probs[best_index]),
                "market_probability": float(market_probs[best_index]),
                "edge": float(edges[best_index]),
                "stake_units": decision.stake_units,
                "mode": decision.state,
                "code_version": code_version,
                "data_manifest": snapshot.data_manifest,
            })
        rows.append({
            "Date": fixture_date_text,
            "HomeTeam": home,
            "AwayTeam": away,
            "Bet": labels[best_index],
            "State": decision.state,
            "Model": round(float(model_probs[best_index]) * 100, 1),
            "Edge": round(float(edges[best_index]) * 100, 1),
            "Odds": selected_odds,
            "StakeUnits": decision.stake_units,
            "NoBetReasons": " | ".join(decision.reasons),
            "home_prob": round(home_prob, 4),
            "draw_prob": round(draw_prob, 4),
            "away_prob": round(away_prob, 4),
            "uncertainty": round(entropy, 4),
            "prediction_set": ",".join(labels[index] for index in prediction_set),
            "cross_market_consistent": consistent,
            "expected_home_goals": round(score_model.expected_goals(home, away)[0], 3),
            "expected_away_goals": round(score_model.expected_goals(home, away)[1], 3),
            "over_2_5_probability": round(score_markets["over"], 4),
            "btts_probability": round(score_markets["btts_yes"], 4),
            "data_manifest": snapshot.data_manifest,
        })
    output = pd.DataFrame(rows)
    output.to_csv(TODAY_PATH, index=False)
    print(f"[MLS] Wrote {len(output)} governed selections to {TODAY_PATH}")


if __name__ == "__main__":
    main()
