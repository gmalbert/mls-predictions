from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.roadmap import (
    compute_gpaa,
    conference_table,
    expected_points_table,
    playoff_simulation,
    series_win_probability,
)
from models.backtesting import DixonColesBaseline, evaluate_release_gate, rolling_backtest, score_metrics, select_deployment_candidate


def synthetic_matches() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    teams = ["LA Galaxy", "LAFC", "Seattle Sounders", "Austin FC"]
    rows = []
    for season in (2023, 2024, 2025):
        for match in range(80):
            home = teams[match % len(teams)]
            away = teams[(match + 1 + match // len(teams)) % len(teams)]
            if home == away:
                away = teams[(teams.index(home) + 1) % len(teams)]
            home_goals = int(rng.poisson(1.6))
            away_goals = int(rng.poisson(1.2))
            result = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")
            rows.append({
                "MatchDate": f"{season}-{1 + match // 28:02d}-{1 + match % 28:02d}",
                "HomeTeam": home,
                "AwayTeam": away,
                "HomeGoals": home_goals,
                "AwayGoals": away_goals,
                "Result": result,
                "home_xg_l5": 1.5 + rng.normal(0, 0.2),
                "away_xg_l5": 1.2 + rng.normal(0, 0.2),
                "home_pts_l5": rng.uniform(0.5, 2.5),
                "away_pts_l5": rng.uniform(0.5, 2.5),
                "home_is_turf": int(home == "Seattle Sounders"),
                "is_long_haul": int({home, away} == {"Seattle Sounders", "Austin FC"}),
                "is_cross_conference": 0,
            })
    return pd.DataFrame(rows)


def test_series_and_goalkeeper_analytics() -> None:
    assert series_win_probability(0.5) == 0.5
    keepers = compute_gpaa(pd.DataFrame([{"shots_on_target_faced": 10, "goals_conceded": 2, "minutes_played": 90}]))
    assert keepers.loc[0, "gpaa_90"] == 1.2


def test_tables_and_playoff_simulator() -> None:
    matches = synthetic_matches()
    table = conference_table(matches[matches["MatchDate"].str.startswith("2025")], 2025)
    assert not table.empty
    simulation = playoff_simulation(table, simulations=100, random_seed=1)
    assert not simulation.empty
    assert simulation["mls_cup"].sum() <= 1.01
    assert not expected_points_table(matches.head(20)).empty


def test_multiclass_metrics_are_finite() -> None:
    metrics = score_metrics(np.array([0, 1, 2]), np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]))
    assert metrics.log_loss > 0
    assert 0 <= metrics.ece <= 1


def test_rolling_backtest_preserves_untouched_fold() -> None:
    predictions, report, model = rolling_backtest(synthetic_matches(), first_test_season=2025)
    assert len(predictions) > 40
    assert report["folds"][0]["trained_through"] == 2024
    assert report["folds"][0]["calibration_matches"] > 0
    assert report["aggregate"]["model"]["samples"] == len(predictions)
    assert hasattr(model, "predict_proba")


def test_rolling_backtest_locks_calibration_for_later_seasons() -> None:
    matches = synthetic_matches()
    extra = matches[matches["MatchDate"].str.startswith("2025")].copy()
    extra["MatchDate"] = extra["MatchDate"].str.replace("2025-", "2026-", regex=False)
    _, report, _ = rolling_backtest(pd.concat([matches, extra], ignore_index=True), first_test_season=2025)
    assert [fold["calibration_matches"] for fold in report["folds"]] == [16, 0]
    assert report["folds"][1]["full_untouched_season"] is False  # only 80 matches, not a full MLS season


def test_release_gate_requires_explicit_full_untouched_season() -> None:
    incomplete = {"folds": [{"untouched_matches": 500, "full_untouched_season": False}], "aggregate": {}}
    complete = {"folds": [{"untouched_matches": 500, "full_untouched_season": True}], "aggregate": {}}
    assert evaluate_release_gate(incomplete).untouched_season is False
    assert evaluate_release_gate(complete).untouched_season is True


def test_release_gate_can_pass_with_complete_evidence() -> None:
    report = {
        "folds": [{"full_untouched_season": True}],
        "market_comparison": {
            "model": {"brier": 0.60, "log_loss": 1.0},
            "market": {"brier": 0.65, "log_loss": 1.1},
        },
    }
    ledger = pd.DataFrame([{
        "selection_id": f"s-{index}", "selection_time": "2026-01-01T00:00:00Z",
        "book": "book", "odds": 120, "limits": 100, "result": "H",
        "settlement": "win", "code_version": "sha", "clv": 0.01,
    } for index in range(300)])
    gate = evaluate_release_gate(report, ledger)
    assert gate.passed is True


def test_red_card_score_distribution_is_normalized_and_sensitive() -> None:
    matches = synthetic_matches()
    model = DixonColesBaseline().fit(matches)
    base = model.score_grid("LA Galaxy", "LAFC")
    adjusted = model.red_card_score_grid("LA Galaxy", "LAFC", home_red_probability=0.3, away_red_probability=0.0)
    assert adjusted.sum() == pytest.approx(1.0)
    assert not np.allclose(base, adjusted)


def test_release_gate_fails_without_market_and_ledger() -> None:
    _, report, _ = rolling_backtest(synthetic_matches(), first_test_season=2025)
    gate = evaluate_release_gate(report)
    assert gate.passed is False
    assert gate.frozen_selections == 0


def test_deployment_selection_requires_model_to_beat_elo_on_both_metrics() -> None:
    assert select_deployment_candidate({"model": {"log_loss": 1.0, "brier": 0.6}, "elo": {"log_loss": 1.1, "brier": 0.61}})["eligible"]
    rejected = select_deployment_candidate({"model": {"log_loss": 1.0, "brier": 0.62}, "elo": {"log_loss": 1.1, "brier": 0.61}})
    assert rejected["selected_candidate"] == "elo"
    assert rejected["eligible"] is False
