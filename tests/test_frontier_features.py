from __future__ import annotations

import pandas as pd
import pytest

from models.frontier_features import _phase, add_frontier_features, haversine_km, travel_load, validate_point_in_time


def test_travel_and_altitude_features_are_populated() -> None:
    matches = pd.DataFrame([
        {"MatchDate": "2025-03-01", "HomeTeam": "Colorado Rapids", "AwayTeam": "Seattle Sounders FC", "HomeGoals": 1, "AwayGoals": 0, "Result": "H"},
    ])
    output = add_frontier_features(matches)
    assert haversine_km("Seattle Sounders", "Colorado Rapids") > 1_500
    assert output.loc[0, "home_altitude_ft"] >= 5_000
    assert output.loc[0, "altitude_gain_ft"] > 5_000
    assert output.loc[0, "travel_load"] > 0


def test_frontier_rolling_features_use_only_prior_matches() -> None:
    matches = pd.DataFrame([
        {"MatchDate": "2025-03-01", "HomeTeam": "LA Galaxy", "AwayTeam": "LAFC", "HomeGoals": 1, "AwayGoals": 0, "home_xgoals": 2.0, "away_xgoals": 0.5, "Result": "H"},
        {"MatchDate": "2025-03-08", "HomeTeam": "LA Galaxy", "AwayTeam": "Austin FC", "HomeGoals": 0, "AwayGoals": 1, "home_xgoals": 0.2, "away_xgoals": 1.8, "Result": "A"},
    ])
    output = add_frontier_features(matches)
    assert output.loc[0, "home_matches_prior"] == 0
    assert output.loc[1, "home_matches_prior"] == 1
    # With one prior game, recent and season differential are identical.
    assert output.loc[1, "home_xg_differential_momentum"] == pytest.approx(0.0)


def test_competition_phase_is_never_invented_when_stage_is_absent() -> None:
    assert _phase(pd.Series({"Competition": "Leagues Cup"})) == "Leagues Cup"
    assert _phase(pd.Series({"stage_name": "MLS Cup Playoffs"})) == "Playoffs"
    assert _phase(pd.Series({"stage_name": "Regular Season"})) == "Regular Season"
    assert _phase(pd.Series({})) == "Unknown"


def test_hierarchical_club_rating_updates_only_after_result() -> None:
    matches = pd.DataFrame([
        {"MatchDate": "2025-03-01", "HomeTeam": "LA Galaxy", "AwayTeam": "LAFC", "HomeGoals": 2, "AwayGoals": 0, "Result": "H"},
        {"MatchDate": "2025-03-08", "HomeTeam": "LA Galaxy", "AwayTeam": "LAFC", "HomeGoals": 0, "AwayGoals": 0, "Result": "D"},
    ])
    output = add_frontier_features(matches)
    assert output.loc[0, "hierarchical_rating_edge"] == pytest.approx(0.0)
    assert output.loc[1, "hierarchical_rating_edge"] > 0


def test_availability_after_kickoff_is_not_used() -> None:
    matches = pd.DataFrame([
        {"MatchDate": "2025-03-01 19:00Z", "HomeTeam": "Inter Miami CF", "AwayTeam": "Orlando City", "HomeGoals": 1, "AwayGoals": 1, "Result": "D"},
    ])
    availability = pd.DataFrame([
        {"team": "Inter Miami CF", "available_at": "2025-03-01 20:00Z", "event_date": "2025-03-01", "roster_mechanism": "DP", "status": "out", "replacement_value": 1.0},
    ])
    output = add_frontier_features(matches, {"availability": availability})
    assert output.loc[0, "home_dp_available"] == pytest.approx(0.5)
    assert output.loc[0, "roster_data_missing"] == 1


def test_available_roster_mechanisms_are_minutes_weighted() -> None:
    matches = pd.DataFrame([
        {"MatchDate": "2025-03-01 19:00Z", "HomeTeam": "Inter Miami CF", "AwayTeam": "Orlando City", "HomeGoals": 1, "AwayGoals": 1, "Result": "D"},
    ])
    availability = pd.DataFrame([
        {"team": "Inter Miami CF", "available_at": "2025-02-28 20:00Z", "event_date": "2025-02-28", "roster_mechanism": "DP", "status": "out", "projected_minutes": 90, "replacement_value": 1.0},
        {"team": "Inter Miami CF", "available_at": "2025-02-28 20:00Z", "event_date": "2025-02-28", "roster_mechanism": "DP", "status": "available", "projected_minutes": 45, "replacement_value": 1.0},
    ])
    output = add_frontier_features(matches, {"availability": availability})
    assert output.loc[0, "home_dp_available"] == pytest.approx(1 / 3)
    assert output.loc[0, "home_roster_missing_impact"] == pytest.approx(1.0)


def test_international_duty_and_goals_added_are_point_in_time() -> None:
    matches = pd.DataFrame([
        {"MatchDate": "2025-03-01 19:00Z", "HomeTeam": "Inter Miami CF", "AwayTeam": "Orlando City", "HomeGoals": 1, "AwayGoals": 1, "Result": "D"},
    ])
    availability = pd.DataFrame([
        {"team": "Inter Miami CF", "available_at": "2025-02-28 20:00Z", "event_date": "2025-02-28", "roster_mechanism": "DP", "status": "international duty", "replacement_value": 1.0},
    ])
    goals_added = pd.DataFrame([
        {"team": "Inter Miami CF", "available_at": "2025-02-25 12:00Z", "event_date": "2025-02-24", "team_goals_added": 0.7},
        {"team": "Inter Miami CF", "available_at": "2025-03-02 12:00Z", "event_date": "2025-02-28", "team_goals_added": 99.0},
    ])
    output = add_frontier_features(matches, {"availability": availability, "goals_added": goals_added})
    assert output.loc[0, "home_international_absences"] == 1
    assert output.loc[0, "home_dp_available"] == 0
    assert output.loc[0, "home_team_goals_added_l10"] == pytest.approx(0.7)


def test_superdraft_integration_ramps_with_time() -> None:
    matches = pd.DataFrame([
        {"MatchDate": "2025-03-01 19:00Z", "HomeTeam": "LA Galaxy", "AwayTeam": "LAFC", "HomeGoals": 1, "AwayGoals": 1, "Result": "D"},
        {"MatchDate": "2025-06-01 19:00Z", "HomeTeam": "LA Galaxy", "AwayTeam": "LAFC", "HomeGoals": 1, "AwayGoals": 1, "Result": "D"},
    ])
    draft = pd.DataFrame([
        {"team": "LA Galaxy", "available_at": "2025-01-02 12:00Z", "draft_date": "2025-01-01", "draft_impact": 1.0, "rookie_minutes_share": 1.0},
    ])
    output = add_frontier_features(matches, {"draft": draft})
    assert 0 < output.loc[0, "home_superdraft_integration"] < output.loc[1, "home_superdraft_integration"] < 1


def test_validate_point_in_time_drops_late_corrections() -> None:
    frame = pd.DataFrame([
        {"MatchDate": "2025-03-01T19:00:00Z", "available_at": "2025-03-01T18:00:00Z", "value": 1},
        {"MatchDate": "2025-03-01T19:00:00Z", "available_at": "2025-03-01T20:00:00Z", "value": 2},
    ])
    safe = validate_point_in_time(frame)
    assert safe["value"].tolist() == [1]


def test_travel_load_penalizes_time_zone_and_altitude() -> None:
    baseline = travel_load(1_000, 0, 120, 0)
    burdened = travel_load(1_000, 3, 72, 4_000)
    assert burdened > baseline
