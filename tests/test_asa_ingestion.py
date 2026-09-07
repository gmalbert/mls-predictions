import pandas as pd

from fetch_asa_data import _normalise_game_xgoals, archive_team_goals_added_snapshot
from prepare_model_data import add_rolling_features


def test_game_xgoals_normalises_official_v21_schema() -> None:
    payload = pd.DataFrame(
        [
            {
                "game_id": "match-1",
                "home_team_xgoals": 1.75,
                "away_team_xgoals": 0.62,
                "home_player_xgoals": 1.80,
            }
        ]
    )

    result = _normalise_game_xgoals(payload)

    assert result.to_dict("records") == [
        {"game_id": "match-1", "home_xgoals": 1.75, "away_xgoals": 0.62}
    ]


def test_missing_match_xg_has_explicit_indicator_and_safe_fallback() -> None:
    matches = pd.DataFrame(
        [
            {
                "MatchDate": "2025-03-01",
                "HomeTeam": "LA Galaxy",
                "AwayTeam": "LAFC",
                "HomeGoals": 2,
                "AwayGoals": 1,
                "home_xgoals": float("nan"),
                "away_xgoals": float("nan"),
                "Result": "H",
            },
            {
                "MatchDate": "2025-03-08",
                "HomeTeam": "LA Galaxy",
                "AwayTeam": "Austin FC",
                "HomeGoals": 0,
                "AwayGoals": 0,
                "home_xgoals": 0.9,
                "away_xgoals": 0.7,
                "Result": "D",
            },
        ]
    )
    matches["MatchDate"] = pd.to_datetime(matches["MatchDate"])

    result = add_rolling_features(matches, lookbacks=[5])

    assert result.loc[0, "xg_data_missing"] == 1.0
    assert result.loc[1, "xg_data_missing"] == 0.0
    assert result.loc[1, "home_xg_l5"] == 2.0


def test_goals_added_does_not_duplicate_a_transferred_player(monkeypatch, tmp_path) -> None:
    payload = pd.DataFrame(
        [
            {
                "player_id": "player-1",
                "team_id": ["club-a", "club-b"],
                "data": [{"goals_added_above_avg": 2.0}],
            }
        ]
    )
    monkeypatch.setattr("fetch_asa_data._get_team_id_map", lambda: {"club-a": "Club A", "club-b": "Club B"})
    monkeypatch.setattr("fetch_asa_data.RAW_DIR", str(tmp_path))

    snapshot = archive_team_goals_added_snapshot(payload)

    values = snapshot.set_index("team")["team_goals_added_cumulative"].to_dict()
    assert values == {"Club A": 1.0, "Club B": 1.0}
