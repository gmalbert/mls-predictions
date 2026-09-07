from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from database.db_manager import DatabaseManager
from automation.settle_ledger import settle_open_selections
from models.governance import make_snapshot


def test_snapshot_odds_and_ledger_roundtrip(tmp_path) -> None:
    database = DatabaseManager(str(tmp_path / "test.db"))
    now = datetime.now(timezone.utc).isoformat()
    odds = pd.DataFrame([{
        "fixture_id": "fixture-1", "snapshot_at": now, "book": "book", "market": "h2h",
        "selection": "home", "line": None, "american_odds": 120, "decimal_odds": 2.2,
        "limit_amount": 100, "available_at": now,
    }])
    assert database.store_market_odds(odds) == 1
    assert len(database.get_market_odds("fixture-1")) == 1

    snapshot = make_snapshot("fixture-1", {"home": 0.5, "draw": 0.25, "away": 0.25}, "v1", {"data": 1})
    database.store_prediction_snapshot(snapshot)
    assert database.get_prediction_snapshots("fixture-1").iloc[0]["probabilities"]["home"] == 0.5

    database.freeze_selection({
        "selection_id": "selection-1", "fixture_id": "fixture-1", "selection_time": now,
        "prediction_time": now, "book": "book", "market": "1x2", "selection": "home",
        "line": None, "odds": 120, "limits": 100, "model_probability": 0.55,
        "market_probability": 0.45, "edge": 0.1, "stake_units": 0.0, "mode": "PAPER",
        "code_version": "v1", "data_manifest": snapshot.data_manifest,
    })
    database.settle_selection("selection-1", "H", "win", 1.2, closing_odds=110, clv=0.02)
    ledger = database.get_bet_ledger(settled_only=True)
    assert ledger.iloc[0]["settlement"] == "win"
    assert ledger.iloc[0]["clv"] == 0.02


def test_open_paper_selection_settles_from_results_and_close(tmp_path) -> None:
    database = DatabaseManager(str(tmp_path / "settlement.db"))
    database.freeze_selection({
        "selection_id": "paper-1", "fixture_id": "fixture-2", "match_date": "2026-03-01",
        "home_team": "LA Galaxy", "away_team": "LAFC",
        "selection_time": "2026-02-28T12:00:00Z", "prediction_time": "2026-02-28T12:00:00Z",
        "book": "draftkings", "market": "1x2", "selection": "home", "line": None,
        "odds": 120, "limits": 100, "model_probability": 0.55, "market_probability": 0.45,
        "edge": 0.1, "stake_units": 0.0, "mode": "PAPER", "code_version": "v1", "data_manifest": "hash",
    })
    database.store_market_odds(pd.DataFrame([{
        "fixture_id": "fixture-2", "snapshot_at": "2026-03-01T18:45:00Z", "is_closing": True,
        "book": "draftkings", "market": "h2h", "selection": "LA Galaxy", "line": None,
        "american_odds": 110, "decimal_odds": 2.1, "limit_amount": 100, "available_at": "2026-03-01T18:45:00Z",
    }]))
    history = pd.DataFrame([{
        "MatchDate": "2026-03-01", "HomeTeam": "LA Galaxy", "AwayTeam": "LAFC",
        "HomeGoals": 2, "AwayGoals": 1, "Result": "H",
    }])
    assert settle_open_selections(database, history) == 1
    row = database.get_bet_ledger(settled_only=True).iloc[0]
    assert row["settlement"] == "win"
    assert row["closing_odds"] == 110
    assert row["clv"] > 0
