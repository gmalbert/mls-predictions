from __future__ import annotations

import pandas as pd
import pytest

import prepare_model_data


def test_archived_market_overlay_uses_only_pre_kickoff_prices(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prepare_model_data, "RAW_DIR", str(tmp_path))
    prices = []
    for side, odds in (("LA Galaxy", 120), ("Draw", 240), ("LAFC", 210)):
        prices.append({
            "fixture_id": "f1", "commence_time": "2026-03-02T00:00:00Z",
            "home_team": "LA Galaxy", "away_team": "LAFC",
            "snapshot_at": "2026-03-01T12:00:00Z", "available_at": "2026-03-01T12:00:00Z",
            "is_closing": False, "book": "draftkings", "market": "h2h",
            "selection": side, "american_odds": odds,
        })
        prices.append({
            "fixture_id": "f1", "commence_time": "2026-03-02T00:00:00Z",
            "home_team": "LA Galaxy", "away_team": "LAFC",
            "snapshot_at": "2026-03-01T23:50:00Z", "available_at": "2026-03-01T23:50:00Z",
            "is_closing": True, "book": "draftkings", "market": "h2h",
            "selection": side, "american_odds": odds - 10,
        })
        prices.append({
            "fixture_id": "f1", "commence_time": "2026-03-02T00:00:00Z",
            "home_team": "LA Galaxy", "away_team": "LAFC",
            "snapshot_at": "2026-03-02T00:05:00Z", "available_at": "2026-03-02T00:05:00Z",
            "is_closing": True, "book": "draftkings", "market": "h2h",
            "selection": side, "american_odds": 999,
        })
    pd.DataFrame(prices).to_csv(tmp_path / "multi_book_odds.csv", index=False)
    matches = pd.DataFrame([{
        "MatchDate": "2026-03-02", "HomeTeam": "LA Galaxy", "AwayTeam": "LAFC",
        "odds_data_available": 0,
    }])
    output = prepare_model_data.add_archived_market_features(matches)
    assert output.loc[0, "odds_data_available"] == 1
    assert output.loc[0, "odds_home_value"] == 2.2
    assert output.loc[0, "closing_home_value"] == 2.1
    assert output.loc[0, ["selection_implied_home_prob", "selection_implied_draw_prob", "selection_implied_away_prob"]].sum() == pytest.approx(1.0)
    assert output.loc[0, ["odds_implied_home_prob", "odds_implied_draw_prob", "odds_implied_away_prob"]].sum() == pytest.approx(1.0)
