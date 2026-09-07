"""Settle frozen MLS selections from completed results and archived closing prices."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager
from models.governance import american_to_decimal, settle_market
from team_name_mapping import normalize_team_name


def _closing_price(
    database: DatabaseManager,
    fixture_id: str,
    market: str,
    selection: str,
    book: str,
    home_team: str,
    away_team: str,
) -> float | None:
    prices = database.get_market_odds(fixture_id)
    if prices.empty:
        return None
    market_aliases = {"1x2": {"1x2", "h2h"}, "moneyline": {"1x2", "h2h"}}
    accepted = market_aliases.get(market, {market})
    prices = prices[prices["market"].astype(str).str.lower().isin(accepted)].copy()
    closing = pd.to_numeric(prices.get("is_closing", 0), errors="coerce").fillna(0).astype(bool)
    prices = prices[closing]
    if prices.empty:
        return None
    target = {
        "home": normalize_team_name(home_team),
        "away": normalize_team_name(away_team),
        "draw": "draw",
    }.get(selection, selection)
    names = prices["selection"].astype(str).map(normalize_team_name).str.lower()
    prices = prices[(names == str(target).lower()) | (names == selection.lower())]
    if prices.empty:
        return None
    same_book = prices[prices["book"].astype(str).str.lower() == book.lower()]
    chosen = same_book if not same_book.empty else prices
    chosen = chosen.sort_values("snapshot_at")
    value = pd.to_numeric(chosen.iloc[-1].get("american_odds"), errors="coerce")
    return float(value) if pd.notna(value) else None


def settle_open_selections(
    database: DatabaseManager,
    historical: pd.DataFrame,
) -> int:
    ledger = database.get_bet_ledger()
    if ledger.empty:
        return 0
    unsettled = ledger[ledger["settled_at"].isna()].copy()
    if unsettled.empty:
        return 0
    results = historical.copy()
    results["_date"] = pd.to_datetime(results["MatchDate"], errors="coerce").dt.date.astype(str)
    results["_home"] = results["HomeTeam"].astype(str).map(normalize_team_name)
    results["_away"] = results["AwayTeam"].astype(str).map(normalize_team_name)
    settled = 0
    for row in unsettled.itertuples(index=False):
        home = normalize_team_name(str(getattr(row, "home_team", "")))
        away = normalize_team_name(str(getattr(row, "away_team", "")))
        match_date = str(getattr(row, "match_date", ""))[:10]
        candidates = results[
            (results["_date"] == match_date)
            & (results["_home"] == home)
            & (results["_away"] == away)
        ]
        if candidates.empty:
            continue
        result_row = candidates.iloc[-1]
        home_goals = pd.to_numeric(result_row.get("HomeGoals"), errors="coerce")
        away_goals = pd.to_numeric(result_row.get("AwayGoals"), errors="coerce")
        if pd.isna(home_goals) or pd.isna(away_goals):
            continue
        market = str(row.market).lower()
        selection = str(row.selection).lower()
        settlement = settle_market(
            market,
            selection,
            float(row.odds),
            float(row.stake_units),
            int(home_goals),
            int(away_goals),
            line=float(row.line) if pd.notna(row.line) else None,
        )
        closing_odds = _closing_price(
            database,
            str(row.fixture_id),
            market,
            selection,
            str(row.book),
            home,
            away,
        )
        clv = None
        if closing_odds is not None:
            clv = american_to_decimal(float(row.odds)) / american_to_decimal(closing_odds) - 1.0
        actual = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")
        database.settle_selection(
            str(row.selection_id),
            actual,
            settlement.status,
            settlement.profit_units,
            closing_odds=closing_odds,
            clv=clv,
        )
        settled += 1
    return settled


def main() -> None:
    historical_path = ROOT / "data_files" / "combined_historical_data.csv"
    if not historical_path.exists():
        raise FileNotFoundError(f"Historical data not found: {historical_path}")
    database = DatabaseManager(str(ROOT / "data_files" / "mls.db"))
    count = settle_open_selections(database, pd.read_csv(historical_path))
    print(f"[MLS] Settled {count} frozen selection(s)")


if __name__ == "__main__":
    main()
