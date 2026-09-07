"""Refresh point-in-time roster, news, weather, event, and multi-book context.

Provider-specific URLs are configured through environment variables. Every record
gets a separate ``available_at`` timestamp so corrected statistics or confirmed
lineups cannot leak into earlier predictions.
"""
from __future__ import annotations

import io
import json
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.db_manager import DatabaseManager
from models.data_quality import audit_market_odds, audit_optional_sources
from models.frontier_features import STADIUMS, load_optional_sources
from team_name_mapping import normalize_team_name

RAW_DIR = ROOT / "data_files" / "raw"
FIXTURES_PATH = ROOT / "data_files" / "upcoming_fixtures.csv"
USER_AGENT = "mls-predictions/1.0 point-in-time research"

FEEDS = {
    "MLS_ROSTER_FEED_URL": "roster_availability.csv",
    "MLS_TRANSACTIONS_FEED_URL": "transactions.csv",
    "MLS_MANAGER_FEED_URL": "manager_tenures.csv",
    "MLS_EVENT_FEED_URL": "event_features.csv",
    "MLS_GOALKEEPER_FEED_URL": "goalkeeper_stats.csv",
    "MLS_ATTENDANCE_FEED_URL": "attendance.csv",
    "MLS_SUPERDRAFT_FEED_URL": "superdraft.csv",
    "MLS_TEAM_RATINGS_FEED_URL": "team_ratings.csv",
    "MLS_REFEREE_FEED_URL": "referee_crews.csv",
    "MLS_LINEUP_FEED_URL": "lineups.csv",
    "MLS_COMPETITION_FEED_URL": "competition_phases.csv",
    "MLS_TRAVEL_FEED_URL": "travel_estimates.csv",
}


def _fetch_frame(url: str, timeout: int = 30) -> pd.DataFrame:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "json" in content_type or url.lower().endswith(".json"):
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("results", payload.get("items", [payload])))
        return pd.json_normalize(payload)
    return pd.read_csv(io.StringIO(response.text))


def _normalize_feed(frame: pd.DataFrame, fetched_at: str) -> pd.DataFrame:
    output = frame.copy()
    team_col = next((column for column in ("team", "Team", "team_name", "club") if column in output.columns), None)
    if team_col:
        output["team"] = output[team_col].astype(str).map(normalize_team_name)
    if "available_at" not in output.columns:
        output["available_at"] = fetched_at
    else:
        parsed = pd.to_datetime(output["available_at"], errors="coerce", utc=True)
        output["available_at"] = parsed.fillna(pd.Timestamp(fetched_at)).astype(str)
    output["fetched_at"] = fetched_at
    return output


def refresh_configured_feeds() -> dict[str, int]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    results: dict[str, int] = {}
    for environment_key, filename in FEEDS.items():
        url = os.getenv(environment_key, "").strip()
        if not url:
            results[filename] = 0
            continue
        frame = _normalize_feed(_fetch_frame(url), fetched_at)
        destination = RAW_DIR / filename
        if destination.exists():
            previous = pd.read_csv(destination)
            frame = pd.concat([previous, frame], ignore_index=True).drop_duplicates()
        frame.to_csv(destination, index=False)
        results[filename] = len(frame)
    return results


def _fixture_timestamp(row: pd.Series) -> pd.Timestamp:
    raw_date = row.get("Date", row.get("MatchDate"))
    raw_time = row.get("Time", "12:00")
    return pd.to_datetime(f"{raw_date} {raw_time}", errors="coerce", utc=True)


def refresh_weather() -> int:
    """Fetch no-key hourly forecasts from Open-Meteo for upcoming outdoor games."""
    if not FIXTURES_PATH.exists():
        return 0
    fixtures = pd.read_csv(FIXTURES_PATH)
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for _, fixture in fixtures.iterrows():
        home = normalize_team_name(str(fixture.get("HomeTeam", "")))
        stadium = STADIUMS.get(home)
        kickoff = _fixture_timestamp(fixture)
        if stadium is None or stadium.dome or pd.isna(kickoff):
            continue
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": stadium.latitude,
                "longitude": stadium.longitude,
                "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
                "forecast_days": 16,
            },
            timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 200:
            continue
        hourly = response.json().get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), errors="coerce", utc=True)
        if len(times) == 0:
            continue
        nearest = int(np.argmin(np.abs((times - kickoff).total_seconds())))
        rows.append({
            "team": home,
            "event_date": kickoff.isoformat(),
            "temperature_f": hourly.get("temperature_2m", [None] * len(times))[nearest],
            "precipitation_probability": hourly.get("precipitation_probability", [None] * len(times))[nearest],
            "wind_mph": hourly.get("wind_speed_10m", [None] * len(times))[nearest],
            "available_at": fetched_at,
        })
    if rows:
        destination = RAW_DIR / "weather.csv"
        frame = pd.DataFrame(rows)
        if destination.exists():
            frame = pd.concat([pd.read_csv(destination), frame], ignore_index=True).drop_duplicates()
        frame.to_csv(destination, index=False)
    return len(rows)


def refresh_multi_book_odds() -> int:
    """Archive 1X2, Asian handicap/spread, and totals prices from The Odds API."""
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    provider_url = os.getenv("MLS_MARKET_ODDS_FEED_URL", "").strip()
    if not api_key and not provider_url:
        return 0
    snapshot_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    if api_key:
        response = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_usa_mls/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=30,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        for fixture in response.json():
            commence_at = pd.to_datetime(fixture.get("commence_time"), errors="coerce", utc=True)
            snapshot_timestamp = pd.Timestamp(snapshot_at)
            minutes_to_kickoff = (
                (commence_at - snapshot_timestamp).total_seconds() / 60.0
                if pd.notna(commence_at)
                else float("inf")
            )
            is_closing = 0.0 <= minutes_to_kickoff <= 40.0
            for bookmaker in fixture.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        american = float(outcome.get("price"))
                        decimal = 1 + (american / 100 if american >= 100 else 100 / abs(american))
                        rows.append({
                            "fixture_id": fixture.get("id"),
                            "commence_time": fixture.get("commence_time"),
                            "home_team": normalize_team_name(str(fixture.get("home_team", ""))),
                            "away_team": normalize_team_name(str(fixture.get("away_team", ""))),
                            "snapshot_at": snapshot_at,
                            "is_closing": is_closing,
                            "book": bookmaker.get("key", bookmaker.get("title", "unknown")),
                            "market": market.get("key"),
                            "selection": outcome.get("name"),
                            "line": outcome.get("point"),
                            "american_odds": american,
                            "decimal_odds": decimal,
                            "limit_amount": outcome.get("limit"),
                            "available_at": market.get("last_update", bookmaker.get("last_update", snapshot_at)),
                        })
    if provider_url:
        provider = _normalize_feed(_fetch_frame(provider_url), snapshot_at)
        required = {"fixture_id", "commence_time", "home_team", "away_team", "book", "market", "selection"}
        missing = required.difference(provider.columns)
        if missing:
            raise ValueError(f"Configured market feed is missing columns: {sorted(missing)}")
        if not {"american_odds", "decimal_odds"}.intersection(provider.columns):
            raise ValueError("Configured market feed requires american_odds or decimal_odds.")
        if "snapshot_at" not in provider.columns:
            provider["snapshot_at"] = snapshot_at
        if "limit_amount" not in provider.columns and "limits" in provider.columns:
            provider["limit_amount"] = provider["limits"]
        if "decimal_odds" not in provider.columns and "american_odds" in provider.columns:
            american = pd.to_numeric(provider["american_odds"], errors="coerce")
            provider["decimal_odds"] = np.where(
                american >= 100, 1.0 + american / 100.0, 1.0 + 100.0 / american.abs()
            )
        if "american_odds" not in provider.columns and "decimal_odds" in provider.columns:
            decimal = pd.to_numeric(provider["decimal_odds"], errors="coerce")
            provider["american_odds"] = np.where(
                decimal >= 2.0, (decimal - 1.0) * 100.0, -100.0 / (decimal - 1.0)
            )
        if "is_closing" not in provider.columns:
            commence = pd.to_datetime(provider.get("commence_time"), errors="coerce", utc=True)
            snapshot = pd.to_datetime(provider["snapshot_at"], errors="coerce", utc=True)
            provider["is_closing"] = (commence - snapshot).dt.total_seconds().div(60).between(0, 40)
        for team_column in ("home_team", "away_team"):
            if team_column in provider.columns:
                provider[team_column] = provider[team_column].astype(str).map(normalize_team_name)
        rows.extend(provider.to_dict("records"))
    if not rows:
        return 0
    frame = pd.DataFrame(rows)
    archive = RAW_DIR / "multi_book_odds.csv"
    if archive.exists():
        frame = pd.concat([pd.read_csv(archive), frame], ignore_index=True).drop_duplicates()
    frame.to_csv(archive, index=False)
    DatabaseManager(str(ROOT / "data_files" / "mls.db")).store_market_odds(pd.DataFrame(rows))
    return len(rows)


def refresh_news_sentiment() -> int:
    """Parse a configured MLS news feed into timestamped injury/lineup signals."""
    feed_url = (
        os.getenv("MLS_SOCIAL_FEED_URL", "").strip()
        or os.getenv("MLS_NEWS_FEED_URL", "").strip()
    )
    if not feed_url:
        return 0
    import feedparser

    feed = feedparser.parse(feed_url)
    positive = {"available", "returns", "cleared", "fit", "signed"}
    negative = {"injury", "injured", "suspended", "out", "doubt", "absence"}
    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for entry in feed.entries:
        text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
        score = sum(word in text for word in positive) - sum(word in text for word in negative)
        mentioned = [team for team in STADIUMS if team.lower() in text]
        for team in mentioned:
            rows.append({
                "team": team,
                "headline": entry.get("title", ""),
                "url": entry.get("link", ""),
                "sentiment_signal": score,
                "event_date": entry.get("published", fetched_at),
                "available_at": fetched_at,
            })
    if rows:
        destination = RAW_DIR / "news_sentiment.csv"
        frame = pd.DataFrame(rows)
        if destination.exists():
            frame = pd.concat([pd.read_csv(destination), frame], ignore_index=True).drop_duplicates(subset=["team", "headline", "url"])
        frame.to_csv(destination, index=False)
    return len(rows)


def main(odds_only: bool = False) -> None:
    load_dotenv(ROOT / ".env")
    if odds_only:
        print(json.dumps({"multi_book_odds": refresh_multi_book_odds()}, indent=2))
        return
    results: dict[str, Any] = {"configured_feeds": refresh_configured_feeds()}
    for name, function in {
        "weather": refresh_weather,
        "multi_book_odds": refresh_multi_book_odds,
        "news_sentiment": refresh_news_sentiment,
    }.items():
        try:
            results[name] = function()
        except Exception as exc:
            results[name] = {"error": str(exc)}
    sources = load_optional_sources(RAW_DIR)
    odds_path = RAW_DIR / "multi_book_odds.csv"
    odds = pd.read_csv(odds_path) if odds_path.exists() else pd.DataFrame()
    readiness = {"context": audit_optional_sources(sources), "market_odds": audit_market_odds(odds)}
    quality_path = ROOT / "data_files" / "quality" / "latest_feed_quality.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(readiness, indent=2, default=str), encoding="utf-8")
    results["quality_report"] = str(quality_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odds-only", action="store_true", help="Archive only current multi-book prices.")
    main(odds_only=parser.parse_args().odds_only)
