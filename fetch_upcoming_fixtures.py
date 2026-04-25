"""
fetch_upcoming_fixtures.py
Pull upcoming MLS fixtures from ESPN's public API and save to
data_files/upcoming_fixtures.csv.

Also fetches DraftKings moneylines from The Odds API and appends
best_home_odds, best_draw_odds, best_away_odds columns when
ODDS_API_KEY is set in .env.

Usage:
    python fetch_upcoming_fixtures.py
    python fetch_upcoming_fixtures.py --days 60   # look further ahead
    python fetch_upcoming_fixtures.py --no-odds   # skip odds fetch
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from os import path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually if not installed

from team_name_mapping import normalize_team_name

# ── Constants ──────────────────────────────────────────────────────────────────
ESPN_MLS_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/scoreboard"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_usa_mls/odds"
DATA_DIR = "data_files"
OUTPUT_PATH = path.join(DATA_DIR, "upcoming_fixtures.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

TURF_STADIUMS = {
    "New England Revolution",
    "Portland Timbers",
    "Seattle Sounders",
    "Vancouver Whitecaps",
    "FC Cincinnati",
}

EASTERN_CONF = {
    "Atlanta United", "CF Montréal", "Charlotte FC", "Chicago Fire",
    "Columbus Crew", "D.C. United", "FC Cincinnati", "Inter Miami CF",
    "Nashville SC", "New England Revolution", "New York City FC",
    "New York Red Bulls", "Orlando City", "Philadelphia Union", "Toronto FC",
}

WESTERN_CONF = {
    "Austin FC", "Colorado Rapids", "FC Dallas", "Houston Dynamo",
    "LA Galaxy", "LAFC", "Minnesota United", "Portland Timbers",
    "Real Salt Lake", "San Jose Earthquakes", "Seattle Sounders",
    "Sporting Kansas City", "St. Louis City SC", "Vancouver Whitecaps",
}

STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "Atlanta United": (33.7557, -84.4010),
    "Austin FC": (30.3874, -97.7185),
    "CF Montréal": (45.5623, -73.5517),
    "Charlotte FC": (35.2258, -80.8528),
    "Chicago Fire": (41.8623, -87.6167),
    "Colorado Rapids": (39.8059, -104.8917),
    "Columbus Crew": (39.9685, -83.0176),
    "D.C. United": (38.8682, -77.0122),
    "FC Cincinnati": (39.1110, -84.5260),
    "FC Dallas": (33.1548, -97.0641),
    "Houston Dynamo": (29.7524, -95.3513),
    "Inter Miami CF": (25.9580, -80.2390),
    "LA Galaxy": (33.8644, -118.2611),
    "LAFC": (34.0131, -118.2845),
    "Minnesota United": (44.9536, -93.1669),
    "Nashville SC": (36.1306, -86.7715),
    "New England Revolution": (42.0910, -71.2643),
    "New York City FC": (40.8274, -73.9262),
    "New York Red Bulls": (40.7369, -74.1503),
    "Orlando City": (28.5411, -81.3894),
    "Philadelphia Union": (39.8327, -75.3799),
    "Portland Timbers": (45.5215, -122.6917),
    "Real Salt Lake": (40.5829, -111.8929),
    "San Jose Earthquakes": (37.3512, -121.9253),
    "Seattle Sounders": (47.5952, -122.3316),
    "Sporting Kansas City": (39.1212, -94.8235),
    "St. Louis City SC": (38.6328, -90.1924),
    "Toronto FC": (43.6332, -79.4189),
    "Vancouver Whitecaps": (49.2772, -123.1124),
}


def _haversine_miles(home: str, away: str) -> float:
    if home not in STADIUM_COORDS or away not in STADIUM_COORDS:
        return 0.0
    lat1, lon1 = map(radians, STADIUM_COORDS[home])
    lat2, lon2 = map(radians, STADIUM_COORDS[away])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return round(3958.8 * 2 * atan2(sqrt(a), sqrt(1 - a)), 1)


def _utc_to_et(utc_str: str) -> str:
    """Convert ESPN UTC date string (e.g. '2026-05-01T00:00Z') to ET time string."""
    try:
        dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%MZ")
        # Approximate: EDT = UTC-4 (Mar–Nov), EST = UTC-5 (Nov–Mar)
        offset = 4 if 3 <= dt.month <= 11 else 5
        et = dt - timedelta(hours=offset)
        return et.strftime("%I:%M %p")
    except Exception:
        return ""


def fetch_day(date_str: str) -> list[dict]:
    """Fetch all scheduled MLS matches for a single YYYYMMDD date string."""
    url = f"{ESPN_MLS_URL}?dates={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except requests.RequestException as exc:
        print(f"  [WARN] ESPN request failed for {date_str}: {exc}")
        return []

    fixtures = []
    for event in data.get("events", []):
        status_name = event.get("status", {}).get("type", {}).get("name", "")
        # Skip live and completed matches
        if status_name not in ("STATUS_SCHEDULED",):
            continue

        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])

        home_raw = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away_raw = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home_raw or not away_raw:
            continue

        home = normalize_team_name(home_raw["team"]["displayName"])
        away = normalize_team_name(away_raw["team"]["displayName"])

        utc_date = event.get("date", "")
        et_time = _utc_to_et(utc_date)
        match_date = utc_date[:10] if len(utc_date) >= 10 else date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]

        home_conf = "Eastern" if home in EASTERN_CONF else ("Western" if home in WESTERN_CONF else "Unknown")
        away_conf = "Eastern" if away in EASTERN_CONF else ("Western" if away in WESTERN_CONF else "Unknown")

        dist = _haversine_miles(home, away)
        venue = comp.get("venue", {}).get("fullName", "")

        fixtures.append({
            "Date": match_date,
            "Time": et_time,
            "HomeTeam": home,
            "AwayTeam": away,
            "HomeConference": home_conf,
            "AwayConference": away_conf,
            "CrossConference": "Yes" if home_conf != away_conf and home_conf != "Unknown" else "No",
            "HomeSurface": "Turf" if home in TURF_STADIUMS else "Grass",
            "AwayTravelMiles": dist,
            "IsLongHaul": "Yes" if dist > 1500 else "No",
            "Venue": venue,
            "ESPN_ID": event.get("id", ""),
        })

    return fixtures


# ── Odds API helpers ───────────────────────────────────────────────────────────

# The Odds API uses its own team name conventions; map them to our canonical names.
_ODDS_TEAM_MAP: dict[str, str] = {
    "Atlanta United FC": "Atlanta United",
    "Austin FC": "Austin FC",
    "CF Montreal": "CF Montréal",
    "CF Montréal": "CF Montréal",
    "Charlotte FC": "Charlotte FC",
    "Chicago Fire FC": "Chicago Fire",
    "Colorado Rapids": "Colorado Rapids",
    "Columbus Crew": "Columbus Crew",
    "DC United": "D.C. United",
    "D.C. United": "D.C. United",
    "FC Cincinnati": "FC Cincinnati",
    "FC Dallas": "FC Dallas",
    "Houston Dynamo FC": "Houston Dynamo",
    "Inter Miami CF": "Inter Miami CF",
    "LA Galaxy": "LA Galaxy",
    "Los Angeles FC": "LAFC",
    "LAFC": "LAFC",
    "Minnesota United FC": "Minnesota United",
    "Nashville SC": "Nashville SC",
    "New England Revolution": "New England Revolution",
    "New York City FC": "New York City FC",
    "New York Red Bulls": "New York Red Bulls",
    "Orlando City SC": "Orlando City",
    "Philadelphia Union": "Philadelphia Union",
    "Portland Timbers": "Portland Timbers",
    "Real Salt Lake": "Real Salt Lake",
    "San Jose Earthquakes": "San Jose Earthquakes",
    "Seattle Sounders FC": "Seattle Sounders",
    "Sporting Kansas City": "Sporting Kansas City",
    "St. Louis City SC": "St. Louis City SC",
    "Toronto FC": "Toronto FC",
    "Vancouver Whitecaps FC": "Vancouver Whitecaps",
}


def _normalise_odds_team(raw: str) -> str:
    return _ODDS_TEAM_MAP.get(raw, normalize_team_name(raw))


def _decimal_to_american(dec: float) -> int:
    """Convert decimal odds to American moneyline."""
    if dec >= 2.0:
        return round((dec - 1) * 100)
    else:
        return round(-100 / (dec - 1))


def _enrich_with_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Call The Odds API and join best-available DraftKings moneylines onto
    the fixture DataFrame.  Requires ODDS_API_KEY in environment.

    Adds columns: best_home_odds, best_draw_odds, best_away_odds (American).
    If the API key is absent or the call fails, columns are set to pd.NA.
    """
    api_key = os.environ.get("ODDS_API_KEY", "").strip()

    for col in ("best_home_odds", "best_draw_odds", "best_away_odds"):
        df[col] = pd.NA

    if not api_key:
        print("  [ODDS] ODDS_API_KEY not set — skipping odds enrichment.")
        print("         Add ODDS_API_KEY=<key> to .env to enable EV analysis.")
        return df

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "bookmakers": "draftkings,fanduel,betmgm,pointsbet,caesars",
    }

    try:
        resp = requests.get(ODDS_API_URL, params=params, timeout=20)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        print(f"  [ODDS] API usage: {used} used / {remaining} remaining")

        if resp.status_code == 401:
            print("  [ODDS] Invalid or expired ODDS_API_KEY — skipping.")
            return df
        if resp.status_code == 429:
            print("  [ODDS] Rate limit hit — skipping odds for this run.")
            return df
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [ODDS] Request failed: {exc} — skipping odds enrichment.")
        return df

    events = resp.json()
    if not isinstance(events, list):
        print("  [ODDS] Unexpected API response format.")
        return df

    # Build lookup: (home_canonical, away_canonical) → {home_odds, draw_odds, away_odds}
    odds_lookup: dict[tuple[str, str], dict[str, float]] = {}

    for event in events:
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        home_key = _normalise_odds_team(home_raw)
        away_key = _normalise_odds_team(away_raw)

        best: dict[str, float] = {}  # outcome_label → best decimal odds across books

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = _normalise_odds_team(outcome.get("name", ""))
                    price = float(outcome.get("price", 0))
                    if price <= 1.0:
                        continue
                    # Map outcome name to role
                    if name == home_key:
                        role = "home"
                    elif name == away_key:
                        role = "away"
                    else:
                        role = "draw"  # three-way market: third outcome is draw
                    if role not in best or price > best[role]:
                        best[role] = price

        if best:
            odds_lookup[(home_key, away_key)] = best

    matched = 0
    for idx, row in df.iterrows():
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        entry = odds_lookup.get((home, away))
        if entry:
            if "home" in entry:
                df.at[idx, "best_home_odds"] = _decimal_to_american(entry["home"])
            if "draw" in entry:
                df.at[idx, "best_draw_odds"] = _decimal_to_american(entry["draw"])
            if "away" in entry:
                df.at[idx, "best_away_odds"] = _decimal_to_american(entry["away"])
            matched += 1

    print(f"  [ODDS] Matched odds for {matched}/{len(df)} fixtures.")
    return df


def fetch_upcoming_fixtures(days_ahead: int = 45, include_odds: bool = True) -> pd.DataFrame:
    """Fetch upcoming MLS fixtures for the next *days_ahead* days."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_fixtures: list[dict] = []

    today = datetime.now()
    print(f"Fetching MLS schedule for next {days_ahead} days...")

    for day_offset in range(days_ahead):
        target = today + timedelta(days=day_offset)
        date_str = target.strftime("%Y%m%d")
        day_fixtures = fetch_day(date_str)
        if day_fixtures:
            print(f"  {target.strftime('%Y-%m-%d')}: {len(day_fixtures)} match(es)")
            all_fixtures.extend(day_fixtures)
        # Be polite to ESPN's API
        time.sleep(0.15)

    if not all_fixtures:
        print("No upcoming MLS fixtures found (possible offseason or API issue).")
        # Write empty file with correct columns so the app doesn't break
        df = pd.DataFrame(columns=[
            "Date", "Time", "HomeTeam", "AwayTeam",
            "HomeConference", "AwayConference", "CrossConference",
            "HomeSurface", "AwayTravelMiles", "IsLongHaul", "Venue", "ESPN_ID",
            "best_home_odds", "best_draw_odds", "best_away_odds",
        ])
    else:
        df = pd.DataFrame(all_fixtures).drop_duplicates(
            subset=["Date", "HomeTeam", "AwayTeam"]
        ).sort_values("Date").reset_index(drop=True)

    # ── Odds enrichment ────────────────────────────────────────────────────────
    if include_odds:
        df = _enrich_with_odds(df)
    else:
        for col in ("best_home_odds", "best_draw_odds", "best_away_odds"):
            if col not in df.columns:
                df[col] = pd.NA

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(df)} upcoming fixtures → {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch upcoming MLS fixtures from ESPN.")
    parser.add_argument("--days", type=int, default=45, help="Days ahead to look (default: 45)")
    parser.add_argument("--no-odds", action="store_true", help="Skip The Odds API enrichment")
    args = parser.parse_args()
    fetch_upcoming_fixtures(days_ahead=args.days, include_odds=not args.no_odds)
