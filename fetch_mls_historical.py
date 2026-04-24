"""
fetch_mls_historical.py
Pull MLS historical match results from football-data.org (supplemental source).

Requires a free API key from https://www.football-data.org — store it in .env
as FD_API_KEY.  If no key is present the script prints guidance and exits
gracefully; the rest of the pipeline will still work via ASA data.

Outputs:
  data_files/raw/fdorg_matches_{season}.csv  — one file per season

Usage:
    python fetch_mls_historical.py
    python fetch_mls_historical.py --seasons 2022 2023 2024 2025
"""

import argparse
import os
import time
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

from team_name_mapping import normalize_team_name

load_dotenv()

FD_API_KEY: str | None = os.getenv("FD_API_KEY")
FD_BASE = "https://api.football-data.org/v4"
MLS_COMPETITION = "MLS"   # football-data.org competition code for MLS
RAW_DIR = os.path.join("data_files", "raw")

HEADERS_FD = {"X-Auth-Token": FD_API_KEY or ""}


def _get(url: str, params: dict | None = None) -> dict | None:
    """Single GET with basic error handling."""
    if not FD_API_KEY:
        return None
    try:
        resp = requests.get(url, headers=HEADERS_FD, params=params, timeout=20)
        if resp.status_code == 429:
            # Rate limit hit — wait and retry once
            print("  [WAIT] Rate limit hit, sleeping 70 s…")
            time.sleep(70)
            resp = requests.get(url, headers=HEADERS_FD, params=params, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        print(f"  [WARN] HTTP {resp.status_code} for {url}")
    except requests.RequestException as exc:
        print(f"  [WARN] Request error: {exc}")
    return None


def fetch_season_matches(season: int) -> pd.DataFrame:
    """Return a DataFrame of MLS matches for a single season year."""
    url = f"{FD_BASE}/competitions/{MLS_COMPETITION}/matches"
    data = _get(url, params={"season": season, "status": "FINISHED"})
    if not data:
        return pd.DataFrame()

    matches = data.get("matches", [])
    if not matches:
        print(f"  No finished matches found for {season}.")
        return pd.DataFrame()

    rows = []
    for m in matches:
        home_raw = m.get("homeTeam", {}).get("name", "")
        away_raw = m.get("awayTeam", {}).get("name", "")
        score = m.get("score", {})
        ft = score.get("fullTime", {})
        home_goals = ft.get("home")
        away_goals = ft.get("away")

        if home_goals is None or away_goals is None:
            continue

        home_goals = int(home_goals)
        away_goals = int(away_goals)

        if home_goals > away_goals:
            result = "H"
        elif away_goals > home_goals:
            result = "A"
        else:
            result = "D"

        rows.append({
            "MatchDate": m.get("utcDate", "")[:10],
            "Season": season,
            "HomeTeam": normalize_team_name(home_raw),
            "AwayTeam": normalize_team_name(away_raw),
            "HomeGoals": home_goals,
            "AwayGoals": away_goals,
            "Result": result,
            "Matchday": m.get("matchday"),
            "FD_ID": m.get("id"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
        df = df.sort_values("MatchDate").reset_index(drop=True)

    return df


def main(seasons: list[int] | None = None) -> None:
    if not FD_API_KEY:
        print(
            "⚠️  FD_API_KEY not set in .env.\n"
            "   Get a free key at https://www.football-data.org and add:\n"
            "     FD_API_KEY=your_key_here\n"
            "   to your .env file.  Skipping football-data.org pull.\n"
            "   The pipeline will still work using ASA data as the primary source."
        )
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    if seasons is None:
        current_year = datetime.now().year
        seasons = list(range(2017, current_year + 1))

    all_frames: list[pd.DataFrame] = []

    for season in seasons:
        print(f"Fetching football-data.org MLS season {season}…")
        df = fetch_season_matches(season)
        if not df.empty:
            out_path = os.path.join(RAW_DIR, f"fdorg_matches_{season}.csv")
            df.to_csv(out_path, index=False)
            print(f"  ✅ {len(df)} matches → {out_path}")
            all_frames.append(df)
        else:
            print(f"  (no data for {season})")
        # football-data.org free tier: 10 req/min
        time.sleep(6)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = os.path.join(RAW_DIR, "fdorg_matches_all.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n✅ Combined {len(combined)} matches → {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLS historical data from football-data.org.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Season years to fetch (default: 2017 to current year)",
    )
    args = parser.parse_args()
    main(seasons=args.seasons)
