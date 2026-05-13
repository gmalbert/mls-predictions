"""
fetch_asa_data.py
Pull MLS historical game data and xG metrics from American Soccer Analysis
via the itscalledsoccer Python package (free, no API key required).

Outputs:
  data_files/raw/asa_games.csv      — game-level results + xG
  data_files/raw/asa_team_xg.csv   — team xG aggregates (current season)
  data_files/raw/asa_player_ga.csv — player goals added (current season)

Usage:
    python fetch_asa_data.py
    python fetch_asa_data.py --seasons 2020 2021 2022 2023 2024 2025

Notes on itscalledsoccer compatibility:
  - v1.3.x: get_games() used `seasons=` (list of strings)
  - v2.0.0: get_games() renamed that parameter to `season_name=` (str | list[str])
    Also added a `status=` parameter for server-side filtering ("FullTime", "PreMatch", "Abandoned")
  - v2.0.0: get_teams() is now stable; REST fallback kept for defensive compatibility
  - get_game_xgoals() returns 500 from ASA server → skipped
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import requests

from team_name_mapping import normalize_team_name

RAW_DIR = os.path.join("data_files", "raw")
ASA_BASE = "https://app.americansocceranalysis.com/api/v1"


def _ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)


def _get_team_id_map() -> dict[str, str]:
    """
    Build team_id → canonical_name mapping by calling the ASA REST API
    directly (the itscalledsoccer get_teams() method is broken in v1.3.5
    due to an empty internal teams cache).
    """
    try:
        resp = requests.get(f"{ASA_BASE}/mls/teams", timeout=15)
        resp.raise_for_status()
        teams = resp.json()
        mapping = {}
        for t in teams:
            tid = t.get("team_id", "")
            name = t.get("team_name", "")
            if tid and name:
                mapping[tid] = normalize_team_name(name)
        print(f"  Loaded {len(mapping)} team IDs from ASA REST API.")
        return mapping
    except Exception as exc:
        print(f"  [WARN] Could not fetch ASA teams via REST: {exc}")
        return {}


def fetch_asa_games(seasons: list[int]) -> pd.DataFrame:
    """Pull game-level data (results) from ASA for a list of seasons."""
    from itscalledsoccer.client import AmericanSoccerAnalysis

    team_id_map = _get_team_id_map()

    # season_name accepts str or list[str] in itscalledsoccer v2.0.0
    # (was `seasons=` in v1.3.x — renamed in the v2.0.0 breaking release)
    str_seasons = [str(s) for s in seasons]
    print(f"Fetching ASA game data for seasons: {str_seasons}…")

    asa = AmericanSoccerAnalysis()
    games_df = asa.get_games(leagues="mls", season_name=str_seasons, status="FullTime")

    if games_df.empty:
        print("  [WARN] ASA returned no game data.")
        return pd.DataFrame()

    print(f"  Retrieved {len(games_df)} games from ASA.")

    # Known ASA v1.3.5 columns: game_id, date_time_utc, home_score, away_score,
    # home_team_id, away_team_id, season_name, matchday, status
    # (no game-level xG — leave as NaN; prepare_model_data uses goals as proxy)
    col_map_candidates: dict[str, list[str]] = {
        "date": ["date_time_utc", "date", "match_date"],
        "home_team_id": ["home_team_id", "home_team"],
        "away_team_id": ["away_team_id", "away_team"],
        "home_goals": ["home_score", "home_goals", "home_goal_count"],
        "away_goals": ["away_score", "away_goals", "away_goal_count"],
    }

    normalized: dict[str, pd.Series] = {}
    for std_col, candidates in col_map_candidates.items():
        for cand in candidates:
            if cand in games_df.columns:
                normalized[std_col] = games_df[cand]
                break

    clean = pd.DataFrame()
    date_series = normalized.get("date", pd.Series(dtype=str))
    clean["MatchDate"] = (
        pd.to_datetime(date_series, errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    home_ids = normalized.get("home_team_id", pd.Series(dtype=str)).astype(str)
    away_ids = normalized.get("away_team_id", pd.Series(dtype=str)).astype(str)

    clean["HomeTeam"] = home_ids.map(team_id_map).fillna(home_ids).apply(normalize_team_name)
    clean["AwayTeam"] = away_ids.map(team_id_map).fillna(away_ids).apply(normalize_team_name)

    clean["HomeGoals"] = pd.to_numeric(
        normalized.get("home_goals", pd.Series(dtype=float)), errors="coerce"
    )
    clean["AwayGoals"] = pd.to_numeric(
        normalized.get("away_goals", pd.Series(dtype=float)), errors="coerce"
    )
    clean["home_xgoals"] = float("nan")
    clean["away_xgoals"] = float("nan")

    if "season_name" in games_df.columns:
        clean["Season"] = games_df["season_name"].values
    if "game_id" in games_df.columns:
        clean["game_id"] = games_df["game_id"].values
    if "status" in games_df.columns:
        clean["status"] = games_df["status"].values

    # Only keep finished games
    if "status" in clean.columns:
        clean = clean[clean["status"] == "FullTime"].copy()

    def _result(row) -> str:
        hg, ag = row["HomeGoals"], row["AwayGoals"]
        if pd.isna(hg) or pd.isna(ag):
            return ""
        hg, ag = int(hg), int(ag)
        return "H" if hg > ag else ("A" if ag > hg else "D")

    clean["Result"] = clean.apply(_result, axis=1)
    clean = clean.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam"])
    clean = clean[clean["HomeTeam"] != ""]
    clean = clean.sort_values("MatchDate").reset_index(drop=True)
    return clean


def fetch_asa_team_xg(season_name: str | None = None) -> pd.DataFrame:
    """Pull team xG aggregates for the current (or specified) season."""
    from itscalledsoccer.client import AmericanSoccerAnalysis

    season_name = season_name or str(datetime.now().year)
    asa = AmericanSoccerAnalysis()
    print(f"Fetching ASA team xG for {season_name}…")
    df = asa.get_team_xgoals(leagues="mls", season_name=season_name)
    if df.empty:
        print("  [WARN] No team xG data returned.")
    else:
        # Enrich with team names from REST API
        team_id_map = _get_team_id_map()
        if "team_id" in df.columns:
            df = df.copy()
            df["team_name"] = df["team_id"].map(team_id_map).fillna(df["team_id"])
        print(f"  Retrieved xG data for {len(df)} teams.")
    return df


def fetch_asa_player_goals_added(season_name: str | None = None) -> pd.DataFrame:
    """Pull player goals added for the current (or specified) season."""
    from itscalledsoccer.client import AmericanSoccerAnalysis

    season_name = season_name or str(datetime.now().year)
    asa = AmericanSoccerAnalysis()
    print(f"Fetching ASA player goals added for {season_name}…")
    df = asa.get_player_goals_added(leagues="mls", season_name=season_name)
    if df.empty:
        print("  [WARN] No player goals added data returned.")
    else:
        print(f"  Retrieved data for {len(df)} player-season records.")
    return df


def main(seasons: list[int] | None = None) -> None:
    _ensure_dirs()

    if seasons is None:
        current_year = datetime.now().year
        seasons = list(range(2017, current_year + 1))

    # ── Game data ──────────────────────────────────────────────────────────────
    games_df = fetch_asa_games(seasons)
    if not games_df.empty:
        out_path = os.path.join(RAW_DIR, "asa_games.csv")
        games_df.to_csv(out_path, index=False)
        print(f"✅ Saved {len(games_df)} games → {out_path}")
    else:
        print("❌ No ASA game data saved.")

    # ── Team xG (current season) ───────────────────────────────────────────────
    current_season = str(max(seasons))
    try:
        team_xg_df = fetch_asa_team_xg(current_season)
        if not team_xg_df.empty:
            out_path = os.path.join(RAW_DIR, "asa_team_xg.csv")
            team_xg_df.to_csv(out_path, index=False)
            print(f"✅ Saved team xG → {out_path}")
    except Exception as exc:
        print(f"  [WARN] Could not fetch team xG: {exc}")

    # ── Player goals added (current season) ───────────────────────────────────
    try:
        player_ga_df = fetch_asa_player_goals_added(current_season)
        if not player_ga_df.empty:
            out_path = os.path.join(RAW_DIR, "asa_player_goals_added.csv")
            player_ga_df.to_csv(out_path, index=False)
            print(f"✅ Saved player goals added → {out_path}")
    except Exception as exc:
        print(f"  [WARN] Could not fetch player goals added: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLS data from American Soccer Analysis.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Season years to fetch (default: 2017 to current year)",
    )
    args = parser.parse_args()
    main(seasons=args.seasons)
