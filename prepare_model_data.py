"""
prepare_model_data.py
Combine raw MLS data sources, engineer MLS-specific features, and write the
final training CSV consumed by predictions.py.

Data priority:
  1. ASA games (xG + results)  — primary
  2. football-data.org          — supplement / cross-check
  3. Hardcoded MLS lookup tables (surface, conference, stadium coords)

Output: data_files/combined_historical_data.csv

Usage:
    python prepare_model_data.py
"""

import os
from collections import defaultdict
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd

from team_name_mapping import normalize_team_name

# ── Directory paths ────────────────────────────────────────────────────────────
DATA_DIR = "data_files"
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_PATH = os.path.join(DATA_DIR, "combined_historical_data.csv")

# ── MLS lookup tables (mirrors predictions.py constants) ──────────────────────
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

TURF_STADIUMS = {
    "New England Revolution", "Portland Timbers",
    "Seattle Sounders", "Vancouver Whitecaps", "FC Cincinnati",
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

# MLS playoff format: top 9 in each conference qualify (as of 2024 format)
MLS_PLAYOFF_SPOTS = 9


# ── Utility functions ──────────────────────────────────────────────────────────

def _haversine_miles(home: str, away: str) -> float:
    if home not in STADIUM_COORDS or away not in STADIUM_COORDS:
        return 0.0
    lat1, lon1 = map(radians, STADIUM_COORDS[home])
    lat2, lon2 = map(radians, STADIUM_COORDS[away])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 3958.8 * 2 * atan2(sqrt(a), sqrt(1 - a))


def _conference(team: str) -> str:
    if team in EASTERN_CONF:
        return "Eastern"
    if team in WESTERN_CONF:
        return "Western"
    return "Unknown"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """Load all available raw data and combine into a single DataFrame."""
    frames: list[pd.DataFrame] = []

    # ── ASA games (primary) ────────────────────────────────────────────────────
    asa_path = os.path.join(RAW_DIR, "asa_games.csv")
    if os.path.exists(asa_path):
        df_asa = pd.read_csv(asa_path)
        df_asa["Source"] = "ASA"
        frames.append(df_asa)
        print(f"  Loaded {len(df_asa)} rows from ASA games.")
    else:
        print(f"  [INFO] ASA games not found at {asa_path}. Run fetch_asa_data.py first.")

    # ── football-data.org (supplement) ────────────────────────────────────────
    fdo_path = os.path.join(RAW_DIR, "fdorg_matches_all.csv")
    if os.path.exists(fdo_path):
        df_fdo = pd.read_csv(fdo_path)
        df_fdo["Source"] = "FDO"
        # Only add FDO rows not already covered by ASA
        if frames:
            asa_keys = set(
                zip(
                    frames[0]["MatchDate"].astype(str),
                    frames[0]["HomeTeam"],
                    frames[0]["AwayTeam"],
                )
            )
            mask = ~df_fdo.apply(
                lambda r: (str(r["MatchDate"])[:10], r["HomeTeam"], r["AwayTeam"]) in asa_keys,
                axis=1,
            )
            df_fdo_new = df_fdo[mask]
            if not df_fdo_new.empty:
                frames.append(df_fdo_new)
                print(f"  Added {len(df_fdo_new)} supplemental rows from football-data.org.")
        else:
            frames.append(df_fdo)
            print(f"  Loaded {len(df_fdo)} rows from football-data.org (no ASA data found).")

    if not frames:
        print("❌ No raw data found. Run fetch_asa_data.py and/or fetch_mls_historical.py first.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    df = df.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam"])
    df = df.sort_values("MatchDate").reset_index(drop=True)

    print(f"  Combined: {len(df)} rows, {df['MatchDate'].min().year}–{df['MatchDate'].max().year}")
    return df


# ── Static MLS feature engineering ────────────────────────────────────────────

def add_static_mls_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-invariant MLS structural features to each row."""
    df = df.copy()

    df["HomeConference"] = df["HomeTeam"].apply(_conference)
    df["AwayConference"] = df["AwayTeam"].apply(_conference)
    df["is_cross_conference"] = (
        (df["HomeConference"] != df["AwayConference"]) &
        (df["HomeConference"] != "Unknown")
    ).astype(int)

    df["home_is_turf"] = df["HomeTeam"].isin(TURF_STADIUMS).astype(int)
    df["away_is_turf"] = df["AwayTeam"].isin(TURF_STADIUMS).astype(int)

    df["away_travel_miles"] = df.apply(
        lambda r: _haversine_miles(r["HomeTeam"], r["AwayTeam"]), axis=1
    )
    df["is_long_haul"] = (df["away_travel_miles"] > 1500).astype(int)

    return df


# ── Rolling team features (O(n), no look-ahead leakage) ───────────────────────

def add_rolling_features(df: pd.DataFrame, lookbacks: list[int] = [5, 10]) -> pd.DataFrame:
    """
    For each match, compute rolling stats for home and away teams using only
    their previous matches.  Uses a running dict to avoid O(n²) lookups.
    """
    df = df.sort_values("MatchDate").reset_index(drop=True)

    # team_history[team] = deque of dicts {goals_for, goals_against, xg_for, xg_against, points}
    team_history: dict[str, list[dict]] = defaultdict(list)

    results_home: list[dict] = []
    results_away: list[dict] = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        def _rolling_stats(team: str) -> dict:
            hist = team_history[team]
            stats: dict[str, float] = {}
            for lb in lookbacks:
                window = hist[-lb:] if len(hist) >= lb else hist
                n = len(window)
                if n == 0:
                    stats[f"xg_l{lb}"] = np.nan
                    stats[f"pts_l{lb}"] = np.nan
                    stats[f"goals_for_l{lb}"] = np.nan
                    stats[f"goals_against_l{lb}"] = np.nan
                    stats[f"xg_against_l{lb}"] = np.nan
                else:
                    xg_vals = [h["xg_for"] for h in window if not np.isnan(h["xg_for"])]
                    stats[f"xg_l{lb}"] = float(np.mean(xg_vals)) if xg_vals else np.nan
                    stats[f"pts_l{lb}"] = float(np.mean([h["points"] for h in window]))
                    stats[f"goals_for_l{lb}"] = float(np.mean([h["goals_for"] for h in window]))
                    stats[f"goals_against_l{lb}"] = float(np.mean([h["goals_against"] for h in window]))
                    xga_vals = [h["xg_against"] for h in window if not np.isnan(h["xg_against"])]
                    stats[f"xg_against_l{lb}"] = float(np.mean(xga_vals)) if xga_vals else np.nan
            # Rest days
            if hist:
                last_date = hist[-1]["date"]
                rest = (row["MatchDate"] - last_date).days if pd.notna(last_date) else np.nan
                stats["rest_days"] = rest
            else:
                stats["rest_days"] = np.nan
            return stats

        home_stats = _rolling_stats(home)
        away_stats = _rolling_stats(away)

        results_home.append(home_stats)
        results_away.append(away_stats)

        # Update histories AFTER reading stats (no leakage)
        result = row.get("Result", "")
        home_pts = 3 if result == "H" else (1 if result == "D" else 0)
        away_pts = 3 if result == "A" else (1 if result == "D" else 0)

        hg = float(row.get("HomeGoals", 0) or 0)
        ag = float(row.get("AwayGoals", 0) or 0)
        hxg = float(row.get("home_xgoals", hg) or hg)
        axg = float(row.get("away_xgoals", ag) or ag)

        team_history[home].append({
            "date": row["MatchDate"],
            "goals_for": hg, "goals_against": ag,
            "xg_for": hxg, "xg_against": axg,
            "points": home_pts,
        })
        team_history[away].append({
            "date": row["MatchDate"],
            "goals_for": ag, "goals_against": hg,
            "xg_for": axg, "xg_against": hxg,
            "points": away_pts,
        })

    # Rename and attach columns
    home_df = pd.DataFrame(results_home).add_prefix("home_")
    away_df = pd.DataFrame(results_away).add_prefix("away_")

    df = pd.concat([df.reset_index(drop=True), home_df, away_df], axis=1)
    return df


# ── Head-to-head rolling feature ──────────────────────────────────────────────

def add_h2h_features(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Add head-to-head win rate for the home team over the last N meetings."""
    df = df.sort_values("MatchDate").reset_index(drop=True)

    h2h_records: dict[frozenset, list[dict]] = defaultdict(list)
    home_h2h_win_rate: list[float] = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        key = frozenset([home, away])
        hist = h2h_records[key][-lookback:]

        if hist:
            wins = sum(1 for h in hist if h["home_winner"] == home)
            rate = wins / len(hist)
        else:
            rate = 0.5  # neutral prior when no H2H history

        home_h2h_win_rate.append(rate)

        result = row.get("Result", "")
        if result == "H":
            winner = home
        elif result == "A":
            winner = away
        else:
            winner = None

        h2h_records[key].append({"date": row["MatchDate"], "home_winner": winner})

    df["home_h2h_win_rate_l5"] = home_h2h_win_rate
    return df


# ── Intra-season standings features ───────────────────────────────────────────

def add_standings_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match date, compute a running conference standings snapshot and
    derive `games_from_playoff` (positive = inside, negative = outside).
    """
    df = df.sort_values("MatchDate").reset_index(drop=True)

    # Running points tally per season
    team_season_pts: dict[tuple[str, int], int] = defaultdict(int)
    team_season_played: dict[tuple[str, int], int] = defaultdict(int)

    home_gap: list[int] = []
    away_gap: list[int] = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        season = row["MatchDate"].year

        home_conf = _conference(home)
        away_conf = _conference(away)

        def _playoff_gap(team: str, team_season: int) -> int:
            conf = _conference(team)
            if conf == "Unknown":
                return 0
            conf_teams = EASTERN_CONF if conf == "Eastern" else WESTERN_CONF
            conf_pts = {
                t: team_season_pts[(t, team_season)]
                for t in conf_teams
            }
            # Sort by points descending
            sorted_pts = sorted(conf_pts.values(), reverse=True)
            team_pts = conf_pts.get(team, 0)
            rank = sum(1 for p in sorted_pts if p > team_pts) + 1
            return MLS_PLAYOFF_SPOTS - rank  # positive = inside, negative = outside

        home_gap.append(_playoff_gap(home, season))
        away_gap.append(_playoff_gap(away, season))

        # Update pts AFTER reading
        result = row.get("Result", "")
        team_season_pts[(home, season)] += 3 if result == "H" else (1 if result == "D" else 0)
        team_season_pts[(away, season)] += 3 if result == "A" else (1 if result == "D" else 0)
        team_season_played[(home, season)] += 1
        team_season_played[(away, season)] += 1

    df["home_games_from_playoff"] = home_gap
    df["away_games_from_playoff"] = away_gap
    return df


# ── Designated player placeholder ─────────────────────────────────────────────

def add_dp_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Designated player availability flag.
    Default 0.5 (unknown) until live injury/roster data is integrated.
    Future: replace with real-time roster availability via MLS API.
    """
    df["home_dp_available"] = 0.5
    df["away_dp_available"] = 0.5
    return df


# ── Season recency weight ──────────────────────────────────────────────────────

def add_season_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per the MLS roadmap: weight 2020+ seasons more heavily in training.
    Encodes as a numeric feature so tree-based models can exploit it.
    """
    current_year = datetime.now().year
    df["season_weight"] = df["MatchDate"].dt.year.apply(
        lambda y: 1.0 if y < 2020 else round(1.0 + (y - 2020) * 0.15, 2)
    )
    return df


# ── Combine & save ─────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=== prepare_model_data.py — MLS Feature Engineering ===\n")

    # 1. Load raw data
    print("Step 1: Loading raw data…")
    df = load_raw_data()
    if df.empty:
        print("No data to process. Exiting.")
        return

    # 2. Standardise Result column
    result_col = next((c for c in ["Result", "FullTimeResult", "FTR"] if c in df.columns), None)
    if result_col and result_col != "Result":
        df["Result"] = df[result_col]
    elif "Result" not in df.columns:
        # Derive from goals
        hg_col = next((c for c in ["HomeGoals", "home_goals", "FTHG"] if c in df.columns), None)
        ag_col = next((c for c in ["AwayGoals", "away_goals", "FTAG"] if c in df.columns), None)
        if hg_col and ag_col:
            df["Result"] = np.where(
                df[hg_col] > df[ag_col], "H",
                np.where(df[ag_col] > df[hg_col], "A", "D")
            )
        else:
            print("  [WARN] Cannot determine Result; predictions will be unavailable.")
            df["Result"] = ""

    # Standardise goal columns
    for raw_col, std_col in [
        (["HomeGoals", "home_goals", "FTHG"], "HomeGoals"),
        (["AwayGoals", "away_goals", "FTAG"], "AwayGoals"),
    ]:
        if std_col not in df.columns:
            for c in raw_col:
                if c in df.columns:
                    df[std_col] = pd.to_numeric(df[c], errors="coerce")
                    break

    # Standardise xG columns
    for raw_col, std_col in [
        (["home_xgoals", "xg_home", "HomeXG"], "home_xgoals"),
        (["away_xgoals", "xg_away", "AwayXG"], "away_xgoals"),
    ]:
        if std_col not in df.columns:
            for c in raw_col:
                if c in df.columns:
                    df[std_col] = pd.to_numeric(df[c], errors="coerce")
                    break
            else:
                # Fall back to actual goals as a proxy for xG
                proxy = "HomeGoals" if "home" in std_col.lower() else "AwayGoals"
                df[std_col] = df.get(proxy, 0)

    # Drop rows without a valid result for training
    df_clean = df[df["Result"].isin(["H", "D", "A"])].copy()
    print(f"  Rows with valid results: {len(df_clean)} / {len(df)}")

    # 3. Static MLS features
    print("\nStep 2: Adding static MLS features (conference, surface, travel)…")
    df_clean = add_static_mls_features(df_clean)

    # 4. Rolling team stats
    print("Step 3: Computing rolling team statistics (form, xG, rest days)…")
    df_clean = add_rolling_features(df_clean)

    # 5. H2H
    print("Step 4: Computing head-to-head features…")
    df_clean = add_h2h_features(df_clean)

    # 6. Standings / playoff race
    print("Step 5: Computing playoff race features…")
    df_clean = add_standings_features(df_clean)

    # 7. Designated player placeholder
    print("Step 6: Adding designated player flags (placeholder)…")
    df_clean = add_dp_flag(df_clean)

    # 8. Season recency weight
    print("Step 7: Adding season recency weights…")
    df_clean = add_season_weight(df_clean)

    # 9. Final clean-up: numeric conversion, fill NaN for model features
    feature_cols = [c for c in df_clean.columns if c not in (
        "MatchDate", "HomeTeam", "AwayTeam", "Result", "Source",
        "HomeConference", "AwayConference", "Season", "game_id", "FD_ID",
        "Matchday", "status",
    )]
    for col in feature_cols:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        except (TypeError, ValueError):
            pass

    # Fill NaN in rolling features with column medians
    rolling_cols = [c for c in df_clean.columns if any(
        x in c for x in ["_l5", "_l10", "rest_days", "h2h", "playoff"]
    )]
    for col in rolling_cols:
        med = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(med if pd.notna(med) else 0)

    df_clean = df_clean.sort_values("MatchDate").reset_index(drop=True)

    # 10. Save
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(df_clean)} rows → {OUTPUT_PATH}")
    print(f"   Columns: {len(df_clean.columns)}")
    print(f"   Date range: {df_clean['MatchDate'].min().date()} → {df_clean['MatchDate'].max().date()}")
    print(f"   Seasons: {sorted(df_clean['MatchDate'].dt.year.unique().tolist())}")

    # 11. Quick summary
    result_counts = df_clean["Result"].value_counts()
    print(f"\n   Result distribution:")
    for r, n in result_counts.items():
        print(f"     {r}: {n} ({n/len(df_clean):.1%})")


if __name__ == "__main__":
    main()
