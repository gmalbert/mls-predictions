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

import ast
import os
import sys
from collections import defaultdict
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd

from team_name_mapping import normalize_team_name
from models.frontier_features import (
    EASTERN_CONF as FRONTIER_EASTERN_CONF,
    STADIUMS as FRONTIER_STADIUMS,
    TURF_STADIUMS as FRONTIER_TURF_STADIUMS,
    WESTERN_CONF as FRONTIER_WESTERN_CONF,
    add_frontier_features,
    load_optional_sources,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Directory paths ────────────────────────────────────────────────────────────
DATA_DIR = "data_files"
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_PATH = os.path.join(DATA_DIR, "combined_historical_data.csv")

# ── Centralized MLS metadata ──────────────────────────────────────────────────
EASTERN_CONF = set(FRONTIER_EASTERN_CONF)
WESTERN_CONF = set(FRONTIER_WESTERN_CONF)
TURF_STADIUMS = set(FRONTIER_TURF_STADIUMS)
STADIUM_COORDS = {
    team: (stadium.latitude, stadium.longitude)
    for team, stadium in FRONTIER_STADIUMS.items()
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
    df["HomeTeam"] = df["HomeTeam"].astype(str).apply(normalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].astype(str).apply(normalize_team_name)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")

    # Optional phase/competition manifest is joined only when it was available
    # before kickoff, allowing regular season, playoffs, and Leagues Cup to remain
    # distinct in the rolling audit.
    phase_path = os.path.join(RAW_DIR, "competition_phases.csv")
    if os.path.exists(phase_path):
        try:
            phases = pd.read_csv(phase_path)
            if "available_at" in phases.columns:
                phases["available_at"] = pd.to_datetime(phases["available_at"], errors="coerce", utc=True).dt.tz_localize(None)
                phase_date_col = next((c for c in ("MatchDate", "event_date", "date") if c in phases.columns), None)
                if phase_date_col:
                    phases[phase_date_col] = pd.to_datetime(phases[phase_date_col], errors="coerce")
                    phases = phases[phases["available_at"] < phases[phase_date_col]]
                phase_columns = [column for column in ("Competition", "Phase") if column in phases.columns]
                if phase_columns:
                    if "game_id" in phases.columns and "game_id" in df.columns:
                        df = df.merge(phases[["game_id", *phase_columns]].drop_duplicates("game_id"), on="game_id", how="left", suffixes=("", "_manifest"))
                    elif phase_date_col and {"HomeTeam", "AwayTeam"}.issubset(phases.columns):
                        phases["HomeTeam"] = phases["HomeTeam"].map(normalize_team_name)
                        phases["AwayTeam"] = phases["AwayTeam"].map(normalize_team_name)
                        phases = phases.rename(columns={phase_date_col: "MatchDate"})
                        keys = ["MatchDate", "HomeTeam", "AwayTeam"]
                        df = df.merge(phases[[*keys, *phase_columns]].drop_duplicates(keys), on=keys, how="left", suffixes=("", "_manifest"))
                    for column in phase_columns:
                        manifest_column = f"{column}_manifest"
                        if manifest_column in df.columns:
                            df[column] = df.get(column, pd.Series(index=df.index, dtype=object)).fillna(df[manifest_column])
                            df = df.drop(columns=[manifest_column])
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            print(f"  [WARN] Could not apply competition phase manifest: {exc}")
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
    home_xg_source = df["home_xgoals"] if "home_xgoals" in df.columns else pd.Series(np.nan, index=df.index)
    away_xg_source = df["away_xgoals"] if "away_xgoals" in df.columns else pd.Series(np.nan, index=df.index)
    home_xg_raw = pd.to_numeric(home_xg_source, errors="coerce")
    away_xg_raw = pd.to_numeric(away_xg_source, errors="coerce")
    df["xg_data_missing"] = (home_xg_raw.isna() | away_xg_raw.isna()).astype(float)

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
                    def _recency_weighted(key: str) -> float:
                        values = [h[key] for h in window if not np.isnan(h[key])]
                        if not values:
                            return np.nan
                        # Latest match receives weight 1.0; each older observation
                        # decays by 15%, matching the roadmap's recency emphasis.
                        weights = np.power(0.85, np.arange(len(values) - 1, -1, -1))
                        return float(np.average(values, weights=weights))

                    stats[f"xg_l{lb}"] = _recency_weighted("xg_for")
                    stats[f"pts_l{lb}"] = float(np.mean([h["points"] for h in window]))
                    stats[f"goals_for_l{lb}"] = float(np.mean([h["goals_for"] for h in window]))
                    stats[f"goals_against_l{lb}"] = float(np.mean([h["goals_against"] for h in window]))
                    stats[f"xg_against_l{lb}"] = _recency_weighted("xg_against")
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
        # A missing feed value falls back to the observed score only so rolling
        # histories remain numerically usable.  xg_data_missing makes that
        # degraded input explicit to training, evaluation, and the UI.
        hxg_raw = pd.to_numeric(row.get("home_xgoals"), errors="coerce")
        axg_raw = pd.to_numeric(row.get("away_xgoals"), errors="coerce")
        hxg = hg if pd.isna(hxg_raw) else float(hxg_raw)
        axg = ag if pd.isna(axg_raw) else float(axg_raw)

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
    season_participants: dict[int, set[str]] = defaultdict(set)
    for season, season_rows in df.groupby(df["MatchDate"].dt.year):
        season_participants[int(season)] = set(season_rows["HomeTeam"]).union(season_rows["AwayTeam"])

    home_gap: list[float] = []
    away_gap: list[float] = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        season = row["MatchDate"].year

        home_conf = _conference(home)
        away_conf = _conference(away)

        def _playoff_gap(team: str, team_season: int) -> float:
            conf = _conference(team)
            if conf == "Unknown":
                return 0
            current_conference = EASTERN_CONF if conf == "Eastern" else WESTERN_CONF
            conf_teams = current_conference.intersection(season_participants[team_season])
            conf_pts = {
                t: team_season_pts[(t, team_season)]
                for t in conf_teams
            }
            # Compare points to the ninth-place cutoff and express the margin in
            # three-point-win equivalents. Positive is inside/ahead of the line.
            sorted_pts = sorted(conf_pts.values(), reverse=True)
            team_pts = conf_pts.get(team, 0)
            cutoff_index = min(MLS_PLAYOFF_SPOTS - 1, len(sorted_pts) - 1)
            cutoff_points = sorted_pts[cutoff_index] if sorted_pts else 0
            return round((team_pts - cutoff_points) / 3.0, 3)

        home_gap.append(_playoff_gap(home, season))
        away_gap.append(_playoff_gap(away, season))

        # Update pts AFTER reading
        result = row.get("Result", "")
        team_season_pts[(home, season)] += 3 if result == "H" else (1 if result == "D" else 0)
        team_season_pts[(away, season)] += 3 if result == "A" else (1 if result == "D" else 0)

    df["home_games_from_playoff"] = home_gap
    df["away_games_from_playoff"] = away_gap
    return df


# ── Designated player placeholder ─────────────────────────────────────────────

def add_dp_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated compatibility shim; frontier feeds own availability features."""
    return df.copy()


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


# ── Venue-specific stadium features ───────────────────────────────────────────

MLS_STADIUM_CAPACITY = {
    team: stadium.capacity for team, stadium in FRONTIER_STADIUMS.items()
}

_MEDIAN_CAPACITY = 22000  # fallback for unknown teams


def add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add venue-specific stadium features:
      - home_stadium_capacity  : absolute capacity
      - home_crowd_pressure    : capacity / league-median (normalised, higher = louder)
      - home_is_large_stadium  : capacity > 28 000 (top ~10% in MLS)
    """
    df = df.copy()
    df["home_stadium_capacity"] = (
        df["HomeTeam"].map(MLS_STADIUM_CAPACITY).fillna(_MEDIAN_CAPACITY).astype(int)
    )
    df["home_crowd_pressure"] = (
        df["home_stadium_capacity"] / _MEDIAN_CAPACITY
    ).round(3)
    df["home_is_large_stadium"] = (df["home_stadium_capacity"] > 28000).astype(int)
    return df


# ── Enhanced historical odds features ─────────────────────────────────────────

_BOOK_HOME_COLS  = ["B365H", "BWH", "IWH", "PSH", "WHH", "VCH"]
_BOOK_DRAW_COLS  = ["B365D", "BWD", "IWD", "PSD", "WHD", "VCD"]
_BOOK_AWAY_COLS  = ["B365A", "BWA", "IWA", "PSA", "WHA", "VCA"]


def _american_to_decimal(v: float) -> float:
    """Convert American moneyline to decimal odds."""
    if v >= 100:
        return round(v / 100 + 1, 4)
    elif v <= -100:
        return round(100 / abs(v) + 1, 4)
    return 1.0


def add_enhanced_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bookmaker-implied probability features when historical odds columns
    (e.g. from football-data.org: B365H/D/A) are present in the data.

    New columns added (when odds present):
      - odds_implied_home_prob : vig-free home win implied probability
      - odds_implied_draw_prob : vig-free draw implied probability
      - odds_implied_away_prob : vig-free away win implied probability
      - odds_market_margin     : total book overround (1 = no vig, >1 = vig)
      - odds_home_value        : decimal home odds (best available)
      - odds_draw_value        : decimal draw odds (best available)
      - odds_away_value        : decimal away odds (best available)
    """
    df = df.copy()

    # Identify which bookmaker columns actually exist
    home_cols = [c for c in _BOOK_HOME_COLS if c in df.columns]
    draw_cols = [c for c in _BOOK_DRAW_COLS if c in df.columns]
    away_cols = [c for c in _BOOK_AWAY_COLS if c in df.columns]

    if not (home_cols and draw_cols and away_cols):
        # No odds columns present — fill with neutral priors
        df["odds_data_available"] = 0
        df["odds_implied_home_prob"] = 0.45
        df["odds_implied_draw_prob"] = 0.26
        df["odds_implied_away_prob"] = 0.29
        df["odds_market_margin"]     = 1.0
        df["odds_home_value"]        = 2.22
        df["odds_draw_value"]        = 3.50
        df["odds_away_value"]        = 3.10
        return df

    df["odds_data_available"] = 1

    # Best available decimal odds = highest decimal (most favourable for bettor)
    df["odds_home_value"] = df[home_cols].apply(
        lambda row: row.dropna().max() if not row.dropna().empty else np.nan, axis=1
    )
    df["odds_draw_value"] = df[draw_cols].apply(
        lambda row: row.dropna().max() if not row.dropna().empty else np.nan, axis=1
    )
    df["odds_away_value"] = df[away_cols].apply(
        lambda row: row.dropna().max() if not row.dropna().empty else np.nan, axis=1
    )

    # Implied probabilities (raw, still includes vig)
    imp_h = 1.0 / df["odds_home_value"].replace(0, np.nan)
    imp_d = 1.0 / df["odds_draw_value"].replace(0, np.nan)
    imp_a = 1.0 / df["odds_away_value"].replace(0, np.nan)
    total = (imp_h + imp_d + imp_a).replace(0, np.nan)

    df["odds_market_margin"]     = total.round(4)
    # Vig-removed probabilities (sum to 1.0)
    df["odds_implied_home_prob"] = (imp_h / total).round(4)
    df["odds_implied_draw_prob"] = (imp_d / total).round(4)
    df["odds_implied_away_prob"] = (imp_a / total).round(4)

    # Fill rows where odds were absent with neutral priors
    fill_vals = {
        "odds_implied_home_prob": 0.45,
        "odds_implied_draw_prob": 0.26,
        "odds_implied_away_prob": 0.29,
        "odds_market_margin":     1.0,
        "odds_home_value":        2.22,
        "odds_draw_value":        3.50,
        "odds_away_value":        3.10,
    }
    for col, fill in fill_vals.items():
        df[col] = df[col].fillna(fill)

    return df


# ── Transfer market proxy features ────────────────────────────────────────────

def add_transfer_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deprecated compatibility shim for the former season-level goals-added join.

    Season aggregates can contain post-match information and are therefore not
    safe model inputs. Point-in-time rolling g+ is now created by
    ``models.frontier_features.add_frontier_features`` from rows carrying an
    ``available_at`` timestamp. This shim returns explicit missing-data values so
    an old caller cannot silently reintroduce target leakage.

    Columns added:
      - home_goals_added_avg : home team's mean player goals-added per season
      - away_goals_added_avg : away team's mean player goals-added per season
      - goals_added_edge     : home − away (positive favours home)

    The legacy implementation below is intentionally unreachable and retained
    temporarily only to keep historical blame context compact.
    """
    df = df.copy()
    df["home_goals_added_avg"] = 0.0
    df["away_goals_added_avg"] = 0.0
    df["goals_added_edge"] = 0.0
    df["goals_added_data_missing"] = 1
    return df


def add_archived_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Overlay genuinely archived selection/closing multi-book prices.

    The archive is populated before kickoff by ``automation/refresh_context.py``.
    Rows lacking event identity or a valid pre-kickoff ``available_at`` timestamp
    are ignored; neutral odds placeholders therefore never become market evidence.
    """
    archive_path = os.path.join(RAW_DIR, "multi_book_odds.csv")
    if not os.path.exists(archive_path):
        return df
    try:
        prices = pd.read_csv(archive_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return df
    required = {
        "commence_time", "home_team", "away_team", "snapshot_at", "available_at",
        "market", "selection", "american_odds",
    }
    if prices.empty or not required.issubset(prices.columns):
        return df
    prices = prices[prices["market"].astype(str).str.lower().isin({"h2h", "1x2"})].copy()
    prices["_kickoff"] = pd.to_datetime(prices["commence_time"], errors="coerce", utc=True)
    prices["_snapshot"] = pd.to_datetime(prices["snapshot_at"], errors="coerce", utc=True)
    prices["_available"] = pd.to_datetime(prices["available_at"], errors="coerce", utc=True)
    prices = prices[
        prices["_kickoff"].notna()
        & prices["_snapshot"].notna()
        & prices["_available"].notna()
        & (prices["_available"] < prices["_kickoff"])
        & (prices["_snapshot"] < prices["_kickoff"])
    ].copy()
    if prices.empty:
        return df
    prices["_date"] = prices["_kickoff"].dt.date.astype(str)
    prices["_home"] = prices["home_team"].astype(str).map(normalize_team_name)
    prices["_away"] = prices["away_team"].astype(str).map(normalize_team_name)
    selection_names = prices["selection"].astype(str).map(normalize_team_name).str.lower()
    prices["_side"] = np.where(
        (selection_names == prices["_home"].str.lower()) | (selection_names == "home"), "home",
        np.where(
            (selection_names == prices["_away"].str.lower()) | (selection_names == "away"),
            "away",
            np.where(selection_names == "draw", "draw", ""),
        ),
    )
    prices = prices[prices["_side"].isin({"home", "draw", "away"})]
    prices["_decimal"] = pd.to_numeric(prices["american_odds"], errors="coerce").map(
        lambda value: _american_to_decimal(float(value)) if pd.notna(value) else np.nan
    )
    prices = prices[prices["_decimal"] > 1.0]
    closing_flag = pd.to_numeric(
        prices.get("is_closing", pd.Series(0, index=prices.index)), errors="coerce"
    ).fillna(0).astype(bool)
    keys = ["_date", "_home", "_away", "_side"]
    selection_prices = prices.sort_values("_snapshot").groupby(keys, as_index=False).first()
    closing_prices = prices[closing_flag].sort_values("_snapshot").groupby(keys, as_index=False).last()
    selection_wide = selection_prices.pivot(index=["_date", "_home", "_away"], columns="_side", values="_decimal")
    closing_wide = closing_prices.pivot(index=["_date", "_home", "_away"], columns="_side", values="_decimal") if not closing_prices.empty else pd.DataFrame()

    output = df.copy()
    match_dates = pd.to_datetime(output["MatchDate"], errors="coerce").dt.date.astype(str)
    match_keys = list(zip(
        match_dates,
        output["HomeTeam"].astype(str).map(normalize_team_name),
        output["AwayTeam"].astype(str).map(normalize_team_name),
    ))
    available = []
    for index, key in zip(output.index, match_keys):
        if key not in selection_wide.index:
            available.append(0)
            continue
        selection_row = selection_wide.loc[key]
        close_row = closing_wide.loc[key] if not closing_wide.empty and key in closing_wide.index else selection_row
        if not all(pd.notna(selection_row.get(side)) for side in ("home", "draw", "away")):
            available.append(0)
            continue
        closing_values: dict[str, float] = {}
        for side in ("home", "draw", "away"):
            output.loc[index, f"odds_{side}_value"] = float(selection_row[side])
            closing_candidate = close_row.get(side, np.nan)
            closing_values[side] = float(closing_candidate) if pd.notna(closing_candidate) else float(selection_row[side])
            output.loc[index, f"closing_{side}_value"] = closing_values[side]
        selection_baseline = np.array([float(selection_row[side]) for side in ("home", "draw", "away")])
        selection_implied = 1.0 / selection_baseline
        selection_implied /= selection_implied.sum()
        output.loc[index, [
            "selection_implied_home_prob", "selection_implied_draw_prob", "selection_implied_away_prob"
        ]] = selection_implied
        baseline = np.array([closing_values[side] for side in ("home", "draw", "away")])
        implied = 1.0 / baseline
        implied /= implied.sum()
        output.loc[index, ["odds_implied_home_prob", "odds_implied_draw_prob", "odds_implied_away_prob"]] = implied
        output.loc[index, "odds_market_margin"] = float((1.0 / baseline).sum())
        available.append(1)
    output["odds_data_available"] = available
    return output

    # Legacy implementation retained below for source-history context only.
    # It is not executable because season-wide player snapshots are not
    # point-in-time safe.
    ga_path = os.path.join(RAW_DIR, "asa_player_goals_added.csv")

    if not os.path.exists(ga_path):
        df["home_goals_added_avg"] = 0.0
        df["away_goals_added_avg"] = 0.0
        df["goals_added_edge"]     = 0.0
        return df

    try:
        ga_raw = pd.read_csv(ga_path)

        # Identify the team and season columns (ASA schema varies slightly by version)
        team_col   = next((c for c in ["team_name", "team", "Team"] if c in ga_raw.columns), None)
        season_col = next((c for c in ["season_name", "season", "Season"] if c in ga_raw.columns), None)
        # Goals-added total: prefer goals_added_total, then goals_added
        ga_col     = next(
            (c for c in ["goals_added_total", "goals_added", "g+"] if c in ga_raw.columns),
            None,
        )

        # itscalledsoccer v2 returns one nested action list per player. Flatten
        # that schema and map team_id through the team xG snapshot.
        if ga_col is None and "data" in ga_raw.columns:
            def _sum_goals_added(value: object) -> float:
                try:
                    actions = ast.literal_eval(str(value))
                    return float(sum(float(action.get("goals_added_above_avg", action.get("goals_added_raw", 0))) for action in actions))
                except (ValueError, SyntaxError, TypeError):
                    return 0.0

            ga_raw["goals_added_total"] = ga_raw["data"].apply(_sum_goals_added)
            ga_col = "goals_added_total"
        if team_col is None and "team_id" in ga_raw.columns:
            team_xg_path = os.path.join(RAW_DIR, "asa_team_xg.csv")
            if os.path.exists(team_xg_path):
                team_xg = pd.read_csv(team_xg_path)
                if {"team_id", "team_name"}.issubset(team_xg.columns):
                    team_map = team_xg.drop_duplicates("team_id").set_index("team_id")["team_name"]
                    ga_raw["team_name"] = ga_raw["team_id"].map(team_map)
                    team_col = "team_name"

        if team_col is None or ga_col is None:
            raise ValueError("Required columns not found in ASA player goals-added file.")

        ga_raw[ga_col] = pd.to_numeric(ga_raw[ga_col], errors="coerce").fillna(0)

        # Normalise team names for joining
        ga_raw["team_norm"] = ga_raw[team_col].apply(normalize_team_name)

        group_cols = ["team_norm"]
        if season_col:
            ga_raw[season_col] = pd.to_numeric(ga_raw[season_col], errors="coerce")
            group_cols.append(season_col)

        team_ga = (
            ga_raw.groupby(group_cols)[ga_col]
            .mean()
            .rename("goals_added_avg")
            .reset_index()
        )

        df["season_year"] = df["MatchDate"].dt.year
        df["home_norm"]   = df["HomeTeam"].apply(normalize_team_name)
        df["away_norm"]   = df["AwayTeam"].apply(normalize_team_name)

        if season_col in team_ga.columns:
            home_merge = df.merge(
                team_ga.rename(columns={"team_norm": "home_norm", season_col: "season_year",
                                        "goals_added_avg": "home_goals_added_avg"}),
                on=["home_norm", "season_year"], how="left",
            )
            merged = home_merge.merge(
                team_ga.rename(columns={"team_norm": "away_norm", season_col: "season_year",
                                        "goals_added_avg": "away_goals_added_avg"}),
                on=["away_norm", "season_year"], how="left",
            )
        else:
            team_ga_simple = team_ga.set_index("team_norm")["goals_added_avg"]
            df["home_goals_added_avg"] = df["home_norm"].map(team_ga_simple)
            df["away_goals_added_avg"] = df["away_norm"].map(team_ga_simple)
            merged = df

        df["home_goals_added_avg"] = (
            merged.get("home_goals_added_avg", pd.Series(dtype=float))
            .fillna(0.0).values
        )
        df["away_goals_added_avg"] = (
            merged.get("away_goals_added_avg", pd.Series(dtype=float))
            .fillna(0.0).values
        )

    except Exception as exc:
        print(f"  [WARN] Transfer market features failed: {exc} — defaulting to 0.")
        df["home_goals_added_avg"] = 0.0
        df["away_goals_added_avg"] = 0.0

    df["goals_added_edge"] = (
        df["home_goals_added_avg"] - df["away_goals_added_avg"]
    ).round(4)

    # Drop temp columns if they crept in
    for c in ["season_year", "home_norm", "away_norm"]:
        if c in df.columns:
            df = df.drop(columns=[c])

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

    # 7. Season recency weight
    print("Step 6: Adding season recency weights…")
    df_clean = add_season_weight(df_clean)

    # 8b. Venue-specific stadium features
    print("Step 8b: Adding venue / stadium capacity features…")
    df_clean = add_venue_features(df_clean)

    # 8c. Enhanced historical odds features
    print("Step 8c: Adding enhanced historical odds features…")
    df_clean = add_enhanced_odds_features(df_clean)
    df_clean = add_archived_market_features(df_clean)

    # 8d. Frontier point-in-time context: altitude, time zones, roster mechanisms,
    # tactical/event quality, coaching/expansion priors, GPAA, attendance and more.
    print("Step 8d: Adding frontier point-in-time MLS features…")
    frontier_sources = load_optional_sources(RAW_DIR)
    df_clean = add_frontier_features(df_clean, frontier_sources)

    # 8e. Derived rest-day features (back-to-back flag, rest advantage)
    if "home_rest_days" in df_clean.columns and "away_rest_days" in df_clean.columns:
        df_clean["home_is_back_to_back"] = (df_clean["home_rest_days"] <= 3).astype(int)
        df_clean["away_is_back_to_back"] = (df_clean["away_rest_days"] <= 3).astype(int)
        df_clean["rest_advantage"] = df_clean["home_rest_days"] - df_clean["away_rest_days"]
        print("Step 9: Added back-to-back flags and rest_advantage feature.")

    # 9. Final clean-up: numeric conversion, fill NaN for model features
    feature_cols = [c for c in df_clean.columns if c not in (
        "MatchDate", "HomeTeam", "AwayTeam", "Result", "Source",
        "HomeConference", "AwayConference", "Season", "game_id", "FD_ID",
        "Matchday", "status", "competition_phase",
    )]
    for col in feature_cols:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        except (TypeError, ValueError):
            pass

    # Fill NaN in rolling features with column medians
    rolling_cols = [c for c in df_clean.columns if any(
        x in c for x in ["_l5", "_l10", "rest_days", "h2h", "playoff",
                          "back_to_back", "rest_advantage"]
    )]
    for col in rolling_cols:
        values = df_clean[col].dropna()
        med = values.median() if not values.empty else 0.0
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
