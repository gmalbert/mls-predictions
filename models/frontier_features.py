"""Point-in-time MLS feature engineering for the frontier model.

The functions in this module deliberately accept optional, normalized data frames.
That keeps the training pipeline reproducible when a live provider is unavailable,
while still enforcing ``available_at < kickoff`` whenever contextual data exists.
Unknown information is represented by neutral values plus explicit missing flags;
it is never silently treated as a confirmed absence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import atan2, cos, exp, log1p, radians, sin, sqrt
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from team_name_mapping import normalize_team_name


@dataclass(frozen=True)
class Stadium:
    latitude: float
    longitude: float
    altitude_ft: int
    surface: str
    utc_offset: int
    conference: str
    dome: bool = False
    capacity: int = 22_000
    expansion_year: int = 1996


# Operational metadata is intentionally centralized here.  ``unknown`` is handled
# explicitly by the feature builder, so expansion/venue changes do not invent data.
STADIUMS: dict[str, Stadium] = {
    "Atlanta United": Stadium(33.7557, -84.4010, 1_050, "turf", -5, "Eastern", True, 42_500, 2017),
    "Austin FC": Stadium(30.3874, -97.7185, 620, "grass", -6, "Western", False, 20_738, 2021),
    "CF Montréal": Stadium(45.5623, -73.5517, 118, "grass", -5, "Eastern", False, 19_619, 2012),
    "Charlotte FC": Stadium(35.2258, -80.8528, 751, "turf", -5, "Eastern", False, 74_867, 2022),
    "Chicago Fire": Stadium(41.8623, -87.6167, 594, "grass", -6, "Eastern", False, 24_955, 1998),
    "Colorado Rapids": Stadium(39.8059, -104.8917, 5_200, "grass", -7, "Western", False, 18_061, 1996),
    "Columbus Crew": Stadium(39.9685, -83.0176, 725, "grass", -5, "Eastern", False, 20_371, 1996),
    "D.C. United": Stadium(38.8682, -77.0122, 20, "grass", -5, "Eastern", False, 20_000, 1996),
    "FC Cincinnati": Stadium(39.1110, -84.5260, 500, "grass", -5, "Eastern", False, 26_000, 2019),
    "FC Dallas": Stadium(33.1548, -97.0641, 650, "grass", -6, "Western", False, 19_096, 1996),
    "Houston Dynamo": Stadium(29.7524, -95.3513, 43, "grass", -6, "Western", False, 22_039, 2006),
    "Inter Miami CF": Stadium(25.9580, -80.2390, 10, "grass", -5, "Eastern", False, 21_550, 2020),
    "LA Galaxy": Stadium(33.8644, -118.2611, 46, "grass", -8, "Western", False, 27_000, 1996),
    "LAFC": Stadium(34.0131, -118.2845, 180, "grass", -8, "Western", False, 22_000, 2018),
    "Minnesota United": Stadium(44.9536, -93.1669, 889, "grass", -6, "Western", False, 19_400, 2017),
    "Nashville SC": Stadium(36.1306, -86.7715, 492, "grass", -6, "Eastern", False, 30_000, 2020),
    "New England Revolution": Stadium(42.0910, -71.2643, 269, "turf", -5, "Eastern", False, 20_000, 1996),
    "New York City FC": Stadium(40.8274, -73.9262, 55, "grass", -5, "Eastern", False, 30_321, 2015),
    "New York Red Bulls": Stadium(40.7369, -74.1503, 10, "grass", -5, "Eastern", False, 25_000, 1996),
    "Orlando City": Stadium(28.5411, -81.3894, 85, "grass", -5, "Eastern", False, 25_500, 2015),
    "Philadelphia Union": Stadium(39.8327, -75.3799, 10, "grass", -5, "Eastern", False, 18_500, 2010),
    "Portland Timbers": Stadium(45.5215, -122.6917, 50, "turf", -8, "Western", False, 25_218, 2011),
    "Real Salt Lake": Stadium(40.5829, -111.8929, 4_450, "grass", -7, "Western", False, 20_213, 2005),
    "San Diego FC": Stadium(32.7840, -117.1220, 46, "grass", -8, "Western", False, 35_000, 2025),
    "San Jose Earthquakes": Stadium(37.3512, -121.9253, 62, "grass", -8, "Western", False, 18_000, 1996),
    "Seattle Sounders": Stadium(47.5952, -122.3316, 20, "turf", -8, "Western", False, 40_000, 2009),
    "Sporting Kansas City": Stadium(39.1212, -94.8235, 900, "grass", -6, "Western", False, 18_467, 1996),
    "St. Louis City SC": Stadium(38.6328, -90.1924, 466, "grass", -6, "Western", False, 22_500, 2023),
    "Toronto FC": Stadium(43.6332, -79.4189, 250, "grass", -5, "Eastern", False, 30_000, 2007),
    "Vancouver Whitecaps": Stadium(49.2772, -123.1124, 7, "turf", -8, "Western", True, 22_120, 2011),
}

EASTERN_CONF = {team for team, stadium in STADIUMS.items() if stadium.conference == "Eastern"}
WESTERN_CONF = {team for team, stadium in STADIUMS.items() if stadium.conference == "Western"}
TURF_STADIUMS = {team for team, stadium in STADIUMS.items() if stadium.surface == "turf"}

OPTIONAL_SOURCE_FILES: Mapping[str, str] = {
    "availability": "roster_availability.csv",
    "transactions": "transactions.csv",
    "managers": "manager_tenures.csv",
    "events": "event_features.csv",
    "goalkeepers": "goalkeeper_stats.csv",
    "weather": "weather.csv",
    "attendance": "attendance.csv",
    "draft": "superdraft.csv",
    "ratings": "team_ratings.csv",
    "goals_added": "team_goals_added.csv",
    "lineups": "lineups.csv",
    "referees": "referee_crews.csv",
    "news": "news_sentiment.csv",
    "travel": "travel_estimates.csv",
}


def _stadium(team: str) -> Stadium | None:
    return STADIUMS.get(normalize_team_name(str(team)))


def haversine_km(team_a: str, team_b: str) -> float:
    """Great-circle distance between the clubs' primary stadiums."""
    a, b = _stadium(team_a), _stadium(team_b)
    if a is None or b is None:
        return 0.0
    lat1, lon1, lat2, lon2 = map(radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    arc = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6_371.0088 * 2 * atan2(sqrt(arc), sqrt(max(1.0 - arc, 0.0)))


def travel_load(km: float, tz_shift: float, rest_hours: float, altitude_gain: float) -> float:
    """League-specific travel load interaction from the frontier blueprint."""
    return float(
        log1p(max(km, 0.0))
        + 0.8 * abs(tz_shift)
        + 0.002 * max(altitude_gain, 0.0)
        - 0.03 * min(max(rest_hours, 0.0), 120.0)
    )


def validate_point_in_time(source: pd.DataFrame, kickoff_col: str = "MatchDate") -> pd.DataFrame:
    """Remove observations that were not knowable at prediction time.

    Sources without a valid ``available_at`` are retained for display/audit and
    tagged unknown; feature extraction excludes those rows so corrected statistics
    cannot be backfilled into the past.
    """
    if source.empty:
        return source.copy()
    out = source.copy()
    if kickoff_col in out.columns:
        out[kickoff_col] = pd.to_datetime(out[kickoff_col], errors="coerce", utc=True)
    if "available_at" not in out.columns:
        out["available_at_missing"] = 1
        return out
    out["available_at"] = pd.to_datetime(out["available_at"], errors="coerce", utc=True)
    out["available_at_missing"] = out["available_at"].isna().astype(int)
    if kickoff_col in out.columns:
        safe = out["available_at"].isna() | (out["available_at"] < out[kickoff_col])
        out = out.loc[safe].copy()
    return out


def load_optional_sources(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load normalized optional feeds without requiring any provider."""
    root = Path(raw_dir)
    sources: dict[str, pd.DataFrame] = {}
    for name, filename in OPTIONAL_SOURCE_FILES.items():
        file_path = root / filename
        try:
            frame = pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()
            if not frame.empty:
                frame = validate_point_in_time(frame, kickoff_col="__no_fixture_cutoff__")
            sources[name] = frame
        except (OSError, pd.errors.ParserError):
            sources[name] = pd.DataFrame()
    return sources


def _numeric(row: pd.Series, names: Iterable[str], default: float = np.nan) -> float:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return float(default)


def _fixture_cutoff(row: pd.Series) -> pd.Timestamp:
    raw = row.get("Kickoff", row.get("MatchDate"))
    timestamp = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.Timestamp("1970-01-01", tz="UTC")
    return timestamp


def _available_rows(source: pd.DataFrame, team: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if source.empty:
        return source
    frame = source.copy()
    team_col = next((c for c in ("team", "Team", "team_name") if c in frame.columns), None)
    if team_col and team:
        frame = frame[frame[team_col].astype(str).map(normalize_team_name) == normalize_team_name(team)]
    if "available_at" not in frame.columns:
        return frame.iloc[0:0].copy()
    available = pd.to_datetime(frame["available_at"], errors="coerce", utc=True)
    frame = frame[available.notna() & (available < cutoff)]
    event_col = next((c for c in ("event_date", "MatchDate", "date", "started_at") if c in frame.columns), None)
    if event_col:
        event_at = pd.to_datetime(frame[event_col], errors="coerce", utc=True)
        frame = frame[event_at.notna() & (event_at <= cutoff)]
    return frame


def _roster_context(source: pd.DataFrame, team: str, cutoff: pd.Timestamp) -> dict[str, float]:
    rows = _available_rows(source, team, cutoff)
    if rows.empty:
        return {
            "dp_available": 0.5,
            "u22_available": 0.5,
            "tam_available": 0.5,
            "international_absences": 0.0,
            "roster_missing_impact": 0.0,
            "roster_data_missing": 1.0,
        }
    sort_col = next((column for column in ("available_at", "event_date", "date") if column in rows.columns), None)
    if sort_col:
        rows = rows.assign(_sort=pd.to_datetime(rows[sort_col], errors="coerce", utc=True)).sort_values("_sort")
    player_col = next((column for column in ("player_id", "player", "player_name", "name") if column in rows.columns), None)
    if player_col:
        rows = rows.drop_duplicates(player_col, keep="last")
    else:
        snapshot_col = next((column for column in ("roster_mechanism", "mechanism", "designation") if column in rows.columns), None)
        if snapshot_col and "_sort" in rows.columns:
            latest_by_mechanism = rows.groupby(snapshot_col)["_sort"].transform("max")
            rows = rows[rows["_sort"] == latest_by_mechanism]
    mechanism_col = next((c for c in ("roster_mechanism", "mechanism", "designation") if c in rows.columns), None)
    status_col = next((c for c in ("status", "availability", "is_available") if c in rows.columns), None)
    minutes_col = next((c for c in ("projected_minutes", "minutes", "minutes_share") if c in rows.columns), None)
    value_col = next((c for c in ("replacement_value", "goals_added_90", "war") if c in rows.columns), None)
    if mechanism_col is None:
        rows = rows.assign(_mechanism="OTHER")
        mechanism_col = "_mechanism"
    mechanism = rows[mechanism_col].astype(str).str.upper()
    if status_col is None:
        available = pd.Series(0.5, index=rows.index)
        international_absences = 0.0
    else:
        raw = rows[status_col]
        reason_col = next((column for column in ("reason", "absence_reason", "detail") if column in rows.columns), None)
        international_text = raw.astype(str)
        if reason_col:
            international_text = international_text + " " + rows[reason_col].astype(str)
        international_absences = float(
            international_text.str.lower().str.contains("international|national team|country duty", regex=True).sum()
        )
        available = raw.map(
            lambda value: 1.0
            if str(value).lower() in {"1", "1.0", "true", "available", "confirmed", "probable"}
            else (0.5 if str(value).lower() in {"questionable", "unknown", "nan"} else 0.0)
        )
    minutes = pd.to_numeric(rows[minutes_col], errors="coerce").fillna(90.0) if minutes_col else pd.Series(90.0, index=rows.index)
    replacement = pd.to_numeric(rows[value_col], errors="coerce").fillna(0.0) if value_col else pd.Series(0.0, index=rows.index)
    weight = (minutes.clip(lower=0, upper=90) / 90.0) * replacement.abs()
    lost = ((1.0 - available) * weight).sum()

    def mechanism_availability(label: str) -> float:
        mask = mechanism.str.contains(label)
        if not mask.any():
            return 0.5
        denom = weight[mask].sum()
        return float((available[mask] * weight[mask]).sum() / denom) if denom > 0 else float(available[mask].mean())

    return {
        "dp_available": mechanism_availability("DP"),
        "u22_available": mechanism_availability("U22"),
        "tam_available": mechanism_availability("TAM"),
        "international_absences": international_absences,
        "roster_missing_impact": float(lost),
        "roster_data_missing": float(any(not mechanism.str.contains(label).any() for label in ("DP", "U22", "TAM"))),
    }


def _recent_count(source: pd.DataFrame, team: str, cutoff: pd.Timestamp, days: int) -> int:
    rows = _available_rows(source, team, cutoff)
    if rows.empty:
        return 0
    event_col = next((c for c in ("event_date", "date", "MatchDate") if c in rows.columns), None)
    if event_col is None:
        return int(len(rows))
    event_at = pd.to_datetime(rows[event_col], errors="coerce", utc=True)
    return int(((event_at >= cutoff - pd.Timedelta(days=days)) & (event_at <= cutoff)).sum())


def _latest_numeric(
    source: pd.DataFrame,
    team: str,
    cutoff: pd.Timestamp,
    candidates: Iterable[str],
    default: float = 0.0,
) -> float:
    rows = _available_rows(source, team, cutoff)
    if rows.empty:
        return default
    sort_col = next((c for c in ("available_at", "event_date", "date", "MatchDate") if c in rows.columns), None)
    if sort_col:
        rows = rows.assign(_sort=pd.to_datetime(rows[sort_col], errors="coerce", utc=True)).sort_values("_sort")
    for column in candidates:
        if column in rows.columns:
            values = pd.to_numeric(rows[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[-1])
    return default


def _recent_numeric_mean(
    source: pd.DataFrame,
    team: str,
    cutoff: pd.Timestamp,
    candidates: Iterable[str],
    limit: int = 10,
) -> tuple[float, float]:
    """Return a strictly available recent mean and an explicit missing flag."""
    rows = _available_rows(source, team, cutoff)
    if rows.empty:
        return 0.0, 1.0
    event_col = next((c for c in ("event_date", "date", "MatchDate", "available_at") if c in rows.columns), None)
    if event_col:
        rows = rows.assign(_event_at=pd.to_datetime(rows[event_col], errors="coerce", utc=True)).sort_values("_event_at")
    column = next((candidate for candidate in candidates if candidate in rows.columns), None)
    if column is None:
        return 0.0, 1.0
    values = pd.to_numeric(rows[column], errors="coerce").dropna().tail(limit)
    return (float(values.mean()), 0.0) if not values.empty else (0.0, 1.0)


def _superdraft_context(source: pd.DataFrame, team: str, cutoff: pd.Timestamp) -> tuple[float, float]:
    """Minutes/value-weighted rookie contribution with a 90-day integration ramp."""
    rows = _available_rows(source, team, cutoff)
    if rows.empty:
        return 0.0, 1.0
    date_col = next((c for c in ("signed_at", "draft_date", "event_date", "date") if c in rows.columns), None)
    impact_col = next((c for c in ("draft_impact", "pick_value", "projected_war", "rookie_minutes_share") if c in rows.columns), None)
    if date_col is None or impact_col is None:
        return 0.0, 1.0
    event_at = pd.to_datetime(rows[date_col], errors="coerce", utc=True)
    impact = pd.to_numeric(rows[impact_col], errors="coerce").fillna(0.0)
    minutes = pd.to_numeric(rows.get("rookie_minutes_share", 1.0), errors="coerce")
    if not isinstance(minutes, pd.Series):
        minutes = pd.Series(float(minutes), index=rows.index)
    minutes = minutes.fillna(0.0).clip(0.0, 1.0)
    days = (cutoff - event_at).dt.total_seconds().div(86_400.0).clip(lower=0.0)
    ramp = 1.0 - np.exp(-days / 90.0)
    valid = event_at.notna()
    if not valid.any():
        return 0.0, 1.0
    return float((impact[valid] * minutes[valid] * ramp[valid]).sum()), 0.0


def _manager_context(source: pd.DataFrame, team: str, cutoff: pd.Timestamp) -> tuple[float, float]:
    rows = _available_rows(source, team, cutoff)
    if rows.empty:
        return 180.0, 1.0
    start_col = next((c for c in ("started_at", "start_date", "event_date") if c in rows.columns), None)
    if start_col is None:
        return 180.0, 1.0
    starts = pd.to_datetime(rows[start_col], errors="coerce", utc=True).dropna()
    if starts.empty:
        return 180.0, 1.0
    tenure = max((cutoff - starts.max()).total_seconds() / 86_400.0, 0.0)
    return float(tenure), 0.0


def _event_context(source: pd.DataFrame, team: str, cutoff: pd.Timestamp) -> dict[str, float]:
    rows = _available_rows(source, team, cutoff)
    if rows.empty:
        return {
            "set_piece_xg": 0.0,
            "direct_attack_ratio": 0.0,
            "xgott_against": 0.0,
            "corners_earned": 0.0,
            "dangerous_free_kicks": 0.0,
            "set_piece_specialist_available": 0.5,
        }
    sort_col = next((column for column in ("event_date", "date", "available_at") if column in rows.columns), None)
    if sort_col:
        rows = rows.assign(_sort=pd.to_datetime(rows[sort_col], errors="coerce", utc=True)).sort_values("_sort")
    if len(rows) > 10:
        rows = rows.tail(10)
    result: dict[str, float] = {}
    for output, candidates in {
        "set_piece_xg": ("set_piece_xg", "set_piece_goals", "dead_ball_xg"),
        "direct_attack_ratio": ("direct_attack_ratio", "central_chain_share", "possession_chain_threat"),
        "xgott_against": ("xgott_against", "opponent_xgott", "xg_on_target_against"),
        "corners_earned": ("corners_earned", "corners", "corner_kicks"),
        "dangerous_free_kicks": ("dangerous_free_kicks", "free_kicks_final_third", "free_kick_xg"),
        "set_piece_specialist_available": ("set_piece_specialist_available", "specialist_available"),
    }.items():
        column = next((c for c in candidates if c in rows.columns), None)
        default = 0.5 if output == "set_piece_specialist_available" else 0.0
        result[output] = float(pd.to_numeric(rows[column], errors="coerce").mean()) if column else default
    return result


def _phase(row: pd.Series) -> str:
    competition = str(row.get("Competition", row.get("competition", ""))).lower()
    stage = str(row.get("Phase", row.get("stage", row.get("stage_name", "")))).lower()
    if "league" in competition and "cup" in competition:
        return "Leagues Cup"
    if "all-star" in competition or "all star" in competition:
        return "All-Star"
    if any(word in stage for word in ("playoff", "round", "conference final", "mls cup")):
        return "Playoffs"
    if "regular" in stage or "regular" in competition:
        return "Regular Season"
    return "Unknown"


def add_frontier_features(
    matches: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Add every MLS-specific roadmap feature with strict temporal ordering.

    The method is intentionally deterministic and provider-agnostic.  It consumes
    optional normalized feeds from :func:`load_optional_sources`; unavailable feeds
    produce neutral values and missingness flags so live predictions remain honest.
    """
    if matches.empty:
        return matches.copy()
    source_map = {name: frame.copy() for name, frame in (sources or {}).items()}
    for name in OPTIONAL_SOURCE_FILES:
        source_map.setdefault(name, pd.DataFrame())

    df = matches.copy()
    date_col = "MatchDate" if "MatchDate" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    df["HomeTeam"] = df["HomeTeam"].astype(str).map(normalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].astype(str).map(normalize_team_name)

    history: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    cross_conf_points: dict[tuple[str, int], list[float]] = defaultdict(list)
    club_ratings: dict[str, float] = defaultdict(lambda: 1_500.0)
    feature_rows: list[dict[str, float | str]] = []

    for _, row in df.iterrows():
        home, away = str(row["HomeTeam"]), str(row["AwayTeam"])
        cutoff = _fixture_cutoff(row)
        season = int(pd.Timestamp(row[date_col]).year) if pd.notna(row[date_col]) else datetime.now().year
        home_meta, away_meta = _stadium(home), _stadium(away)
        travel_km = haversine_km(away, home)
        altitude_gain = max((home_meta.altitude_ft if home_meta else 0) - (away_meta.altitude_ft if away_meta else 0), 0)
        tz_shift = (home_meta.utc_offset - away_meta.utc_offset) if home_meta and away_meta else 0
        home_hist, away_hist = history[(home, season)], history[(away, season)]
        home_rest_hours = max((cutoff - home_hist[-1]["date"]).total_seconds() / 3_600, 0.0) if home_hist else 120.0
        away_rest_hours = max((cutoff - away_hist[-1]["date"]).total_seconds() / 3_600, 0.0) if away_hist else 120.0

        def rolling(team_history: list[dict[str, float]], prefix: str) -> dict[str, float]:
            recent3 = team_history[-3:]
            recent10 = team_history[-10:]
            season_hist = team_history

            def mean(key: str, values: list[dict[str, float]]) -> float:
                selected = [v[key] for v in values if key in v and pd.notna(v[key])]
                return float(np.mean(selected)) if selected else 0.0

            diff3 = mean("xg_for", recent3) - mean("xg_against", recent3)
            diff_season = mean("xg_for", season_hist) - mean("xg_against", season_hist)
            return {
                f"{prefix}_xg_differential_momentum": diff3 - diff_season,
                f"{prefix}_xgott_against_l10": mean("xgott_against", recent10),
                f"{prefix}_set_piece_xg_l10": mean("set_piece_xg", recent10),
                f"{prefix}_direct_attack_ratio_l10": mean("direct_attack_ratio", recent10),
                f"{prefix}_home_xg_l10_split": mean("xg_for", [v for v in recent10 if v.get("was_home") == 1.0]),
                f"{prefix}_away_xg_l10_split": mean("xg_for", [v for v in recent10 if v.get("was_home") == 0.0]),
                f"{prefix}_xg_home_away_diff": (
                    mean("xg_for", [v for v in recent10 if v.get("was_home") == 1.0])
                    - mean("xg_for", [v for v in recent10 if v.get("was_home") == 0.0])
                ),
                f"{prefix}_matches_prior": float(len(season_hist)),
                f"{prefix}_early_season_reliability": float(len(season_hist) / (len(season_hist) + 5.0)),
            }

        home_roll = rolling(home_hist, "home")
        away_roll = rolling(away_hist, "away")
        home_roster = _roster_context(source_map["availability"], home, cutoff)
        away_roster = _roster_context(source_map["availability"], away, cutoff)
        home_event = _event_context(source_map["events"], home, cutoff)
        away_event = _event_context(source_map["events"], away, cutoff)
        home_goals_added, home_goals_added_missing = _recent_numeric_mean(
            source_map["goals_added"], home, cutoff, ("team_goals_added", "goals_added", "goals_added_total")
        )
        away_goals_added, away_goals_added_missing = _recent_numeric_mean(
            source_map["goals_added"], away, cutoff, ("team_goals_added", "goals_added", "goals_added_total")
        )
        home_tenure, home_manager_missing = _manager_context(source_map["managers"], home, cutoff)
        away_tenure, away_manager_missing = _manager_context(source_map["managers"], away, cutoff)
        home_conf = home_meta.conference if home_meta else "Unknown"
        away_conf = away_meta.conference if away_meta else "Unknown"
        home_conf_strength = float(np.mean(cross_conf_points[(home_conf, season)])) if cross_conf_points[(home_conf, season)] else 0.5
        away_conf_strength = float(np.mean(cross_conf_points[(away_conf, season)])) if cross_conf_points[(away_conf, season)] else 0.5
        temperature = _latest_numeric(source_map["weather"], home, cutoff, ("temperature_f", "temp_f"), 70.0)
        attendance = _latest_numeric(source_map["attendance"], home, cutoff, ("attendance",), 0.0)
        capacity = float(home_meta.capacity if home_meta else 22_000)
        attendance_ratio = min(max(attendance / capacity, 0.0), 1.25) if attendance > 0 else 0.75
        gpaa_home = _latest_numeric(source_map["goalkeepers"], home, cutoff, ("gpaa_90", "goals_prevented_90"), 0.0)
        gpaa_away = _latest_numeric(source_map["goalkeepers"], away, cutoff, ("gpaa_90", "goals_prevented_90"), 0.0)
        draft_home, draft_home_missing = _superdraft_context(source_map["draft"], home, cutoff)
        draft_away, draft_away_missing = _superdraft_context(source_map["draft"], away, cutoff)
        transfer_home = _latest_numeric(source_map["ratings"], home, cutoff, ("transfer_war_delta", "war_delta"), 0.0)
        transfer_away = _latest_numeric(source_map["ratings"], away, cutoff, ("transfer_war_delta", "war_delta"), 0.0)
        rating_home = _latest_numeric(source_map["ratings"], home, cutoff, ("club_rating", "elo"), club_ratings[home])
        rating_away = _latest_numeric(source_map["ratings"], away, cutoff, ("club_rating", "elo"), club_ratings[away])
        lineup_home = _latest_numeric(source_map["lineups"], home, cutoff, ("confirmed", "lineup_confirmed"), 0.0)
        lineup_away = _latest_numeric(source_map["lineups"], away, cutoff, ("confirmed", "lineup_confirmed"), 0.0)
        news_home = _latest_numeric(source_map["news"], home, cutoff, ("sentiment_signal",), 0.0)
        news_away = _latest_numeric(source_map["news"], away, cutoff, ("sentiment_signal",), 0.0)
        charter_away = _latest_numeric(source_map["travel"], away, cutoff, ("chartered", "charter_probability"), 0.5)
        referee_red_cards = _latest_numeric(source_map["referees"], "", cutoff, ("red_cards_per_game",), 0.18)
        referee_penalties = _latest_numeric(source_map["referees"], "", cutoff, ("penalties_per_game",), 0.24)
        weather_data_missing = float(_available_rows(source_map["weather"], home, cutoff).empty)
        goalkeeper_data_missing = float(
            _available_rows(source_map["goalkeepers"], home, cutoff).empty
            or _available_rows(source_map["goalkeepers"], away, cutoff).empty
        )
        attendance_data_missing = float(_available_rows(source_map["attendance"], home, cutoff).empty)
        rating_data_missing = float(
            _available_rows(source_map["ratings"], home, cutoff).empty
            or _available_rows(source_map["ratings"], away, cutoff).empty
        )
        lineup_data_missing = float(
            _available_rows(source_map["lineups"], home, cutoff).empty
            or _available_rows(source_map["lineups"], away, cutoff).empty
        )
        referee_data_missing = float(_available_rows(source_map["referees"], "", cutoff).empty)
        news_data_missing = float(
            _available_rows(source_map["news"], home, cutoff).empty
            or _available_rows(source_map["news"], away, cutoff).empty
        )
        expansion_prior_home = exp(-max(season - (home_meta.expansion_year if home_meta else season), 0) / 2.0)
        expansion_prior_away = exp(-max(season - (away_meta.expansion_year if away_meta else season), 0) / 2.0)
        coaching_prior_home = exp(-max(home_tenure, 0.0) / 90.0) if not home_manager_missing else 0.0
        coaching_prior_away = exp(-max(away_tenure, 0.0) / 90.0) if not away_manager_missing else 0.0
        shrunk_rating_home = 1_500.0 + (rating_home - 1_500.0) * (1.0 - 0.35 * expansion_prior_home) * (1.0 - 0.25 * coaching_prior_home)
        shrunk_rating_away = 1_500.0 + (rating_away - 1_500.0) * (1.0 - 0.35 * expansion_prior_away) * (1.0 - 0.25 * coaching_prior_away)
        phase = _phase(row)

        features: dict[str, float | str] = {
            **home_roll,
            **away_roll,
            "travel_km": travel_km,
            "away_travel_miles_frontier": travel_km * 0.621371,
            "time_zone_shift": float(abs(tz_shift)),
            "home_altitude_ft": float(home_meta.altitude_ft if home_meta else 0),
            "altitude_gain_ft": float(altitude_gain),
            "travel_load": travel_load(travel_km, tz_shift, away_rest_hours, altitude_gain),
            "travel_turnaround_interaction": float(log1p(travel_km) / max(away_rest_hours, 24.0)),
            "travel_surface_interaction": float(
                log1p(travel_km) * bool(away_meta and home_meta and away_meta.surface != home_meta.surface)
            ),
            "travel_temperature_interaction": float(
                log1p(travel_km) * abs(temperature - 70.0) / 30.0
                if home_meta and not home_meta.dome else 0.0
            ),
            "turf_to_grass_visitor": float(bool(away_meta and home_meta and away_meta.surface == "turf" and home_meta.surface == "grass")),
            "grass_to_turf_visitor": float(bool(away_meta and home_meta and away_meta.surface == "grass" and home_meta.surface == "turf")),
            "home_surface_turf": float(bool(home_meta and home_meta.surface == "turf")),
            "home_is_dome": float(bool(home_meta and home_meta.dome)),
            "temperature_f": temperature,
            "extreme_weather_outdoor": float(bool(home_meta and not home_meta.dome and (temperature < 35 or temperature > 90))),
            "home_rest_hours": home_rest_hours,
            "away_rest_hours": away_rest_hours,
            "home_short_rest": float(home_rest_hours <= 72),
            "away_short_rest": float(away_rest_hours <= 72),
            "is_cross_conference_frontier": float(home_conf != away_conf and "Unknown" not in (home_conf, away_conf)),
            "conference_strength_edge": home_conf_strength - away_conf_strength,
            "home_manager_tenure_days": home_tenure,
            "away_manager_tenure_days": away_tenure,
            "manager_tenure_data_missing": max(home_manager_missing, away_manager_missing),
            "home_expansion_prior": expansion_prior_home,
            "away_expansion_prior": expansion_prior_away,
            "home_coaching_regime_prior": coaching_prior_home,
            "away_coaching_regime_prior": coaching_prior_away,
            "hierarchical_rating_edge": shrunk_rating_home - shrunk_rating_away,
            "home_mid_season_roster_changes": float(_recent_count(source_map["transactions"], home, cutoff, 30)),
            "away_mid_season_roster_changes": float(_recent_count(source_map["transactions"], away, cutoff, 30)),
            "home_dp_available": home_roster["dp_available"],
            "away_dp_available": away_roster["dp_available"],
            "home_u22_available": home_roster["u22_available"],
            "away_u22_available": away_roster["u22_available"],
            "home_tam_available": home_roster["tam_available"],
            "away_tam_available": away_roster["tam_available"],
            "home_international_absences": home_roster["international_absences"],
            "away_international_absences": away_roster["international_absences"],
            "home_roster_missing_impact": home_roster["roster_missing_impact"],
            "away_roster_missing_impact": away_roster["roster_missing_impact"],
            "roster_data_missing": max(home_roster["roster_data_missing"], away_roster["roster_data_missing"]),
            "home_team_goals_added_l10": home_goals_added,
            "away_team_goals_added_l10": away_goals_added,
            "goals_added_data_missing": max(home_goals_added_missing, away_goals_added_missing),
            "xg_data_missing": float(
                pd.isna(pd.to_numeric(row.get("home_xgoals"), errors="coerce"))
                or pd.isna(pd.to_numeric(row.get("away_xgoals"), errors="coerce"))
            ),
            "home_set_piece_xg_context": home_event["set_piece_xg"],
            "away_set_piece_xg_context": away_event["set_piece_xg"],
            "set_piece_corners_edge": home_event["corners_earned"] - away_event["corners_earned"],
            "dangerous_free_kick_edge": home_event["dangerous_free_kicks"] - away_event["dangerous_free_kicks"],
            "set_piece_specialist_edge": home_event["set_piece_specialist_available"] - away_event["set_piece_specialist_available"],
            "tactical_matchup_edge": home_event["direct_attack_ratio"] - away_event["direct_attack_ratio"],
            "home_xgott_against_context": home_event["xgott_against"],
            "away_xgott_against_context": away_event["xgott_against"],
            "home_gpaa_90": gpaa_home,
            "away_gpaa_90": gpaa_away,
            "gpaa_edge": gpaa_home - gpaa_away,
            "home_superdraft_integration": draft_home,
            "away_superdraft_integration": draft_away,
            "superdraft_data_missing": max(draft_home_missing, draft_away_missing),
            "transfer_war_edge": transfer_home - transfer_away,
            "home_lineup_confirmed": lineup_home,
            "away_lineup_confirmed": lineup_away,
            "lineup_uncertainty": 1.0 - min(lineup_home, lineup_away),
            "news_sentiment_edge": news_home - news_away,
            "weather_data_missing": weather_data_missing,
            "goalkeeper_data_missing": goalkeeper_data_missing,
            "attendance_data_missing": attendance_data_missing,
            "rating_data_missing": rating_data_missing,
            "lineup_data_missing": lineup_data_missing,
            "referee_data_missing": referee_data_missing,
            "news_data_missing": news_data_missing,
            "away_charter_probability": charter_away,
            "travel_estimate_uncertainty": float(source_map["travel"].empty),
            "referee_red_cards_per_game": referee_red_cards,
            "referee_penalties_per_game": referee_penalties,
            "attendance_ratio": attendance_ratio,
            "attendance_home_advantage": (attendance_ratio - 0.75) * 0.08,
            "competition_phase": phase,
            "is_playoffs": float(phase == "Playoffs"),
            "is_leagues_cup": float(phase == "Leagues Cup"),
            "is_all_star": float(phase == "All-Star"),
        }
        feature_rows.append(features)

        # Outcome updates happen after all features for this fixture are frozen.
        home_goals = _numeric(row, ("HomeGoals", "home_goals", "FTHG"), 0.0)
        away_goals = _numeric(row, ("AwayGoals", "away_goals", "FTAG"), 0.0)
        home_xg = _numeric(row, ("home_xgoals", "HomeXG", "xg_home"), home_goals)
        away_xg = _numeric(row, ("away_xgoals", "AwayXG", "xg_away"), away_goals)
        event_home = {
            "set_piece_xg": _numeric(row, ("home_set_piece_xg",), home_event["set_piece_xg"]),
            "direct_attack_ratio": _numeric(row, ("home_direct_attack_ratio",), home_event["direct_attack_ratio"]),
            "xgott_against": _numeric(row, ("away_xgott", "home_xgott_against"), home_event["xgott_against"]),
        }
        event_away = {
            "set_piece_xg": _numeric(row, ("away_set_piece_xg",), away_event["set_piece_xg"]),
            "direct_attack_ratio": _numeric(row, ("away_direct_attack_ratio",), away_event["direct_attack_ratio"]),
            "xgott_against": _numeric(row, ("home_xgott", "away_xgott_against"), away_event["xgott_against"]),
        }
        history[(home, season)].append({
            "date": cutoff,
            "xg_for": home_xg,
            "xg_against": away_xg,
            "was_home": 1.0,
            **event_home,
        })
        history[(away, season)].append({
            "date": cutoff,
            "xg_for": away_xg,
            "xg_against": home_xg,
            "was_home": 0.0,
            **event_away,
        })
        expected_home = 1.0 / (1.0 + 10 ** ((club_ratings[away] - (club_ratings[home] + 60.0)) / 400.0))
        actual_home = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        rating_change = 20.0 * (actual_home - expected_home)
        club_ratings[home] += rating_change
        club_ratings[away] -= rating_change
        if home_conf != away_conf and "Unknown" not in (home_conf, away_conf):
            if home_goals > away_goals:
                cross_conf_points[(home_conf, season)].append(1.0)
                cross_conf_points[(away_conf, season)].append(0.0)
            elif home_goals < away_goals:
                cross_conf_points[(home_conf, season)].append(0.0)
                cross_conf_points[(away_conf, season)].append(1.0)
            else:
                cross_conf_points[(home_conf, season)].append(0.5)
                cross_conf_points[(away_conf, season)].append(0.5)

    feature_df = pd.DataFrame(feature_rows, index=df.index)
    for duplicate in set(feature_df.columns).intersection(df.columns):
        # Frontier values replace placeholders such as the old 0.5 DP flag.
        df = df.drop(columns=[duplicate])
    return pd.concat([df, feature_df], axis=1)


FRONTIER_NUMERIC_FEATURES = [
    "home_xg_differential_momentum",
    "away_xg_differential_momentum",
    "home_xg_home_away_diff",
    "away_xg_home_away_diff",
    "home_team_goals_added_l10",
    "away_team_goals_added_l10",
    "home_xgott_against_l10",
    "away_xgott_against_l10",
    "travel_load",
    "time_zone_shift",
    "home_altitude_ft",
    "altitude_gain_ft",
    "turf_to_grass_visitor",
    "grass_to_turf_visitor",
    "home_short_rest",
    "away_short_rest",
    "travel_turnaround_interaction",
    "travel_surface_interaction",
    "travel_temperature_interaction",
    "conference_strength_edge",
    "hierarchical_rating_edge",
    "home_early_season_reliability",
    "away_early_season_reliability",
    "home_expansion_prior",
    "away_expansion_prior",
    "home_coaching_regime_prior",
    "away_coaching_regime_prior",
    "home_manager_tenure_days",
    "away_manager_tenure_days",
    "manager_tenure_data_missing",
    "home_mid_season_roster_changes",
    "away_mid_season_roster_changes",
    "home_dp_available",
    "away_dp_available",
    "home_u22_available",
    "away_u22_available",
    "home_tam_available",
    "away_tam_available",
    "home_international_absences",
    "away_international_absences",
    "home_roster_missing_impact",
    "away_roster_missing_impact",
    "roster_data_missing",
    "goals_added_data_missing",
    "xg_data_missing",
    "home_set_piece_xg_l10",
    "away_set_piece_xg_l10",
    "set_piece_corners_edge",
    "dangerous_free_kick_edge",
    "set_piece_specialist_edge",
    "tactical_matchup_edge",
    "gpaa_edge",
    "home_superdraft_integration",
    "away_superdraft_integration",
    "superdraft_data_missing",
    "transfer_war_edge",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "lineup_uncertainty",
    "news_sentiment_edge",
    "away_charter_probability",
    "referee_red_cards_per_game",
    "referee_penalties_per_game",
    "attendance_home_advantage",
    "weather_data_missing",
    "goalkeeper_data_missing",
    "attendance_data_missing",
    "rating_data_missing",
    "lineup_data_missing",
    "referee_data_missing",
    "news_data_missing",
    "travel_estimate_uncertainty",
    "extreme_weather_outdoor",
    "is_playoffs",
    "is_leagues_cup",
    "is_all_star",
]
