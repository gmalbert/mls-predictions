"""Roadmap analytics: playoffs, players, form, xG timing, and regression tables."""
from __future__ import annotations

from collections import defaultdict
from math import exp
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson

from models.frontier_features import STADIUMS, haversine_km
from team_name_mapping import normalize_team_name


def series_win_probability(game_win_probability: float, games_to_win: int = 2) -> float:
    """Probability of winning a best-of-three first-round series."""
    if games_to_win != 2:
        raise ValueError("The current MLS implementation models a best-of-three series.")
    probability = float(np.clip(game_win_probability, 0.0, 1.0))
    return probability**2 + 2.0 * probability**2 * (1.0 - probability)


def compute_gpaa(keeper_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute goals prevented above average per 90, preferring post-shot xG."""
    frame = keeper_stats.copy()
    minutes = pd.to_numeric(frame.get("minutes_played", 0), errors="coerce").fillna(0).clip(lower=1)
    conceded = pd.to_numeric(frame.get("goals_conceded", 0), errors="coerce").fillna(0)
    if "xg_on_target_faced" in frame.columns:
        expected = pd.to_numeric(frame["xg_on_target_faced"], errors="coerce").fillna(0)
    else:
        shots = pd.to_numeric(frame.get("shots_on_target_faced", 0), errors="coerce").fillna(0)
        expected = shots * 0.32
    frame["expected_to_concede"] = expected
    frame["gpaa_90"] = ((expected - conceded) / minutes * 90.0).round(4)
    return frame.sort_values("gpaa_90", ascending=False)


def player_prop_probabilities(players: pd.DataFrame, team_expected_goals: float) -> pd.DataFrame:
    """Estimate goal/assist probabilities from target share and per-shot quality.

    This produces research estimates only. Product betting decisions keep player
    props disabled until a sufficiently large timestamped prop ledger exists.
    """
    frame = players.copy()
    shots_90 = pd.to_numeric(frame.get("shots_per_90", 0), errors="coerce").fillna(0).clip(lower=0)
    xg_shot = pd.to_numeric(frame.get("xg_per_shot", 0.1), errors="coerce").fillna(0.1).clip(0, 1)
    minutes = pd.to_numeric(frame.get("projected_minutes", 75), errors="coerce").fillna(75).clip(0, 120)
    target_share = pd.to_numeric(frame.get("target_share", 0), errors="coerce").fillna(0).clip(0, 1)
    raw_xg = shots_90 * xg_shot * minutes / 90.0
    if raw_xg.sum() > 0:
        allocated = raw_xg / raw_xg.sum() * max(float(team_expected_goals), 0.0)
    else:
        allocated = target_share / max(target_share.sum(), 1.0) * max(float(team_expected_goals), 0.0)
    xa_90 = pd.to_numeric(frame.get("xa_per_90", 0), errors="coerce").fillna(0).clip(lower=0)
    frame["projected_xg"] = allocated.round(4)
    frame["anytime_goal_probability"] = (1.0 - np.exp(-allocated)).round(4)
    frame["anytime_assist_probability"] = (1.0 - np.exp(-xa_90 * minutes / 90.0)).round(4)
    odds_col = next((column for column in ("draftkings_odds", "anytime_goal_odds", "odds") if column in frame.columns), None)
    if odds_col:
        odds = pd.to_numeric(frame[odds_col], errors="coerce")
        implied = np.where(
            odds <= -100,
            np.abs(odds) / (np.abs(odds) + 100.0),
            np.where(odds >= 100, 100.0 / (odds + 100.0), np.where(odds > 1.0, 1.0 / odds, np.nan)),
        )
        frame["draftkings_implied_probability"] = np.round(implied, 4)
        frame["research_edge"] = (
            frame["anytime_goal_probability"] - frame["draftkings_implied_probability"]
        ).round(4)
        frame["research_value_over_3pct"] = frame["research_edge"] > 0.03
    frame["market_status"] = "RESEARCH ONLY — PROP BETTING DISABLED"
    return frame.sort_values("anytime_goal_probability", ascending=False)


def expected_points_from_xg(home_xg: float, away_xg: float, max_goals: int = 9) -> tuple[float, float]:
    home = poisson.pmf(np.arange(max_goals + 1), max(float(home_xg), 0.05))
    away = poisson.pmf(np.arange(max_goals + 1), max(float(away_xg), 0.05))
    grid = np.outer(home, away)
    grid /= grid.sum()
    home_win = float(np.tril(grid, -1).sum())
    draw = float(np.trace(grid))
    away_win = float(np.triu(grid, 1).sum())
    return 3.0 * home_win + draw, 3.0 * away_win + draw


def expected_points_table(matches: pd.DataFrame) -> pd.DataFrame:
    records: dict[str, dict[str, float]] = defaultdict(lambda: {"played": 0, "actual_points": 0.0, "expected_points": 0.0})
    for row in matches.itertuples(index=False):
        home, away = normalize_team_name(str(row.HomeTeam)), normalize_team_name(str(row.AwayTeam))
        home_goals = float(getattr(row, "HomeGoals", 0) or 0)
        away_goals = float(getattr(row, "AwayGoals", 0) or 0)
        home_xg = float(getattr(row, "home_xgoals", home_goals) or home_goals)
        away_xg = float(getattr(row, "away_xgoals", away_goals) or away_goals)
        home_xpts, away_xpts = expected_points_from_xg(home_xg, away_xg)
        records[home]["played"] += 1
        records[away]["played"] += 1
        records[home]["expected_points"] += home_xpts
        records[away]["expected_points"] += away_xpts
        if home_goals > away_goals:
            records[home]["actual_points"] += 3
        elif home_goals < away_goals:
            records[away]["actual_points"] += 3
        else:
            records[home]["actual_points"] += 1
            records[away]["actual_points"] += 1
    frame = pd.DataFrame([{"Team": team, **values} for team, values in records.items()])
    if frame.empty:
        return frame
    frame["xpts_delta"] = frame["actual_points"] - frame["expected_points"]
    frame["regression_signal"] = np.where(frame["xpts_delta"] > 4, "Likely pullback", np.where(frame["xpts_delta"] < -4, "Bounce-back candidate", "In line"))
    return frame.sort_values("expected_points", ascending=False).reset_index(drop=True)


def conference_table(matches: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    frame = matches.copy()
    frame["MatchDate"] = pd.to_datetime(frame["MatchDate"], errors="coerce")
    if season is None and frame["MatchDate"].notna().any():
        season = int(frame["MatchDate"].dt.year.max())
    if season is not None:
        frame = frame[frame["MatchDate"].dt.year == season]
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0})
    for row in frame.itertuples(index=False):
        home, away = normalize_team_name(str(row.HomeTeam)), normalize_team_name(str(row.AwayTeam))
        hg, ag = int(getattr(row, "HomeGoals", 0) or 0), int(getattr(row, "AwayGoals", 0) or 0)
        for team, goals_for, goals_against in ((home, hg, ag), (away, ag, hg)):
            stats[team]["P"] += 1
            stats[team]["GF"] += goals_for
            stats[team]["GA"] += goals_against
        if hg > ag:
            stats[home]["W"] += 1; stats[home]["Pts"] += 3; stats[away]["L"] += 1
        elif hg < ag:
            stats[away]["W"] += 1; stats[away]["Pts"] += 3; stats[home]["L"] += 1
        else:
            stats[home]["D"] += 1; stats[away]["D"] += 1; stats[home]["Pts"] += 1; stats[away]["Pts"] += 1
    output = pd.DataFrame([{"Team": team, "Conference": STADIUMS.get(team).conference if STADIUMS.get(team) else "Unknown", **values} for team, values in stats.items()])
    if output.empty:
        return output
    output["GD"] = output["GF"] - output["GA"]
    output = output.sort_values(["Conference", "Pts", "GD", "GF"], ascending=[True, False, False, False])
    output["Position"] = output.groupby("Conference").cumcount() + 1
    output["Playoff"] = output["Position"] <= 9
    return output.reset_index(drop=True)


def playoff_simulation(
    standings: pd.DataFrame,
    ratings: Mapping[str, float] | None = None,
    remaining_fixtures: pd.DataFrame | None = None,
    simulations: int = 10_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Simulate conference brackets and MLS Cup from current seeds."""
    if standings.empty:
        return pd.DataFrame(columns=["Team", "make_playoffs", "conference_win", "mls_cup"])
    rng = np.random.default_rng(random_seed)
    ratings = dict(ratings or {})
    teams = standings.sort_values(["Conference", "Position"])
    counters = {team: np.zeros(3, dtype=float) for team in teams["Team"]}
    remaining = remaining_fixtures.copy() if remaining_fixtures is not None else pd.DataFrame()
    if not remaining.empty:
        required = {"HomeTeam", "AwayTeam"}
        if not required.issubset(remaining.columns):
            remaining = pd.DataFrame()
        else:
            remaining["HomeTeam"] = remaining["HomeTeam"].astype(str).map(normalize_team_name)
            remaining["AwayTeam"] = remaining["AwayTeam"].astype(str).map(normalize_team_name)

    def win_probability(team_a: str, team_b: str) -> float:
        return 1.0 / (1.0 + 10 ** ((ratings.get(team_b, 1_500.0) - ratings.get(team_a, 1_500.0)) / 400.0))

    def play(team_a: str, team_b: str, series: bool = False) -> str:
        probability = win_probability(team_a, team_b)
        if series:
            probability = series_win_probability(probability)
        return team_a if rng.random() < probability else team_b

    def regular_season_result(home: str, away: str) -> str:
        home_rating = ratings.get(home, 1_500.0) + 60.0
        away_rating = ratings.get(away, 1_500.0)
        decisive_home = 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))
        draw_probability = 0.26 * exp(-abs(home_rating - away_rating) / 650.0)
        random_value = rng.random()
        home_probability = decisive_home * (1.0 - draw_probability)
        if random_value < home_probability:
            return "H"
        if random_value < home_probability + draw_probability:
            return "D"
        return "A"

    for _ in range(max(int(simulations), 1)):
        simulated_points = {
            str(row.Team): float(row.Pts) for row in teams.itertuples(index=False)
        }
        if not remaining.empty:
            for fixture in remaining.itertuples(index=False):
                home, away = str(fixture.HomeTeam), str(fixture.AwayTeam)
                if home not in simulated_points or away not in simulated_points:
                    continue
                outcome = regular_season_result(home, away)
                if outcome == "H":
                    simulated_points[home] += 3
                elif outcome == "A":
                    simulated_points[away] += 3
                else:
                    simulated_points[home] += 1
                    simulated_points[away] += 1
        conference_champions: list[str] = []
        for conference in ("Eastern", "Western"):
            conference_teams = teams[teams["Conference"] == conference]["Team"].tolist()
            seeded = sorted(
                conference_teams,
                key=lambda team: (simulated_points.get(team, 0.0), ratings.get(team, 1_500.0)),
                reverse=True,
            )[:9]
            if not seeded:
                continue
            for team in seeded:
                counters[team][0] += 1
            if len(seeded) >= 9:
                wild_card = play(seeded[7], seeded[8])
                bracket = [seeded[0], wild_card, seeded[3], seeded[4], seeded[2], seeded[5], seeded[1], seeded[6]]
            else:
                bracket = seeded[:8]
            while len(bracket) > 1:
                next_round = []
                for index in range(0, len(bracket) - 1, 2):
                    next_round.append(play(bracket[index], bracket[index + 1], series=len(bracket) == 8))
                if len(bracket) % 2:
                    next_round.append(bracket[-1])
                bracket = next_round
            if bracket:
                champion = bracket[0]
                counters[champion][1] += 1
                conference_champions.append(champion)
        if len(conference_champions) == 2:
            champion = play(conference_champions[0], conference_champions[1])
            counters[champion][2] += 1
    return pd.DataFrame(
        [
            {
                "Team": team,
                "make_playoffs": values[0] / simulations,
                "conference_win": values[1] / simulations,
                "mls_cup": values[2] / simulations,
            }
            for team, values in counters.items()
        ]
    ).sort_values("mls_cup", ascending=False)


def form_heatmap(matches: pd.DataFrame, weeks: int = 20) -> pd.DataFrame:
    frame = matches.copy()
    frame["MatchDate"] = pd.to_datetime(frame["MatchDate"], errors="coerce")
    frame = frame.dropna(subset=["MatchDate"]).sort_values("MatchDate")
    frame["week"] = frame["MatchDate"].dt.isocalendar().week.astype(int)
    rows = []
    for row in frame.itertuples(index=False):
        result = str(getattr(row, "Result", ""))
        hg, ag = float(getattr(row, "HomeGoals", 0) or 0), float(getattr(row, "AwayGoals", 0) or 0)
        margin = min(abs(hg - ag), 4.0)
        home_value = (1 + margin / 4) if result == "H" else (-1 - margin / 4 if result == "A" else 0)
        rows.extend([
            {"Team": normalize_team_name(str(row.HomeTeam)), "Week": int(row.week), "Form": home_value},
            {"Team": normalize_team_name(str(row.AwayTeam)), "Week": int(row.week), "Form": -home_value},
        ])
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    recent = sorted(output["Week"].unique())[-weeks:]
    return output[output["Week"].isin(recent)].pivot_table(index="Team", columns="Week", values="Form", aggfunc="mean").fillna(0)


def xg_timeline(events: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    frame = events.copy()
    team_col = next((c for c in ("team", "Team", "team_name") if c in frame.columns), None)
    minute_col = next((c for c in ("minute", "event_minute") if c in frame.columns), None)
    xg_col = next((c for c in ("xg", "shot_xg") if c in frame.columns), None)
    if team_col is None or minute_col is None or xg_col is None:
        return pd.DataFrame(columns=["Team", "time_bucket", "shots", "xg", "xg_share"])
    frame["Team"] = frame[team_col].astype(str).map(normalize_team_name)
    if team:
        frame = frame[frame["Team"] == normalize_team_name(team)]
    minutes = pd.to_numeric(frame[minute_col], errors="coerce").fillna(0).clip(0, 120)
    frame["time_bucket"] = pd.cut(minutes, [0, 15, 30, 45, 60, 75, 90, 120], include_lowest=True).astype(str)
    frame["_xg"] = pd.to_numeric(frame[xg_col], errors="coerce").fillna(0)
    output = frame.groupby(["Team", "time_bucket"], observed=True).agg(shots=("_xg", "size"), xg=("_xg", "sum")).reset_index()
    totals = output.groupby("Team")["xg"].transform("sum").replace(0, np.nan)
    output["xg_share"] = (output["xg"] / totals).fillna(0)
    return output


def shot_location_zones(events: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    frame = events.copy()
    team_col = next((c for c in ("team", "Team", "team_name") if c in frame.columns), None)
    x_col = next((c for c in ("x", "shot_x") if c in frame.columns), None)
    y_col = next((c for c in ("y", "shot_y") if c in frame.columns), None)
    xg_col = next((c for c in ("xg", "shot_xg") if c in frame.columns), None)
    if None in (team_col, x_col, y_col, xg_col):
        return pd.DataFrame(columns=["Team", "Zone", "shots", "xg"])
    frame["Team"] = frame[team_col].astype(str).map(normalize_team_name)
    if team:
        frame = frame[frame["Team"] == normalize_team_name(team)]
    x = pd.to_numeric(frame[x_col], errors="coerce").fillna(50)
    y = pd.to_numeric(frame[y_col], errors="coerce").fillna(50)
    frame["Zone"] = np.select(
        [(x >= 83) & y.between(33, 67), (x >= 83) & (y < 33), (x >= 83) & (y > 67), x.between(67, 83)],
        ["Central box", "Left channel", "Right channel", "Zone 14"],
        default="Long range",
    )
    frame["_xg"] = pd.to_numeric(frame[xg_col], errors="coerce").fillna(0)
    return frame.groupby(["Team", "Zone"]).agg(shots=("_xg", "size"), xg=("_xg", "sum")).reset_index()


def travel_history(matches: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    frame = matches.copy()
    frame["MatchDate"] = pd.to_datetime(frame["MatchDate"], errors="coerce")
    if season is not None:
        frame = frame[frame["MatchDate"].dt.year == season]
    rows = [
        {"Team": normalize_team_name(str(row.AwayTeam)), "travel_miles": haversine_km(str(row.AwayTeam), str(row.HomeTeam)) * 0.621371}
        for row in frame.itertuples(index=False)
    ]
    if not rows:
        return pd.DataFrame(columns=["Team", "season_travel_miles", "long_haul_trips"])
    output = pd.DataFrame(rows)
    return output.groupby("Team").agg(season_travel_miles=("travel_miles", "sum"), long_haul_trips=("travel_miles", lambda values: int((values > 1_500).sum()))).reset_index()


def home_away_rates(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team in sorted(set(matches.get("HomeTeam", [])).union(matches.get("AwayTeam", []))):
        home = matches[matches["HomeTeam"] == team]
        away = matches[matches["AwayTeam"] == team]
        home_xg = pd.to_numeric(home.get("home_xgoals", home.get("HomeGoals")), errors="coerce").mean()
        away_xg = pd.to_numeric(away.get("away_xgoals", away.get("AwayGoals")), errors="coerce").mean()
        rows.append({"Team": team, "home_xg_pg": home_xg, "away_xg_pg": away_xg, "home_advantage_xg": home_xg - away_xg})
    return pd.DataFrame(rows).sort_values("home_advantage_xg", ascending=False)


def expansion_tracker(matches: pd.DataFrame) -> pd.DataFrame:
    frame = matches.copy()
    frame["MatchDate"] = pd.to_datetime(frame["MatchDate"], errors="coerce")
    rows = []
    for team, metadata in STADIUMS.items():
        first_year = frame[(frame["MatchDate"].dt.year == metadata.expansion_year) & ((frame["HomeTeam"] == team) | (frame["AwayTeam"] == team))]
        if first_year.empty:
            continue
        wins = ((first_year["HomeTeam"] == team) & (first_year["Result"] == "H")) | ((first_year["AwayTeam"] == team) & (first_year["Result"] == "A"))
        rows.append({"Team": team, "Expansion Year": metadata.expansion_year, "Matches": len(first_year), "Win Rate": wins.mean()})
    return pd.DataFrame(rows).sort_values("Expansion Year", ascending=False) if rows else pd.DataFrame()


def designated_player_impact(matches: pd.DataFrame) -> pd.DataFrame:
    required = {"home_dp_available", "away_dp_available"}
    if not required.issubset(matches.columns):
        return pd.DataFrame(columns=["Team", "DP State", "Matches", "Points Per Match"])
    rows = []
    for match in matches.itertuples(index=False):
        result = str(getattr(match, "Result", ""))
        for side in ("home", "away"):
            team = getattr(match, "HomeTeam" if side == "home" else "AwayTeam")
            availability = float(getattr(match, f"{side}_dp_available", 0.5))
            points = 3 if result == ("H" if side == "home" else "A") else (1 if result == "D" else 0)
            rows.append({"Team": team, "DP State": "Available" if availability >= 0.75 else ("Unavailable" if availability <= 0.25 else "Uncertain"), "Points": points})
    return pd.DataFrame(rows).groupby(["Team", "DP State"]).agg(Matches=("Points", "size"), **{"Points Per Match": ("Points", "mean")}).reset_index()


def all_star_probability(mls_rating: float, opponent_rating: float, motivation: float = 0.0) -> float:
    """Special-event All-Star estimate, kept separate from club training data."""
    adjusted_gap = float(mls_rating - opponent_rating + 80.0 * np.clip(motivation, -1, 1))
    return float(1.0 / (1.0 + exp(-adjusted_gap / 180.0)))
