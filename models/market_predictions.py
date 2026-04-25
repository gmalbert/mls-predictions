"""
models/market_predictions.py
Poisson-based betting market predictions for MLS fixtures.

Provides:
  - btts_probability(home_xg, away_xg) -> float
  - over_under_probabilities(home_xg, away_xg, thresholds) -> dict
  - correct_score_distribution(home_xg, away_xg, max_goals, rho) -> dict
  - top_correct_scores(home_xg, away_xg, top_n) -> list[tuple[str, float]]
  - fit_rho_from_historical(df) -> float
  - get_team_xg_averages(team_xg_df, lookback_games) -> dict
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

# MLS-specific surface calibration — turf stadiums have ~8% higher BTTS rate
TURF_BTTS_MULTIPLIER = 1.08

# Default Dixon-Coles rho (correlation correction for low scores).
# Negative: 0-0 and 1-1 are slightly less likely than pure independence predicts.
DEFAULT_RHO = -0.10


# ── Core Poisson utilities ─────────────────────────────────────────────────────

def btts_probability(
    home_xg: float,
    away_xg: float,
    home_is_turf: bool = False,
) -> float:
    """
    P(BTTS=Yes) = P(home scores ≥1) × P(away scores ≥1).
    Applies a surface multiplier when the home stadium uses artificial turf.
    """
    if home_xg <= 0 or away_xg <= 0:
        return 0.0
    home_xg_adj = home_xg * TURF_BTTS_MULTIPLIER if home_is_turf else home_xg
    p_home = 1.0 - poisson.pmf(0, home_xg_adj)
    p_away = 1.0 - poisson.pmf(0, away_xg)
    return round(float(p_home * p_away), 4)


def over_under_probabilities(
    home_xg: float,
    away_xg: float,
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    """
    P(total goals > threshold) using independent Poisson assumption.
    Returns a dict keyed like 'over_2.5', 'under_2.5', etc.
    """
    if thresholds is None:
        thresholds = [1.5, 2.5, 3.5]

    total_xg = max(home_xg + away_xg, 0.01)
    results: dict[str, float] = {}

    for t in thresholds:
        k_max = int(t)  # thresholds are half-integers, so floor = integer below
        p_under = float(sum(poisson.pmf(k, total_xg) for k in range(k_max + 1)))
        p_over = 1.0 - p_under
        key = str(t).replace(".", "_")
        results[f"over_{key}"] = round(p_over, 4)
        results[f"under_{key}"] = round(p_under, 4)

    return results


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def correct_score_distribution(
    home_xg: float,
    away_xg: float,
    max_goals: int = 6,
    rho: float = DEFAULT_RHO,
) -> dict[tuple[int, int], float]:
    """
    Full score probability matrix using Dixon-Coles correction.

    Returns:
        Dict mapping (home_goals, away_goals) → probability.
    """
    home_xg = max(home_xg, 0.01)
    away_xg = max(away_xg, 0.01)

    scores: dict[tuple[int, int], float] = {}
    total = 0.0

    for h, a in product(range(max_goals + 1), range(max_goals + 1)):
        p = (
            poisson.pmf(h, home_xg)
            * poisson.pmf(a, away_xg)
            * _tau(h, a, home_xg, away_xg, rho)
        )
        scores[(h, a)] = p
        total += p

    if total <= 0:
        return scores

    return {k: round(v / total, 5) for k, v in scores.items()}


def top_correct_scores(
    home_xg: float,
    away_xg: float,
    top_n: int = 5,
    rho: float = DEFAULT_RHO,
) -> list[tuple[str, float]]:
    """
    Return the N most-likely scorelines as (label, percent) tuples.
    Example: [('1-1', 14.2), ('2-1', 11.8), ...]
    """
    dist = correct_score_distribution(home_xg, away_xg, rho=rho)
    sorted_scores = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    return [(f"{h}-{a}", round(p * 100, 1)) for (h, a), p in sorted_scores[:top_n]]


# ── rho calibration ────────────────────────────────────────────────────────────

def fit_rho_from_historical(df: pd.DataFrame) -> float:
    """
    Fit Dixon-Coles rho on MLS historical data.
    Requires columns: home_xgoals, away_xgoals, HomeGoals (int), AwayGoals (int).
    Returns rho in [-0.5, 0.0]. Falls back to DEFAULT_RHO on failure.
    """
    required = {"home_xgoals", "away_xgoals", "HomeGoals", "AwayGoals"}
    if not required.issubset(df.columns):
        return DEFAULT_RHO

    sub = df[list(required)].dropna()
    if len(sub) < 50:
        return DEFAULT_RHO

    def neg_log_likelihood(rho: float) -> float:
        ll = 0.0
        for _, row in sub.iterrows():
            lam = max(float(row["home_xgoals"]), 0.01)
            mu = max(float(row["away_xgoals"]), 0.01)
            h = int(row["HomeGoals"])
            a = int(row["AwayGoals"])
            p = (
                poisson.pmf(h, lam)
                * poisson.pmf(a, mu)
                * _tau(h, a, lam, mu, rho)
            )
            ll += np.log(max(p, 1e-12))
        return -ll

    try:
        result = minimize_scalar(
            neg_log_likelihood, bounds=(-0.5, 0.0), method="bounded"
        )
        return float(result.x) if result.success else DEFAULT_RHO
    except Exception:
        return DEFAULT_RHO


# ── xG lookup helpers ──────────────────────────────────────────────────────────

def get_team_xg_averages(
    team_xg_df: pd.DataFrame,
    name_col: str = "team_name",
) -> dict[str, dict[str, float]]:
    """
    Build a lookup dict: canonical_team_name → {xg_for_pg, xga_pg, gp}.
    Handles both ASA column naming conventions.
    """
    col_map = {
        "xgoals_for": "xg_for",
        "xgoals_against": "xga",
        "xg": "xg_for",
        "xga": "xga",
        "goals_for": "goals_for",
        "goals_against": "goals_against",
        "count_games": "gp",
    }

    df = team_xg_df.rename(columns={k: v for k, v in col_map.items() if k in team_xg_df.columns})

    result: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name:
            continue
        gp = max(float(row.get("gp", 1) or 1), 1)
        xg_raw = row.get("xg_for", row.get("goals_for", 1.2))
        xga_raw = row.get("xga", row.get("goals_against", 1.2))

        result[name] = {
            "xg_for_pg": round(float(xg_raw) / gp, 3),
            "xga_pg": round(float(xga_raw) / gp, 3),
            "gp": int(gp),
        }

    return result


def resolve_team_xg(
    team: str,
    xg_lookup: dict[str, dict[str, float]],
    fallback_xg: float = 1.25,
) -> tuple[float, float]:
    """
    Return (xg_for_pg, xga_pg) for a team.
    Tries exact match, then case-insensitive partial match.
    Falls back to league-average if not found.
    """
    if team in xg_lookup:
        d = xg_lookup[team]
        return d["xg_for_pg"], d["xga_pg"]

    # Partial case-insensitive match (handles ESPN vs ASA naming gaps)
    team_lower = team.lower()
    for key, d in xg_lookup.items():
        if team_lower in key.lower() or key.lower() in team_lower:
            return d["xg_for_pg"], d["xga_pg"]

    return fallback_xg, fallback_xg


def compute_fixture_xg(
    home_team: str,
    away_team: str,
    xg_lookup: dict[str, dict[str, float]],
    home_advantage: float = 0.12,
) -> tuple[float, float]:
    """
    Estimate expected goals for a single upcoming fixture.

    home_xg = avg(home xg_for, away xga) * (1 + home_advantage)
    away_xg = avg(away xg_for, home xga) * (1 - home_advantage/2)
    """
    h_for, h_against = resolve_team_xg(home_team, xg_lookup)
    a_for, a_against = resolve_team_xg(away_team, xg_lookup)

    home_xg = ((h_for + a_against) / 2) * (1 + home_advantage)
    away_xg = ((a_for + h_against) / 2) * (1 - home_advantage / 2)

    return round(max(home_xg, 0.1), 3), round(max(away_xg, 0.1), 3)
