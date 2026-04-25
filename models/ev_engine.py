"""
models/ev_engine.py
Expected Value (EV) engine and team rankings for MLS predictions.

Provides:
  - american_to_decimal(american_odds) -> float
  - implied_probability(american_odds) -> float
  - remove_vig(home_odds, draw_odds, away_odds) -> tuple[float, float, float]
  - compute_ev(model_prob, american_odds) -> float
  - edge_percentage(model_prob, book_implied_prob) -> float
  - compute_team_rankings(team_xg_df) -> pd.DataFrame
  - matchup_narrative(home_team, away_team, rankings) -> str
  - compute_rest_days(team, match_date, historical_df) -> int
  - rest_advantage_label(home_rest, away_rest) -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Odds conversion ────────────────────────────────────────────────────────────

def american_to_decimal(american_odds: int | float) -> float:
    """Convert American moneyline to decimal odds."""
    odds = float(american_odds)
    if odds >= 100:
        return round((odds / 100) + 1.0, 4)
    elif odds <= -100:
        return round((100 / abs(odds)) + 1.0, 4)
    else:
        # Handle odds between -100 and +100 (edge case; treat as even)
        return 2.0


def implied_probability(american_odds: int | float) -> float:
    """Book-implied win probability from American moneyline (raw, includes vig)."""
    decimal = american_to_decimal(american_odds)
    return round(1.0 / decimal, 4)


def remove_vig(
    home_odds: int | float,
    draw_odds: int | float,
    away_odds: int | float,
) -> tuple[float, float, float]:
    """
    Remove sportsbook overround (vig) to get fair implied probabilities.
    Returns (home_prob, draw_prob, away_prob) normalised to sum to 1.0.
    """
    raw = [implied_probability(o) for o in (home_odds, draw_odds, away_odds)]
    total = sum(raw)
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return tuple(round(p / total, 4) for p in raw)  # type: ignore[return-value]


# ── EV calculation ─────────────────────────────────────────────────────────────

def compute_ev(model_prob: float, american_odds: int | float) -> float:
    """
    Expected Value percentage for a single market outcome.

    EV = (model_prob × decimal_odds) - 1

    Positive EV means the model believes the true probability exceeds
    the book-implied probability — a +EV bet.

    Returns EV as a decimal (e.g. 0.058 = +5.8%).
    """
    decimal = american_to_decimal(american_odds)
    return round((model_prob * decimal) - 1.0, 4)


def edge_percentage(model_prob: float, book_implied_prob: float) -> float:
    """
    StatSniper-style edge: model prob minus book implied prob, as a percentage.
    Positive = model likes the outcome more than the book does.
    """
    return round((model_prob - book_implied_prob) * 100, 1)


# ── Team rankings ──────────────────────────────────────────────────────────────

def compute_team_rankings(team_xg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute offensive and defensive rankings 1-N for all MLS teams.

    Offensive rank: xG per game (1 = best attack).
    Defensive rank: xGA per game (1 = stingiest defence = fewest xG allowed).

    Compatible with both ASA column naming conventions.
    """
    df = team_xg_df.copy()

    # Normalise column names — use first matching alias only to avoid duplicate columns
    xg_candidates = ["xgoals_for", "xg", "goals_for"]
    xga_candidates = ["xgoals_against", "xga", "goals_against"]

    xg_src = next((c for c in xg_candidates if c in df.columns), None)
    xga_src = next((c for c in xga_candidates if c in df.columns), None)

    rename_map: dict[str, str] = {}
    if xg_src:
        rename_map[xg_src] = "total_xg"
    if xga_src:
        rename_map[xga_src] = "total_xga"

    df = df.rename(columns=rename_map)

    gp_col = "count_games" if "count_games" in df.columns else None
    name_col = next((c for c in ["team_name", "team"] if c in df.columns), None)

    if name_col is None or "total_xg" not in df.columns:
        return pd.DataFrame()

    df = df[[name_col, "total_xg", "total_xga"] + ([gp_col] if gp_col else [])].copy()
    df = df.rename(columns={name_col: "Team"})

    if gp_col:
        df["gp"] = pd.to_numeric(df[gp_col], errors="coerce").fillna(1).clip(lower=1)
    else:
        df["gp"] = 1.0

    df["total_xg"] = pd.to_numeric(df["total_xg"], errors="coerce").fillna(0)
    df["total_xga"] = pd.to_numeric(df["total_xga"], errors="coerce").fillna(0)

    df["xg_per_game"] = (df["total_xg"] / df["gp"]).round(3)
    df["xga_per_game"] = (df["total_xga"] / df["gp"]).round(3)

    df["offense_rank"] = df["xg_per_game"].rank(ascending=False, method="min").astype(int)
    df["defense_rank"] = df["xga_per_game"].rank(ascending=True, method="min").astype(int)

    return df[["Team", "gp", "xg_per_game", "xga_per_game", "offense_rank", "defense_rank"]].sort_values(
        "offense_rank"
    )


def matchup_narrative(
    home_team: str,
    away_team: str,
    rankings: pd.DataFrame,
    home_rest: int | None = None,
    away_rest: int | None = None,
) -> str:
    """
    Generate a plain-English matchup summary from offense/defense rankings
    and optional rest-day context. Returns empty string if rankings unavailable.
    """
    h_row = rankings[rankings["Team"] == home_team]
    a_row = rankings[rankings["Team"] == away_team]

    if h_row.empty or a_row.empty:
        return ""

    h = h_row.iloc[0]
    a = a_row.iloc[0]
    n = len(rankings)

    lines: list[str] = []

    # Offensive vs defensive mismatches
    if h["offense_rank"] <= 5 and a["defense_rank"] >= n - 4:
        lines.append(
            f"**{home_team}** has the league's #{h['offense_rank']} attack "
            f"against {away_team}'s #{a['defense_rank']} defence — strong home scoring edge."
        )
    elif a["offense_rank"] <= 5 and h["defense_rank"] >= n - 4:
        lines.append(
            f"**{away_team}** brings a #{a['offense_rank']} attack "
            f"into {home_team}'s #{h['defense_rank']}-ranked defence — away side can hurt here."
        )
    else:
        lines.append(
            f"{home_team}: attack #{h['offense_rank']} / defence #{h['defense_rank']}. "
            f"{away_team}: attack #{a['offense_rank']} / defence #{a['defense_rank']}."
        )

    # Rest context
    if home_rest is not None and away_rest is not None:
        if away_rest <= 3 and home_rest >= 5:
            lines.append(
                f"🛌 Rest edge: {home_team} has {home_rest} days rest; "
                f"{away_team} is on a back-to-back ({away_rest} days)."
            )
        elif home_rest <= 3 and away_rest >= 5:
            lines.append(
                f"⚠️ {home_team} is on a back-to-back ({home_rest} days rest) "
                f"while {away_team} has {away_rest} days."
            )

    return "  \n".join(lines)


# ── Rest days helpers ──────────────────────────────────────────────────────────

def compute_rest_days(
    team: str,
    match_date: pd.Timestamp,
    historical_df: pd.DataFrame,
) -> int | None:
    """
    Days since team's most recent completed match before match_date.
    Returns None if no prior match found in the dataset.
    """
    date_col = next((c for c in ["MatchDate", "Date"] if c in historical_df.columns), None)
    if date_col is None:
        return None

    home_col = "HomeTeam" if "HomeTeam" in historical_df.columns else None
    away_col = "AwayTeam" if "AwayTeam" in historical_df.columns else None
    if not home_col or not away_col:
        return None

    dates = pd.to_datetime(historical_df[date_col], errors="coerce")
    mask = (
        ((historical_df[home_col] == team) | (historical_df[away_col] == team))
        & (dates < match_date)
    )
    prior = dates[mask]
    if prior.empty:
        return None
    last = prior.max()
    return (match_date - last).days


def rest_advantage_label(home_rest: int | None, away_rest: int | None) -> str:
    """Short label for the rest-day situation in a fixture card."""
    if home_rest is None or away_rest is None:
        return ""
    diff = home_rest - away_rest
    if away_rest <= 3 and home_rest >= 5:
        return f"🛌 Home rested ({home_rest}d) vs Away B2B ({away_rest}d)"
    if home_rest <= 3 and away_rest >= 5:
        return f"⚠️ Home B2B ({home_rest}d) vs Away rested ({away_rest}d)"
    if abs(diff) >= 3:
        if diff > 0:
            return f"🛌 Home +{diff}d rest advantage"
        return f"⚠️ Away +{abs(diff)}d rest advantage"
    return ""


# ── Best Bets filter ───────────────────────────────────────────────────────────

def generate_best_bets(
    upcoming_df: pd.DataFrame,
    model_probs: dict[str, tuple[float, float, float]],
    min_ev: float = 0.04,
    min_confidence: float = 0.52,
) -> pd.DataFrame:
    """
    Filter upcoming fixtures to only those with ≥min_ev EV and ≥min_confidence
    model probability.

    upcoming_df must have: HomeTeam, AwayTeam, Date.
    It may optionally have: best_home_odds, best_draw_odds, best_away_odds,
    best_home_book, best_draw_book, best_away_book (American moneyline integers).

    model_probs: dict keyed by "{HomeTeam}|{AwayTeam}" → (home_prob, draw_prob, away_prob)

    Returns a DataFrame of +EV picks, sorted by EV descending.
    """
    odds_cols = {
        "H": ("best_home_odds", "best_home_book"),
        "D": ("best_draw_odds", "best_draw_book"),
        "A": ("best_away_odds", "best_away_book"),
    }
    label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    prob_idx = {"H": 0, "D": 1, "A": 2}

    rows: list[dict] = []

    for _, row in upcoming_df.iterrows():
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        key = f"{home}|{away}"

        probs = model_probs.get(key)
        if probs is None:
            continue

        for outcome, (odds_col, book_col) in odds_cols.items():
            if odds_col not in upcoming_df.columns:
                continue
            american_odds = row.get(odds_col)
            if pd.isna(american_odds):
                continue

            model_prob = probs[prob_idx[outcome]]
            ev = compute_ev(model_prob, int(american_odds))
            book_prob = implied_probability(int(american_odds))

            if ev >= min_ev and model_prob >= min_confidence:
                rows.append(
                    {
                        "Match": f"{home} vs {away}",
                        "Date": row.get("Date", ""),
                        "Pick": label_map[outcome],
                        "Model Prob": f"{model_prob:.0%}",
                        "Best Odds": (
                            f"+{int(american_odds)}"
                            if int(american_odds) > 0
                            else str(int(american_odds))
                        ),
                        "Book": row.get(book_col, "—"),
                        "EV": f"+{ev * 100:.1f}%",
                        "_ev_sort": ev,
                    }
                )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("_ev_sort", ascending=False)
        .drop(columns=["_ev_sort"])
        .reset_index(drop=True)
    )
