"""
pages/6_Markets.py
BTTS, Over/Under, Correct Score, and Shots-on-Target predictions for upcoming MLS fixtures.
Powered by Poisson / Dixon-Coles model using ASA xG data.

Naming convention: plain ASCII filename, icon set via st.navigation() or sidebar label.
No st.set_page_config() here — called once in predictions.py (entry point).
"""

import os
import sys
from os import path

import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure project root is on the path ────────────────────────────────────────
_root = path.dirname(path.dirname(path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from models.market_predictions import (
    btts_probability,
    compute_fixture_xg,
    fit_rho_from_historical,
    get_team_xg_averages,
    over_under_probabilities,
    top_correct_scores,
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = path.join(_root, "data_files")
FIXTURES_PATH = path.join(DATA_DIR, "upcoming_fixtures.csv")
HIST_PATH = path.join(DATA_DIR, "combined_historical_data.csv")
ASA_XG_PATH = path.join(DATA_DIR, "raw", "asa_team_xg.csv")

TURF_STADIUMS = {
    "New England Revolution", "Portland Timbers",
    "Seattle Sounders", "Vancouver Whitecaps", "FC Cincinnati",
}


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_fixtures() -> pd.DataFrame:
    if not path.exists(FIXTURES_PATH):
        return pd.DataFrame()
    return pd.read_csv(FIXTURES_PATH)


@st.cache_data(ttl=3600)
def _load_team_xg() -> pd.DataFrame:
    if path.exists(ASA_XG_PATH):
        return pd.read_csv(ASA_XG_PATH)
    try:
        from itscalledsoccer.client import AmericanSoccerAnalysis
        from datetime import datetime
        asa = AmericanSoccerAnalysis()
        return asa.get_team_xgoals(leagues="mls", season_name=str(datetime.now().year))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=7200)
def _fit_rho() -> float:
    if not path.exists(HIST_PATH):
        return -0.10
    try:
        df = pd.read_csv(HIST_PATH)
        return fit_rho_from_historical(df)
    except Exception:
        return -0.10


# ── Probability bar helper ─────────────────────────────────────────────────────

def _prob_bar(label: str, prob: float, color: str = "#4CAF50") -> str:
    """Return an HTML progress-bar style display for a probability."""
    pct = round(prob * 100, 1)
    bar_width = round(prob * 100)
    return (
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:0.85em;'>{label}</span><br>"
        f"<div style='background:#eee;border-radius:4px;height:14px;width:100%;'>"
        f"<div style='background:{color};border-radius:4px;height:14px;width:{bar_width}%;'></div>"
        f"</div>"
        f"<span style='font-size:0.8em;color:#555'>{pct}%</span>"
        f"</div>"
    )


def _compute_sot_ou(
    home_xg: float,
    away_xg: float,
    conversion_rate: float = 0.32,
) -> dict:
    """
    Estimate shots-on-target O/U probabilities using Poisson distributions.

    MLS average: goals ≈ 32% of shots on target (conversion_rate = 0.32).
    Expected SoT per team = team_xg / conversion_rate.
    Total expected SoT drawn independently from two Poisson distributions.

    Returns a dict with keys: expected, home_sot, away_sot,
    over_7_5, over_8_5, over_9_5, over_10_5.
    """
    from scipy.stats import poisson  # noqa: PLC0415

    home_sot = max(home_xg / conversion_rate, 1.0)
    away_sot = max(away_xg / conversion_rate, 1.0)
    expected_total = home_sot + away_sot

    results = {
        "expected": round(expected_total, 2),
        "home_sot": round(home_sot, 2),
        "away_sot": round(away_sot, 2),
    }

    for line in [7.5, 8.5, 9.5, 10.5]:
        key = f"over_{str(line).replace('.', '_')}"
        # P(total > line) = 1 - P(total <= floor(line))
        cutoff = int(line)
        p_over = 0.0
        # Convolve two independent Poisson distributions up to a reasonable max
        max_total = cutoff + 20
        for h in range(max_total + 1):
            for a in range(max_total + 1 - h):
                if h + a > cutoff:
                    p_over += poisson.pmf(h, home_sot) * poisson.pmf(a, away_sot)
        results[key] = round(min(p_over, 1.0), 4)

    return results


# ── League stats panel ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _league_stats(hist_path: str, season: int) -> dict | None:
    if not path.exists(hist_path):
        return None
    df = pd.read_csv(hist_path)
    date_col = next((c for c in ["MatchDate", "Date"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[df[date_col].dt.year == season]
    if df.empty:
        return None

    res_col = next((c for c in ["Result", "FullTimeResult", "FTR"] if c in df.columns), None)
    hg_col = next((c for c in ["HomeGoals", "FTHG"] if c in df.columns), None)
    ag_col = next((c for c in ["AwayGoals", "FTAG"] if c in df.columns), None)

    if not res_col or not hg_col or not ag_col:
        return None

    n = len(df)
    if n == 0:
        return None

    hg = pd.to_numeric(df[hg_col], errors="coerce").fillna(0)
    ag = pd.to_numeric(df[ag_col], errors="coerce").fillna(0)
    total = hg + ag

    return {
        "n": n,
        "home_win_pct": (df[res_col] == "H").sum() / n,
        "draw_pct": (df[res_col] == "D").sum() / n,
        "away_win_pct": (df[res_col] == "A").sum() / n,
        "avg_home_goals": hg.mean(),
        "avg_away_goals": ag.mean(),
        "avg_total_goals": total.mean(),
        "btts_pct": ((hg > 0) & (ag > 0)).sum() / n,
        "over_1_5_pct": (total > 1.5).sum() / n,
        "over_2_5_pct": (total > 2.5).sum() / n,
        "over_3_5_pct": (total > 3.5).sum() / n,
        "clean_sheet_pct": ((hg == 0) | (ag == 0)).sum() / n,
    }


# ── Main page ──────────────────────────────────────────────────────────────────

st.title("📊 Markets")
st.caption("BTTS · Over/Under · Correct Score — powered by ASA xG + Poisson model")

# ── League stats banner ────────────────────────────────────────────────────────
from datetime import datetime
current_season = datetime.now().year
stats = _league_stats(HIST_PATH, current_season)

if stats:
    with st.expander(f"📈 MLS {current_season} Season Base Rates ({stats['n']} matches)", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Home Win", f"{stats['home_win_pct']:.0%}")
        c2.metric("Draw", f"{stats['draw_pct']:.0%}")
        c3.metric("Away Win", f"{stats['away_win_pct']:.0%}")
        c4.metric("Avg Goals", f"{stats['avg_total_goals']:.2f}")
        c5.metric("BTTS", f"{stats['btts_pct']:.0%}")
        c6.metric("Over 2.5", f"{stats['over_2_5_pct']:.0%}")

        st.markdown("**Goals distribution:**")
        gcols = st.columns(4)
        gcols[0].metric("Over 1.5", f"{stats['over_1_5_pct']:.0%}")
        gcols[1].metric("Over 2.5", f"{stats['over_2_5_pct']:.0%}")
        gcols[2].metric("Over 3.5", f"{stats['over_3_5_pct']:.0%}")
        gcols[3].metric("Clean Sheets", f"{stats['clean_sheet_pct']:.0%}")

st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
fixtures = _load_fixtures()
team_xg_df = _load_team_xg()

if fixtures.empty:
    st.warning(
        "No upcoming fixtures found. "
        "Run `python fetch_upcoming_fixtures.py` to populate `data_files/upcoming_fixtures.csv`."
    )
    st.stop()

if team_xg_df.empty:
    st.warning(
        "No xG data available. Markets use league-average xG (1.25 per team). "
        "Run `python fetch_asa_data.py` for accurate predictions."
    )

# Fit rho from historical data (cached, ~0.5s first run)
rho = _fit_rho()
xg_lookup = get_team_xg_averages(team_xg_df) if not team_xg_df.empty else {}

# ── Fixture loop ───────────────────────────────────────────────────────────────
st.subheader("⚽ Upcoming Fixtures — All Markets")
st.caption(
    f"Dixon-Coles rho={rho:.3f} (calibrated on MLS history). "
    "Probabilities are model estimates, not guaranteed outcomes."
)

if "HomeTeam" not in fixtures.columns or "AwayTeam" not in fixtures.columns:
    st.error("Fixtures file is missing HomeTeam / AwayTeam columns.")
    st.stop()

for _, fix in fixtures.iterrows():
    home = str(fix.get("HomeTeam", "?"))
    away = str(fix.get("AwayTeam", "?"))
    date_str = str(fix.get("Date", ""))
    time_str = str(fix.get("Time", ""))
    is_turf = home in TURF_STADIUMS

    home_xg, away_xg = compute_fixture_xg(home, away, xg_lookup)

    # BTTS
    btts_yes = btts_probability(home_xg, away_xg, home_is_turf=is_turf)
    btts_no = round(1.0 - btts_yes, 4)

    # Over/Under
    ou = over_under_probabilities(home_xg, away_xg)

    # Correct Score
    top_scores = top_correct_scores(home_xg, away_xg, top_n=5, rho=rho)

    # 1X2 derived from Poisson (for reference)
    from models.market_predictions import correct_score_distribution
    dist = correct_score_distribution(home_xg, away_xg, rho=rho)
    home_win_p = sum(p for (h, a), p in dist.items() if h > a)
    draw_p = sum(p for (h, a), p in dist.items() if h == a)
    away_win_p = sum(p for (h, a), p in dist.items() if a > h)

    label = f"⚽ **{home}** vs **{away}** — {date_str} {time_str}"
    if is_turf:
        label += "  🟫 Turf"

    with st.expander(label, expanded=False):
        # Row 1: 1X2
        st.markdown("**1X2 (Poisson-derived)**")
        col_h, col_d, col_a = st.columns(3)
        col_h.metric(f"🏠 {home} Win", f"{home_win_p:.1%}")
        col_d.metric("🤝 Draw", f"{draw_p:.1%}")
        col_a.metric(f"✈️ {away} Win", f"{away_win_p:.1%}")

        st.divider()

        # Row 2: BTTS + O/U
        st.markdown("**Goals Markets**")
        gm1, gm2, gm3, gm4, gm5 = st.columns(5)

        btts_color = "normal" if btts_yes >= 0.55 else "inverse"
        gm1.metric(
            "BTTS Yes",
            f"{btts_yes:.1%}",
            delta="✅ Lean Yes" if btts_yes >= 0.55 else "❌ Lean No",
            delta_color="off",
        )
        gm2.metric("Over 1.5", f"{ou.get('over_1_5', 0):.1%}")
        gm3.metric("Over 2.5", f"{ou.get('over_2_5', 0):.1%}")
        gm4.metric("Under 2.5", f"{ou.get('under_2_5', 0):.1%}")
        gm5.metric("Over 3.5", f"{ou.get('over_3_5', 0):.1%}")

        st.divider()

        # Row 3: Correct Score
        st.markdown("**Top 5 Correct Scores**")
        cs_cols = st.columns(5)
        for i, (score, pct) in enumerate(top_scores):
            cs_cols[i].metric(score, f"{pct}%")

        # Model xG footnote
        st.caption(
            f"Model inputs: {home} xG={home_xg:.2f}, {away} xG={away_xg:.2f}. "
            f"{'Turf boost applied to BTTS. ' if is_turf else ''}"
            "Poisson independence + Dixon-Coles low-score correction."
        )

        st.divider()

        # Row 4: Shots on Target
        st.markdown("**Shots on Target — Total Market**")
        sot_probs = _compute_sot_ou(home_xg, away_xg)
        sot_cols = st.columns(5)
        sot_cols[0].metric("Exp. Total SoT", f"{sot_probs['expected']:.1f}")
        sot_cols[1].metric("Over 7.5", f"{sot_probs['over_7_5']:.1%}")
        sot_cols[2].metric("Over 8.5", f"{sot_probs['over_8_5']:.1%}")
        sot_cols[3].metric("Over 9.5", f"{sot_probs['over_9_5']:.1%}")
        sot_cols[4].metric("Over 10.5", f"{sot_probs['over_10_5']:.1%}")
        st.caption(
            f"Estimated SoT: {home} ≈{sot_probs['home_sot']:.1f}, "
            f"{away} ≈{sot_probs['away_sot']:.1f}. "
            "Model: SoT ≈ xG ÷ 0.32 (MLS avg conversion rate). Poisson distribution."
        )
