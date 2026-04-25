"""
pages/7_Best_Bets.py
Expected Value engine, team offense/defense rankings, rest-day analysis,
and +EV best bets for upcoming MLS fixtures.

Naming convention: plain ASCII filename, no emoji, icon set via sidebar label.
No st.set_page_config() here — called once in predictions.py (entry point).
"""

import sys
from os import path
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Ensure project root is on the path ────────────────────────────────────────
_root = path.dirname(path.dirname(path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from models.ev_engine import (
    american_to_decimal,
    compute_ev,
    compute_rest_days,
    compute_team_rankings,
    edge_percentage,
    generate_best_bets,
    implied_probability,
    matchup_narrative,
    remove_vig,
    rest_advantage_label,
)
from models.market_predictions import (
    compute_fixture_xg,
    correct_score_distribution,
    get_team_xg_averages,
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = path.join(_root, "data_files")
FIXTURES_PATH = path.join(DATA_DIR, "upcoming_fixtures.csv")
HIST_PATH = path.join(DATA_DIR, "combined_historical_data.csv")
ASA_XG_PATH = path.join(DATA_DIR, "raw", "asa_team_xg.csv")


# ── Cached loaders ─────────────────────────────────────────────────────────────

def _fixtures_mtime() -> float:
    return path.getmtime(FIXTURES_PATH) if path.exists(FIXTURES_PATH) else 0.0


@st.cache_data(ttl=3600)
def _load_fixtures(_mtime: float = 0.0) -> pd.DataFrame:
    if not path.exists(FIXTURES_PATH):
        return pd.DataFrame()
    return pd.read_csv(FIXTURES_PATH)


@st.cache_data(ttl=3600)
def _load_historical() -> pd.DataFrame:
    if not path.exists(HIST_PATH):
        return pd.DataFrame()
    df = pd.read_csv(HIST_PATH)
    date_col = next((c for c in ["MatchDate", "Date"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if date_col != "MatchDate":
            df = df.rename(columns={date_col: "MatchDate"})
    return df


@st.cache_data(ttl=3600)
def _load_team_xg() -> pd.DataFrame:
    if path.exists(ASA_XG_PATH):
        return pd.read_csv(ASA_XG_PATH)
    try:
        from itscalledsoccer.client import AmericanSoccerAnalysis
        asa = AmericanSoccerAnalysis()
        return asa.get_team_xgoals(leagues="mls", season_name=str(datetime.now().year))
    except Exception:
        return pd.DataFrame()


# ── Page header ────────────────────────────────────────────────────────────────

st.title("💰 Best Bets")
st.caption(
    "Expected Value engine · Team Rankings · Rest-Day Analysis  \n"
    "⚠️ Model-derived probabilities for informational purposes only. Bet responsibly."
)

# ── Load data ──────────────────────────────────────────────────────────────────
fixtures = _load_fixtures(_fixtures_mtime())
hist_df = _load_historical()
team_xg_df = _load_team_xg()

if fixtures.empty:
    st.warning(
        "No upcoming fixtures found. "
        "Run `python fetch_upcoming_fixtures.py` first."
    )
    st.stop()

xg_lookup = get_team_xg_averages(team_xg_df) if not team_xg_df.empty else {}
rankings = compute_team_rankings(team_xg_df) if not team_xg_df.empty else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Team Rankings
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("📊 Team Offense & Defense Rankings")
st.caption("Based on xG per game from American Soccer Analysis (current season).")

if rankings.empty:
    st.info(
        "No xG data available for rankings. "
        "Run `python fetch_asa_data.py` or check `data_files/raw/asa_team_xg.csv`."
    )
else:
    n_teams = len(rankings)
    top5_cutoff = 5
    bot5_cutoff = n_teams - 4

    def _style_rank(val: int) -> str:
        if val <= top5_cutoff:
            return "background-color:#d4edda;color:#155724"
        if val >= bot5_cutoff:
            return "background-color:#f8d7da;color:#721c24"
        return ""

    display_rankings = rankings.copy()
    display_rankings.columns = ["Team", "GP", "xG/Game", "xGA/Game", "Attack Rank", "Defense Rank"]

    styled = display_rankings.style.map(
        _style_rank, subset=["Attack Rank", "Defense Rank"]
    ).format({"xG/Game": "{:.3f}", "xGA/Game": "{:.3f}"})

    st.dataframe(styled, hide_index=True, height=min(35 * len(display_rankings) + 40, 600))

    with st.expander("ℹ️ How to read the rankings"):
        st.markdown(
            """
- **Attack Rank #1** = highest xG per game in the league (best attack).
- **Defense Rank #1** = lowest xGA per game allowed (best defence).
- 🟢 Green = top 5 in that category. 🔴 Red = bottom 5.
- Rankings update when `fetch_asa_data.py` runs (nightly pipeline).
            """
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Matchup Breakdown (rest + narrative)
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("🔍 Fixture Matchup Analysis")
st.caption("Combines attack/defence rankings with rest-day context.")

if "HomeTeam" not in fixtures.columns or "AwayTeam" not in fixtures.columns:
    st.error("Fixtures file is missing HomeTeam / AwayTeam columns.")
else:
    for _, fix in fixtures.iterrows():
        home = str(fix.get("HomeTeam", "?"))
        away = str(fix.get("AwayTeam", "?"))
        date_str = str(fix.get("Date", ""))

        # Rest days (from historical data)
        home_rest: int | None = None
        away_rest: int | None = None
        if not hist_df.empty:
            try:
                match_dt = pd.Timestamp(date_str) if date_str else None
                if match_dt:
                    home_rest = compute_rest_days(home, match_dt, hist_df)
                    away_rest = compute_rest_days(away, match_dt, hist_df)
            except Exception:
                pass

        # EV context (only if odds columns exist)
        home_xg, away_xg = compute_fixture_xg(home, away, xg_lookup)
        dist = correct_score_distribution(home_xg, away_xg)
        home_win_p = sum(p for (h, a), p in dist.items() if h > a)
        draw_p = sum(p for (h, a), p in dist.items() if h == a)
        away_win_p = sum(p for (h, a), p in dist.items() if a > h)

        rest_label = rest_advantage_label(home_rest, away_rest)
        narrative = matchup_narrative(home, away, rankings, home_rest, away_rest) if not rankings.empty else ""

        card_label = f"**{home}** vs **{away}** — {date_str}"
        if rest_label:
            card_label += f"  {rest_label}"

        with st.expander(card_label, expanded=False):
            # Poisson probs
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric(f"🏠 {home}", f"{home_win_p:.1%}")
            pc2.metric("Draw", f"{draw_p:.1%}")
            pc3.metric(f"✈️ {away}", f"{away_win_p:.1%}")

            if narrative:
                st.markdown(narrative)

            # Rest days
            if home_rest is not None or away_rest is not None:
                rest_cols = st.columns(2)
                rest_cols[0].metric(
                    f"{home} Rest",
                    f"{home_rest}d" if home_rest is not None else "—",
                    delta="B2B ⚠️" if (home_rest is not None and home_rest <= 3) else None,
                    delta_color="inverse" if (home_rest is not None and home_rest <= 3) else "off",
                )
                rest_cols[1].metric(
                    f"{away} Rest",
                    f"{away_rest}d" if away_rest is not None else "—",
                    delta="B2B ⚠️" if (away_rest is not None and away_rest <= 3) else None,
                    delta_color="inverse" if (away_rest is not None and away_rest <= 3) else "off",
                )

            # Odds-based EV (only if odds columns present)
            odds_cols_present = any(
                c in fixtures.columns
                for c in ["best_home_odds", "best_draw_odds", "best_away_odds"]
            )
            if odds_cols_present:
                st.markdown("**EV vs. Current Lines**")
                ev_rows = []
                for outcome, prob, odds_col, label in [
                    ("H", home_win_p, "best_home_odds", f"Home Win ({home})"),
                    ("D", draw_p, "best_draw_odds", "Draw"),
                    ("A", away_win_p, "best_away_odds", f"Away Win ({away})"),
                ]:
                    american = fix.get(odds_col)
                    if pd.notna(american):
                        ev = compute_ev(prob, int(american))
                        book_p = implied_probability(int(american))
                        edge = edge_percentage(prob, book_p)
                        ev_rows.append({
                            "Outcome": label,
                            "Model Prob": f"{prob:.1%}",
                            "Odds": f"+{int(american)}" if int(american) > 0 else str(int(american)),
                            "Book Impl.": f"{book_p:.1%}",
                            "Edge": f"{edge:+.1f}%",
                            "EV": f"{ev * 100:+.1f}%",
                        })
                if ev_rows:
                    ev_df = pd.DataFrame(ev_rows)
                    st.dataframe(ev_df, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Best Bets (only when odds data available)
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("🎯 Today's Best Bets")

odds_available = any(
    c in fixtures.columns for c in ["best_home_odds", "best_draw_odds", "best_away_odds"]
)

if not odds_available:
    st.info(
        "Live odds data not yet available in `upcoming_fixtures.csv`.  \n"
        "When The Odds API is integrated and `fetch_upcoming_fixtures.py` writes "
        "`best_home_odds`, `best_draw_odds`, `best_away_odds` columns, this section "
        "will automatically display +EV picks ≥4% edge.  \n\n"
        "**How to enable:** Add `ODDS_API_KEY` to `.env` and update "
        "`fetch_upcoming_fixtures.py` to append odds columns."
    )
else:
    # Build model_probs dict for all fixtures
    model_probs: dict[str, tuple[float, float, float]] = {}
    for _, fix in fixtures.iterrows():
        home = str(fix.get("HomeTeam", ""))
        away = str(fix.get("AwayTeam", ""))
        home_xg, away_xg = compute_fixture_xg(home, away, xg_lookup)
        dist = correct_score_distribution(home_xg, away_xg)
        hp = sum(p for (h, a), p in dist.items() if h > a)
        dp = sum(p for (h, a), p in dist.items() if h == a)
        ap = sum(p for (h, a), p in dist.items() if a > h)
        model_probs[f"{home}|{away}"] = (hp, dp, ap)

    min_ev = st.slider("Minimum EV threshold (%)", 1, 15, 4, 1) / 100
    min_conf = st.slider("Minimum model confidence (%)", 45, 70, 52, 1) / 100

    best_df = generate_best_bets(fixtures, model_probs, min_ev=min_ev, min_confidence=min_conf)

    if best_df.empty:
        st.info(
            f"No picks meet the current filters (EV ≥{min_ev:.0%}, confidence ≥{min_conf:.0%}).  \n"
            "Try lowering the thresholds or check back when lines move."
        )
    else:
        st.success(f"Found **{len(best_df)}** +EV pick(s) meeting your criteria.")
        st.dataframe(best_df, hide_index=True)
        st.caption(
            "EV = (Model Prob × Decimal Odds) − 1. Positive EV means the model "
            "estimates a higher probability than the book implies. Past performance "
            "does not guarantee future results."
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Odds Explainer
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("📖 How EV is calculated"):
    st.markdown(
        """
### Expected Value (EV)

$$
\\text{EV} = (p_{\\text{model}} \\times d_{\\text{decimal}}) - 1
$$

Where $d_{\\text{decimal}}$ is the decimal equivalent of the American moneyline:

- Favourite (e.g. −235): $d = \\frac{100}{235} + 1 = 1.426$
- Underdog (e.g. +180): $d = \\frac{180}{100} + 1 = 2.80$

**Positive EV** means the model believes the outcome is more likely than the
book-implied probability. A +5% EV bet means for every $100 wagered, the
model expects a $5 long-run profit.

### Edge %

$$
\\text{Edge} = (p_{\\text{model}} - p_{\\text{book}}) \\times 100
$$

Edge strips vig from the book's implied probability first and compares directly
to the model's estimate.

### Vig removal

The book-implied probability for a three-way market (H/D/A) sums to more than
1.0 because of the overround. The "fair" probability normalises the three raw
implied probabilities to sum to exactly 1.0.
        """
    )
