import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from os import path
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Optional MLS-specific data package ────────────────────────────────────────
try:
    from itscalledsoccer.client import AmericanSoccerAnalysis

    ASA_AVAILABLE = True
except ImportError:
    ASA_AVAILABLE = False

# ── Team name normalizer ───────────────────────────────────────────────────────
try:
    from team_name_mapping import normalize_team_name
except ImportError:
    def normalize_team_name(name: str) -> str:  # type: ignore[misc]
        return name.strip() if isinstance(name, str) else name

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = "data_files/"

EASTERN_CONF = {
    "Atlanta United",
    "CF Montréal",
    "Charlotte FC",
    "Chicago Fire",
    "Columbus Crew",
    "D.C. United",
    "FC Cincinnati",
    "Inter Miami CF",
    "Nashville SC",
    "New England Revolution",
    "New York City FC",
    "New York Red Bulls",
    "Orlando City",
    "Philadelphia Union",
    "Toronto FC",
}

WESTERN_CONF = {
    "Austin FC",
    "Colorado Rapids",
    "FC Dallas",
    "Houston Dynamo",
    "LA Galaxy",
    "LAFC",
    "Minnesota United",
    "Portland Timbers",
    "Real Salt Lake",
    "San Jose Earthquakes",
    "Seattle Sounders",
    "Sporting Kansas City",
    "St. Louis City SC",
    "Vancouver Whitecaps",
}

# Artificial-turf home stadiums (2024 season)
TURF_STADIUMS = {
    "New England Revolution",
    "Portland Timbers",
    "Seattle Sounders",
    "Vancouver Whitecaps",
    "FC Cincinnati",
}

# Stadium GPS coordinates for travel-distance calculations
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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLS Predictor",
    layout="wide",
    page_icon="⚽",
)

st.image(path.join(DATA_DIR, "logo.png"), width=250)
st.title("MLS Predictor")

fixtures_file = path.join(DATA_DIR, "upcoming_fixtures.csv")
if path.exists(fixtures_file):
    last_updated = datetime.fromtimestamp(os.path.getmtime(fixtures_file))
    st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %I:%M %p ET')}")

# ── MLS helper utilities ───────────────────────────────────────────────────────

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

def get_conference(team: str) -> str:
    if team in EASTERN_CONF:
        return "Eastern"
    if team in WESTERN_CONF:
        return "Western"
    return "Unknown"


def is_turf(team: str) -> bool:
    return team in TURF_STADIUMS


def travel_distance_miles(home_team: str, away_team: str) -> float:
    """Great-circle distance in miles between two MLS stadiums."""
    if home_team not in STADIUM_COORDS or away_team not in STADIUM_COORDS:
        return 0.0
    lat1, lon1 = map(radians, STADIUM_COORDS[home_team])
    lat2, lon2 = map(radians, STADIUM_COORDS[away_team])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 3958.8 * 2 * atan2(sqrt(a), sqrt(1 - a))


def get_dataframe_height(
    df, row_height: int = 35, header_height: int = 38, padding: int = 2, max_height: int = 600
) -> int:
    calculated = len(df) * row_height + header_height + padding
    return min(calculated, max_height) if max_height else calculated


def create_simple_ensemble():
    return VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(eval_metric="mlogloss", random_state=42, verbosity=0)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ],
        voting="soft",
    )


def calculate_prediction_risk(home_prob: float, draw_prob: float, away_prob: float):
    probs = np.clip(np.array([home_prob, draw_prob, away_prob]), 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    normalized_entropy = entropy / np.log(3)
    confidence_score = 1 - normalized_entropy
    variance = np.sum((probs - 1 / 3) ** 2) / 3
    risk_score = min(100, max(0, (1 - confidence_score) * 50 + variance * 50))
    return risk_score, confidence_score


def risk_category(score: float) -> str:
    if score > 47:
        return "🚨 Critical"
    if score > 40:
        return "🔴 High"
    if score > 30:
        return "🟡 Moderate"
    return "🟢 Low"


def betting_recommendation(home_prob, draw_prob, away_prob, risk_score) -> str:
    max_prob = max(home_prob, draw_prob, away_prob)
    if max_prob >= 0.60 and risk_score <= 30:
        if home_prob == max_prob:
            return "💰 Bet Home Win"
        if draw_prob == max_prob:
            return "💰 Bet Draw"
        return "💰 Bet Away Win"
    if max_prob >= 0.50 and risk_score <= 50:
        if home_prob == max_prob:
            return "🤔 Consider Home"
        if draw_prob == max_prob:
            return "🤔 Consider Draw"
        return "🤔 Consider Away"
    return "❌ Avoid Betting"


# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_historical_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = next((c for c in ["MatchDate", "Date"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if date_col != "MatchDate":
            df.rename(columns={date_col: "MatchDate"}, inplace=True)
    return df


@st.cache_data(ttl=3600)
def load_upcoming_fixtures(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data(ttl=7200)
def fetch_asa_xg_data(seasons: tuple):
    """Pull xG/goals-added data from American Soccer Analysis.
    Loads from pre-fetched CSV first; falls back to live API.
    """
    team_xg_csv = path.join(DATA_DIR, "raw", "asa_team_xg.csv")
    games_csv = path.join(DATA_DIR, "raw", "asa_games.csv")

    # Load pre-fetched files if available (fast path)
    team_xg = pd.read_csv(team_xg_csv) if path.exists(team_xg_csv) else pd.DataFrame()
    games_xg = pd.read_csv(games_csv) if path.exists(games_csv) else pd.DataFrame()

    if not team_xg.empty:
        return games_xg, team_xg

    # Fallback: live API (seasons must be strings for itscalledsoccer)
    if not ASA_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame()
    try:
        asa = AmericanSoccerAnalysis()
        str_seasons = [str(s) for s in seasons]
        team_xg = asa.get_team_xgoals(leagues="mls", season_name=str_seasons[-1])
        games_xg = asa.get_games(leagues="mls", seasons=str_seasons)
        return games_xg, team_xg
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def compute_mls_standings(df: pd.DataFrame, season_start: str = "2025-01-01") -> pd.DataFrame:
    if "MatchDate" not in df.columns or df.empty:
        return pd.DataFrame()
    current = df[df["MatchDate"] >= season_start].copy()
    if current.empty:
        return pd.DataFrame()

    result_col = next((c for c in ["Result", "FullTimeResult", "FTR"] if c in current.columns), None)
    home_goals_col = next((c for c in ["HomeGoals", "FullTimeHomeGoals", "FTHG"] if c in current.columns), None)
    away_goals_col = next((c for c in ["AwayGoals", "FullTimeAwayGoals", "FTAG"] if c in current.columns), None)

    if not result_col:
        return pd.DataFrame()

    records = []
    for _, row in current.iterrows():
        home = row.get("HomeTeam", "")
        away = row.get("AwayTeam", "")
        hg = int(row.get(home_goals_col, 0) or 0) if home_goals_col else 0
        ag = int(row.get(away_goals_col, 0) or 0) if away_goals_col else 0
        result = row.get(result_col, "")
        hw = hd = hl = aw = ad = al = 0
        if result == "H":
            hw = 1; al = 1
        elif result == "D":
            hd = 1; ad = 1
        elif result == "A":
            hl = 1; aw = 1
        records.append({"Team": home, "GF": hg, "GA": ag, "W": hw, "D": hd, "L": hl, "MatchDate": row["MatchDate"]})
        records.append({"Team": away, "GF": ag, "GA": hg, "W": aw, "D": ad, "L": al, "MatchDate": row["MatchDate"]})

    mdf = pd.DataFrame(records)
    standings = mdf.groupby("Team").agg(
        Played=("GF", "count"),
        Win=("W", "sum"),
        Draw=("D", "sum"),
        Lose=("L", "sum"),
        GoalsFor=("GF", "sum"),
        GoalsAgainst=("GA", "sum"),
    ).reset_index()
    standings["GD"] = standings["GoalsFor"] - standings["GoalsAgainst"]
    standings["Points"] = standings["Win"] * 3 + standings["Draw"]
    standings["Conference"] = standings["Team"].apply(get_conference)
    standings = standings.sort_values(
        ["Points", "GD", "GoalsFor"], ascending=False
    ).reset_index(drop=True)
    standings["Rank"] = standings.index + 1

    def _form(team):
        rows = mdf[mdf["Team"] == team].sort_values("MatchDate").tail(5)
        return "".join("W" if r["W"] else ("D" if r["D"] else "L") for _, r in rows.iterrows())

    standings["Form"] = standings["Team"].apply(_form)
    return standings[["Rank", "Team", "Conference", "Played", "Win", "Draw", "Lose",
                       "GoalsFor", "GoalsAgainst", "GD", "Points", "Form"]]


@st.cache_data(ttl=3600)
def load_and_process_data(csv_path: str):
    """Load historical CSV and build ML training arrays."""
    df = pd.read_csv(csv_path)
    date_col = next((c for c in ["MatchDate", "Date"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    result_col = next((c for c in ["Result", "FullTimeResult", "FTR"] if c in df.columns), None)
    if result_col is None:
        return None, None, None, None, [], df

    target_map = {"H": 0, "D": 1, "A": 2}
    df_model = df[df[result_col].isin(target_map)].copy()
    df_model["target"] = df_model[result_col].map(target_map)

    drop_cols = [
        result_col, "target", "MatchDate", "Date", "HomeTeam", "AwayTeam",
        "HomeGoals", "AwayGoals", "FullTimeHomeGoals", "FullTimeAwayGoals",
        "FTHG", "FTAG", "Season", "Round",
    ]
    X = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], errors="ignore")
    y = df_model["target"]

    X_numeric = X.select_dtypes(include=[np.number])
    cat_cols = X.select_dtypes(include=["object"]).columns
    X_cat = pd.DataFrame()
    for col in cat_cols:
        le = LabelEncoder()
        X_cat[col] = le.fit_transform(X[col].astype(str))

    X_final = pd.concat([X_numeric, X_cat], axis=1).fillna(0)
    X_final.columns = [f"feature_{i}" for i in range(X_final.shape[1])]
    feature_names = X_final.columns.tolist()

    X_arr = X_final.values
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names, df


# ── Main CSV path ──────────────────────────────────────────────────────────────
csv_path = path.join(DATA_DIR, "combined_historical_data.csv")

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗓️ Upcoming Matches",
    "🎯 Upcoming Predictions",
    "📊 Statistics",
    "🔬 Team Deep Dive",
    "📁 Raw Data",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPCOMING MATCHES
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Current standings
    if path.exists(csv_path):
        df_main = load_historical_data(csv_path)
        standings = compute_mls_standings(df_main)
        if not standings.empty:
            st.subheader("📊 Current MLS Standings")
            east = standings[standings["Conference"] == "Eastern"].reset_index(drop=True)
            west = standings[standings["Conference"] == "Western"].reset_index(drop=True)
            col_e, col_w = st.columns(2)
            with col_e:
                st.markdown("**Eastern Conference**")
                st.dataframe(east, height=get_dataframe_height(east), hide_index=True)
            with col_w:
                st.markdown("**Western Conference**")
                st.dataframe(west, height=get_dataframe_height(west), hide_index=True)
            st.divider()

    # Upcoming fixtures
    if not path.exists(fixtures_file):
        st.warning(
            f"No upcoming fixtures found at `{fixtures_file}`. "
            "Run `python fetch_upcoming_fixtures.py` to populate."
        )
    else:
        upcoming_raw = load_upcoming_fixtures(fixtures_file)
        st.subheader("🗓️ Upcoming MLS Fixtures")
        st.caption("*Times shown in Eastern Time (ET)*")

        for _, fixture in upcoming_raw.iterrows():
            home = str(fixture.get("HomeTeam", "?"))
            away = str(fixture.get("AwayTeam", "?"))
            date_str = fixture.get("Date", "")
            time_str = fixture.get("Time", "")

            home_conf = get_conference(home)
            away_conf = get_conference(away)
            cross_conf = (home_conf != away_conf) and (home_conf != "Unknown")
            surface = "🟫 Artificial Turf" if is_turf(home) else "🟢 Natural Grass"
            dist = travel_distance_miles(home, away)

            label = f"**{home}** vs **{away}**  —  {date_str} {time_str}"
            if cross_conf:
                label += "  🔀 Cross-Conference"

            with st.expander(label, expanded=False):
                c1, c2, c3 = st.columns([2, 1, 2])
                with c1:
                    st.markdown(f"**🏠 {home}**")
                    st.caption(f"Conference: {home_conf}")
                with c2:
                    st.markdown("### VS")
                    st.caption(surface)
                with c3:
                    st.markdown(f"**✈️ {away}**")
                    st.caption(f"Conference: {away_conf}")

                if dist > 0:
                    haul = " — long-haul cross-country trip" if dist > 1500 else ""
                    st.caption(f"✈️ Away travel distance: **{dist:,.0f} miles**{haul}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPCOMING PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not path.exists(csv_path):
        st.warning(
            f"Historical data not found at `{csv_path}`. "
            "Add your MLS historical data CSV to enable predictions."
        )
        st.stop()

    if not path.exists(fixtures_file):
        st.warning("No upcoming fixtures found. Run `python fetch_upcoming_fixtures.py`.")
        st.stop()

    with st.spinner("Processing historical data…"):
        result = load_and_process_data(csv_path)

    if result[0] is None:
        st.warning(
            "Could not parse historical data. "
            "Ensure the CSV includes a `Result` or `FullTimeResult` column."
        )
        st.stop()

    X_train, X_test, y_train, y_test, feature_names, df_hist = result

    # Train ensemble model
    with st.spinner("Training prediction model…"):
        model = create_simple_ensemble()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{acc:.1%}", f"{acc - 0.5:.1%} vs random")
    col2.metric("Mean Absolute Error", f"{mae:.3f}")
    col3.metric("Test Predictions", len(y_test))
    st.divider()

    # Full-dataset model for upcoming predictions
    X_full = np.concatenate([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    model_full = create_simple_ensemble()
    model_full.fit(X_full, y_full)

    upcoming_pred_df = load_upcoming_fixtures(fixtures_file)
    n_features = X_train.shape[1]

    # Feature matrix for upcoming matches (placeholder zeros until data pipeline is built)
    X_upcoming = np.zeros((len(upcoming_pred_df), n_features))
    proba = model_full.predict_proba(X_upcoming)

    # Build display DataFrame
    display_cols_needed = ["Date", "Time", "HomeTeam", "AwayTeam"]
    base_cols = [c for c in display_cols_needed if c in upcoming_pred_df.columns]
    display_df = upcoming_pred_df[base_cols].copy()

    display_df["Home Win %"] = (proba[:, 0] * 100).round(1)
    display_df["Draw %"] = (proba[:, 1] * 100).round(1)
    display_df["Away Win %"] = (proba[:, 2] * 100).round(1)

    risk_scores, conf_scores = [], []
    for i in range(len(upcoming_pred_df)):
        r, c = calculate_prediction_risk(proba[i, 0], proba[i, 1], proba[i, 2])
        risk_scores.append(r)
        conf_scores.append(c)

    display_df["Risk Score"] = [round(r, 1) for r in risk_scores]
    display_df["Risk Level"] = [risk_category(r) for r in risk_scores]
    display_df["Confidence %"] = [round(c * 100, 1) for c in conf_scores]

    # MLS-specific columns
    if "HomeTeam" in display_df.columns and "AwayTeam" in display_df.columns:
        display_df["Surface"] = display_df["HomeTeam"].apply(
            lambda t: "Turf" if is_turf(t) else "Grass"
        )
        display_df["Travel (mi)"] = display_df.apply(
            lambda r: round(travel_distance_miles(r["HomeTeam"], r["AwayTeam"])), axis=1
        )
        display_df["Cross-Conf"] = display_df.apply(
            lambda r: "Yes"
            if get_conference(r["HomeTeam"]) != get_conference(r["AwayTeam"])
            and get_conference(r["HomeTeam"]) != "Unknown"
            else "No",
            axis=1,
        )
        display_df["Betting Tip"] = [
            betting_recommendation(proba[i, 0], proba[i, 1], proba[i, 2], risk_scores[i])
            for i in range(len(upcoming_pred_df))
        ]

    # Risk filter
    st.subheader("🎯 Upcoming Match Predictions with Risk Assessment")
    st.caption(
        "*Predictions use a zeroed feature matrix until historical MLS data is loaded. "
        "Accuracy improves significantly once `combined_historical_data.csv` is populated.*"
    )

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    show_all = col_f1.button("📊 All", use_container_width=True)
    show_low = col_f2.button("🟢 Low Risk", use_container_width=True)
    show_mod = col_f3.button("🟡 Moderate", use_container_width=True)
    show_high = col_f4.button("🔴 High/Critical", use_container_width=True)

    if show_low:
        filtered = display_df[display_df["Risk Score"] <= 30]
    elif show_mod:
        filtered = display_df[(display_df["Risk Score"] > 30) & (display_df["Risk Score"] <= 40)]
    elif show_high:
        filtered = display_df[display_df["Risk Score"] > 40]
    else:
        filtered = display_df

    def color_risk_rows(row):
        s = row["Risk Score"]
        if s <= 30:
            return ["background-color: #d4edda; color: #155724"] * len(row)
        if s <= 40:
            return ["background-color: #fff3cd; color: #856404"] * len(row)
        if s <= 47:
            return ["background-color: #f8d7da; color: #721c24"] * len(row)
        return ["background-color: #f5c6cb; color: #721c24"] * len(row)

    if len(filtered) > 0:
        styled = filtered.style.apply(color_risk_rows, axis=1)
        st.dataframe(styled, width="stretch", hide_index=True, height=get_dataframe_height(filtered))
    else:
        st.info("No matches found for the selected risk filter.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 MLS Statistics")

    # ── American Soccer Analysis xG data ──────────────────────────────────────
    st.markdown("### ⚽ Expected Goals (xG) — American Soccer Analysis")
    if not ASA_AVAILABLE:
        st.info(
            "`itscalledsoccer` is not installed. "
            "Run `pip install itscalledsoccer` to enable xG and goals-added data."
        )
    else:
        current_year = datetime.now().year
        with st.spinner("Fetching xG data from American Soccer Analysis…"):
            _, team_xg = fetch_asa_xg_data((current_year - 1, current_year))

        if team_xg.empty:
            st.warning("Could not load xG data. Check your internet connection.")
        else:
            # ASA column names vary by source; map to display names
            col_map = {
                "xgoals_for": "xG For",
                "xgoals_against": "xG Against",
                "xgoal_difference": "xG Diff",
                "goals_for": "Goals",
                "goals_against": "Goals Against",
                "goal_difference": "Goal Diff",
                "xpoints": "xPoints",
                "points": "Points",
                "count_games": "GP",
                # legacy names
                "xg": "xG For",
                "xga": "xG Against",
                "xg_diff": "xG Diff",
            }
            display_cols = ["team_name"] + [
                c for c in col_map if c in team_xg.columns
            ]
            df_xg = team_xg[display_cols].rename(columns=col_map)
            sort_col = "xG For" if "xG For" in df_xg.columns else df_xg.columns[1]
            st.dataframe(
                df_xg.sort_values(sort_col, ascending=False).rename(
                    columns={"team_name": "Team"}
                ),
                hide_index=True,
                height=get_dataframe_height(df_xg),
            )

    st.divider()

    if not path.exists(csv_path):
        st.info("Add `data_files/combined_historical_data.csv` to unlock historical statistics.")
    else:
        df_stats = load_historical_data(csv_path)
        result_col_s = next(
            (c for c in ["Result", "FullTimeResult", "FTR"] if c in df_stats.columns), None
        )

        # ── Team form ─────────────────────────────────────────────────────────
        st.markdown("### 📈 Recent Team Form (Last 5 Matches)")
        if result_col_s and "HomeTeam" in df_stats.columns and "MatchDate" in df_stats.columns:
            all_teams = sorted(
                set(df_stats["HomeTeam"].dropna()) | set(df_stats.get("AwayTeam", pd.Series()).dropna())
            )
            form_rows = []
            for team in all_teams:
                home_m = df_stats[df_stats["HomeTeam"] == team][["MatchDate", result_col_s]].assign(
                    won=lambda d: d[result_col_s] == "H",
                    draw=lambda d: d[result_col_s] == "D",
                )
                away_m = (
                    df_stats[df_stats["AwayTeam"] == team][["MatchDate", result_col_s]].assign(
                        won=lambda d: d[result_col_s] == "A",
                        draw=lambda d: d[result_col_s] == "D",
                    )
                    if "AwayTeam" in df_stats.columns
                    else pd.DataFrame()
                )
                all_m = (
                    pd.concat([home_m, away_m]).sort_values("MatchDate").tail(5)
                    if not away_m.empty
                    else home_m.tail(5)
                )
                form_str = "".join(
                    "W" if r["won"] else ("D" if r["draw"] else "L")
                    for _, r in all_m.iterrows()
                )
                pts = sum(3 if c == "W" else (1 if c == "D" else 0) for c in form_str)
                form_rows.append({
                    "Team": team,
                    "Conference": get_conference(team),
                    "Surface": "Turf" if is_turf(team) else "Grass",
                    "Last 5": form_str,
                    "Points": pts,
                })
            form_df = pd.DataFrame(form_rows).sort_values("Points", ascending=False)
            st.dataframe(form_df, hide_index=True, height=get_dataframe_height(form_df, max_height=500))

        st.divider()

        # ── Head-to-Head ──────────────────────────────────────────────────────
        st.markdown("### 🏆 Head-to-Head Analyzer")
        if result_col_s and "HomeTeam" in df_stats.columns and "AwayTeam" in df_stats.columns:
            h2h_teams = sorted(df_stats["HomeTeam"].dropna().unique())
            hc1, hc2 = st.columns(2)
            with hc1:
                t1 = st.selectbox("Team 1", h2h_teams, key="h2h_t1")
            with hc2:
                t2 = st.selectbox("Team 2", [t for t in h2h_teams if t != t1], key="h2h_t2")

            if st.button("🔍 Analyze Head-to-Head"):
                mask = (
                    ((df_stats["HomeTeam"] == t1) & (df_stats["AwayTeam"] == t2)) |
                    ((df_stats["HomeTeam"] == t2) & (df_stats["AwayTeam"] == t1))
                )
                h2h_df = df_stats[mask].sort_values("MatchDate", ascending=False).head(10)

                if len(h2h_df) > 0:
                    st.success(f"Found {len(h2h_df)} historical matches between {t1} and {t2}.")

                    goal_cols = [
                        c for c in ["HomeGoals", "FullTimeHomeGoals", "FTHG",
                                    "AwayGoals", "FullTimeAwayGoals", "FTAG"]
                        if c in h2h_df.columns
                    ]
                    show_cols = ["MatchDate", "HomeTeam", "AwayTeam", result_col_s] + goal_cols
                    st.dataframe(
                        h2h_df[[c for c in show_cols if c in h2h_df.columns]],
                        hide_index=True,
                        height=get_dataframe_height(h2h_df),
                    )
                else:
                    st.info(f"No historical matches found between {t1} and {t2}.")

        st.divider()

        # ── MLS-specific: Surface & conference analysis ────────────────────────
        st.markdown("### 🏟️ Surface & Conference Analysis")
        if result_col_s and "HomeTeam" in df_stats.columns:
            df_stats["HomeSurface"] = df_stats["HomeTeam"].apply(
                lambda t: "Turf" if is_turf(t) else "Grass"
            )
            surface_stats = df_stats.groupby("HomeSurface").agg(
                Matches=(result_col_s, "count"),
                HomeWins=(result_col_s, lambda x: (x == "H").sum()),
                Draws=(result_col_s, lambda x: (x == "D").sum()),
                AwayWins=(result_col_s, lambda x: (x == "A").sum()),
            ).reset_index()
            surface_stats["Home Win %"] = (surface_stats["HomeWins"] / surface_stats["Matches"] * 100).round(1)
            surface_stats["Draw %"] = (surface_stats["Draws"] / surface_stats["Matches"] * 100).round(1)
            surface_stats["Away Win %"] = (surface_stats["AwayWins"] / surface_stats["Matches"] * 100).round(1)
            st.dataframe(surface_stats, hide_index=True, width=600)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEAM DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔬 Team Deep Dive")
    st.caption("Detailed performance analysis including MLS-specific factors.")

    all_teams_dd = sorted(EASTERN_CONF | WESTERN_CONF)
    selected_team = st.selectbox("Select a team:", all_teams_dd, key="deep_dive_team")

    conf = get_conference(selected_team)
    surf = "Artificial Turf" if is_turf(selected_team) else "Natural Grass"

    k1, k2, k3 = st.columns(3)
    k1.metric("Conference", conf)
    k2.metric("Home Surface", surf)

    # Travel profile
    travel_rows = [
        {
            "Visiting Team (Away)": opp,
            "Their Conference": get_conference(opp),
            "Miles to Travel": round(travel_distance_miles(selected_team, opp)),
        }
        for opp in all_teams_dd
        if opp != selected_team
    ]
    travel_df = pd.DataFrame(travel_rows).sort_values("Miles to Travel", ascending=False)
    long_haul = (travel_df["Miles to Travel"] > 1500).sum()
    k3.metric("Long-Haul Away Trips (>1500 mi)", long_haul)

    st.divider()

    if path.exists(csv_path):
        df_dd = load_historical_data(csv_path)
        result_col_dd = next(
            (c for c in ["Result", "FullTimeResult", "FTR"] if c in df_dd.columns), None
        )
        if result_col_dd and "HomeTeam" in df_dd.columns and "AwayTeam" in df_dd.columns:
            home_m = df_dd[df_dd["HomeTeam"] == selected_team]
            away_m = df_dd[df_dd["AwayTeam"] == selected_team]
            total = len(home_m) + len(away_m)
            if total > 0:
                wins = (
                    (home_m[result_col_dd] == "H").sum() +
                    (away_m[result_col_dd] == "A").sum()
                )
                draws = (
                    (home_m[result_col_dd] == "D").sum() +
                    (away_m[result_col_dd] == "D").sum()
                )
                st.markdown("**All-Time Record (dataset)**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Matches", total)
                m2.metric("Wins", int(wins))
                m3.metric("Draws", int(draws))
                m4.metric("Win Rate", f"{wins / total:.1%}")
            else:
                st.info(f"No historical data found for {selected_team}.")

    st.divider()
    st.markdown("**Away Travel Profile — visiting opponents**")
    st.dataframe(travel_df, hide_index=True, height=get_dataframe_height(travel_df, max_height=400))

    # ASA xG for selected team
    if ASA_AVAILABLE:
        st.divider()
        st.markdown("**⚽ xG Data — American Soccer Analysis**")
        current_year = datetime.now().year
        with st.spinner("Fetching team xG…"):
            _, team_xg_dd = fetch_asa_xg_data((current_year,))
        if not team_xg_dd.empty:
            name_col = next((c for c in ["team_name", "team"] if c in team_xg_dd.columns), None)
            if name_col:
                team_xg_row = team_xg_dd[
                    team_xg_dd[name_col].str.contains(
                        selected_team.split()[0], case=False, na=False
                    )
                ]
                if not team_xg_row.empty:
                    _xg_col_map = {
                        "team_name": "Team", "count_games": "GP",
                        "goals_for": "Goals", "goals_against": "Goals Against",
                        "goal_difference": "Goal Diff",
                        "xgoals_for": "xG For", "xgoals_against": "xG Against",
                        "xgoal_difference": "xG Diff",
                        "goal_difference_minus_xgoal_difference": "GD-xGD",
                        "points": "Points", "xpoints": "xPoints",
                    }
                    _keep = [c for c in _xg_col_map if c in team_xg_row.columns]
                    st.dataframe(
                        team_xg_row[_keep].rename(columns=_xg_col_map),
                        hide_index=True,
                    )
                else:
                    st.info(f"No xG data found for {selected_team} this season.")
        else:
            st.info("Could not load ASA xG data.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📁 Historical MLS Match Data")
    if not path.exists(csv_path):
        st.warning(
            f"No data file found at `{csv_path}`. "
            "Add your MLS historical match CSV to view data here."
        )
    else:
        df_raw = load_historical_data(csv_path)
        if "MatchDate" in df_raw.columns:
            df_raw = df_raw.sort_values("MatchDate", ascending=False)
            # Strip time component when it is midnight (00:00:00)
            dates = pd.to_datetime(df_raw["MatchDate"], errors="coerce")
            df_raw["MatchDate"] = dates.apply(
                lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) and d.hour == 0 and d.minute == 0 and d.second == 0
                else (d.strftime("%Y-%m-%d %H:%M") if pd.notna(d) else "")
            )

        _raw_col_map = {
            "MatchDate": "Date",
            "HomeTeam": "Home Team",
            "AwayTeam": "Away Team",
            "HomeGoals": "Home Goals",
            "AwayGoals": "Away Goals",
            "home_xgoals": "Home xG",
            "away_xgoals": "Away xG",
            "Season": "Season",
            "Result": "Result",
            "Source": "Source",
            "HomeConference": "Home Conf",
            "AwayConference": "Away Conf",
            "is_cross_conference": "Cross-Conf",
            "home_is_turf": "Home Turf",
            "away_is_turf": "Away Turf",
            "away_travel_miles": "Away Travel (mi)",
            "is_long_haul": "Long Haul",
            "home_xg_l5": "Home xG L5",
            "away_xg_l5": "Away xG L5",
            "home_pts_l5": "Home Pts L5",
            "away_pts_l5": "Away Pts L5",
            "home_goals_for_l5": "Home GF L5",
            "away_goals_for_l5": "Away GF L5",
            "home_goals_against_l5": "Home GA L5",
            "away_goals_against_l5": "Away GA L5",
            "home_xg_l10": "Home xG L10",
            "away_xg_l10": "Away xG L10",
            "home_pts_l10": "Home Pts L10",
            "away_pts_l10": "Away Pts L10",
            "home_rest_days": "Home Rest Days",
            "away_rest_days": "Away Rest Days",
            "home_h2h_win_rate_l5": "H2H Win Rate L5",
            "home_games_from_playoff": "Home Games From Playoff",
            "away_games_from_playoff": "Away Games From Playoff",
            "home_dp_available": "Home DP Available",
            "away_dp_available": "Away DP Available",
            "season_weight": "Season Weight",
        }
        # Keep columns in a sensible order; drop internal IDs
        _priority = [
            "MatchDate", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals",
            "Result", "Season", "home_xgoals", "away_xgoals",
            "HomeConference", "AwayConference", "is_cross_conference",
            "home_is_turf", "away_travel_miles", "is_long_haul",
            "home_xg_l5", "away_xg_l5", "home_pts_l5", "away_pts_l5",
            "home_xg_l10", "away_xg_l10", "home_pts_l10", "away_pts_l10",
            "home_rest_days", "away_rest_days", "home_h2h_win_rate_l5",
            "home_games_from_playoff", "away_games_from_playoff",
            "home_dp_available", "away_dp_available", "Source",
        ]
        _display_cols = [c for c in _priority if c in df_raw.columns]
        df_display = df_raw[_display_cols].rename(columns=_raw_col_map)

        st.write(f"**{len(df_display):,} matches** in dataset")
        st.dataframe(
            df_display,
            height=get_dataframe_height(df_display),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("📖 Data Dictionary"):
            st.markdown("""
| Column | Description |
|---|---|
| **Date** | Match date (YYYY-MM-DD). Time shown only if not midnight. |
| **Home Team** | Name of the home club. |
| **Away Team** | Name of the away club. |
| **Home Goals** | Goals scored by the home team at full time. |
| **Away Goals** | Goals scored by the away team at full time. |
| **Result** | Match outcome from the home team's perspective: **H** = home win, **A** = away win, **D** = draw. |
| **Season** | MLS season year. |
| **Home xG** | Expected goals for the home team (from American Soccer Analysis). NaN for seasons where game-level xG is unavailable. |
| **Away xG** | Expected goals for the away team (from American Soccer Analysis). |
| **Home Conf** | Conference of the home team (Eastern / Western). |
| **Away Conf** | Conference of the away team (Eastern / Western). |
| **Cross-Conf** | 1 if the teams are from different conferences, 0 otherwise. Cross-conference H2H history is thinner due to MLS's unbalanced schedule. |
| **Home Turf** | 1 if the home stadium uses artificial turf, 0 for natural grass. |
| **Away Travel (mi)** | Great-circle distance (miles) the away team travelled to reach the home stadium. |
| **Long Haul** | 1 if away travel exceeded 1,500 miles — a meaningful fatigue signal in MLS. |
| **Home xG L5** | Average xG per game for the home team over their last 5 matches. |
| **Away xG L5** | Average xG per game for the away team over their last 5 matches. |
| **Home Pts L5** | Points accumulated by the home team in their last 5 matches (W=3, D=1, L=0). |
| **Away Pts L5** | Points accumulated by the away team in their last 5 matches. |
| **Home xG L10** | Average xG per game for the home team over their last 10 matches. |
| **Away xG L10** | Average xG per game for the away team over their last 10 matches. |
| **Home Pts L10** | Points accumulated by the home team in their last 10 matches. |
| **Away Pts L10** | Points accumulated by the away team in their last 10 matches. |
| **Home Rest Days** | Days since the home team's previous match. |
| **Away Rest Days** | Days since the away team's previous match. |
| **H2H Win Rate L5** | Home team's win rate in the last 5 head-to-head meetings between these two clubs. |
| **Home Games From Playoff** | How many places the home team sits from the playoff cut line at the time of the match. Positive = inside playoffs, negative = outside. |
| **Away Games From Playoff** | Same metric for the away team. |
| **Home DP Available** | 1 if the home team's designated player(s) were available for this match. |
| **Away DP Available** | 1 if the away team's designated player(s) were available for this match. |
| **Source** | Data origin: **ASA** = American Soccer Analysis, **FD** = football-data.org. |
""")
        
