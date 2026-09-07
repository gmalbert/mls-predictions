"""Frontier MLS analytics and model-governance product surface."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.roadmap import (
    all_star_probability,
    conference_table,
    designated_player_impact,
    expansion_tracker,
    expected_points_table,
    form_heatmap,
    home_away_rates,
    player_prop_probabilities,
    playoff_simulation,
    shot_location_zones,
    travel_history,
    xg_timeline,
)
from database.db_manager import DatabaseManager
from footer import add_betting_oracle_footer
from models.frontier_features import STADIUMS, haversine_km, load_optional_sources
from themes import apply_theme

DATA_DIR = ROOT / "data_files"
RAW_DIR = DATA_DIR / "raw"

st.set_page_config(page_title="Frontier Analytics | MLS Predictor", page_icon="⚽", layout="wide")
apply_theme()


@st.cache_data(ttl=3_600)
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


@st.cache_data(ttl=3_600)
def load_report(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def frame_height(frame: pd.DataFrame, maximum: int = 500) -> int:
    return min(38 + max(len(frame), 1) * 35, maximum)


def last_rest_days(history: pd.DataFrame, team: str, fixture_date: pd.Timestamp) -> int | None:
    if history.empty or pd.isna(fixture_date):
        return None
    dates = history.loc[
        ((history["HomeTeam"] == team) | (history["AwayTeam"] == team))
        & (history["MatchDate"] < fixture_date),
        "MatchDate",
    ]
    return int((fixture_date - dates.max()).days) if not dates.empty else None


def fixture_total_estimate(history: pd.DataFrame, home: str, away: str) -> tuple[float, float]:
    """Return expected total goals and P(over 2.5) from strictly prior form."""
    if history.empty:
        expected = 2.8
    else:
        values = []
        for team in (home, away):
            recent = history[(history["HomeTeam"] == team) | (history["AwayTeam"] == team)].tail(10)
            team_values = []
            for row in recent.itertuples(index=False):
                is_home = str(row.HomeTeam) == team
                goal = float(getattr(row, "HomeGoals" if is_home else "AwayGoals", 0) or 0)
                xg = getattr(row, "home_xgoals" if is_home else "away_xgoals", goal)
                team_values.append(float(xg) if pd.notna(xg) else goal)
            values.append(float(np.mean(team_values)) if team_values else 1.4)
        expected = float(sum(values))
    return expected, float(1.0 - poisson.cdf(2, max(expected, 0.1)))


def release_panel(report: dict) -> None:
    gate = report.get("release_gate", {})
    passed = bool(gate.get("passed", False))
    if passed:
        st.success("RELEASED — all untouched-season, market-relative, CLV, and ledger gates passed.")
    else:
        st.warning("NO BET / PAPER ONLY — the production release gate has not passed.")
        reasons = gate.get("reasons", ["Run scripts/run_backtest.py to generate the first measured release report."])
        for reason in reasons:
            st.caption(f"• {reason}")
    deployment = report.get("deployment", {})
    if deployment:
        candidate = str(deployment.get("selected_candidate", "none")).upper()
        reason = deployment.get("reason", "No deployment comparison is available.")
        if deployment.get("eligible", False):
            st.success(f"Deployment candidate: {candidate}. {reason}")
        else:
            st.warning(f"Deployment candidate: {candidate}; model picks disabled. {reason}")


history = load_csv(str(DATA_DIR / "combined_historical_data.csv"))
fixtures = load_csv(str(DATA_DIR / "upcoming_fixtures.csv"))
picks = load_csv(str(DATA_DIR / "picks_today.csv"))
report = load_report(str(DATA_DIR / "backtests" / "latest_report.json"))
readiness = load_report(str(DATA_DIR / "quality" / "latest_readiness.json"))
sources = load_optional_sources(RAW_DIR)

if not history.empty:
    history["MatchDate"] = pd.to_datetime(history["MatchDate"], errors="coerce")
    history = history.dropna(subset=["MatchDate"]).sort_values("MatchDate")
if not fixtures.empty:
    date_column = next((column for column in ("Date", "MatchDate", "date") if column in fixtures.columns), None)
    fixtures["_date"] = pd.to_datetime(fixtures[date_column], errors="coerce") if date_column else pd.NaT

st.title("⚽ Frontier MLS Analytics")
st.caption("Point-in-time league context, transparent uncertainty, and market-aware evaluation")
release_panel(report)
if readiness:
    summary = readiness.get("context", {}).get("summary", {})
    odds_quality = readiness.get("market_odds", {})
    st.caption(
        "Data readiness — "
        f"context: {summary.get('ready', 0)} ready, {summary.get('stale', 0)} stale, {summary.get('missing', 0)} missing; "
        f"market odds: {odds_quality.get('status', 'unknown')} ({odds_quality.get('closing_rows', 0)} closing rows)."
    )

slate_tab, team_tab, betting_tab, league_tab, analysis_tab, frontier_tab = st.tabs(
    ["Match Day", "Team Profiles", "Betting Tools", "League & Playoffs", "Historical Analysis", "Frontier Lab"]
)

with slate_tab:
    st.subheader("Today's slate and travel environment")
    if fixtures.empty:
        st.info("No fixture file is available. Run `python fetch_upcoming_fixtures.py` to refresh it.")
    else:
        today = pd.Timestamp.now().normalize()
        slate = fixtures[fixtures["_date"].dt.normalize() == today]
        if slate.empty:
            future = fixtures[fixtures["_date"] >= today].sort_values("_date")
            slate = future.head(8) if not future.empty else fixtures.sort_values("_date").tail(8)
            st.caption("No match is listed today; showing the nearest available fixture records.")
        for _, fixture in slate.iterrows():
            home, away = str(fixture.get("HomeTeam", "")), str(fixture.get("AwayTeam", ""))
            home_meta, away_meta = STADIUMS.get(home), STADIUMS.get(away)
            miles = haversine_km(away, home) * 0.621371
            home_rest = last_rest_days(history, home, fixture["_date"])
            away_rest = last_rest_days(history, away, fixture["_date"])
            cross = bool(home_meta and away_meta and home_meta.conference != away_meta.conference)
            surface = home_meta.surface.title() if home_meta else str(fixture.get("HomeSurface", "Unknown"))
            altitude = home_meta.altitude_ft if home_meta else 0
            expected_total, over_probability = fixture_total_estimate(history, home, away)
            pick_match = picks[
                (picks.get("HomeTeam", pd.Series(dtype=str)).astype(str) == home)
                & (picks.get("AwayTeam", pd.Series(dtype=str)).astype(str) == away)
            ] if not picks.empty and {"HomeTeam", "AwayTeam"}.issubset(picks.columns) else pd.DataFrame()
            with st.container(border=True):
                left, middle, right = st.columns([2.2, 2, 1.4])
                with left:
                    st.markdown(f"### {away} at {home}")
                    st.caption(f"{fixture['_date']:%a, %b %d} · {fixture.get('Time', 'TBD')} · {fixture.get('Venue', 'Venue TBD')}")
                with middle:
                    badges = [f"{surface} surface", f"{altitude:,} ft altitude", f"{miles:,.0f} away miles"]
                    if cross:
                        badges.append("Cross-conference")
                    if away_rest is not None and away_rest <= 3:
                        badges.append("Away short rest")
                    if home_rest is not None and home_rest <= 3:
                        badges.append("Home short rest")
                    st.write(" · ".join(badges))
                    if home_meta and away_meta and away_meta.surface == "turf" and home_meta.surface == "grass":
                        st.warning("Turf-to-grass visitor transition")
                with right:
                    if not pick_match.empty:
                        snapshot_row = pick_match.iloc[-1]
                        st.write(
                            f"H {float(snapshot_row.get('home_prob', 0)):.0%} · "
                            f"D {float(snapshot_row.get('draw_prob', 0)):.0%} · "
                            f"A {float(snapshot_row.get('away_prob', 0)):.0%}"
                        )
                    st.metric("Expected total", f"{expected_total:.2f}", f"Over 2.5: {over_probability:.0%}")
                    for label, column in (("H", "best_home_odds"), ("D", "best_draw_odds"), ("A", "best_away_odds")):
                        if pd.notna(fixture.get(column)):
                            st.metric(f"{label} price", f"{float(fixture[column]):+g}")

    st.subheader("Probability and status snapshot")
    if picks.empty:
        st.info("No governed pick snapshot exists for today's slate. The match cards remain informational.")
    else:
        columns = [column for column in (
            "HomeTeam", "AwayTeam", "Bet", "State", "Model", "Edge",
            "over_2_5_probability", "btts_probability", "prediction_set",
            "cross_market_consistent", "StakeUnits", "NoBetReasons",
        ) if column in picks.columns]
        st.dataframe(picks[columns], hide_index=True, use_container_width=True, height=frame_height(picks))

with team_tab:
    st.subheader("Team profile")
    teams = sorted(set(history.get("HomeTeam", pd.Series(dtype=str))).union(history.get("AwayTeam", pd.Series(dtype=str))).union(STADIUMS))
    if not teams:
        st.info("Historical match data is required for team profiles.")
    else:
        selected_team = st.selectbox("Club", teams, key="frontier_team")
        metadata = STADIUMS.get(selected_team)
        team_matches = history[(history["HomeTeam"] == selected_team) | (history["AwayTeam"] == selected_team)].copy()
        team_matches = team_matches.sort_values("MatchDate")
        recent = team_matches.tail(10)
        metric_columns = st.columns(4)
        metric_columns[0].metric("Conference", metadata.conference if metadata else "Unknown")
        metric_columns[1].metric("Home altitude", f"{metadata.altitude_ft:,} ft" if metadata else "Unknown")
        metric_columns[2].metric("Surface", metadata.surface.title() if metadata else "Unknown")
        metric_columns[3].metric("Expansion year", metadata.expansion_year if metadata else "Unknown")

        if not team_matches.empty:
            xg_rows = []
            for match in team_matches.itertuples(index=False):
                is_home = str(match.HomeTeam) == selected_team
                goals = float(getattr(match, "HomeGoals" if is_home else "AwayGoals", 0) or 0)
                xg_value = getattr(match, "home_xgoals" if is_home else "away_xgoals", goals)
                xg_rows.append({"MatchDate": match.MatchDate, "xG": float(xg_value) if pd.notna(xg_value) else goals})
            xg_frame = pd.DataFrame(xg_rows)
            xg_frame["Rolling xG"] = xg_frame["xG"].rolling(5, min_periods=1).mean()
            st.plotly_chart(px.line(xg_frame, x="MatchDate", y=["xG", "Rolling xG"], title="ASA xG trend"), use_container_width=True)
            left, right = st.columns(2)
            with left:
                st.markdown("#### Recent results")
                result_columns = [column for column in ("MatchDate", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "Result") if column in recent.columns]
                st.dataframe(recent[result_columns].sort_values("MatchDate", ascending=False), hide_index=True, use_container_width=True, height=frame_height(recent, 410))
            with right:
                st.markdown("#### Home advantage")
                current_season = int(history["MatchDate"].dt.year.max())
                season_history = history[history["MatchDate"].dt.year == current_season]
                rates = home_away_rates(season_history)
                team_rate = rates[rates["Team"] == selected_team]
                st.dataframe(team_rate, hide_index=True, use_container_width=True, height=frame_height(team_rate, 180))
                travel = travel_history(history, current_season)
                st.dataframe(travel[travel["Team"] == selected_team], hide_index=True, use_container_width=True, height=150)

        roster = sources.get("availability", pd.DataFrame())
        if not roster.empty and "team" in roster.columns:
            roster_view = roster[roster["team"] == selected_team].copy()
            st.markdown("#### DP / U22 / TAM availability and replacement waterfall")
            if roster_view.empty:
                st.info("No roster mechanism records are available for this club.")
            else:
                value_col = next((column for column in ("replacement_value", "goals_added_90", "war") if column in roster_view.columns), None)
                name_col = next((column for column in ("player", "player_name", "name") if column in roster_view.columns), None)
                if value_col and name_col:
                    chart = px.bar(roster_view, x=name_col, y=value_col, color=roster_view.get("status"), title="Minutes-weighted replacement impact")
                    st.plotly_chart(chart, use_container_width=True)
                st.dataframe(roster_view, hide_index=True, use_container_width=True, height=frame_height(roster_view))
        else:
            st.info("Roster feed not configured. Unknown availability is modeled as uncertainty, not as healthy.")
        news = sources.get("news", pd.DataFrame())
        if not news.empty and "team" in news.columns:
            st.markdown("#### Timestamped news / sentiment signals")
            st.dataframe(news[news["team"] == selected_team].tail(20), hide_index=True, use_container_width=True, height=frame_height(news.tail(20), 350))

with betting_tab:
    st.subheader("Value finder and release controls")
    release_panel(report)
    if not picks.empty:
        minimum_edge = st.slider("Minimum displayed edge", 0.0, 15.0, 3.0, 0.5)
        edge_values = pd.to_numeric(picks.get("Edge", 0), errors="coerce").fillna(0)
        value = picks[edge_values >= minimum_edge]
        st.dataframe(value, hide_index=True, use_container_width=True, height=frame_height(value))
    else:
        st.caption("The 3% value rule is enforced when a governed snapshot is generated.")

    st.markdown("#### Totals environment filter")
    if not fixtures.empty:
        environment_rows = []
        for _, fixture in fixtures.iterrows():
            home = str(fixture.get("HomeTeam", ""))
            metadata = STADIUMS.get(home)
            environment_rows.append({
                "Date": fixture.get("Date", fixture.get("MatchDate")),
                "Match": f"{fixture.get('AwayTeam', '')} at {home}",
                "Surface": metadata.surface.title() if metadata else fixture.get("HomeSurface", "Unknown"),
                "Altitude": metadata.altitude_ft if metadata else 0,
                "Dome": metadata.dome if metadata else False,
                "Totals Context": "High-altitude" if metadata and metadata.altitude_ft >= 3_500 else ("Turf" if metadata and metadata.surface == "turf" else "Standard"),
            })
        environment = pd.DataFrame(environment_rows)
        selected_context = st.multiselect("Environment", sorted(environment["Totals Context"].unique()), default=sorted(environment["Totals Context"].unique()))
        st.dataframe(environment[environment["Totals Context"].isin(selected_context)], hide_index=True, use_container_width=True, height=frame_height(environment))

    st.markdown("#### Frozen selection ledger")
    db_path = DATA_DIR / "mls.db"
    if db_path.exists():
        ledger = DatabaseManager(str(db_path)).get_bet_ledger()
        if ledger.empty:
            st.info("No frozen paper selections yet.")
        else:
            st.dataframe(ledger, hide_index=True, use_container_width=True, height=frame_height(ledger))
    else:
        st.info("The ledger will be created on the first generated snapshot.")

with league_tab:
    st.subheader("Conference table and playoff race")
    if history.empty:
        st.info("Historical results are required.")
    else:
        available_seasons = sorted(history["MatchDate"].dt.year.unique(), reverse=True)
        season = st.selectbox("Season", available_seasons, key="league_season")
        season_matches = history[history["MatchDate"].dt.year == season]
        standings = conference_table(season_matches, season)
        east, west = st.columns(2)
        with east:
            st.markdown("#### Eastern Conference")
            st.dataframe(standings[standings["Conference"] == "Eastern"], hide_index=True, use_container_width=True, height=frame_height(standings, 600))
        with west:
            st.markdown("#### Western Conference")
            st.dataframe(standings[standings["Conference"] == "Western"], hide_index=True, use_container_width=True, height=frame_height(standings, 600))
        st.markdown("#### 10,000-run playoff bracket simulation")
        if st.checkbox("Run the 10,000-path simulation", key=f"run_playoff_sim_{season}"):
            ppg = standings.set_index("Team").apply(
                lambda row: float(row["Pts"]) / max(float(row["P"]), 1.0), axis=1
            ) if not standings.empty else pd.Series(dtype=float)
            league_ppg = float(ppg.mean()) if not ppg.empty else 1.4
            ratings = {team: 1_500.0 + 100.0 * (value - league_ppg) for team, value in ppg.items()}
            remaining = pd.DataFrame()
            if not fixtures.empty:
                season_end = pd.Timestamp(f"{season}-12-31")
                latest_result = season_matches["MatchDate"].max()
                remaining = fixtures[(fixtures["_date"] > latest_result) & (fixtures["_date"] <= season_end)].copy()
            simulations = playoff_simulation(
                standings,
                ratings=ratings,
                remaining_fixtures=remaining,
                simulations=10_000,
            )
            if not simulations.empty:
                st.plotly_chart(px.bar(simulations.head(15), x="Team", y="mls_cup", color="conference_win", title="MLS Cup probability"), use_container_width=True)
                st.dataframe(simulations, hide_index=True, use_container_width=True, height=frame_height(simulations))
        else:
            st.caption("Run on demand to keep the dashboard responsive; the simulation always uses 10,000 paths.")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Expected points vs actual")
            xpts = expected_points_table(season_matches)
            st.dataframe(xpts, hide_index=True, use_container_width=True, height=frame_height(xpts))
        with right:
            st.markdown("#### Expansion-team tracker")
            expansions = expansion_tracker(history)
            st.dataframe(expansions, hide_index=True, use_container_width=True, height=frame_height(expansions))
        st.markdown("#### Designated-player on/off impact")
        dp_impact = designated_player_impact(season_matches)
        if dp_impact.empty:
            st.info("Confirmed availability history is not yet large enough for an on/off comparison.")
        else:
            st.dataframe(dp_impact, hide_index=True, use_container_width=True, height=frame_height(dp_impact))

with analysis_tab:
    st.subheader("Backtest, calibration, and feature segments")
    if not report:
        st.info("Run `python scripts/run_backtest.py` to generate the reproducible audit.")
    else:
        aggregate = report.get("aggregate", {})
        aggregate_frame = pd.DataFrame(aggregate).T.reset_index(names="Baseline") if aggregate else pd.DataFrame()
        if not aggregate_frame.empty:
            st.dataframe(aggregate_frame, hide_index=True, use_container_width=True, height=frame_height(aggregate_frame))
            st.plotly_chart(px.bar(aggregate_frame, x="Baseline", y=["log_loss", "brier", "rps"], barmode="group", title="Model vs Elo, market, Dixon–Coles, and home-field baselines"), use_container_width=True)
        betting = report.get("betting", {})
        if betting:
            st.markdown("#### Paper-trading recap")
            cols = st.columns(4)
            cols[0].metric("Selections", betting.get("selections", 0))
            cols[1].metric("Turnover", f"{betting.get('turnover_units', 0):.1f}u")
            roi = betting.get("roi")
            cols[2].metric("ROI", "Unavailable" if roi is None else f"{roi:.1%}")
            drawdown = betting.get("max_drawdown_units")
            cols[3].metric("Max drawdown", "Unavailable" if drawdown is None else f"{drawdown:.2f}u")
        folds = report.get("folds", [])
        if folds:
            season_rows = []
            for fold in folds:
                model_fold = fold.get("model", {})
                betting_fold = fold.get("betting", {})
                season_rows.append({
                    "Season": fold.get("season"),
                    "Trained Through": fold.get("trained_through"),
                    "Calibration Matches": fold.get("calibration_matches"),
                    "Untouched Matches": fold.get("untouched_matches"),
                    "Full Untouched Season": fold.get("full_untouched_season", False),
                    "Accuracy": model_fold.get("accuracy"),
                    "Log Loss": model_fold.get("log_loss"),
                    "Brier": model_fold.get("brier"),
                    "ROI": betting_fold.get("roi"),
                    "Selections": betting_fold.get("selections", 0),
                })
            st.markdown("#### Season-by-season accuracy and ROI")
            st.dataframe(pd.DataFrame(season_rows), hide_index=True, use_container_width=True, height=frame_height(pd.DataFrame(season_rows)))
        segments = report.get("segments", {})
        if segments:
            segment_name = st.selectbox("Accuracy segment", list(segments), key="segment_name")
            segment_frame = pd.DataFrame(segments[segment_name]).T.reset_index(names="Bucket")
            st.dataframe(segment_frame, hide_index=True, use_container_width=True, height=frame_height(segment_frame))
            st.plotly_chart(px.bar(segment_frame, x="Bucket", y=["accuracy", "ece"], barmode="group", title=f"{segment_name} calibration split"), use_container_width=True)

    if not history.empty:
        st.markdown("#### Team × week form heatmap")
        heatmap_data = form_heatmap(history, weeks=20)
        if not heatmap_data.empty:
            figure = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns.astype(str), y=heatmap_data.index, colorscale="RdYlGn", zmid=0))
            figure.update_layout(height=max(500, len(heatmap_data) * 22), xaxis_title="Week", yaxis_title="Team")
            st.plotly_chart(figure, use_container_width=True)

        st.markdown("#### Salary-cap roster mechanism comparison")
        st.caption("This comparison uses DP/U22/TAM availability tiers—not squad value, wages, or transfer fees.")
        salary_segment = report.get("segments", {}).get("salary_cap_roster_tier", {}) if report else {}
        if salary_segment:
            salary_frame = pd.DataFrame(salary_segment).T.reset_index(names="Roster Tier")
            st.dataframe(salary_frame, hide_index=True, use_container_width=True, height=frame_height(salary_frame))
        else:
            st.info("Timestamped DP/U22/TAM history is not yet sufficient for a roster-tier accuracy split.")

with frontier_tab:
    st.subheader("Scenarios, probability changes, event shape, and special models")
    db_path = DATA_DIR / "mls.db"
    if db_path.exists():
        db = DatabaseManager(str(db_path))
        with db._connect() as connection:
            snapshot_ids = pd.read_sql_query("SELECT DISTINCT fixture_id FROM prediction_snapshots ORDER BY fixture_id", connection)
        if not snapshot_ids.empty:
            fixture_id = st.selectbox("Snapshot fixture", snapshot_ids["fixture_id"].tolist())
            snapshots = db.get_prediction_snapshots(fixture_id)
            probability_rows = []
            for _, snapshot in snapshots.iterrows():
                probability_rows.append({"prediction_time": snapshot["prediction_time"], "scenario": snapshot["lineup_scenario"], **snapshot["probabilities"]})
            probability_frame = pd.DataFrame(probability_rows)
            st.plotly_chart(px.line(probability_frame, x="prediction_time", y=[column for column in ("home", "draw", "away") if column in probability_frame], markers=True, title="Probability-change feed"), use_container_width=True)
            if len(snapshots) >= 2:
                previous_probabilities = snapshots.iloc[-2]["probabilities"]
                current_probabilities = snapshots.iloc[-1]["probabilities"]
                changes = pd.DataFrame([
                    {
                        "Outcome": outcome.title(),
                        "Previous": previous_probabilities.get(outcome, 0.0),
                        "Current": current_probabilities.get(outcome, 0.0),
                        "Change (pp)": 100.0 * (current_probabilities.get(outcome, 0.0) - previous_probabilities.get(outcome, 0.0)),
                    }
                    for outcome in ("home", "draw", "away")
                ])
                st.dataframe(changes, hide_index=True, use_container_width=True, height=frame_height(changes, 180))
                explanations = snapshots.iloc[-1].get("explanations", [])
                for explanation in explanations:
                    st.caption(f"• {explanation}")
            st.dataframe(snapshots, hide_index=True, use_container_width=True, height=frame_height(snapshots))
        else:
            st.info("Generate a pick snapshot to start the probability-change feed.")
    else:
        st.info("Generate a pick snapshot to start the probability-change feed.")

    st.markdown("#### Questionable-starter scenario mixture")
    if not picks.empty and {"home_prob", "draw_prob", "away_prob"}.issubset(picks.columns):
        scenario_match = st.selectbox(
            "Fixture scenario",
            list(range(len(picks))),
            format_func=lambda index: f"{picks.iloc[index].get('AwayTeam', '')} at {picks.iloc[index].get('HomeTeam', '')}",
        )
        base = picks.iloc[scenario_match]
        start_probability = st.slider("Questionable starter plays", 0.0, 1.0, 0.5, 0.05)
        home_impact = st.slider("Home win impact if absent (percentage points)", -15.0, 15.0, -4.0, 0.5) / 100
        available = np.array([base["home_prob"], base["draw_prob"], base["away_prob"]], dtype=float)
        absent = available.copy()
        absent[0] = np.clip(absent[0] + home_impact, 0, 1)
        remainder = 1.0 - absent[0]
        other = available[1:].sum()
        absent[1:] = available[1:] / other * remainder if other > 0 else remainder / 2
        mixture = start_probability * available + (1.0 - start_probability) * absent
        scenario_frame = pd.DataFrame({"Outcome": ["Home", "Draw", "Away"], "Projected lineup": available, "Starter absent": absent, "Mixture": mixture})
        st.plotly_chart(px.bar(scenario_frame, x="Outcome", y=["Projected lineup", "Starter absent", "Mixture"], barmode="group"), use_container_width=True)
    else:
        st.info("Generate today's snapshot to explore questionable-starter scenarios.")

    events = sources.get("events", pd.DataFrame())
    if not events.empty:
        event_team_col = next((column for column in ("team", "Team", "team_name") if column in events.columns), None)
        event_teams = sorted(events[event_team_col].dropna().astype(str).unique()) if event_team_col else []
        event_team = st.selectbox("Event-data club", ["All clubs", *event_teams]) if event_teams else "All clubs"
        timeline = xg_timeline(events, None if event_team == "All clubs" else event_team)
        zones = shot_location_zones(events, None if event_team == "All clubs" else event_team)
        if not timeline.empty:
            st.plotly_chart(px.bar(timeline, x="time_bucket", y="xg_share", color="Team", title="When teams create xG"), use_container_width=True)
        if not zones.empty:
            st.plotly_chart(px.treemap(zones, path=["Team", "Zone"], values="xg", color="xg", title="Shot-location xG zones"), use_container_width=True)
    else:
        st.info("Possession-chain and shot event feed not configured; timing and pitch-zone charts remain data-empty.")

    st.markdown("#### Player goal and assist research")
    players = load_csv(str(RAW_DIR / "player_props.csv"))
    if players.empty:
        st.warning("Player props remain disabled: no timestamped shots/target-share/market history is available.")
    else:
        team_xg = st.number_input("Team expected goals", 0.1, 5.0, 1.5, 0.1)
        props = player_prop_probabilities(players, team_xg)
        st.dataframe(props, hide_index=True, use_container_width=True, height=frame_height(props))

    st.markdown("#### MLS All-Star special-event model")
    col_a, col_b, col_c = st.columns(3)
    mls_rating = col_a.number_input("MLS XI rating", 1_000, 2_500, 1_650)
    opponent_rating = col_b.number_input("Opponent rating", 1_000, 2_500, 1_750)
    motivation = col_c.slider("Motivation adjustment", -1.0, 1.0, 0.0, 0.1)
    st.metric("MLS All-Star win probability", f"{all_star_probability(mls_rating, opponent_rating, motivation):.1%}")
    st.caption("This special-event estimate is isolated from MLS club training and is never added to the betting ledger automatically.")

add_betting_oracle_footer()
