"""
database/db_manager.py
SQLite persistence layer for MLS match history, fixtures, predictions, and odds snapshots.

Usage:
    from database.db_manager import DatabaseManager

    db = DatabaseManager()
    db.migrate_from_csv("data_files/combined_historical_data.csv")
    matches = db.query_matches(home_team="LA Galaxy", limit=10)
    h2h     = db.get_head_to_head("LA Galaxy", "LAFC", limit=10)
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

DB_PATH = os.path.join("data_files", "mls.db")


class DatabaseManager:
    """SQLite database manager for MLS prediction data."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._initialize_schema()

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS matches (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_date  TEXT,
                    home_team   TEXT,
                    away_team   TEXT,
                    home_goals  REAL,
                    away_goals  REAL,
                    result      TEXT,
                    home_xg     REAL,
                    away_xg     REAL,
                    season      INTEGER,
                    source      TEXT,
                    UNIQUE(match_date, home_team, away_team)
                );
                CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_team);
                CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_team);
                CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);

                CREATE TABLE IF NOT EXISTS fixtures (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_date      TEXT,
                    home_team       TEXT,
                    away_team       TEXT,
                    kick_off        TEXT,
                    venue           TEXT,
                    home_surface    TEXT,
                    travel_miles    REAL,
                    best_home_odds  REAL,
                    best_draw_odds  REAL,
                    best_away_odds  REAL,
                    fetched_at      TEXT,
                    UNIQUE(match_date, home_team, away_team)
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_date       TEXT,
                    home_team        TEXT,
                    away_team        TEXT,
                    home_win_prob    REAL,
                    draw_prob        REAL,
                    away_win_prob    REAL,
                    predicted_result TEXT,
                    confidence_pct   REAL,
                    risk_score       REAL,
                    model_version    TEXT,
                    created_at       TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS odds_snapshots (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_at    TEXT,
                    match_date     TEXT,
                    home_team      TEXT,
                    away_team      TEXT,
                    best_home_odds REAL,
                    best_draw_odds REAL,
                    best_away_odds REAL
                );
                """
            )

    # ── Match queries ──────────────────────────────────────────────────────────

    def query_matches(
        self,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        season: Optional[int] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Query historical matches with optional filters."""
        sql = "SELECT * FROM matches WHERE 1=1"
        params: list = []
        if home_team:
            sql += " AND home_team = ?"
            params.append(home_team)
        if away_team:
            sql += " AND away_team = ?"
            params.append(away_team)
        if season:
            sql += " AND season = ?"
            params.append(season)
        sql += " ORDER BY match_date DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_head_to_head(
        self,
        team_a: str,
        team_b: str,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Return the last N meetings between two teams regardless of home/away."""
        sql = """
            SELECT * FROM matches
            WHERE (home_team = ? AND away_team = ?)
               OR (home_team = ? AND away_team = ?)
            ORDER BY match_date DESC
            LIMIT ?
        """
        with self._connect() as conn:
            return pd.read_sql_query(
                sql, conn, params=[team_a, team_b, team_b, team_a, limit]
            )

    def get_recent_form(self, team: str, n: int = 5) -> pd.DataFrame:
        """Return the team's last N matches (home or away)."""
        sql = """
            SELECT * FROM matches
            WHERE home_team = ? OR away_team = ?
            ORDER BY match_date DESC
            LIMIT ?
        """
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=[team, team, n])

    def get_team_stats(self, team: str, season: Optional[int] = None) -> dict:
        """Return aggregate win/draw/loss stats for a team."""
        season_filter = ""
        base_params: list = [team, team, team, team]
        if season:
            season_filter = " AND season = ?"
            base_params.append(season)
        sql = f"""
            SELECT
                COUNT(*) AS played,
                SUM(
                    CASE
                        WHEN (home_team = ? AND result = 'H')
                          OR (away_team = ? AND result = 'A') THEN 1
                        ELSE 0
                    END
                ) AS wins,
                SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) AS draws
            FROM matches
            WHERE (home_team = ? OR away_team = ?){season_filter}
        """
        with self._connect() as conn:
            row = conn.execute(sql, base_params).fetchone()
            if row:
                played = int(row["played"] or 0)
                wins = int(row["wins"] or 0)
                draws = int(row["draws"] or 0)
                return {
                    "played": played,
                    "wins": wins,
                    "draws": draws,
                    "losses": played - wins - draws,
                    "win_rate": round(wins / played, 3) if played > 0 else 0.0,
                }
        return {}

    # ── Prediction storage ─────────────────────────────────────────────────────

    def store_prediction(
        self,
        match_date: str,
        home_team: str,
        away_team: str,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        confidence_pct: float,
        risk_score: float,
        model_version: str = "ensemble_v1",
    ) -> None:
        """Persist a single prediction row."""
        probs = [home_win_prob, draw_prob, away_win_prob]
        result_labels = ["H", "D", "A"]
        predicted = result_labels[probs.index(max(probs))]
        sql = """
            INSERT OR REPLACE INTO predictions
                (match_date, home_team, away_team, home_win_prob, draw_prob, away_win_prob,
                 predicted_result, confidence_pct, risk_score, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(
                sql,
                [
                    match_date, home_team, away_team, home_win_prob, draw_prob,
                    away_win_prob, predicted, confidence_pct, risk_score, model_version,
                ],
            )

    def get_prediction_accuracy(self) -> dict:
        """Compare stored predictions against actual match results in the matches table."""
        sql = """
            SELECT
                COUNT(*) AS total,
                SUM(
                    CASE WHEN p.predicted_result = m.result THEN 1 ELSE 0 END
                ) AS correct
            FROM predictions p
            JOIN matches m
                ON p.match_date  = m.match_date
               AND p.home_team   = m.home_team
               AND p.away_team   = m.away_team
        """
        with self._connect() as conn:
            row = conn.execute(sql).fetchone()
            if row and row["total"]:
                total = int(row["total"])
                correct = int(row["correct"] or 0)
                return {
                    "total": total,
                    "correct": correct,
                    "accuracy": round(correct / total, 3),
                }
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    # ── Odds snapshots ─────────────────────────────────────────────────────────

    def store_odds_snapshot(self, fixtures_df: pd.DataFrame) -> None:
        """Persist current odds for all fixtures (line-movement tracking)."""
        now = datetime.utcnow().isoformat()
        rows = []
        for _, row in fixtures_df.iterrows():
            rows.append(
                (
                    now,
                    str(row.get("Date", "")),
                    str(row.get("HomeTeam", "")),
                    str(row.get("AwayTeam", "")),
                    row.get("best_home_odds"),
                    row.get("best_draw_odds"),
                    row.get("best_away_odds"),
                )
            )
        sql = """
            INSERT INTO odds_snapshots
                (snapshot_at, match_date, home_team, away_team,
                 best_home_odds, best_draw_odds, best_away_odds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.executemany(sql, rows)

    def get_line_movement(self, home_team: str, away_team: str) -> pd.DataFrame:
        """Return timestamped odds history for a specific fixture."""
        sql = """
            SELECT snapshot_at, best_home_odds, best_draw_odds, best_away_odds
            FROM odds_snapshots
            WHERE home_team = ? AND away_team = ?
            ORDER BY snapshot_at
        """
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=[home_team, away_team])

    # ── CSV migration ──────────────────────────────────────────────────────────

    def migrate_from_csv(self, csv_path: str) -> int:
        """
        One-time import of combined_historical_data.csv into the matches table.
        Returns the number of rows inserted.
        """
        if not os.path.exists(csv_path):
            print(f"[DB] CSV not found: {csv_path}")
            return 0
        df = pd.read_csv(csv_path)
        col_map = {
            "MatchDate":    "match_date",
            "HomeTeam":     "home_team",
            "AwayTeam":     "away_team",
            "HomeGoals":    "home_goals",
            "AwayGoals":    "away_goals",
            "Result":       "result",
            "home_xgoals":  "home_xg",
            "away_xgoals":  "away_xg",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "match_date" not in df.columns:
            print("[DB] match_date column not found — skipping migration.")
            return 0

        df["match_date"] = df["match_date"].astype(str).str[:10]
        if "season" not in df.columns:
            df["season"] = pd.to_datetime(df["match_date"], errors="coerce").dt.year

        keep = [
            "match_date", "home_team", "away_team", "home_goals", "away_goals",
            "result", "home_xg", "away_xg", "season", "source",
        ]
        df = df[[c for c in keep if c in df.columns]]

        inserted = 0
        with self._connect() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO matches
                            (match_date, home_team, away_team, home_goals, away_goals,
                             result, home_xg, away_xg, season, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            row.get("match_date"), row.get("home_team"),
                            row.get("away_team"),  row.get("home_goals"),
                            row.get("away_goals"), row.get("result"),
                            row.get("home_xg"),    row.get("away_xg"),
                            row.get("season"),     row.get("source", "CSV"),
                        ],
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass  # duplicate row — skip
        print(f"[DB] Migrated {inserted} rows from {csv_path} → {self.db_path}")
        return inserted
