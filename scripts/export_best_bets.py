"""
scripts/export_best_bets.py — MLS (mls-predictions)
Reads data_files/picks_today.csv (written by scripts/generate_picks.py) and writes
data_files/best_bets_today.json in the unified Sports Picks Grid schema.
"""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

SPORT = "MLS"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH = Path("data_files/best_bets_today.json")
SRC_PATH = Path("data_files/picks_today.csv")


def _write(bets: list, notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _tier_from_edge(edge: float) -> str:
    if edge >= 0.08:
        return "Elite"
    elif edge >= 0.04:
        return "Strong"
    elif edge >= 0.02:
        return "Good"
    return "Standard"


def _decimal_to_american(dec: float) -> int | None:
    try:
        dec = float(dec)
        if dec >= 2.0:
            return round((dec - 1) * 100)
        else:
            return round(-100 / (dec - 1))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def main() -> None:
    today = date.today()

    # MLS season roughly March–November (regular season), Dec–Jan (off)
    month = today.month
    if month in [12, 1, 2]:
        _write([], "MLS off-season")
        return

    if not SRC_PATH.exists():
        _write([], f"picks_today.csv not found — run generate_picks.py first")
        return

    try:
        df = pd.read_csv(SRC_PATH)
    except Exception as e:
        _write([], f"Failed to read {SRC_PATH}: {e}")
        return

    if df.empty:
        _write([], f"No MLS picks for {today}")
        return

    bets = []
    for _, row in df.iterrows():
        edge_raw = row.get("Edge", 0)
        try:
            edge = float(edge_raw) / 100 if abs(float(edge_raw)) > 1 else float(edge_raw)
        except (TypeError, ValueError):
            edge = 0.0

        if edge < 0.02:
            continue

        tier = _tier_from_edge(edge)
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        game = f"{away} @ {home}"
        bet_outcome = str(row.get("Bet", "Home Win"))

        conf_raw = row.get("Model", 0)
        try:
            conf = float(conf_raw) / 100 if float(conf_raw) > 1 else float(conf_raw)
        except (TypeError, ValueError):
            conf = None

        odds = _decimal_to_american(row.get("Odds"))

        bet: dict = {
            "game_date": str(today),
            "game_time": None,
            "game": game,
            "home_team": home,
            "away_team": away,
            "bet_type": "Match Result",
            "pick": bet_outcome,
            "confidence": conf,
            "edge": edge,
            "tier": tier,
            "odds": odds,
            "line": None,
            "league": "MLS",
        }
        bets.append(bet)

    _write(bets, "" if bets else f"No qualifying MLS picks for {today}")


if __name__ == "__main__":
    main()
