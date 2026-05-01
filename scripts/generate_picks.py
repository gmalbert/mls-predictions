"""
scripts/generate_picks.py — MLS (mls-predictions)
Runs the VotingClassifier ensemble against today's upcoming fixtures and
writes data_files/picks_today.csv for export_best_bets.py to consume.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TODAY_PATH = ROOT / "data_files" / "picks_today.csv"
FIXTURES_PATH = ROOT / "data_files" / "upcoming_fixtures.csv"
HISTORICAL_PATH = ROOT / "data_files" / "combined_historical_data.csv"
MODEL_PATH = ROOT / "models" / "ensemble_model.pkl"


def _implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    try:
        return 1.0 / float(odds)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.5


def main() -> None:
    today = date.today()

    for p in (FIXTURES_PATH, HISTORICAL_PATH):
        if not p.exists():
            print(f"[MLS] Missing: {p} — writing empty picks")
            pd.DataFrame().to_csv(TODAY_PATH, index=False)
            return

    fixtures = pd.read_csv(FIXTURES_PATH)
    if "date" in fixtures.columns:
        fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.date
        fixtures = fixtures[fixtures["date"] == today]
    elif "Date" in fixtures.columns:
        fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce").dt.date
        fixtures = fixtures[fixtures["Date"] == today]

    if fixtures.empty:
        print(f"[MLS] No fixtures for {today}")
        pd.DataFrame().to_csv(TODAY_PATH, index=False)
        return

    try:
        import pickle
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"[MLS] Could not load model: {e} — writing empty picks")
        pd.DataFrame().to_csv(TODAY_PATH, index=False)
        return

    try:
        from utils import prepare_features_for_upcoming
        X = prepare_features_for_upcoming(fixtures, HISTORICAL_PATH)
    except Exception as e:
        print(f"[MLS] Feature preparation failed: {e}")
        pd.DataFrame().to_csv(TODAY_PATH, index=False)
        return

    probs = model.predict_proba(X)  # [P(Away), P(Draw), P(Home)]
    rows = []
    for i, (_, fixture) in enumerate(fixtures.iterrows()):
        p_away, p_draw, p_home = probs[i]
        outcomes = [("Away Win", p_away), ("Draw", p_draw), ("Home Win", p_home)]
        best_outcome, best_prob = max(outcomes, key=lambda x: x[1])

        odds_col_map = {"Home Win": "OddsHome", "Draw": "OddsDraw", "Away Win": "OddsAway"}
        odds_col = odds_col_map[best_outcome]
        odds_val = fixture.get(odds_col) if odds_col in fixture.index else None
        impl = _implied_prob(odds_val) if odds_val else 0.5
        edge = best_prob - impl

        rows.append({
            "Date": str(today),
            "HomeTeam": fixture.get("HomeTeam", fixture.get("home_team", "")),
            "AwayTeam": fixture.get("AwayTeam", fixture.get("away_team", "")),
            "Bet": best_outcome,
            "Model": round(best_prob * 100, 1),
            "Edge": round(edge * 100, 1),
            "Odds": odds_val,
            "home_prob": round(p_home, 4),
            "draw_prob": round(p_draw, 4),
            "away_prob": round(p_away, 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(TODAY_PATH, index=False)
    print(f"[MLS] Wrote {len(out)} picks → {TODAY_PATH}")


if __name__ == "__main__":
    main()
