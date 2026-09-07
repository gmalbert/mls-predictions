"""Run the season-ordered baseline audit and write reproducible artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backtesting import evaluate_release_gate, rolling_backtest, save_backtest_artifacts
from database.db_manager import DatabaseManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data_files" / "combined_historical_data.csv"))
    parser.add_argument("--first-test-season", type=int, default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "data_files" / "backtests"))
    parser.add_argument("--model", default=str(ROOT / "models" / "frontier_logistic.pkl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Historical data not found: {source}")
    matches = pd.read_csv(source)
    predictions, report, model = rolling_backtest(matches, first_test_season=args.first_test_season)
    ledger_frames: list[pd.DataFrame] = []
    database_path = ROOT / "data_files" / "mls.db"
    if database_path.exists():
        ledger_frames.append(DatabaseManager(str(database_path)).get_bet_ledger(settled_only=True))
    ledger_path = ROOT / "data_files" / "frozen_selections.csv"
    if ledger_path.exists():
        ledger_frames.append(pd.read_csv(ledger_path))
    ledger = pd.concat([frame for frame in ledger_frames if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in ledger_frames
    ) else pd.DataFrame()
    if "selection_id" in ledger.columns:
        ledger = ledger.drop_duplicates("selection_id", keep="last")
    gate = evaluate_release_gate(report, ledger)
    report["release_gate"] = {
        "passed": gate.passed,
        "untouched_season": gate.untouched_season,
        "frozen_selections": gate.frozen_selections,
        "market_relative_brier": gate.market_relative_brier,
        "market_relative_log_loss": gate.market_relative_log_loss,
        "median_clv": gate.median_clv,
        "ledger_complete": gate.ledger_complete,
        "reasons": gate.reasons,
    }
    prediction_path, report_path, model_path = save_backtest_artifacts(
        predictions,
        report,
        model,
        output_dir=args.output_dir,
        model_path=args.model,
    )
    print(json.dumps({
        "predictions": str(prediction_path),
        "report": str(report_path),
        "model": str(model_path),
        "release_gate_passed": gate.passed,
        "release_gate_reasons": gate.reasons,
    }, indent=2))


if __name__ == "__main__":
    main()
