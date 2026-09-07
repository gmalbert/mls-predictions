"""Write reproducible P0/P1 readiness evidence without touching external providers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.data_quality import audit_market_odds, audit_optional_sources
from models.frontier_features import load_optional_sources


def main() -> None:
    raw_dir = ROOT / "data_files" / "raw"
    sources = load_optional_sources(raw_dir)
    odds_path = raw_dir / "multi_book_odds.csv"
    odds = pd.read_csv(odds_path) if odds_path.exists() else pd.DataFrame()
    report_path = ROOT / "data_files" / "backtests" / "latest_report.json"
    backtest = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    evidence = {"context": audit_optional_sources(sources), "market_odds": audit_market_odds(odds), "backtest_release_gate": backtest.get("release_gate", {"passed": False, "reasons": ["Backtest report missing."]}), "deployment": backtest.get("deployment", {})}
    output = ROOT / "data_files" / "quality" / "latest_readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    print(json.dumps(evidence, indent=2, default=str))


if __name__ == "__main__":
    main()
