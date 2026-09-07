"""Rebuild point-in-time data, refit baselines, and refresh evaluation artifacts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(script: Path) -> None:
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def main() -> None:
    _run(ROOT / "prepare_model_data.py")
    _run(ROOT / "scripts" / "run_backtest.py")


if __name__ == "__main__":
    main()

