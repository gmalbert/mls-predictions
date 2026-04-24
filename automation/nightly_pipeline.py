"""
automation/nightly_pipeline.py
Orchestrates the full MLS data refresh pipeline.

Run directly:
    python automation/nightly_pipeline.py

Also called by .github/workflows/nightly.yml on a cron schedule.

Steps:
  1. fetch_asa_data        — pull game results + xG from American Soccer Analysis
  2. fetch_mls_historical  — pull historical results from football-data.org (if key available)
  3. fetch_upcoming_fixtures — pull upcoming schedule from ESPN
  4. prepare_model_data    — engineer MLS features + build combined training CSV
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime

# Ensure repo root is on sys.path regardless of working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(REPO_ROOT, ".env"))

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(REPO_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Step runner ────────────────────────────────────────────────────────────────

def run_step(step_name: str, func, *args, **kwargs) -> bool:
    """Run a single pipeline step; return True on success, False on error."""
    log.info(f"━━━ {step_name} ━━━")
    start = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - start
        log.info(f"✅ {step_name} completed in {elapsed:.1f}s\n")
        return True
    except Exception:
        log.error(f"❌ {step_name} FAILED:\n{traceback.format_exc()}")
        return False


# ── Import pipeline modules ────────────────────────────────────────────────────

def _import_modules():
    """Import pipeline modules lazily (so import errors are caught cleanly)."""
    import importlib

    modules = {}
    for name in [
        "fetch_asa_data",
        "fetch_mls_historical",
        "fetch_upcoming_fixtures",
        "prepare_model_data",
    ]:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError as exc:
            log.error(f"Cannot import {name}: {exc}")
            modules[name] = None
    return modules


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info(f"MLS Prediction Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    os.chdir(REPO_ROOT)
    modules = _import_modules()

    results: dict[str, bool] = {}

    # ── Step 1: ASA games (primary historical data + xG) ──────────────────────
    if modules.get("fetch_asa_data"):
        from datetime import datetime as _dt
        current_year = _dt.now().year
        seasons = list(range(2017, current_year + 1))
        results["fetch_asa"] = run_step(
            "Step 1 — Fetch ASA game data",
            modules["fetch_asa_data"].main,
            seasons=seasons,
        )
    else:
        log.warning("fetch_asa_data module unavailable — skipping.")
        results["fetch_asa"] = False

    # ── Step 2: football-data.org (supplemental historical results) ───────────
    if modules.get("fetch_mls_historical"):
        results["fetch_fdo"] = run_step(
            "Step 2 — Fetch football-data.org historical data",
            modules["fetch_mls_historical"].main,
        )
    else:
        log.warning("fetch_mls_historical module unavailable — skipping.")
        results["fetch_fdo"] = False

    # ── Step 3: Upcoming fixtures (ESPN) ──────────────────────────────────────
    if modules.get("fetch_upcoming_fixtures"):
        results["fetch_fixtures"] = run_step(
            "Step 3 — Fetch upcoming MLS fixtures",
            modules["fetch_upcoming_fixtures"].fetch_upcoming_fixtures,
            days_ahead=45,
        )
    else:
        log.warning("fetch_upcoming_fixtures module unavailable — skipping.")
        results["fetch_fixtures"] = False

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    # Only run if we have some raw game data
    raw_exists = os.path.exists(os.path.join(REPO_ROOT, "data_files", "raw", "asa_games.csv"))
    if raw_exists and modules.get("prepare_model_data"):
        results["prepare_data"] = run_step(
            "Step 4 — Feature engineering",
            modules["prepare_model_data"].main,
        )
    elif not raw_exists:
        log.warning("No raw data found — skipping feature engineering. Pipeline will retry next run.")
        results["prepare_data"] = False
    else:
        log.warning("prepare_model_data module unavailable — skipping.")
        results["prepare_data"] = False

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Pipeline summary:")
    all_ok = True
    for step, ok in results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        log.info(f"  {step:30s} {status}")
        if not ok:
            all_ok = False

    if all_ok:
        log.info("\n🎉 All pipeline steps succeeded.")
    else:
        log.warning("\n⚠️  One or more steps failed. Check logs above.")

    log.info(f"Log saved to: {log_filename}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
