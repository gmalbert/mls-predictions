# Current-model backtest audit (2026)

## Verdict

The prior evidence-gap finding is no longer fully current. The repository now has
season-ordered backtest code, a persisted report/model artifact path, immutable
prediction/market snapshot tables, and a settlement ledger. `best_bets_today.json`
is still a live export, not historical evidence. The current release decision is
therefore **evaluation infrastructure implemented, release evidence incomplete**—not
profitable or unprofitable.

> **Audit update — 2026-08-24:** Backtest and governance code is present in
> `models/backtesting.py`, `models/governance.py`, `scripts/run_backtest.py`, and
> `database/db_manager.py`. The release gate remains fail-closed until a completed
> untouched season, 300+ settled frozen selections, market-relative improvement,
> positive median CLV, and a complete ledger are demonstrated.

## Current artifact finding (2026-08-13 report)

The checked-in report evaluates 701 post-calibration matches (432 from the 2025
holdout and 269 from the in-progress 2026 season). The governed logistic model's
aggregate log loss (1.0453) and Brier score (0.6282) are slightly **worse** than
the rolling Elo baseline (1.0415 and 0.6261). There are no odds selections,
market-relative metrics, ROI, or CLV observations. Accordingly, the immediate
priority is baseline/model selection and evidence collection—not expanding the
feature set or enabling wagering.

## Changes justified by the evidence gap

1. **Create a reproducible baseline first:** rolling Elo plus home advantage, then a multinomial logistic model and Dixon-Coles score model.
2. **Use MLS-specific temporal features:** travel distance/time zones, altitude, surface, short rest, international absences, roster churn, and early-season shrinkage. Compute every rolling feature strictly before kickoff.
3. **Backtest by season and competition phase.** Train through 2024, calibrate on an early 2025 slice if necessary, and preserve a later 2025 holdout; then roll forward. Separate regular season, playoffs, and Leagues Cup.
4. **Make the market a baseline.** Archive de-vigged 1X2, handicap, and totals prices at selection and close. Test whether the model adds residual signal.

## Betting strategy decision

- **1X2/moneyline:** no-bet until probability calibration is measured.
- **Draw-no-bet/Asian handicap:** potentially more robust for MLS draws, but requires line-specific settlement.
- **Totals/BTTS/team totals:** evaluate via a correlated score model, including red-card sensitivity and lineup timing.
- **Props/first scorer/correct score:** insufficient data; remain disabled.
- **Parlays:** do not multiply correlated match probabilities.
- **Staking:** paper trade, then flat 0.25 units after release; capped fractional Kelly only after stable calibration and CLV.

## Release gate

One full untouched season, 300+ frozen selections, market-relative log loss/Brier, positive median CLV, and a ledger with timestamp, book, odds, limits, result, void/push handling, and code version.
