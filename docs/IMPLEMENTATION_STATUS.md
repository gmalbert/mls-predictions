# Roadmap implementation status

This document maps the six requested design documents to executable code. The
the code layer is substantially complete. Provider-backed rows and release evidence
are allowed to remain data-empty rather than being invented. A data-empty feature
carries a missingness or uncertainty signal, and restricted markets remain disabled
exactly as required by the backtest audit.

> **Audit status — 2026-08-24:** `Implemented` means the repository contains the
> code path; it does **not** mean the feature is populated by a live provider or
> approved for betting. See `IMPLEMENTATION_PRIORITIES.md` for the authoritative
> remaining-work order. Local test execution was not possible in this environment
> because no project Python/pytest installation is available.

## Model and audit requirements

| Requirement | Implementation |
|---|---|
| xG differential momentum, xGOT/xGA and team g+ L10 | Strictly prior rolling features in `models/frontier_features.py`; g+ accepts only timestamped team snapshots. |
| DP/U22/TAM, international duty and roster churn | **Partially implemented:** minutes/replacement-value availability waterfall plus 30-day transactions and international-absence counts. Provider feeds still need configuration and historical validation. |
| Playoff line, travel, altitude, surface and turf transition | Pre-kickoff standings features in `prepare_model_data.py`; centralized venue metadata and interactions in `models/frontier_features.py`. |
| Expansion/coaching priors, conference strength, SuperDraft, transfers | Hierarchical/tenure features, prior cross-conference strength, 90-day rookie integration ramp and timestamped WAR deltas. |
| Possession chains, set pieces, GPAA, lineup/referee/weather/attendance | **Data-ready:** provider-neutral point-in-time feed contracts and numeric model features, with missingness where the feed is absent; no claim of populated production feeds. |
| Reproducible baselines | Rolling Elo, recency-weighted multinomial logistic, Dixon–Coles and home-field baselines in `models/backtesting.py`. |
| Rolling evaluation and calibration | Train through 2024, calibrate on the first chronological 2025 slice, preserve the later holdout, then test later seasons in full with locked temperatures. |
| Phase and segment reporting | Regular season/playoffs/Leagues Cup plus travel, surface, altitude, churn, manager tenure, horizon and roster-mechanism metrics. |
| Market baseline and paper performance | **Partially implemented:** de-vigged 1X2 baseline; turnover, ROI, drawdown and CLV reports. Historical selection/closing-price evidence remains incomplete. |
| Joint score and uncertainty | Dixon–Coles score grid, red-card scenario mixture, O/U/BTTS/team totals, conformal prediction sets and visible abstention. |
| Release gate | **Implemented and closed:** requires an explicit completed untouched season, 300+ settled frozen selections, market-relative Brier/log loss, positive median CLV and a complete versioned ledger. |

## Product roadmap (features 1–22)

| # | Feature | Product/code surface |
|---:|---|---|
| 1 | ASA deep integration | `fetch_asa_data.py`, recency-weighted xG/xGA and timestamped g+ archive. |
| 2 | Travel fatigue | Central stadium coordinates, distance, time zones, altitude, turnaround and charter interactions. |
| 3 | DP intelligence | DP/U22/TAM availability and replacement waterfall in model and Team Profiles. |
| 4 | Playoff format | Best-of-three probability and first-round series logic. |
| 5 | Turf advantage | Central surface metadata, grass/turf transitions and Match Day badges. |
| 6 | SuperDraft impact | Timestamped draft feed with minutes/value-weighted 90-day integration ramp. |
| 7 | Conference strength | Dynamic prior cross-conference results and live East/West table. |
| 8 | Set pieces | Corners, dangerous free kicks, specialist availability and set-piece xG. |
| 9 | GPAA | Post-shot-xG preferred GPAA calculation and keeper model edge. |
| 10 | Home/away scoring | Team xG split and home-minus-away differential; Team Profiles comparison. |
| 11 | Playoff simulator | Remaining schedule plus dynamic seeding, playoffs and MLS Cup simulated 10,000 times. |
| 12 | Goal/assist props | Target-share/shots/xG-per-shot goal and xA estimates with DraftKings research edge. Betting remains disabled pending history. |
| 13 | Form heatmap | Team-by-week W/D/L intensity chart. |
| 14 | xG timeline | Timestamped event-feed scoring/xG buckets. |
| 15 | Pitch zones | Shot x/y zone and xG treemap. |
| 16 | Weekly automation | Monday fixtures/context/model workflow and governed best-bet export. |
| 17 | Transfers | Timestamped expected-WAR delta feature. |
| 18 | Dome weather | Dome flag and outdoor extreme-weather interaction. |
| 19 | All-Star model | Isolated rating/motivation model; excluded from club training and ledger. |
| 20 | Sentiment | Timestamped configured news/social feeds with a deliberately low-weight signal. |
| 21 | xPts | Poisson xPts versus actual table and regression flags. |
| 22 | Attendance | Attendance/capacity ratio and home-advantage interaction. |

## Six-month product and automation requirements

`pages/8_Frontier_Analytics.py` contains Today's Match Day slate with 1X2,
O/U, surface, 72-hour rest and conference badges; team xG/DP/results/altitude,
home-away and travel profiles; a 3% DraftKings value filter, totals environment
filter and playoff odds; live conference, expansion and DP views; feature-segment,
season ROI and salary-cap-mechanism analysis; and Frontier scenario/event views.
Friday email, weekly ASA/context refresh, match-window selection/closing-odds snapshots,
automatic result/CLV settlement, and off-season retraining are implemented
under `automation/` and `.github/workflows/roadmap-automation.yml`.

## Data contracts and operating commands

Every optional context row must carry `available_at` independently from its event
date. Raw configured feeds live under `data_files/raw/`; immutable market and model
snapshots plus the settlement ledger live in the persisted `data_files/mls.db`. The supported feed
environment variables are documented in `.env.example`.

```powershell
python prepare_model_data.py
python scripts/run_backtest.py
python scripts/generate_picks.py
python automation/settle_ledger.py
python automation/matchday_email.py
python -m pytest -q
python -m py_compile <all repository Python files>
```

The generated report is `data_files/backtests/latest_report.json`. Its release gate
is authoritative; implementation completion does not override missing profitability
evidence or permit live staking.

## Completion audit (2026-08-13)

- The official ASA v2.1 match-xG endpoint was refreshed for 4,357 matches from
  2017 through 2026; both home and away xG columns are complete in the current
  raw and combined datasets.
- The deployment artifact uses 36 governed features and evaluates 701 untouched
  post-calibration matches across the 2025 holdout and the in-progress 2026 season.
- Missing competition-stage metadata is reported as `Unknown`, never guessed.
  `stage_name` and the timestamped phase-manifest contract activate phase-specific
  reporting when authoritative rows are available.
- The release gate is intentionally closed while the full-season, settled-ledger,
  market-relative, and CLV evidence requirements remain unmet.
