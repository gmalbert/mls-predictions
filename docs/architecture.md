# MLS Predictor — Architecture

> **Implementation audit — 2026-08-24:** The governed/frontier path described
> below is implemented in code. Optional roster, event, keeper, lineup, referee,
> attendance, and sentiment feeds are data-ready rather than confirmed live. The
> release gate remains closed pending historical market and settled-ledger evidence.

## Overview
Streamlit application predicting Major League Soccer match outcomes using machine learning, American Soccer Analysis (ASA) xG data, and MLS-specific structural features (salary cap, travel, turf, conference).

## Data Flow
```
American Soccer Analysis API    football-data.org    ESPN API    The Odds API
(itscalledsoccer)                       ↓               ↓               ↓
fetch_asa_data.py               fetch_mls_historical.py  fixtures  fetch odds
        ↓                               ↓
data_files/combined_historical_data.csv
        ↓
prepare_model_data.py → Feature Engineering
    [xG, travel_miles, is_turf, is_cross_conf, form, playoff_race, DP available]
        ↓
VotingClassifier (XGB + RF + GB + LR, soft voting)
        ↓
models/ensemble_model.pkl
        ↓
predictions.py (Streamlit entry, 5 fixed tabs)
```

## ML Model
- **Target**: `Result` → H=0, D=1, A=2
- **Exploratory ensemble**: the original UI retains its 80/20 diagnostic split.
- **Governed model**: season-ordered multinomial logistic regression, trained through
  the prior season with explicit recency sample weights. The first 2025 slice is used
  for locked calibration; later seasons are evaluated without in-season calibration.
- **Score model**: Dixon–Coles joint score distribution with red-card scenario mixing.
- **Primary features**: rolling xG/xGA/xGOT/g+, travel/time zones/altitude/surface/rest,
  roster mechanisms, coaching/expansion priors, tactical/set-piece and contextual feeds.

### MLS-Specific Engineered Features
| Feature | Notes |
|---------|-------|
| `home_xg_l5`, `away_xg_l5` | ASA xG rolling last 5 games |
| `away_travel_miles` | Great-circle distance (Haversine) |
| `is_long_haul` | Travel > 1,500 miles |
| `home_is_turf` | From `TURF_STADIUMS` set |
| `is_cross_conference` | From `EASTERN_CONF`/`WESTERN_CONF` |
| `home_form_pts_l5` | Points from last 5 games |
| `home_games_from_playoff` | Positive = inside playoff line |
| `home_dp_available` | Minutes/replacement-value weighted DP availability |

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| American Soccer Analysis (`itscalledsoccer`) | xG, goals added, team stats | None (free) |
| football-data.org | Historical MLS matches | `MLS_API_KEY` |
| ESPN (`site.api.espn.com`) | Upcoming fixtures, scores | None (public) |
| The Odds API | DraftKings MLS markets | `ODDS_API_KEY` |

## Key Components
- `predictions.py` — entry and core app layout; the additional Markets, Best Bets,
  and Frontier Analytics pages live under `pages/`.
- `prepare_model_data.py` — feature engineering pipeline
- `team_name_mapping.py` — normalises ESPN/ASA/football-data team names
- `automation/` — nightly data refresh, model retraining, fixture fetch
- `TURF_STADIUMS` set — verify before each season (renovations can change surface)
- `STADIUM_COORDS` dict — `(lat, lon)` per team for Haversine distance
- `EASTERN_CONF`, `WESTERN_CONF` sets — update when expansion teams added

## What NOT to Do
- Do NOT use squad value or wage spend as features (salary cap parity)
- Do NOT train across all seasons without recency weighting
- Do NOT assume balanced schedule (cross-conference H2H is noisy)
- Do NOT hardcode season years — derive from `datetime.now().year`

## Storage
- `data_files/combined_historical_data.csv` — historical MLS matches
- `data_files/upcoming_fixtures.csv` — scheduled fixtures
- `models/` — `.pkl` model artifacts (gitignored)

## Frontier Evaluation and Governance Layer

The original five tabs and ensemble remain available. The expanded system adds a
second, season-ordered evaluation path and a dedicated Frontier Analytics page:

```text
configured roster/news/event/weather feeds
                  |
                  v
automation/refresh_context.py -- available_at enforcement
                  |
                  v
models/frontier_features.py -- point-in-time feature table with explicit missingness
                  |
                  v
models/backtesting.py
  |-- rolling Elo
  |-- multinomial logistic + split calibration
  |-- Dixon-Coles score model
  |-- market-only and home-field baselines
  `-- season/phase/segment/expansion reports
                  |
                  v
data_files/backtests/latest_report.json
                  |
                  v
models/governance.py -- conformal abstention, consistency, release gate
                  |
                  v
prediction_snapshots + market_odds_snapshots + bet_ledger (SQLite)
```

The release gate is fail-closed. Live staking and external bet export remain
disabled until an untouched season, 300 settled frozen selections, market-relative
Brier/log-loss, positive median CLV, and a complete timestamped ledger all pass.

## Point-in-time Storage Contracts

- Context feeds live in `data_files/raw/` and use `available_at` separately from
  `event_date` or kickoff. Late corrections are excluded from historical features.
- Multi-book 1X2, handicap, and totals observations are archived in
  `market_odds_snapshots`, including book, line, limit, selection time, and close flag.
- Each model output is stored as an immutable scenario snapshot with fixture ID,
  prediction time, probabilities, data-manifest hash, and model version.
- The settlement ledger supports voids/pushes, draw-no-bet, quarter-goal Asian
  handicaps, totals, BTTS, and team totals. Props, first scorer, correct score, and
  parlays remain disabled until adequate provider-specific history exists.

## Product and Automation Components

- `pages/8_Frontier_Analytics.py` — match environment cards, team profiles, roster
  waterfall, scenario probabilities, value/no-bet state, tables, playoff simulation,
  xPts, heatmaps, event timing, shot zones, and the isolated All-Star model.
- `analytics/roadmap.py` — playoff, GPAA, player research, form, travel, expansion,
  DP on/off, xG-timing, and shot-location analytics.
- `automation/retrain_model.py` — feature rebuild plus rolling backtest/model artifact.
- `automation/matchday_email.py` — Friday governed slate email or safe HTML preview.
- `.github/workflows/roadmap-automation.yml` — Monday refresh/retrain, Friday brief,
  and manual off-season retraining.
