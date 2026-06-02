# MLS Predictor — Architecture

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
- **Train/test split**: 80/20, `stratify=y`, `random_state=42`, recency-weighted 2020+
- **Primary features**: xG (last 5 + 10 games), away travel miles, turf flag, cross-conference flag

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
| `home_dp_available` | Designated player present (binary) |

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| American Soccer Analysis (`itscalledsoccer`) | xG, goals added, team stats | None (free) |
| football-data.org | Historical MLS matches | `MLS_API_KEY` |
| ESPN (`site.api.espn.com`) | Upcoming fixtures, scores | None (public) |
| The Odds API | DraftKings MLS markets | `ODDS_API_KEY` |

## Key Components
- `predictions.py` — entry, `st.set_page_config`, 5 fixed tabs (do NOT rename)
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
