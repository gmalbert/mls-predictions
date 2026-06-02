> **AI Onboarding Guide** — See also `.github/copilot-instructions.md` for full coding conventions.

# MLS Predictor — Site Summary

## What This App Does

Streamlit app that predicts Major League Soccer match outcomes using a soft-voting ensemble classifier, incorporating MLS-specific structural features that European soccer models ignore: salary cap parity, travel distance, artificial turf, conference structure, and Designated Player availability.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Fetch and prepare data
python automation/fetch_asa_data.py          # American Soccer Analysis (xG, goals added)
python automation/fetch_mls_historical.py    # Historical match results
python automation/fetch_upcoming_fixtures.py # Next 30 days from ESPN

# 3. Run the app
streamlit run predictions.py
```

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (multi-tab) |
| ML | VotingClassifier: XGBoost, RF, GB, LR (soft voting) |
| Data | pandas, NumPy, itscalledsoccer (ASA client) |
| Visualization | Plotly |
| Config | python-dotenv (`.env` file) |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Entry point — app layout, 5 fixed tabs, logo at 250px |
| `prepare_model_data.py` | Feature matrix builder — MLS-specific engineered features |
| `automation/fetch_asa_data.py` | American Soccer Analysis API client (`itscalledsoccer`) |
| `automation/fetch_mls_historical.py` | Historical match results |
| `automation/fetch_upcoming_fixtures.py` | ESPN API — next 30 days of MLS matches |
| `team_name_mapping.py` | Name normalization (ESPN / ASA / football-data.org discrepancies) |
| `data_files/logo.png` | Sidebar logo — always render at `width=250` |

## Data Flow

1. **Historical data**: `fetch_mls_historical.py` → historical match results CSV in `data_files/`
2. **xG data**: `fetch_asa_data.py` using `itscalledsoccer` → team xG, player goals added → `data_files/`
3. **Feature engineering**: `prepare_model_data.py` → MLS-specific features (travel miles, turf, conference, form, DP flag)
4. **Training**: 80/20 stratified split → `VotingClassifier` → `models/*.pkl`
5. **Upcoming fixtures**: `fetch_upcoming_fixtures.py` (ESPN) → merge with team stats → predictions
6. **UI**: Streamlit loads model + predictions → renders 5 tabs

## MLS-Specific Feature Flags

These are critical for model accuracy and unique to MLS — do not remove:

| Feature | Description |
|---|---|
| `away_travel_miles` | Great-circle distance via Haversine formula using `STADIUM_COORDS` |
| `is_long_haul` | `away_travel_miles > 1500` |
| `home_is_turf` | Home stadium is artificial turf (see `TURF_STADIUMS` set) |
| `is_cross_conference` | Home and away teams are in different conferences |
| `home_dp_available` | Designated Player available (injury flag) |
| `home_games_from_playoff` | Integer: positive = inside playoff line |

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `ODDS_API_KEY` | The Odds API — live MLS odds | Optional |
| `MLS_API_KEY` | MLS official API | Optional |

## External APIs & Rate Limits

| API | Notes |
|---|---|
| American Soccer Analysis (`itscalledsoccer`) | Free, no key required; purpose-built for MLS xG data |
| ESPN (site.api.espn.com) | Public JSON endpoint for MLS schedule |
| The Odds API | 500 req/month free tier |

## Critical Conventions

- Squad value and wage spend are **not useful features** — MLS has a hard salary cap
- Do **not** train across all seasons without recency weighting — MLS rosters change aggressively each offseason
- Do **not** assume every team plays every other team twice — unbalanced conference schedule
- Do **not** hardcode season years — derive dynamically from data or `datetime.now().year`
- Update `EASTERN_CONF` and `WESTERN_CONF` sets when expansion teams are added
- Update `TURF_STADIUMS` before each season (stadium surfaces can change)
- Target encoding: H=0, D=1, A=2 (rename columns to `feature_0, feature_1, ...` before XGBoost to avoid special-char issues)

## Common Gotchas

- `itscalledsoccer` is the best free MLS data source — prefer it over manual scraping
- Name mismatches between ESPN, ASA, and football-data.org are common — always route through `team_name_mapping.py`
- Logo must render at `st.image(..., width=250)` — do not change this width
