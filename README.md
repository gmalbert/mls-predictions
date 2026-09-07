# MLS Predictor

<p align="center"><img src="data_files/logo.png" alt="MLS Predictor logo" width="250"></p>

A Streamlit-powered web application that predicts Major League Soccer match outcomes using machine learning, expected-goals (xG) data, and MLS-specific structural features that most public models ignore.

---

## Table of Contents

- [What this project does (for fans)](#what-this-project-does-for-fans)
- [What makes this different from EPL models](#what-makes-this-different-from-epl-models)
- [What's implemented](#whats-implemented)
- [MLS-specific features](#mls-specific-features)
- [Data sources](#data-sources)
- [How to run](#how-to-run)
- [Project structure](#project-structure)
- [Roadmap](#roadmap)

---

## What this project does (for fans)

This project predicts the likely outcome of upcoming MLS matches (home win, draw, or away win) using historical match data, American Soccer Analysis xG metrics, and machine learning. It accounts for structural quirks unique to MLS — salary-cap parity, cross-country travel fatigue, artificial turf surfaces, conference scheduling imbalance, and Designated Player availability — that generic soccer models miss entirely.

The app runs February through November, covering the full MLS season and filling the gap when European leagues are dark (June–August), providing year-round soccer analytics.

[Back to top](#mls-predictor)

---

## What makes this different from EPL models

MLS breaks several assumptions that European league models rely on:

| Factor | EPL | MLS |
|---|---|---|
| Financial parity | No (Man City, etc.) | Yes — hard salary cap |
| Away travel | 2–3 hour bus trip | Up to 6-hour cross-country flight |
| Surface type | Natural grass everywhere | ~25% artificial turf stadiums |
| Schedule balance | Every team plays every other twice | Heavily conference-weighted |
| Season champion | Highest table finish | Knockout playoff bracket |
| Roster depth | 25-man squad depth matters | Single Designated Player can swing a line |

Every one of these "cons" is also an **opportunity**: a model that correctly encodes travel distance, turf, conference structure, and playoff-race incentives has features that public bettors and many sportsbooks underweight in MLS markets.
[Back to top](#mls-predictor)
---

## What's implemented

- A Streamlit app (`predictions.py`) with five tabs:
  - **🗓️ Upcoming Matches** — live fixtures with Eastern/Western conference badges, surface type, travel distance, and cross-conference flags
  - **🎯 Upcoming Predictions** — ensemble model predictions (Home Win %, Draw %, Away Win %) with risk scoring, confidence scores, MLS-specific surface/travel context, and betting tips
  - **📊 Statistics** — American Soccer Analysis xG data, recent team form, head-to-head analyzer, surface & conference win-rate analysis
  - **🔬 Team Deep Dive** — per-team KPIs, full travel profile (miles to every opponent), ASA xG stats
  - **📁 Raw Data** — sortable historical match data viewer
- Ensemble model combining XGBoost, Random Forest, Gradient Boosting, and Logistic Regression with soft voting
- Risk scoring based on entropy and probability distribution variance
- MLS-specific lookup tables: conference membership, artificial-turf stadiums, GPS stadium coordinates for great-circle travel distance
- American Soccer Analysis integration via `itscalledsoccer` Python package (free xG, xGA, goals added)

[Back to top](#mls-predictor)

---

## MLS-specific features

### Encoded in the model
- **xG and xGA** from American Soccer Analysis (last 5 and 10 games, weighted)
- **Travel distance** — great-circle miles between stadiums for the away team
- **Days rest** since last match
- **Surface type** — turf vs. natural grass for the home stadium
- **Form points** — last 5 games, conference and overall
- **Playoff position race flag** — is the team within 2 spots of the playoff line?
- **Designated player availability flag**
- **Cross-conference matchup** flag (Eastern vs. Western)

[Back to top](#mls-predictor)

### Data pull example (American Soccer Analysis)

```python
from itscalledsoccer.client import AmericanSoccerAnalysis

asa = AmericanSoccerAnalysis()

# xG data for all MLS games
xg_data = asa.get_games(leagues="mls", seasons=[2023, 2024])

# Team xG summaries
team_xg = asa.get_team_xgoals(leagues="mls", season_name="2024")

# Player goals added
player_ga = asa.get_player_goals_added(leagues="mls", season_name="2024")
```

---

## Data sources

| Source | What it provides | Cost |
|---|---|---|
| American Soccer Analysis / `itscalledsoccer` | xG, xA, goals added (2013–present) | Free |
| MLS official API | Scores, standings, schedules | Free |
| `football-data.org` | MLS match history | Free tier available |
| FBref | Advanced metrics (scrapeable) | Free |
| The Odds API | Moneylines, spreads, totals | Free tier available |
| ESPN API | Upcoming fixtures | Free (unofficial) |

[Back to top](#mls-predictor)

---

## How to run

### Prerequisites

- Python 3.9+
- A virtual environment (recommended)

### Install

```powershell
python -m venv venv
venv\Scripts\Activate.ps1       # Windows PowerShell
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### Fetch upcoming fixtures (optional)

```powershell
python fetch_upcoming_fixtures.py
```

### Run the app

```powershell
streamlit run predictions.py
```

### Notes for developers

- Add a `data_files/combined_historical_data.csv` with MLS historical match results to unlock predictions, statistics, and the Raw Data tab.
- If you add third-party API keys (Odds API, etc.), store them in a local `.env` file and do **not** commit it.
- The American Soccer Analysis integration requires internet access and the `itscalledsoccer` package.
- Nightly automation scripts (data refresh, model retraining) should be placed in `automation/` and wired to GitHub Actions.

[Back to top](#mls-predictor)

---

## Project structure

```
mls-predictions/
├── predictions.py              # Main Streamlit app entry point
├── data_files/
│   ├── logo.png                # App logo (displayed at 250px width)
│   ├── combined_historical_data.csv   # MLS historical match data (add this)
│   └── upcoming_fixtures.csv  # Upcoming MLS fixtures (generated by fetch script)
├── models/                     # Trained model pickle files (generated at runtime)
├── automation/                 # Data refresh and model retraining scripts
├── docs/
│   └── mls.md                 # MLS league overview and model design notes
├── requirements.txt
├── .gitignore
└── README.md
```

[Back to top](#mls-predictor)

---

## Frontier model, analytics, and governance

The roadmap is implemented as a second, reproducible path alongside the original
five-tab application:

- `models/frontier_features.py` builds strictly point-in-time MLS features for ASA
  xG/xGOT/g+, travel, roster mechanisms, coaching, set pieces, weather, draft,
  transfer, attendance, lineup, referee, and competition context.
- `models/backtesting.py` compares recency-weighted multinomial logistic, rolling
  Elo, Dixon–Coles, home-field, and de-vigged market baselines by season and segment.
- `models/governance.py` owns score-market consistency, conformal abstention,
  paper/live decisions, settlement, and the fail-closed release gate.
- `pages/8_Frontier_Analytics.py` exposes match cards, team profiles, value tools,
  playoff simulation, historical analysis, scenarios, event shape, and special models.
- `automation/refresh_context.py`, `automation/retrain_model.py`, and
  `automation/matchday_email.py` power the scheduled refresh/retrain/Friday workflow.

Run a complete local rebuild and evaluation with:

```powershell
python prepare_model_data.py
python scripts/run_backtest.py
python scripts/generate_picks.py
streamlit run predictions.py
```

Live staking remains disabled until the report proves one full untouched season,
300 settled frozen selections, superiority to the closing market on Brier and log
loss, positive median CLV, and a complete auditable ledger. Missing provider feeds
remain explicit missingness/uncertainty—they are never interpreted as healthy or
zero impact. See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for
the complete requirement mapping and operating contract.

[Back to top](#mls-predictor)

Data analysis and sports betting for Major League Soccer (MLS)
