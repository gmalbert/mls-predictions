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

## Roadmap

**Primary model targets:** Moneyline (1X2), over/under 2.5 goals, both teams to score, MLS Cup futures.

**Planned additions:**
- Nightly data pipeline (fixtures, results, injuries from MLS API)
- Designated player availability tracker (near-real-time injury/availability flags)
- Playoff race context feature (dynamic per round)
- Poisson regression for expected goals scoring
- Conference-specific sub-models (separate Eastern/Western training)
- Historical backtesting dashboard (2017–2024, ~1,750 games)
- DraftKings odds integration via The Odds API
- PDF report export

See [docs/mls.md](docs/mls.md) for the full model design and feature engineering plan.

[Back to top](#mls-predictor)

Data analysis and sports betting for Major League Soccer (MLS)
