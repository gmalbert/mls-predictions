# Copilot Instructions — MLS Predictor

## Project overview

This is a **Streamlit** application that predicts Major League Soccer match outcomes using machine learning, American Soccer Analysis xG data, and MLS-specific structural features. The entry point is `predictions.py`. All outputs and user-facing UI live inside that file.

This project is deliberately modeled after the companion `premier-league` repository but extended with MLS-specific quirks that break standard European soccer model assumptions.

---

## Key architecture rules

- **Entry point is `predictions.py`** — `streamlit run predictions.py` starts the app. Do not rename it.
- **All historical data lives in `data_files/`** — CSVs, pickled models, and the logo all go here.
- **Trained model pickles go in `models/`** — use `models/` as the output directory for any `.pkl` files.
- **Nightly automation scripts go in `automation/`** — data refresh, model retraining, fixture fetch.
- **GitHub Actions workflows go in `.github/workflows/`**.
- **Do not commit `.env`, API keys, or model `.pkl` files** — they are in `.gitignore`.

---

## MLS-specific domain knowledge

### What makes MLS different from European leagues

Every feature below is a meaningful signal that European soccer models miss entirely:

1. **Salary cap parity** — MLS has a hard salary cap. Squad value or wage spend are not useful features. Weight recent form and head-to-head more heavily.
2. **Designated Player rule** — Up to 3 players per team can exceed the cap. A single player's availability (e.g. Messi at Inter Miami) can shift the line significantly. Encode a `designated_player_available` binary flag.
3. **Unbalanced schedule / conference structure** — Eastern and Western Conferences play more intra-conference games. Cross-conference matchups are rarer and head-to-head history is thinner. Always flag cross-conference matches.
4. **Travel distance** — Away trips can be 3,000+ miles (e.g. Seattle to Miami). Use great-circle distance between stadium coordinates as a numeric feature. Flag trips over 1,500 miles as "long-haul".
5. **Artificial turf** — ~25% of MLS home stadiums use artificial turf. Home teams on turf often outperform visiting grass teams. Encode `home_surface` as a binary feature.
6. **Playoff format** — MLS uses a bracket playoff, not the league table, to crown the champion. Late-season games have different incentive structures depending on a team's playoff position. Encode `games_from_playoff_line` as a feature.
7. **Roster volatility** — SuperDraft, expansion teams, and allocation orders make older MLS seasons less predictive. Weight 2020+ seasons more heavily in training.

### Conference membership

Maintain `EASTERN_CONF` and `WESTERN_CONF` sets in `predictions.py` (and any data scripts). When a new expansion team is added, update both sets.

### Artificial turf stadiums

Maintain `TURF_STADIUMS` set in `predictions.py`. Verify before each season — stadium renovations can change surface type.

### Stadium coordinates

`STADIUM_COORDS` dict maps team names to `(lat, lon)` tuples. Used for great-circle travel distance via the Haversine formula. Keep it in sync with the `EASTERN_CONF` / `WESTERN_CONF` sets.

---

## Data sources

| Source | Package / URL | What it provides |
|---|---|---|
| American Soccer Analysis | `itscalledsoccer` | xG, xGA, goals added (2013–present) — free, purpose-built for MLS |
| MLS official API | `site.api.espn.com` | Scores, standings, schedules |
| football-data.org | `requests` | Historical MLS match data |
| The Odds API | `requests` | DraftKings moneylines, totals, player props |
| ESPN API | `requests` | Upcoming fixtures |

The `itscalledsoccer` package is the most valuable free resource for MLS modeling. Use it for xG features before any other source.

```python
from itscalledsoccer.client import AmericanSoccerAnalysis

asa = AmericanSoccerAnalysis()
team_xg  = asa.get_team_xgoals(leagues="mls", season_name="2024")
games_xg = asa.get_games(leagues="mls", seasons=[2023, 2024])
player_ga = asa.get_player_goals_added(leagues="mls", season_name="2024")
```

---

## ML model conventions

- **Primary model:** `VotingClassifier` ensemble (XGBoost + Random Forest + Gradient Boosting + Logistic Regression, soft voting). This mirrors the premier-league repo pattern.
- **Target variable:** `Result` column with values `H` (home win), `D` (draw), `A` (away win). Map to `{H: 0, D: 1, A: 2}`.
- **Feature names:** Rename columns to `feature_0, feature_1, …` before passing to XGBoost to avoid special-character issues.
- **Train/test split:** 80/20, `stratify=y`, `random_state=42`.
- **Backtesting window:** 2017–present. Weight 2020+ more heavily because pre-2020 roster data is less reliable.
- **Model persistence:** Save to `models/` as `.pkl` files via `pickle`. Load with `@st.cache_resource`.

### MLS-specific engineered features (implement in `prepare_model_data.py`)

```python
# xG (weighted last 5 and 10 games)
home_xg_l5, home_xg_l10, away_xg_l5, away_xg_l10

# Travel
away_travel_miles         # great-circle distance
is_long_haul              # bool: travel > 1500 miles

# Surface
home_is_turf              # bool

# Conference
is_cross_conference       # bool

# Form
home_form_pts_l5          # points from last 5 games
away_form_pts_l5

# Playoff race
home_games_from_playoff   # integer: positive = inside, negative = outside
away_games_from_playoff

# Designated player
home_dp_available         # bool
away_dp_available
```

---

## Streamlit UI conventions

- **Logo:** `st.image(path.join(DATA_DIR, 'logo.png'), width=250)` — always at 250px, never change this.
- **Page config:** `layout="wide"`, `page_icon="⚽"`.
- **Caching:** Use `@st.cache_data(ttl=3600)` for data loaders. Use `@st.cache_resource` for ML models.
- **Height helper:** Use `get_dataframe_height(df, max_height=600)` for all `st.dataframe()` calls.
- **Tabs:** The five tabs are fixed — do not rename them without updating `predictions.py`.
- **Footer:** Add a `footer.py` module (similar to the premier-league repo) when the app is ready for production.
- **Missing data:** Always check `path.exists(csv_path)` before loading files and show a friendly `st.warning()` if absent.

---

## Code style

- Use type hints on all new function signatures.
- Keep functions small and focused — one responsibility per function.
- All expensive computations must be wrapped in `@st.cache_data` or `@st.cache_resource`.
- Use `pd.concat()` for adding new columns to DataFrames in batch rather than repeated assignment (avoids fragmentation warnings).
- Normalize team names consistently — a `team_name_mapping.py` module should be created to handle ESPN / ASA / football-data.org name mismatches.
- Do not use `st.experimental_*` APIs — use stable equivalents.

---

## Environment variables

Store all secrets in a `.env` file (loaded via `python-dotenv`). Never commit keys.

```
ODDS_API_KEY=...         # The Odds API
MLS_API_KEY=...          # MLS official API (if required)
```

---

## Testing

- Place unit tests in `tests/` or at the repo root with `test_` prefix.
- Run with `pytest`.
- CI runs tests via `.github/workflows/`.

---

## What NOT to do

- Do not use squad value, wage spend, or transfer fees as features — they are not meaningful in a salary-cap league.
- Do not train a single model across all seasons without recency weighting — MLS rosters change aggressively every offseason.
- Do not assume every team plays every other team twice — the unbalanced schedule makes cross-conference H2H comparisons noisy.
- Do not hardcode season years — derive them dynamically from the data or from `datetime.now().year`.
