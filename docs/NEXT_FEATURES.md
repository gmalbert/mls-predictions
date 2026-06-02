# MLS Predictor — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Travel Distance Feature

**Why:** `STADIUM_COORDS` dict is already defined in `predictions.py` with `(lat, lon)` for every MLS team. Away trips of 3,000+ miles (Seattle → Miami) are a meaningful fatigue signal with no European analog. The Haversine formula is already documented in the copilot-instructions.

**How:**
1. Add `utils/travel.py` with `haversine(lat1, lon1, lat2, lon2) → float` (miles)
2. Compute `away_travel_miles` per upcoming fixture using `STADIUM_COORDS`
3. Add `is_long_haul = away_travel_miles > 1500` binary feature
4. Add both to `FEATURE_COLS` in `prepare_model_data.py` — use `shift(1)` guard for historical data
5. Surface on the Today page as a contextual indicator (not just a model feature)

**Complexity:** Low

---

## Feature 2: Designated Player Availability Flag

**Why:** A single Designated Player (e.g., Messi, Cucho Hernandez, Riqui Puig) can shift the line significantly in a salary-cap league. The DP availability binary (`home_dp_available`, `away_dp_available`) is documented in the copilot-instructions but not yet implemented.

**How:**
1. Add `fetch_injuries.py` using ESPN's MLS injury/availability endpoint (or rotowire via requests)
2. Cross-reference roster with a `data_files/mls_designated_players.csv` list (manually maintained, ~60 players)
3. Set `home_dp_available = 0` if any DP is on the injury list for that team
4. Add as a binary feature in `prepare_model_data.py` for upcoming fixtures

**Complexity:** Medium

---

## Feature 3: Conference Standings Context Feature

**Why:** MLS uses a playoff bracket (not league table) to crown a champion. Whether a team is 1 game inside vs outside the playoff line late in the season dramatically changes their performance incentive. `home_games_from_playoff` (integer: + = inside, − = outside) is documented but not implemented.

**How:**
1. Fetch MLS standings via the ESPN API (`site.api.espn.com/apis/site/v2/sports/soccer/usa.1/standings`)
2. Compute playoff cut-line position per conference (top 9 teams qualify)
3. Add `home_games_from_playoff` and `away_games_from_playoff` to feature set
4. Use `shift(1)` to apply standings as of the last played matchday

**Complexity:** Medium

---

## Feature 4: Value Bet Page

**Why:** The model generates win probabilities but the UI does not compare them against bookmaker odds to surface value bets. A dedicated Value Bets page with Kelly sizing is the most directly actionable output for bettors.

**How:**
1. Add `fetch_odds.py` using The Odds API (`soccer_usa_mls`) with `ODDS_API_KEY` env var
2. Compute edge per fixture: `model_prob - implied_prob`
3. Add `pages/value_bets.py` with a sortable table: Fixture | Model % | DK % | Edge | Kelly Size
4. Apply confidence threshold: only show rows where edge > 2%
5. Color-code by edge tier (Elite/Strong/Good matching other Betting Oracle apps)

**Complexity:** Medium

---

## Feature 5: Artificial Turf Match Report

**Why:** ~25% of MLS home stadiums use turf, and `TURF_STADIUMS` set is already defined but only used as a model feature internally. Surfacing turf vs grass matchup on each prediction card gives users a key contextual factor that public bettors frequently overlook.

**How:**
1. In the Today page prediction cards, add a "Surface" badge: 🌿 Natural Grass or 🟦 Artificial Turf
2. Pull from the existing `TURF_STADIUMS` set — no new data needed
3. Add a historical analysis chart: win rate for visiting grass teams on turf vs grass surfaces
4. This does not require model changes — it is display-only context

**Complexity:** Low
