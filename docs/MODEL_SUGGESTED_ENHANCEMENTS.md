# MLS Predictor — Model Suggested Enhancements

## Priority 1: ASA xG Feature Expansion

### xG Differential Weighting
- Current model uses `xg_l5` and `xg_l10`. Add `xg_differential_momentum`: (last-3 xG differential) − (season-to-date xG differential) to capture form surges.

### xGA Quality
- Break down xGA by shot quality faced: opponent `xgOTT` (on-target xG) is a more robust defence metric than raw xGA.

### Goals Added (gA)
- `get_player_goals_added()` provides per-player contributions. Aggregate to a team-level `team_goals_added_l10` feature.

## Priority 2: MLS-Specific Features

### Designated Player Availability
- `designated_player_available` binary flag. DP absences (injury, international duty, suspension) have an outsized effect in a salary-cap league.
- Integrate MLS news/injury API to automate this flag.

### Allocation Order / Roster Volatility
- Encode `mid_season_roster_changes` (count of transactions in last 30 days). Higher churn = more uncertainty.

### Playoff Race Pressure
- `games_from_playoff_line` feature already planned. Ensure it's computed from the current Eastern/Western Conference standings at prediction time.

## Priority 3: Environmental Features

### Stadium Altitude
- Altitude at stadiums like Colorado's Dick's Sporting Goods Park (~5,280 ft) demonstrably affects match totals.
- Add `home_altitude_ft` as a feature.

### Turf to Grass Transition
- When an away team that primarily plays on turf visits a grass stadium, add a `turf_to_grass_visitor` flag.

## Priority 4: Calibration

- Track historical accuracy split by `home_is_turf`, `is_long_haul`, and `is_cross_conference`.
- Apply calibration separately for each split if systematic bias is found.
