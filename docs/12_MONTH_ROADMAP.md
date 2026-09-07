# MLS Predictions — 12-Month Feature Roadmap

> Generated: 2026-07-31 | Horizon: August 2026 – July 2027

> **Implementation audit — 2026-08-24:** This is now a historical roadmap, not a
> queue. `Implemented` means an auditable repository code path exists. `Data-ready`
> means the code is present but needs a configured, timestamped provider feed;
> `Research-only` remains deliberately unavailable for wagering.

| Features | Status | Remaining condition |
|---|---|---|
| 1, 2, 4, 5, 7, 10, 11, 13, 16, 18, 19, 21 | Implemented | Re-run and validate in the project environment. |
| 3, 6, 8, 9, 14, 15, 17, 20, 22 | Data-ready | Configure authoritative point-in-time feeds and validate coverage/leakage. |
| 12 | Research-only | Timestamped, provider-specific prop history and release review; props remain disabled. |

---

## Q1 (Aug–Oct 2026) — Data Foundation

### Feature 1 — American Soccer Analysis Deep Integration

Use `itscalledsoccer` for xG, xGA, and goals added by player. Build
per-team rolling xG features weighted by recency (2020+ seasons only).

```python
# fetch_asa_advanced.py
from itscalledsoccer.client import AmericanSoccerAnalysis
import pandas as pd

asa = AmericanSoccerAnalysis()

def build_team_xg_features(seasons: list[int]) -> pd.DataFrame:
    dfs = []
    for season in seasons:
        team_xg = asa.get_team_xgoals(leagues="mls", season_name=str(season))
        team_xg["season"] = season
        dfs.append(team_xg)
    combined = pd.concat(dfs)
    return (
        combined.groupby(["team_id", "season"])
        .agg(xgf=("xGoals", "mean"), xga=("xGoalsAgainst", "mean"),
             np_xgf=("np_xGoals", "mean"), np_xga=("np_xGoalsAgainst", "mean"))
        .reset_index()
    )
```

### Feature 2 — Travel Distance Weighted Fatigue

For each away game, compute great-circle distance from home stadium.
Games involving 2,000+ mile trips should penalize the away team.

```python
# utils/travel.py
from math import radians, sin, cos, sqrt, atan2

STADIUM_COORDS = {
    "Inter Miami CF": (25.9580, -80.2390),
    "Seattle Sounders": (47.5952, -122.3316),
    "LA Galaxy": (33.8644, -118.2611),
    "NYCFC": (40.8296, -73.9262),
}

def haversine_miles(coord1: tuple, coord2: tuple) -> float:
    R = 3958.8
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

def get_travel_features(home: str, away: str) -> dict:
    c1 = STADIUM_COORDS.get(away, (39.5, -98.0))
    c2 = STADIUM_COORDS.get(home, (39.5, -98.0))
    dist = haversine_miles(c1, c2)
    return {
        "travel_miles": round(dist),
        "is_long_haul": dist > 1500,
        "travel_fatigue_adj": max(0, (dist - 500) / 2000 * 0.10),
    }
```

### Feature 3 — Designated Player Intelligence

Track DP availability, injury status, and minutes played. A DP absence
is a significant model signal — quantify the WAR lost.

```python
# utils/designated_players.py
import pandas as pd

# Updated each season
DESIGNATED_PLAYERS_2026 = {
    "Inter Miami CF": ["Lionel Messi", "Luis Suarez", "Jordi Alba"],
    "LAFC": ["Giorgio Chiellini", "Riqui Puig"],
    "Atlanta United": ["Xherdan Shaqiri"],
}

def dp_availability_score(team: str, missing_players: list[str]) -> float:
    """Return performance penalty (0-1) when DPs are missing."""
    dps = DESIGNATED_PLAYERS_2026.get(team, [])
    missing_dps = [p for p in missing_players if p in dps]
    if not dps:
        return 1.0
    impact = len(missing_dps) / len(dps)
    return max(0.6, 1.0 - impact * 0.40)  # Max 40% performance reduction
```

### Feature 4 — MLS Playoff Format Predictor

Model differences between regular season and playoffs. Best-of-3 first
round creates high variance — quantify upset probability by seeding.

```python
# analytics/playoff_format.py
import numpy as np
from scipy.stats import binom

def series_win_prob(game_win_prob: float, games_to_win: int = 2) -> float:
    """P(wins best-of-3 series) given per-game win probability."""
    p = game_win_prob
    # Win 2-0 or 2-1
    win_2_0 = p ** 2
    win_2_1 = 2 * p**2 * (1 - p)
    return win_2_0 + win_2_1
```

### Feature 5 — Artificial Turf Home Advantage

For teams playing on artificial turf (e.g., Seattle, Portland, New England),
quantify the home advantage vs visiting grass teams.

```python
# utils/surface_advantage.py
TURF_STADIUMS = {
    "Seattle Sounders", "Portland Timbers", "Vancouver Whitecaps",
    "Philadelphia Union", "New England Revolution",
}

def get_surface_advantage(home_team: str, away_team: str) -> float:
    """Additional home advantage when home team plays on turf."""
    home_on_turf = home_team in TURF_STADIUMS
    away_on_turf = away_team in TURF_STADIUMS
    if home_on_turf and not away_on_turf:
        return 0.05  # 5% win probability boost
    return 0.0
```

---

## Q2 (Nov 2026 – Jan 2027) — Model Enhancement

### Feature 6 — SuperDraft Impact Model

Track end-of-season SuperDraft picks. Rookies on new teams take time to
integrate — model the performance ramp-up time for newly drafted players.

### Feature 7 — Conference Strength Adjustment

Eastern and Western Conferences have different average strength. Adjust
predictions when teams cross conferences.

```python
# utils/conference.py
import pandas as pd

EAST_CONF = {"Inter Miami CF", "NYCFC", "Philadelphia Union", "Atlanta United",
              "New England Revolution", "Columbus Crew", "FC Cincinnati",
              "Toronto FC", "Montreal CF", "Orlando City", "Nashville SC",
              "Charlotte FC", "DC United", "New York Red Bulls", "Chicago Fire"}
WEST_CONF = {"LA Galaxy", "LAFC", "Seattle Sounders", "Portland Timbers",
              "Real Salt Lake", "FC Dallas", "Sporting KC", "Colorado Rapids",
              "Minnesota United", "San Jose Earthquakes", "Vancouver Whitecaps",
              "Austin FC", "Houston Dynamo", "St. Louis City SC"}

def is_cross_conference(home: str, away: str) -> bool:
    home_conf = "East" if home in EAST_CONF else "West"
    away_conf = "East" if away in EAST_CONF else "West"
    return home_conf != away_conf
```

### Feature 8 — Set Piece Goals Model

MLS teams vary significantly in set piece conversion. Build a set piece
goals feature: corners earned, free kick locations, specialist takers.

### Feature 9 — Goalkeeper Metric (Goals Prevented Above Average)

Compute GPAA for each MLS goalkeeper. Teams with GPAA goalkeepers
outperform their xGA significantly — adjust model accordingly.

```python
# analytics/goalkeeper.py
import pandas as pd

def compute_gpaa(keeper_stats: pd.DataFrame) -> pd.DataFrame:
    """Goals prevented above average per 90 min."""
    keeper_stats["expected_to_concede"] = keeper_stats["shots_on_target_faced"] * 0.32
    keeper_stats["gpaa_90"] = (
        (keeper_stats["expected_to_concede"] - keeper_stats["goals_conceded"])
        / keeper_stats["minutes_played"].clip(lower=1) * 90
    )
    return keeper_stats.sort_values("gpaa_90", ascending=False)
```

### Feature 10 — Home vs Away Goal Scoring Rate Differential

MLS has one of the highest home-advantage effects in world soccer. Quantify
the difference in xG/game home vs away per team.

---

## Q3 (Feb–Apr 2027) — Dashboard & Analytics

### Feature 11 — Playoff Bracket Simulator

With dynamic playoff seeding, simulate the remaining regular season +
playoffs 10,000 times. Output MLS Cup probabilities.

### Feature 12 — Player Props: Goals & Assists

Predict anytime goalscorer probability using target share, shots per 90,
and xG/shot. Compare against DraftKings props for value.

### Feature 13 — Team Form Heatmap by Competition

Visual heatmap: team × week colored by win/draw/loss, with intensity
showing margin of victory. Reveals momentum and slumps.

### Feature 14 — xG Timelines (When Teams Score)

Visualize when teams tend to score (early, late, final 10 minutes).
"Late winners" and "early collapses" patterns are exploitable.

### Feature 15 — Distance Metrics for Heat Maps

Show which areas of the pitch each team creates most xG from. Identify
teams that exploit wide areas vs central play.

---

## Q4 (May–Jul 2027) — Automation

### Feature 16 — Automated Fixture Fetch & Model Run

GitHub Action every Monday for upcoming week's fixtures + model predictions.
Auto-export best_bets.json for Sports Picks Grid.

### Feature 17 — Transfer Impact Quantifier

During transfer windows, ingest new signings and compute expected
WAR change per team. Adjust season-end predictions.

### Feature 18 — Weather Dome Impact

Teams in domes (NYCFC, Atlanta) have zero weather impact.
Flag dome vs outdoor as a feature when outdoor conditions are extreme.

### Feature 19 — MLS All-Star Game Prediction

Special model for MLS All-Star vs European club. Historical upsets and
player motivation factors create unique modeling challenges.

### Feature 20 — Social Media Sentiment Signal

Parse Twitter/X for injury news and lineup leaks before confirmed
official announcements. Brief edge before prices adjust.

### Feature 21 — Expected Points Table vs Actual

Show where each team ranks by xPts vs actual points. Teams significantly
above xPts are likely to regress; below = potential bounce-back picks.

### Feature 22 — Fan Attendance Impact on Home Advantage

Quantify how stadium attendance affects home advantage. Packed stadiums
amplify home field; empty stadiums reduce it.

---

## Timeline Summary

| Quarter | Focus | Key Deliverables |
|---------|-------|-----------------|
| Q1 Aug–Oct 2026 | Data foundation | ASA xG, travel fatigue, DP intelligence, playoff format, turf advantage |
| Q2 Nov 2026–Jan 2027 | Model enhancement | SuperDraft impact, conference strength, set piece model, GPAA, home/away xG |
| Q3 Feb–Apr 2027 | Dashboard | Playoff simulator, player props, form heatmap, xG timeline, shot location maps |
| Q4 May–Jul 2027 | Automation | Weekly pipeline, transfer impact, dome flag, All-Star model, sentiment signal, xPts table, attendance impact |
