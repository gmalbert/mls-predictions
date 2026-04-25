# Roadmap: Expanded Betting Markets

**Inspired by:** FootballPredictions.com, PitchPredictions.com, StatSniper.com  
**Status:** Not started  
**Priority:** High — low marginal model cost, high UI/user value

---

## Overview

Every competitor site predicts multiple betting markets simultaneously. The current repo outputs only 1X2 (Home Win / Draw / Away Win). A single additional inference pass over the match-level xG data can generate four more markets with minimal new modeling:

- **BTTS** — Both Teams to Score (Yes/No)
- **Over/Under 2.5 Goals** — total goals threshold
- **Shots on Target total** — Over/Under market
- **Correct Score** — explicit predicted scoreline

These outputs can be added to the existing `predictions.py` UI on the match prediction cards and in the new Tab 6 proposed below.

---

## 1. BTTS — Both Teams to Score

### What It Is
A binary prediction: will both teams score at least one goal in the match?

### Why It's Valuable
- FootballPredictions.com and PitchPredictions.com both feature BTTS on every match card
- Base rate in MLS 2025: ~60% of matches end with both teams scoring (FootballPredictions.com data)
- The market is sensitive to defensive form — a team allowing 2.4 goals per game nearly always concedes. BTTS pricing reflects this poorly in public-facing lines.

### How to Model It

BTTS probability can be derived directly from the team-level Poisson goal model (if implemented) or estimated from xG features:

```python
import numpy as np
from scipy.stats import poisson

def btts_probability(home_xg: float, away_xg: float) -> float:
    """
    Compute BTTS probability using independent Poisson assumption.
    P(BTTS) = P(home scores >= 1) * P(away scores >= 1)
              = (1 - P(home scores 0)) * (1 - P(away scores 0))
    """
    p_home_scores = 1 - poisson.pmf(0, home_xg)
    p_away_scores = 1 - poisson.pmf(0, away_xg)
    return round(p_home_scores * p_away_scores, 4)

def btts_label(prob: float, threshold: float = 0.55) -> str:
    """Return Yes/No signal with confidence."""
    if prob >= threshold:
        return f"Yes ({prob:.0%})"
    elif prob <= (1 - threshold):
        return f"No ({1 - prob:.0%})"
    else:
        return f"No lean ({prob:.0%})"
```

### Feature Dependencies
- `home_xg_l5` / `away_xg_l5` — already in `prepare_model_data.py`
- `home_xga_l5` / `away_xga_l5` — add alongside xG (xG against = defensive proxy)

### MLS-Specific Calibration Note
Turf stadiums tend to have higher BTTS rates — faster surface, higher scoring pace. Add `home_is_turf` as a calibration multiplier:

```python
TURF_BTTS_MULTIPLIER = 1.08  # empirical; recalibrate each season
if home_is_turf:
    p_home_scores *= TURF_BTTS_MULTIPLIER
```

---

## 2. Over/Under Goals (1.5, 2.5, 3.5)

### What It Is
A prediction that total match goals exceed or fall below a threshold. All four competitor sites feature O/U 2.5 as their primary totals market.

### Why It's Valuable
- MLS 2025 base rates (FootballPredictions.com): Over 2.5 = 59%, Under 2.5 = 41%
- The model already has xG data, which is the strongest available predictor of expected total goals
- O/U is often a sharper market than 1X2 because there are only two outcomes

### How to Model It

```python
def over_under_probabilities(home_xg: float, away_xg: float,
                              thresholds: list[float] = [1.5, 2.5, 3.5]) -> dict[str, float]:
    """
    Compute P(total goals > threshold) using Poisson convolution.
    Total goals = Home goals + Away goals (independent Poisson variates).
    """
    total_xg = home_xg + away_xg
    results = {}
    for t in thresholds:
        # P(X > t) where X ~ Poisson(total_xg)
        # = 1 - P(X <= floor(t))
        k_max = int(t)  # threshold is half-integer, so floor is integer below
        p_under = sum(poisson.pmf(k, total_xg) for k in range(k_max + 1))
        results[f"over_{t}"] = round(1 - p_under, 4)
        results[f"under_{t}"] = round(p_under, 4)
    return results
```

### UI Display
Show on each match card as a horizontal bar:

```
Goals: Under 2.5 ██████░░░░ 41%   Over 2.5 ░░░░██████ 59%
Goals: Under 1.5 ████░░░░░░ 22%   Over 1.5 ░░░░░░████ 78%
```

### MLS-Specific Note
Cross-conference road games have slightly lower scoring (tired away sides, unfamiliar opponents). Weight `is_cross_conference` and `away_travel_miles` as dampening factors on `total_xg` before computing probabilities.

---

## 3. Shots on Target (SoT) Total

### What It Is
An Over/Under market on combined shots on target in the match. FootballPredictions.com features this as "Total SoT: Over 9.5" on every card.

### Why It's Valuable
- MLS 2025: average 25.5 shots per match, ~11 shots on target per match (inferred from FootballPredictions data)
- ASA data includes shots data; this market is uniquely well-modeled with the data already fetched
- SoT lines are often less efficient than goals lines — sharper edges available

### Data Source
```python
# Already available from itscalledsoccer
asa = AmericanSoccerAnalysis()
team_xg = asa.get_team_xgoals(leagues="mls", season_name="2025")
# shot_attempts and shots_on_target per team are in this dataset
```

### Model Approach
```python
def sot_probability(home_sot_avg: float, away_sot_avg: float,
                    threshold: float = 9.5) -> dict[str, float]:
    """
    Combined shots on target is approximately Poisson distributed.
    Use team's average SoT per game weighted by last 5 games.
    """
    total_sot_avg = home_sot_avg + away_sot_avg
    # Negative binomial may fit better for overdispersed shot counts
    # but Poisson is a reasonable starting point
    from scipy.stats import poisson
    k_max = int(threshold)
    p_under = sum(poisson.pmf(k, total_sot_avg) for k in range(k_max + 1))
    return {
        "over": round(1 - p_under, 4),
        "under": round(p_under, 4),
        "line": threshold
    }
```

### Features to Add to `prepare_model_data.py`
```python
# Shots on target averages (last 5 games)
home_sot_l5   # average shots on target per game, home team, last 5
away_sot_l5   # average shots on target per game, away team, last 5
home_sot_against_l5  # defensive SoT allowed per game, last 5
away_sot_against_l5
```

---

## 4. Correct Score Prediction

### What It Is
The most precise and highest-payout market — predicting the exact final scoreline. All four competitor sites feature this.

### Why It's Valuable
- FootballPredictions.com lists explicit scorelines ("Correct Score: 2-2") alongside other markets
- Correct score betting has the highest house edge but also highest user engagement — people love specific predictions
- A data-driven correct score is a major differentiator vs. casual prediction sites

### Model Approach: Bivariate Poisson

The Dixon-Coles correction to the independence assumption (accounting for score correlation near 0-0) is recommended:

```python
import numpy as np
from scipy.stats import poisson
from itertools import product

def correct_score_distribution(home_xg: float, away_xg: float,
                                max_goals: int = 6,
                                rho: float = -0.1) -> dict[tuple, float]:
    """
    Generate full score probability matrix using Dixon-Coles correction.
    
    rho: correlation parameter (typically -0.1 for soccer; recalibrate on MLS data)
    home_xg, away_xg: expected goals for home/away team
    """
    
    def tau(x, y, lambda_h, mu_a, rho):
        """Dixon-Coles low-score correction factor."""
        if x == 0 and y == 0:
            return 1 - lambda_h * mu_a * rho
        elif x == 0 and y == 1:
            return 1 + lambda_h * rho
        elif x == 1 and y == 0:
            return 1 + mu_a * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0
    
    scores = {}
    total = 0.0
    
    for h, a in product(range(max_goals + 1), range(max_goals + 1)):
        p = (poisson.pmf(h, home_xg) * 
             poisson.pmf(a, away_xg) * 
             tau(h, a, home_xg, away_xg, rho))
        scores[(h, a)] = p
        total += p
    
    # Normalize
    return {k: round(v / total, 4) for k, v in scores.items()}


def top_correct_scores(home_xg: float, away_xg: float, top_n: int = 5):
    """Return the N most likely scorelines."""
    dist = correct_score_distribution(home_xg, away_xg)
    sorted_scores = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    return [(f"{h}-{a}", round(p * 100, 1)) for (h, a), p in sorted_scores[:top_n]]
```

### Calibrating `rho` on MLS Data
The Dixon-Coles rho parameter should be fit on the historical MLS match data in `data_files/combined_historical_data.csv`. Typical MLS value is likely around -0.08 to -0.12 (games are slightly more independent than European soccer due to tactical differences).

```python
from scipy.optimize import minimize_scalar

def fit_rho(historical_df: pd.DataFrame) -> float:
    """Fit Dixon-Coles rho on historical home/away goal data."""
    def neg_log_likelihood(rho):
        ll = 0
        for _, row in historical_df.iterrows():
            p = correct_score_distribution(row['home_xg'], row['away_xg'], rho=rho)
            ll += np.log(p.get((int(row['FTHG']), int(row['FTAG'])), 1e-10))
        return -ll
    result = minimize_scalar(neg_log_likelihood, bounds=(-0.5, 0.0), method='bounded')
    return result.x
```

---

## 5. Streamlit UI: "Markets" Tab

Add a new **Tab 6: Markets** to `predictions.py` displaying all four markets per upcoming fixture.

### Layout Concept

```python
# Tab 6 — Markets Overview
with tab6:
    st.header("📊 All Markets — Upcoming Fixtures")
    st.caption("Poisson-derived probabilities for BTTS, Over/Under, and Correct Score")
    
    for _, row in upcoming_fixtures.iterrows():
        with st.expander(f"⚽ {row['HomeTeam']} vs {row['AwayTeam']} — {row['Date']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("1X2", f"{row['home_team']} Win {row['home_prob']:.0%}")
                st.metric("Draw", f"{row['draw_prob']:.0%}")
                st.metric("Away Win", f"{row['away_prob']:.0%}")
            
            with col2:
                st.metric("BTTS Yes", f"{row['btts_prob']:.0%}")
                st.metric("Over 2.5", f"{row['over_2_5_prob']:.0%}")
                st.metric("Over 1.5", f"{row['over_1_5_prob']:.0%}")
            
            with col3:
                st.subheader("Top Correct Scores")
                for score, pct in row['top_scores'][:3]:
                    st.write(f"`{score}` — {pct}%")
```

---

## 6. League Stats Panel

Inspired by FootballPredictions.com's stat summary at the top of every league page.

Add a collapsible **MLS Season Stats** panel to the top of the predictions page:

```python
def render_league_stats_panel(historical_df: pd.DataFrame, season: int):
    """Display MLS base-rate stats for the current season."""
    season_df = historical_df[historical_df['Season'] == season]
    
    total_matches = len(season_df)
    home_wins = (season_df['FTR'] == 'H').sum()
    draws = (season_df['FTR'] == 'D').sum()
    away_wins = (season_df['FTR'] == 'A').sum()
    
    avg_home_goals = season_df['FTHG'].mean()
    avg_away_goals = season_df['FTAG'].mean()
    avg_total = avg_home_goals + avg_away_goals
    
    btts = ((season_df['FTHG'] > 0) & (season_df['FTAG'] > 0)).sum()
    over_2_5 = ((season_df['FTHG'] + season_df['FTAG']) > 2.5).sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Home Win %", f"{home_wins/total_matches:.0%}")
    col2.metric("Draw %", f"{draws/total_matches:.0%}")
    col3.metric("Away Win %", f"{away_wins/total_matches:.0%}")
    col4.metric("Avg Goals/Match", f"{avg_total:.2f}")
    
    col5, col6, col7 = st.columns(3)
    col5.metric("BTTS %", f"{btts/total_matches:.0%}")
    col6.metric("Over 2.5 %", f"{over_2_5/total_matches:.0%}")
    col7.metric("Clean Sheets", f"{1 - btts/total_matches:.0%}")
```

---

## Implementation Sequence

1. **Week 1:** Add `home_xga_l5`, `away_xga_l5`, `home_sot_l5`, `away_sot_l5` to `prepare_model_data.py`
2. **Week 1:** Implement `btts_probability()` and `over_under_probabilities()` utility functions in a new `models/market_predictions.py`
3. **Week 2:** Implement `correct_score_distribution()` with Dixon-Coles correction; fit `rho` on historical data
4. **Week 2:** Add league stats panel to `predictions.py` (Tab 1 or sidebar)
5. **Week 3:** Build Tab 6 Markets view in `predictions.py`
6. **Week 3:** Add rest days and back-to-back flag to feature engineering pipeline

---

## Notes

- The Poisson/Dixon-Coles approach is separate from the VotingClassifier ensemble — it runs in parallel and uses xG directly rather than the feature matrix. This is intentional: classification models are not well-suited for exact score distributions.
- Calibrate all Poisson parameters on 2020+ MLS data only (consistent with the recency weighting policy).
- Display confidence intervals alongside probabilities where sample size is small (early season, fewer than 5 games played).
