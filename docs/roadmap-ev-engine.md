# Roadmap: Expected Value Engine & Edge Detection

**Inspired by:** Dimers.com (+EV best bets), StatSniper.com (Edge %)  
**Status:** Not started  
**Priority:** Critical — single highest-value differentiator vs. competitors

---

## Overview

Every prediction site shows probabilities. Few show whether those probabilities represent *value* against available odds. The gap between **model-implied probability** and **book-implied probability** is the Expected Value (EV). A positive EV bet is one where the model believes the true probability of an outcome exceeds the probability implied by the current moneyline.

This repo already fetches live odds via The Odds API. The missing step is computing and surfacing EV per market per game.

---

## 1. The EV Calculation

### Core Formula

```
EV% = (model_prob * decimal_odds) - 1

Where decimal_odds = convert from American moneyline
```

For a positive American moneyline (underdog): `decimal = (moneyline / 100) + 1`  
For a negative American moneyline (favorite): `decimal = (100 / abs(moneyline)) + 1`

### Implementation

```python
def american_to_decimal(american_odds: int) -> float:
    """Convert American moneyline to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1.0
    else:
        return (100 / abs(american_odds)) + 1.0


def implied_probability(american_odds: int) -> float:
    """Book's implied win probability from American moneyline (no vig removed)."""
    decimal = american_to_decimal(american_odds)
    return 1.0 / decimal


def remove_vig(home_odds: int, draw_odds: int, away_odds: int) -> tuple[float, float, float]:
    """
    Remove sportsbook overround (vig) to get fair implied probabilities.
    Normalizes the three implied probabilities to sum to 1.0.
    """
    raw_probs = [implied_probability(o) for o in [home_odds, draw_odds, away_odds]]
    total = sum(raw_probs)
    return tuple(p / total for p in raw_probs)


def compute_ev(model_prob: float, american_odds: int) -> float:
    """
    Compute expected value percentage for a single market.
    
    model_prob: model's estimated win probability (0.0–1.0)
    american_odds: book's moneyline for this outcome
    
    Returns EV as a signed percentage (positive = +EV, negative = -EV)
    """
    decimal = american_to_decimal(american_odds)
    ev = (model_prob * decimal) - 1.0
    return round(ev, 4)


def edge_percentage(model_prob: float, book_implied_prob: float) -> float:
    """
    StatSniper-style edge: difference between model and book probability.
    Positive = model likes this outcome more than the book.
    """
    return round((model_prob - book_implied_prob) * 100, 1)
```

### Example Output
For a match where the model estimates Inter Miami win probability at 70% and BetMGM offers -235:

```
model_prob     = 0.70
american_odds  = -235
decimal_odds   = 1.426
EV             = (0.70 * 1.426) - 1 = -0.002  → roughly break-even
implied_prob   = 1/1.426 = 0.701 → book agrees, no edge

If BetRivers offers -195:
decimal_odds   = 1.513
EV             = (0.70 * 1.513) - 1 = +0.059  → +5.9% EV
edge_pct       = (0.70 - 0.662) * 100 = +3.8%
```

---

## 2. Multi-Book Odds Comparison

The Odds API returns odds from multiple books per game. The EV engine should:
1. Fetch all available books for each market
2. Compute EV at each book
3. Highlight the **best available EV** and which book offers it

```python
from typing import Optional
import pandas as pd

def find_best_ev_line(
    model_probs: dict[str, float],  # {'H': 0.53, 'D': 0.24, 'A': 0.23}
    odds_by_book: list[dict],        # from Odds API response
    min_ev_threshold: float = 0.03   # only flag if EV > 3%
) -> pd.DataFrame:
    """
    Find the best EV available across all books for all three outcomes.
    
    Returns DataFrame with columns:
    outcome, book, american_odds, model_prob, book_implied_prob, ev_pct, edge_pct
    """
    outcome_map = {'H': 'home', 'D': 'draw', 'A': 'away'}
    rows = []
    
    for book in odds_by_book:
        book_name = book['title']
        markets = book.get('markets', [])
        h2h = next((m for m in markets if m['key'] == 'h2h'), None)
        if not h2h:
            continue
        
        outcomes = {o['name']: o['price'] for o in h2h['outcomes']}
        
        for result, model_prob in model_probs.items():
            side = outcome_map[result]
            if side not in outcomes:
                continue
            
            american = outcomes[side]
            ev = compute_ev(model_prob, american)
            book_prob = implied_probability(american)
            edge = edge_percentage(model_prob, book_prob)
            
            rows.append({
                'outcome': result,
                'book': book_name,
                'american_odds': american,
                'model_prob': model_prob,
                'book_implied_prob': round(book_prob, 4),
                'ev_pct': round(ev * 100, 1),
                'edge_pct': edge,
                'is_plus_ev': ev > min_ev_threshold
            })
    
    return pd.DataFrame(rows).sort_values('ev_pct', ascending=False)
```

---

## 3. Rest Days Feature (StatSniper-Inspired)

StatSniper's most cited edge for the April 22 Columbus/Galaxy pick was rest: "Columbus has 4 days, Galaxy on back-to-back and traveling." This is a concrete, measurable advantage that the current feature set lacks.

### Implementation

```python
def compute_rest_days(team: str, match_date: pd.Timestamp,
                      historical_df: pd.DataFrame) -> int:
    """
    Days since team's last completed match.
    Returns 7 if no previous match found in last 30 days (full rest).
    """
    team_matches = historical_df[
        ((historical_df['HomeTeam'] == team) | (historical_df['AwayTeam'] == team)) &
        (historical_df['Date'] < match_date)
    ].sort_values('Date', ascending=False)
    
    if team_matches.empty:
        return 7  # assume fresh
    
    last_match = team_matches.iloc[0]['Date']
    return (match_date - last_match).days


def is_back_to_back(rest_days: int, threshold: int = 3) -> bool:
    """Flag matches where team has fewer than threshold days rest."""
    return rest_days <= threshold


def rest_advantage(home_rest: int, away_rest: int) -> int:
    """
    Signed rest advantage: positive = home team better rested.
    Used as a numeric feature in the model.
    """
    return home_rest - away_rest
```

### Features to Add to `prepare_model_data.py`
```python
home_rest_days          # integer: days since last game
away_rest_days          # integer: days since last game
home_is_back_to_back    # bool: rest_days <= 3
away_is_back_to_back    # bool: rest_days <= 3
rest_advantage          # signed: home_rest - away_rest
```

### MLS-Specific Context
MLS has notoriously compressed mid-season schedules, especially around international windows and playoff pushes. Teams may play 3 games in 8 days. Back-to-back road games (traveling West Coast to East Coast in 3 days) have a measurable negative xG impact.

---

## 4. Team Offense/Defense Rankings (StatSniper-Style)

### Overview
StatSniper ranks every MLS team 1–30 on offense and defense. The matchup narrative writes itself: "Cincinnati's offense ranks 29th facing Charlotte's 8th-ranked defense." These rankings are trivially computed from existing ASA xG data.

### Implementation

```python
def compute_team_rankings(team_xg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute offensive and defensive rankings 1–N for all MLS teams.
    
    Offensive rank: based on xG per game (1 = best attack)
    Defensive rank: based on xGA per game (1 = best defense = fewest goals allowed)
    """
    df = team_xg_df.copy()
    
    # Normalize to per-game rates
    df['xg_per_game'] = df['total_xg'] / df['count_games']
    df['xga_per_game'] = df['total_xga'] / df['count_games']
    
    # Rank (1 = best)
    df['offense_rank'] = df['xg_per_game'].rank(ascending=False).astype(int)
    df['defense_rank'] = df['xga_per_game'].rank(ascending=True).astype(int)  # lower xGA = better
    
    return df[['team_name', 'xg_per_game', 'xga_per_game', 'offense_rank', 'defense_rank']]


def matchup_narrative(home_team: str, away_team: str,
                       rankings: pd.DataFrame) -> str:
    """
    Generate a plain-English matchup summary from rankings.
    StatSniper-style: highlight mismatches between strong offense vs weak defense.
    """
    h = rankings[rankings['team_name'] == home_team].iloc[0]
    a = rankings[rankings['team_name'] == away_team].iloc[0]
    n = len(rankings)
    
    lines = []
    
    # Offensive mismatch: home attack vs away defense
    if h['offense_rank'] <= 5 and a['defense_rank'] >= n - 5:
        lines.append(
            f"{home_team}'s attack (#{h['offense_rank']} in MLS) "
            f"faces {away_team}'s defense (#{a['defense_rank']}). "
            f"Elite offense vs. poor defense — large home scoring edge."
        )
    
    # Defensive mismatch: away attack vs home defense
    if a['offense_rank'] <= 5 and h['defense_rank'] >= n - 5:
        lines.append(
            f"{away_team}'s attack (#{a['offense_rank']}) "
            f"against {home_team}'s defense (#{h['defense_rank']}) — "
            f"away side has a real scoring edge here."
        )
    
    if not lines:
        lines.append(
            f"Relatively balanced matchup: {home_team} offense #{h['offense_rank']}, "
            f"defense #{h['defense_rank']}; {away_team} offense #{a['offense_rank']}, "
            f"defense #{a['defense_rank']}."
        )
    
    return " ".join(lines)
```

### Streamlit UI: Rankings Table

```python
def render_rankings_table(rankings: pd.DataFrame):
    """Display current MLS team rankings in Streamlit."""
    st.subheader("📊 MLS Team Rankings (Current Season)")
    
    display = rankings[['team_name', 'xg_per_game', 'xga_per_game',
                          'offense_rank', 'defense_rank']].copy()
    display.columns = ['Team', 'xG/Game', 'xGA/Game', 'Attack Rank', 'Defense Rank']
    display = display.sort_values('Attack Rank')
    
    def color_rank(val, n=len(rankings)):
        """Green for top 5, red for bottom 5, neutral otherwise."""
        if val <= 5:
            return 'background-color: #d4edda'
        elif val >= n - 4:
            return 'background-color: #f8d7da'
        return ''
    
    styled = display.style.applymap(color_rank, subset=['Attack Rank', 'Defense Rank'])
    st.dataframe(styled, hide_index=True, use_container_width=True)
```

---

## 5. +EV Alert System

Add a "Best Bets" panel (inspired by Dimers' Best Bets product) that surfaces only the highest-EV picks from the upcoming fixture list.

```python
def generate_best_bets(
    upcoming_df: pd.DataFrame,
    min_ev: float = 0.04,       # minimum 4% EV
    min_confidence: float = 0.55 # model must be at least 55% confident
) -> pd.DataFrame:
    """
    Filter upcoming fixtures to only those with +EV and sufficient model confidence.
    
    Columns expected in upcoming_df:
    - HomeTeam, AwayTeam, Date
    - home_prob, draw_prob, away_prob  (model output)
    - best_home_odds, best_draw_odds, best_away_odds  (from Odds API, best book)
    - best_home_book, best_draw_book, best_away_book
    """
    rows = []
    
    for _, row in upcoming_df.iterrows():
        for outcome, prob_col, odds_col, book_col in [
            ('Home Win', 'home_prob', 'best_home_odds', 'best_home_book'),
            ('Draw',     'draw_prob', 'best_draw_odds', 'best_draw_book'),
            ('Away Win', 'away_prob', 'best_away_odds', 'best_away_book'),
        ]:
            model_prob = row.get(prob_col)
            american_odds = row.get(odds_col)
            
            if pd.isna(model_prob) or pd.isna(american_odds):
                continue
            
            ev = compute_ev(model_prob, int(american_odds))
            
            if ev >= min_ev and model_prob >= min_confidence:
                rows.append({
                    'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    'Date': row['Date'],
                    'Pick': outcome,
                    'Model Prob': f"{model_prob:.0%}",
                    'Best Odds': f"+{american_odds}" if american_odds > 0 else str(american_odds),
                    'Book': row.get(book_col, 'N/A'),
                    'EV': f"+{ev*100:.1f}%",
                    'edge_sort': ev
                })
    
    if not rows:
        return pd.DataFrame()
    
    return (pd.DataFrame(rows)
              .sort_values('edge_sort', ascending=False)
              .drop(columns=['edge_sort']))
```

### UI Placement
Add as a highlighted callout box at the top of Tab 1 (Predictions):

```python
st.subheader("🎯 Today's Best Bets")
st.caption("Picks where the model finds ≥4% edge over current book odds")

best_bets_df = generate_best_bets(upcoming_with_odds)
if not best_bets_df.empty:
    st.dataframe(best_bets_df, hide_index=True, use_container_width=True)
else:
    st.info("No +EV edges found in today's slate. Check back as lines move.")
```

---

## 6. Line Movement Tracker (PitchPredictions-Inspired)

Track how odds move from open to close. A line moving from +150 to +120 (favorite getting more action) is a signal; a line moving from -150 to -180 (sharp money on favorite) is a stronger signal.

```python
# In automation/nightly_pipeline.py
# Store odds snapshots with timestamps

def snapshot_odds(fixtures_df: pd.DataFrame, snapshot_time: str) -> pd.DataFrame:
    """Fetch current odds and append to historical odds log."""
    current_odds = fetch_odds_api()  # existing function
    current_odds['snapshot_time'] = snapshot_time
    
    odds_log_path = 'data_files/odds_movement_log.csv'
    if os.path.exists(odds_log_path):
        existing = pd.read_csv(odds_log_path)
        updated = pd.concat([existing, current_odds], ignore_index=True)
    else:
        updated = current_odds
    
    updated.to_csv(odds_log_path, index=False)
    return updated


def compute_line_movement(match_id: str, outcome: str) -> dict:
    """
    Compute opening vs. current odds movement for a specific outcome.
    Returns direction and magnitude of move.
    """
    log = pd.read_csv('data_files/odds_movement_log.csv')
    match_odds = log[log['match_id'] == match_id].sort_values('snapshot_time')
    
    if len(match_odds) < 2:
        return {'direction': 'stable', 'magnitude': 0}
    
    opening = match_odds.iloc[0][f'{outcome}_odds']
    current = match_odds.iloc[-1][f'{outcome}_odds']
    
    open_dec = american_to_decimal(opening)
    cur_dec = american_to_decimal(current)
    
    # Positive magnitude = odds shortened (team got more action)
    magnitude = (1/cur_dec - 1/open_dec) * 100
    direction = 'shortened' if magnitude > 0 else 'drifted'
    
    return {
        'direction': direction,
        'magnitude': round(abs(magnitude), 1),
        'opening': opening,
        'current': current
    }
```

---

## 7. Implementation Roadmap

### Phase 1 — Core EV Engine (1–2 days)
- [ ] Add `ev_engine.py` to `models/` with `compute_ev()`, `find_best_ev_line()`, `edge_percentage()`
- [ ] Update `fetch_upcoming_fixtures.py` to save best available moneyline per outcome per game to `upcoming_fixtures.csv`
- [ ] Add "Best Bets" panel to Tab 1 of `predictions.py`

### Phase 2 — Rest Days & Rankings (1–2 days)
- [ ] Add `compute_rest_days()`, `is_back_to_back()`, `rest_advantage()` to `prepare_model_data.py`
- [ ] Add `home_rest_days`, `away_rest_days`, `home_is_back_to_back`, `away_is_back_to_back`, `rest_advantage` to training feature set
- [ ] Add `compute_team_rankings()` to a new `models/rankings.py`
- [ ] Add rankings table UI to Tab 3 (Team Stats) or a new tab

### Phase 3 — Line Movement (2–3 days)
- [ ] Add `snapshot_odds()` call to `automation/nightly_pipeline.py` at multiple times of day
- [ ] Add `compute_line_movement()` utility
- [ ] Display line movement arrows (↑ shortened, ↓ drifted) on match cards in Tab 1

---

## Notes on Data Quality

- **Odds API free tier** returns ~500 requests/month. Snapshot once per day per game (not real-time). For real-time line movement, a paid plan is needed.
- **Rest days calculation** requires fixture dates from the ESPN or MLS API — the `upcoming_fixtures.csv` should already contain these. Historical match dates are in `combined_historical_data.csv`.
- **EV calculations are sensitive to model calibration.** Run `CalibratedClassifierCV` (already in `roadmap-models-next-steps.md`) before exposing EV figures publicly — uncalibrated probabilities will generate misleading EV signals.
- **Always display a disclaimer** alongside EV figures: "Model-derived probabilities. Not financial advice."
