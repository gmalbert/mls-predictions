# MLS (Major League Soccer)

**Tier: A** | Season: February–November | DraftKings: Full markets (official partner)

## Data Sources

- MLS official API (free — scores, standings, schedules)
- `football-data.org` (covers MLS)
- FBref (scrapeable — advanced metrics)
- American Soccer Analysis / `itscalledsoccer` Python package (free — xG, goals added)
- The Odds API (market lines)

## Overview

MLS is structurally one of the most unique soccer leagues in the world, and that uniqueness is what makes it both challenging and potentially lucrative to model. European soccer models built on the EPL's competitive structure break down in MLS because the underlying incentives, roster mechanics, and scheduling are fundamentally different. A model that accounts for these quirks may have genuine edge over the public and even the sportsbooks, which tend to put less pricing effort into MLS than into EPL.

The `itscalledsoccer` Python package from American Soccer Analysis is a hidden gem — it provides free, clean access to xG, xA, and goals added data purpose-built for MLS, something no equivalent free resource provides for most domestic leagues.

---

## What Makes MLS Different from European Leagues

This is the most important section to understand before building. MLS breaks several assumptions that European league models rely on:

**1. Salary cap and roster parity.** MLS is the only top-flight soccer league in the world with a hard salary cap. This creates genuine competitive parity — upsets are far more common than in Europe, where financial dominance (Man City, PSG, Bayern) is highly predictable. Models that rely on squad value or wage spend as features will underperform in MLS. Weight recent form and head-to-head records more heavily instead.

**2. Designated Player rule.** Each team can sign up to 3 "Designated Players" whose salaries exceed the cap. A single player's availability has outsized model impact in a way that's unusual in European squads with 25 deep contributors. Messi at Inter Miami is the extreme example — his absence/presence can shift the line significantly.

**3. Unbalanced schedule and conference structure.** MLS has Eastern and Western Conferences with a heavily unbalanced schedule — teams play more games against conference opponents. Two teams may only meet once or not at all in a given regular season. Head-to-head history is less meaningful than in EPL where every team plays every other team twice.

**4. Travel distances.** This is a genuinely unique modeling feature with no EPL equivalent. An away trip in EPL is 2–3 hours by bus. An MLS away trip can be a 6-hour cross-country flight from Seattle to Miami. Travel fatigue is a meaningful, encodable feature. Track distance between stadiums and rest days after long-haul away trips.

**5. Turf vs. grass.** A significantly higher proportion of MLS stadiums use artificial turf compared to European leagues. Teams that play on turf at home often perform differently on natural grass, and visiting teams experience the same adjustment. This is a real binary feature worth encoding per match.

**6. Playoff format.** MLS uses a bracket playoff to determine the champion, not the regular season table. This means late-season games have wildly different incentive structures — a team battling for a playoff spot plays very differently from one already eliminated or one that has clinched. European models don't need to account for this.

**7. Roster volatility.** Expansion teams, the SuperDraft, trades, and allocation orders mean MLS rosters change more aggressively each offseason than European squads. Historical data degrades faster — weight recent seasons more heavily in training.

---

## Pros

- **American Soccer Analysis provides free MLS-specific xG data.** The `itscalledsoccer` package gives you clean xG, xA, and goals added for MLS going back to 2013. This is purpose-built for the league and more reliable than using a European xG source.
- **MLS official API has free documented endpoints.** Scores, standings, and schedules are available without scraping.
- **Season runs February–November**, filling the gap when European leagues are dark (June–August) and giving you year-round soccer coverage alongside the European seasons.
- **DraftKings is an official MLS partner.** Full market coverage — moneylines, spreads, totals, futures, player props. Liquidity is solid for major market clubs (Inter Miami, LAFC, Seattle, Atlanta).
- **Less efficient markets = more potential edge.** Sportsbooks invest less pricing sophistication into MLS than EPL. A well-calibrated model may find more value against the line than in more efficient European markets.
- **Structural quirks are encodable.** Every "con" about MLS modeling is also an opportunity — if you correctly account for travel, turf, and cap dynamics, you have features that public bettors (and many books) ignore.

---

## Cons

- **Parity means genuine model uncertainty.** The salary cap works. Any given MLS match has a more compressed win-probability distribution than an EPL match. Your model will produce fewer high-confidence picks, which matters for user experience.
- **Travel and turf data must be manually built.** There's no API that gives you stadium surface types or inter-city distances. You'll need to build a lookup table for all ~30 MLS venues.
- **Thin markets for smaller clubs.** DraftKings depth on major-market clubs (Miami, LA, NY, Seattle, Atlanta) is good. For smaller markets (San Jose, Colorado, D.C.), prop markets are limited.
- **Designated player injury monitoring is high-stakes.** A single player's availability can move a line 10+ points. You need reliable, near-real-time roster news which is harder to automate for MLS than for MLB or NBA.
- **Conference-based scheduling complicates cross-conference predictions.** Eastern teams play very few games against Western teams, making cross-conference statistical comparisons noisier.

---

## Recommended Build Approach

**Primary model targets:** Moneyline (1X2), over/under 2.5 goals, both teams to score, MLS Cup futures.

**Key features to engineer:**
- xG and xGA from American Soccer Analysis (last 5 and 10 games, weighted)
- Travel distance for away team (miles between stadiums)
- Days rest since last match
- Surface type for home stadium (turf vs. grass)
- Form points (last 5 games, conference and overall)
- Playoff position race flag (is team within 2 spots of playoff line?)
- Designated player availability flag
- Eastern vs. Western conference matchup (cross-conference away trips are particularly punishing)

**American Soccer Analysis data pull:**
```python
from itscalledsoccer.client import AmericanSoccerAnalysis

asa = AmericanSoccerAnalysis()

# Get xG data for all MLS games
xg_data = asa.get_games(leagues="mls", seasons=[2023, 2024])

# Get team xG summaries
team_xg = asa.get_team_xgoals(leagues="mls", season_name="2024")

# Get player goals added
player_ga = asa.get_player_goals_added(leagues="mls", season_name="2024")
```

**Suggested model stack:** Same ensemble approach as EPL, but with travel distance, turf, and playoff context as additional features. Consider training separate models for conference games vs. cross-conference games given the scheduling imbalance.

**Backtesting window:** 2017–2023 (7 seasons, ~1,750 games). Older data is less reliable due to heavy expansion activity. Weight 2020–2023 most heavily in training.

---

## Build Priority

**High — especially valuable for schedule coverage.** MLS is active June through August when your European leagues are dark. It's the only addition that gives you continuous soccer content year-round. Build it alongside or immediately after the European league expansion.
