# Competitor Analysis: MLS Prediction Sites

**Reviewed:** April 24, 2026  
**Sites Analyzed:**
- [Dimers.com](https://www.dimers.com/bet-hub/mls/schedule) — Data-driven, probabilistic, multi-sportsbook
- [PitchPredictions.com](https://www.pitchpredictions.com/) — Free stats platform, confidence-rated picks
- [StatSniper.com](https://statsniper.com/daily-picks/mls/) — AI-powered picks, edge/EV focus
- [FootballPredictions.com](https://footballpredictions.com/footballpredictions/mlspredictions/) — Weekly tips, multi-market, league stats

---

## 1. Dimers.com

### What They Do Well
- **Win probabilities as percentages** next to every matchup (e.g., "Toronto 53% — Atlanta 24%"), directly alongside the best available moneyline at that moment from multiple sportsbooks
- **Real-time odds comparison** across FanDuel, DraftKings, BetMGM, Caesars, BetRivers, bet365, Fanatics, theScore Bet, and prop exchange platforms (Kalshi, Novig)
- **Separate product tiers**: free schedule view → Best Bets (+EV picks) → Picks (moneyline/totals) → Futures (championship odds)
- **"Dimebot" AI assistant** — natural language interface over their prediction model
- **MLS Futures modeling** — explicit win probability for MLS Cup at each point in the season
- **Correct Score and Over/Under** predictions per match (visible on individual match pages)
- **Parlay Picker** tool that chains high-confidence picks

### Feature Gaps vs. This Repo
| Dimers Feature | Status in This Repo | Priority |
|---|---|---|
| Win % shown alongside odds | Probabilities exist, no live odds side-by-side | High |
| Best Available Odds aggregation | Odds API integrated but not displayed per-game | High |
| Futures / playoff probability | Not implemented | Medium |
| Parlay builder | Not implemented | Low |
| AI assistant over model | Not implemented | Low |

### Key Insight
Dimers' core differentiator is the **side-by-side view of model probability vs. implied book probability** — making +EV opportunities immediately visible. This is the single highest-value feature gap in this repo.

---

## 2. PitchPredictions.com

### What They Do Well
- **Confidence percentage per prediction** on every match card (e.g., "1, 80% Win Probability")
- **Multi-market tips on a single card**: 1X2 + Over/Under + BTTS + Correct Score + Asian Handicap — all from one model pass
- **League table averages** shown as part of each match card (home/away form averages across the league)
- **Team Comparison tool** — side-by-side head-to-head stat comparison for any two teams
- **Injury/squad news integration** — confirmed absences factored into the prediction
- **Market movement tracking** — flags when betting lines shift significantly (indicating sharp money)
- **700+ leagues** covered through API-Football; demonstrates the architecture for multi-league expansion
- **Live score integration** with in-progress predictions

### Feature Gaps vs. This Repo
| PitchPredictions Feature | Status in This Repo | Priority |
|---|---|---|
| Confidence % displayed on each prediction | Model probabilities computed, not surfaced as % on match cards | High |
| BTTS prediction market | Not modeled | High |
| Over/Under 2.5 Goals | Not modeled | High |
| Correct Score prediction | Not modeled | Medium |
| Asian Handicap / Spread | Not modeled | Medium |
| Team Comparison tool (side-by-side) | Not implemented | Medium |
| Injury/availability factor in model | Designated Player flag exists, no general injury feed | Medium |
| Market movement flag | Odds API integrated but no movement tracking | Low |

### Key Insight
PitchPredictions shows that users expect **all major markets on one match card**. A single model inference pass can output probabilities for H/D/A, BTTS, and O/U simultaneously — there is very little additional ML cost, mainly UI work.

---

## 3. StatSniper.com

### What They Do Well
- **Edge percentage** — compares model-projected spread to current book spread, outputs a signed edge (e.g., "11.1% edge"). This is the clearest possible signal that a bet has value.
- **Confidence % separate from edge** — "Edge 11.1%, Confidence 95%" distinguishes *how big* the edge is from *how certain* the model is
- **Offense/Defense rankings 1–30** per MLS team, updated per game — simple, interpretable, shareable
- **Rest-days-as-a-feature** explicitly — "Columbus has 4 days rest, Galaxy on back-to-back and traveling" is the core of one of their picks
- **Public betting % and "Fade" signal** — surfaces where public money is overcorrecting; "Public loves Cincy off last year's hype, ignoring their 2.4 goals allowed per game"
- **Player prop predictions** — shots on target, total shots, goals, with per-player averages and opponent defensive rank factored in
- **Plain-English narrative generation** — each pick includes a paragraph-length prose explanation of the model's reasoning
- **AI persona ("Chad")** — every pick is attributed to the AI analyst with consistent voice

### Feature Gaps vs. This Repo
| StatSniper Feature | Status in This Repo | Priority |
|---|---|---|
| Edge % (model prob vs implied prob) | Not computed | **Critical** |
| Rest days feature | Travel distance exists; rest days not calculated | High |
| Offense / Defense ranking table (1–30) | xG data available; rankings not surfaced | High |
| Public betting % integration | No public betting data source | Medium |
| Player prop model | ASA player goals added data fetched; no prop model | Medium |
| Narrative generation from model output | Not implemented | Medium |
| Back-to-back flag | Not in feature set | High |

### Key Insight
StatSniper's approach of **ranking every team 1–30 on offense and defense** and then narrating the matchup rank collision ("Cincinnati's offense ranks 29th facing Charlotte's 8th-ranked defense") is a highly legible, low-engineering-cost UI feature that directly uses data already available from ASA xG stats.

---

## 4. FootballPredictions.com

### What They Do Well
- **League-wide stats panel** at the top of every league page:
  - Home win % / Draw % / Away win %
  - Average goals per match (home vs. away)
  - BTTS rate as % of all matches
  - Clean sheet % 
  - Over 0.5/1.5/2.5/3.5/4.5 goal distribution
  - Shots per match (home vs. away), fouls, offsides
- **Correct Score tip** prominently featured on every match card (not just a probability — an explicit recommended scoreline)
- **Shots on Target total** prediction per match ("Total SoT: Over 9.5") — leverages ASA shot data directly
- **BTTS tip on every card** as a binary Yes/No signal
- **Season-wide historical base rates** surfaced in UI, helping users understand base rate calibration
- **Correct Score: 2-2** style explicit pick encourages users to engage with the value calculation

### Feature Gaps vs. This Repo
| FootballPredictions Feature | Status in This Repo | Priority |
|---|---|---|
| League stats dashboard (home win%, draw%, goals avg) | Raw data exists; no UI panel | High |
| Shots on target market prediction | ASA shot data available; not modeled | Medium |
| Fouls/offsides per match stats | Not tracked | Low |
| Explicit correct score recommendation | Not modeled | Medium |
| Season-long base rates displayed in app | Not displayed | Medium |

### Key Insight
The **league stats panel is fast to build and high in user value** — it turns the historical CSV data already in `data_files/` into a contextual reference pane. Users calibrate their expectations against base rates before reading individual predictions.

---

## Summary: Top 10 Feature Gaps to Close

Ranked by (impact × feasibility) against the current repo:

| # | Feature | Source Inspiration | Data Already Available? | Effort |
|---|---|---|---|---|
| 1 | **Edge % / +EV detection** (model prob vs. implied odds prob) | Dimers, StatSniper | Yes — Odds API integrated | Medium |
| 2 | **BTTS market prediction** | PitchPredictions, FootballPredictions | Yes — xG data enables this | Low |
| 3 | **Over/Under 2.5 Goals prediction** | All 4 sites | Yes — Poisson / xG model | Low |
| 4 | **Win probability % displayed on match cards** | Dimers, PitchPredictions | Yes — model already outputs probas | Low |
| 5 | **Team offense/defense rankings (1–30)** | StatSniper | Yes — ASA xG data | Low |
| 6 | **League-wide stats panel** | FootballPredictions | Yes — historical CSV | Low |
| 7 | **Rest days feature** (days since last game) | StatSniper | Partial — fixture dates in CSV | Low |
| 8 | **Back-to-back game flag** | StatSniper | Partial — fixture dates in CSV | Low |
| 9 | **Correct Score prediction** | FootballPredictions, PitchPredictions | Partial — needs Poisson model | Medium |
| 10 | **Player prop predictions** (shots, goals) | StatSniper | Yes — ASA player goals added | High |

---

## Unique Differentiators This Repo Could Own

Features none of the four sites do particularly well for MLS specifically:

1. **Turf/Grass surface impact visualization** — none of the sites model this; ASA + surface data could quantify the turf premium
2. **Conference-adjusted strength ratings** — cross-conference H2H is thin; an Elo-style rating that discounts cross-conference games is a genuine modeling edge
3. **Designated Player availability impact** — none of the sites explicitly model DP presence; the Messi/LAFC effect is measurable in xG pre/post
4. **Travel distance heat map** — a visual showing this week's travel burden across all MLS teams would be uniquely engaging
5. **Playoff probability tracker** — MLS playoff seeding is complex (conference brackets + supporters' shield); a dynamic probability chart updated after each gameweek is absent from all four sites
