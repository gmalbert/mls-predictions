# Frontier Enhancement Blueprint

This supplements the existing MLS, data, model, odds, and EV roadmaps. The proposals emphasize league-specific travel, roster rules, and uncertainty.

> **Implementation audit — 2026-08-24:** The feature-engineering, scenario,
> score-distribution, conformal-abstention, and evaluation structures in this
> blueprint are implemented. Provider-fed inputs remain conditional on configured
> sources and must be validated before they can support a release decision.

## Model enhancements

- Hierarchical club ratings with expansion-team and coaching-regime priors.
- Travel-load interactions: distance, time zones, altitude, surface, temperature, and turnaround.
- Designated Player/U22/TAM availability expressed as minutes-weighted replacement value, not binary injury counts.
- Tactical matchup features from possession chains and set pieces.
- A joint score distribution plus conformal abstention for incoherent or highly uncertain matches.

```python
def travel_load(km, tz_shift, rest_hours, altitude_gain):
    return np.log1p(km) + 0.8 * abs(tz_shift) + 0.002 * max(altitude_gain, 0) - 0.03 * min(rest_hours, 120)
```

## Data enhancements

Ingest stadium surface/altitude, charter/travel estimates, roster mechanisms, player transactions, manager tenures, projected/confirmed lineups, referee crews, event data, and timestamped multi-book prices. Record `available_at` independently from event date to prevent late lineup and corrected-stat leakage.

## Product enhancements

- Travel and environment card on every match.
- Roster-impact waterfall from unavailable player to replacement.
- Scenario probabilities for questionable starters.
- Probability-change feed explaining new data since the previous snapshot.
- Cross-market consistency checks and a visible no-bet state.

```python
class Snapshot(BaseModel):
    fixture_id: str
    prediction_time: datetime
    lineup_scenario: str
    probabilities: dict[str, float]
    data_manifest: str
    model_version: str
```

## Evaluation

Use rolling seasons and expansion-team holdouts. Report log loss, RPS, calibration, CLV, turnover, and drawdown by travel bucket, surface, altitude, roster churn, manager tenure, and prediction horizon. Compare against Elo, market-only, and home-field baselines.
