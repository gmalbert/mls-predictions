# Roadmap: Outstanding Unimplemented Tasks

This document aggregates the roadmap tasks in `docs/roadmap-*.md` that are not yet implemented in the current repository state.

## Notes
- Review was based on the current repo code in `predictions.py`, `prepare_model_data.py`, `models/`, `pages/`, `fetch_upcoming_fixtures.py`, `automation/nightly_pipeline.py`, and `.github/workflows/nightly.yml`.
- Existing implemented features such as `pages/6_Markets.py`, `pages/7_Best_Bets.py`, EV calculations in `models/ev_engine.py`, and the nightly pipeline are excluded from this list.

---

## 1. Critical / High Priority

<!-- 1. **REST API layer for predictions and fixture data**
   - Source: `docs/roadmap-infrastructure.md`, `docs/roadmap-infrastructure-next-steps.md`
   - Not implemented: no `FastAPI`/`uvicorn` service or API endpoints exist in repo. -->

2. **SQLite / database migration**
   - Source: `docs/roadmap-infrastructure.md`, `docs/roadmap-infrastructure-next-steps.md`
   - Not implemented: no `sqlite3` / DB persistence layers, no `database/` package.

3. **Comprehensive model comparison dashboard**
   - Source: `docs/roadmap-models-next-steps.md`
   - Not implemented: no multi-model comparison UI or Plotly dashboard in current app.

4. **Confidence calibration with Platt scaling (`CalibratedClassifierCV`)**
   - Source: `docs/roadmap-models-next-steps.md`
   - Not implemented: no calibration module or calibrated model path exists.

5. **SHAP feature importance analysis**
   - Source: `docs/roadmap-models-next-steps.md`
   - Not implemented: no SHAP analysis or `models/feature_analysis.py`.

6. **Time-series cross-validation**
   - Source: `docs/roadmap-models-next-steps.md`
   - Not implemented: no `TimeSeriesSplit` training workflow.

<!-- 7. **Live match tracker / live scores tab**
   - Source: `docs/roadmap-features.md`, `docs/roadmap-features-next-steps.md`
   - Not implemented: no live match UI or live score auto-refresh panel in the current Streamlit app. -->

<!-- 8. **Interactive match commentary generator**
   - Source: `docs/roadmap-features-next-steps.md`, `docs/roadmap-quick-wins.md`
   - Not implemented: no natural-language preview/commentary generator in the UI. -->

9. **Export predictions to CSV from Streamlit UI**
   - Source: `docs/roadmap-quick-wins.md`
   - Not implemented: no `st.download_button` for predictions export.

<!-- 10. **Email alerts for high-confidence predictions**
    - Source: `docs/roadmap-features-next-steps.md`
    - Not implemented: no email alerting or SMTP integration. -->

---

## 2. Infrastructure / Operations

1. **Redis caching layer**
   - Source: `docs/roadmap-infrastructure.md`, `docs/roadmap-infrastructure-next-steps.md`
   - Not implemented: no `redis` cache module or caching layer exists.

2. **MLflow model versioning / experiment tracking**
   - Source: `docs/roadmap-infrastructure.md`
   - Not implemented: no `mlflow` integration or model tracking.

3. **API-powered database-backed head-to-head and predictions queries**
   - Source: `docs/roadmap-infrastructure.md`, `docs/roadmap-infrastructure-next-steps.md`
   - Not implemented: no DB query helpers or API endpoints.

---

## 3. Data Enrichment

<!-- 1. **Social media sentiment analysis**
   - Source: `docs/roadmap-data-next-steps.md`
   - Not implemented: no sentiment ingestion or sentiment-features pipeline. -->

2. **Transfer market activity features**
   - Source: `docs/roadmap-data-next-steps.md`
   - Not implemented: no transfer activity fetch or feature engine.

3. **Enhanced historical odds coverage**
   - Source: `docs/roadmap-data-next-steps.md`
   - Not implemented: no Asian handicap, O/U, BTTS, or odds movement enrichment in `prepare_model_data.py`.

4. **Venue-specific stadium features**
   - Source: `docs/roadmap-data-next-steps.md`
   - Not implemented: no stadium capacity / pitch / crowd pressure features.

5. **Line movement snapshot and movement tracker**
   - Source: `docs/roadmap-ev-engine.md`
   - Not implemented: no odds snapshot log or line movement analytics pipeline.

---

## 4. Product / UI Enhancements

<!-- 1. **Match commentary text on upcoming predictions**
   - Source: `docs/roadmap-quick-wins.md`
   - Not implemented: no auto-generated commentary strings for upcoming fixtures. -->

2. **Color-coded confidence levels on prediction tables**
   - Source: `docs/roadmap-quick-wins.md`
   - Not implemented: no dedicated confidence color-coding in the prediction output table.

3. **Top feature importance chart**
   - Source: `docs/roadmap-quick-wins.md`
   - Not implemented: no Plotly feature importance visualization in the app.

4. **Date-range filtering for upcoming fixtures**
   - Source: `docs/roadmap-quick-wins.md`
   - Not implemented: no user-facing date filter for `upcoming_fixtures.csv`.

---

## 5. Betting Market Enhancements

1. **Shots-on-target total market**
   - Source: `docs/roadmap-betting-markets.md`
   - Not implemented: `pages/6_Markets.py` currently includes BTTS, Over/Under, and Correct Score, but not Shots on Target.

2. **Explicit multi-book EV comparison page**
   - Source: `docs/roadmap-ev-engine.md`
   - Partially implemented: the app currently uses best available odds, but a richer per-book EV comparison and best-book selection UI remains unbuilt.

---

## 6. Existing Roadmap Claims That Need Validation

These items are documented as completed in the roadmap files but are not present in the current codebase and should be audited further:

- Prediction performance tracker / validation dashboard
- Comprehensive model comparison dashboard
- SHAP feature analysis
- Time-based cross-validation
- MLflow versioning
- Redis caching
- Email alerting

---

## Suggested Next Focus

1. API + database persistence
2. Model calibration / comparison / interpretability
3. Live match tracking + commentary
4. Data enrichment: sentiment, transfers, advanced odds
5. Quick UI wins: export CSV, confidence coloring, feature chart

This file is intended as the single source of roadmap tasks still outstanding after code review.
