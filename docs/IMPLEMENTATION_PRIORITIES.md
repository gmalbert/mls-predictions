# Implementation priorities

> **Created:** 2026-08-24
>
> **Purpose:** Authoritative execution order after reconciling the project roadmaps
> with the repository. This plan distinguishes code completion from data readiness
> and release evidence. It replaces the historical ordering in `NEXT_FEATURES.md`
> and the feature sequencing in the 6- and 12-month roadmaps.

## Current state

- The governed model, point-in-time feature framework, backtest, snapshots,
  settlement ledger, automation, and Frontier Analytics page are implemented.
- Optional context and market feeds are supported but are not evidenced here as
  configured, complete, or historically backfillable.
- The release gate is correctly fail-closed. Do not enable real-money exports,
  staking, player props, correct scores, first scorer, or parlays.
- The current 701-match artifact does not yet clear the model-quality hurdle: its
  governed logistic model is slightly worse than rolling Elo on aggregate log loss
  and Brier score. Treat Elo as the deployment baseline unless a locked evaluation
  reverses that result.

## Implementation record (2026-08-24)

| Priority | Implemented control | Current evidence/state |
|---|---|---|
| P0 | Dependency-backed test/compile verification, reproducible backtest, and deployment-candidate comparison | `pytest` passed (36 tests); `py_compile` passed; the current candidate is rejected in favor of Elo. |
| P1 | `models/data_quality.py`, `scripts/validate_operational_readiness.py`, and refresh-time quality artifacts | 13 optional feeds are missing, goals-added is stale, and market odds are absent—reported explicitly rather than inferred. |
| P2 | Fail-closed model-selection rule and release-gate evidence reporting | Model picks are disabled unless the model beats Elo on both aggregate log loss and Brier; release gate remains closed. |
| P3 | Dashboard readiness and model-selection disclosures | Frontier Analytics exposes release, candidate, and feed-readiness states; unavailable data remains visible. The 10,000-path simulation is now explicitly run on demand to preserve initial page responsiveness. |

The remaining work is operational evidence collection, not unimplemented code. The
release gate will remain closed until its stated historical requirements are met.

## Verification record

Completed in the project environment on 2026-08-24:

- `pytest -q` — 36 passed.
- `python -m py_compile` across every repository Python file — passed.
- `scripts/run_backtest.py` and `scripts/validate_operational_readiness.py` —
  completed and wrote current artifacts.
- Browser smoke test against Streamlit bound to `127.0.0.1` — navigation loaded
  with no Playwright console errors.

The readiness artifact remains intentionally negative until providers are
configured: market odds are absent; 13 optional context feeds are missing; and the
only populated goals-added feed is stale. These are operational prerequisites, not
conditions that code can safely invent or mark complete.

## Priority order

| Priority | Deliverable | Why now | Definition of done |
|---|---|---|---|
| P0 | Reproducible local validation | Establish a trustworthy baseline before adding features. | Install the pinned environment; run `pytest -q`, `py_compile`, data preparation, and `scripts/run_backtest.py`; retain command output and report hashes. |
| P0 | Backtest-artifact audit and baseline decision | Confirm the report reflects only prior-to-kickoff features and correct chronological splits; the current governed model trails Elo on aggregate log loss/Brier. | Versioned report contains train/calibration/holdout dates, feature manifest, sample counts, baselines, calibration, and phase/segment results; no leakage exceptions; deploy only the locked candidate that beats Elo on the predeclared primary metric. |
| P0 | Release-evidence instrumentation | The gate cannot open without frozen selections, market snapshots, settlement, and CLV. | Every eligible paper pick stores fixture ID, selection timestamp, probabilities, manifest/model version, book/line/limit, closing price, result, and settlement. |
| P1 | Historical and live odds coverage | Market-relative skill and CLV are the critical unproven claims. | Multi-book 1X2, totals, and handicap feeds meet documented coverage/quality thresholds; de-vigging and close identification are checked on samples. |
| P1 | Roster/lineup/transaction source | DP/U22/TAM, international duty, churn, and lineup scenarios need actual point-in-time facts. | Select authoritative feeds, document field mapping/licensing, archive `available_at`, backfill safely where possible, and measure missingness by season. |
| P1 | Event, keeper, weather, referee, attendance sources | These features are currently neutral/missing when feeds are absent. | Each source has a contract, freshness and coverage monitor, and a leakage test; retain only features that improve rolling holdouts. |
| P1 | Automation hardening | Scheduled jobs must produce auditable artifacts without silently succeeding on empty inputs. | Dry-run each Monday/Friday/off-season job; fail loudly on required-source failures, publish concise run summaries, and protect secrets. |
| P2 | Model selection and calibration | Use evidence to decide whether enriched features beat simple baselines. | Compare locked candidates against Elo, home-field, market, and logistic baselines by log loss, Brier/RPS, calibration, and market-relative metrics; preregister selection rule. |
| P2 | Segment and drift monitoring | MLS-specific claims should be demonstrated, not assumed. | Report outcomes by travel, altitude, surface, rest, conference, roster churn, manager tenure, phase, and horizon; alert on material data/metric drift. |
| P2 | Product truthfulness pass | UI needs to show availability and no-bet states clearly. | Every card exposes data freshness/missingness, scenario uncertainty, evidence status, and explanation; unavailable markets are visibly disabled. |
| P3 | Research-only feature maturation | Props and richer analytics should follow reliable inputs and model evidence. | Event/prop history supports provider-specific backtests and settlement; obtain explicit release review before enabling any market. |

## Milestones and decision gates

1. **M1 — Validated build:** **Complete.** P0 local validation and artifact audit are complete. No model or UI expansion should bypass the candidate-selection rule.
2. **M2 — Data-qualified paper operation:** P1 market plus core roster feeds have reliable timestamped coverage; paper selections are frozen and settled weekly.
3. **M3 — Evidence-qualified model:** P2 demonstrates a stable, market-relative improvement and acceptable calibration on untouched periods.
4. **M4 — Release review:** A completed untouched season, at least 300 settled frozen selections, positive median CLV, complete ledger, and all implemented release-gate checks pass. Only then consider flat 0.25-unit releases.

## Explicitly deprioritized

- New model features whose inputs are unavailable or not point-in-time safe.
- Real-money staking, Kelly sizing, and external bet export before M4.
- Player props, first scorer, correct score, and parlays before provider-specific
  historical evidence and a separate approval.
- Cosmetic dashboard expansion that obscures current uncertainty or data gaps.

## First implementation sprint

1. Provision the project Python environment and run the P0 commands.
2. Inspect `data_files/backtests/latest_report.json` and the SQLite snapshot/ledger
   tables; document the exact current gate failures.
3. Choose and configure one legal odds provider and one authoritative roster/lineup
   provider, then add coverage/freshness checks.
4. Run a dry Monday-to-Friday paper cycle, settle it, and review the generated
   ledger/report before expanding feature scope.
