from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from models.governance import (
    ConformalAbstainer,
    cross_market_consistency,
    decide_bet,
    make_snapshot,
    probability_change,
    scenario_mixture,
    settle_market,
)


def test_snapshot_scenarios_and_change_feed() -> None:
    now = datetime.now(timezone.utc)
    first = make_snapshot("f1", {"home": 0.5, "draw": 0.25, "away": 0.25}, "v1", {"a": 1}, prediction_time=now)
    second = make_snapshot("f1", {"home": 0.4, "draw": 0.3, "away": 0.3}, "v1", {"a": 2}, prediction_time=now)
    mixture = scenario_mixture([first, second], [0.75, 0.25])
    assert mixture["home"] == pytest.approx(0.475)
    assert probability_change(first, second)[0]["outcome"] == "home"


def test_conformal_abstention_returns_prediction_sets() -> None:
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 10)
    labels = np.array([0, 1, 2] * 10)
    abstainer = ConformalAbstainer(alpha=0.1).fit(probabilities, labels)
    assert abstainer.should_abstain([0.9, 0.05, 0.05]) is False
    assert abstainer.should_abstain([0.34, 0.33, 0.33]) is True


def test_cross_market_consistency_detects_incoherence() -> None:
    grid = np.zeros((3, 3))
    grid[1, 0] = 0.6
    grid[1, 1] = 0.2
    grid[0, 1] = 0.2
    consistent, reasons = cross_market_consistency({"home": 0.2, "draw": 0.2, "away": 0.6}, grid)
    assert not consistent
    assert reasons


def test_release_policy_keeps_props_disabled() -> None:
    decision = decide_bet("player_prop", 0.6, 0.5, 120, True, True, paper_trade=False)
    assert decision.allowed is False
    assert decision.state == "NO BET"


def test_moneyline_requires_calibration() -> None:
    decision = decide_bet("1x2", 0.6, 0.5, 120, False, False)
    assert decision.allowed is False
    assert "calibration" in " ".join(decision.reasons).lower()


def test_valid_unreleased_edge_is_frozen_as_paper() -> None:
    decision = decide_bet("1x2", 0.58, 0.50, 120, True, False, paper_trade=True)
    assert decision.allowed is True
    assert decision.state == "PAPER"
    assert decision.stake_units == 0


def test_live_staking_requires_release_gate() -> None:
    decision = decide_bet("1x2", 0.58, 0.50, 120, True, False, paper_trade=False)
    assert decision.allowed is False
    assert decision.state == "NO BET"


def test_dnb_and_quarter_line_settlement() -> None:
    push = settle_market("dnb", "home", 110, 1.0, 1, 1)
    assert push.status == "push"
    half = settle_market("asian_handicap", "home", 100, 1.0, 1, 1, line=-0.25)
    assert half.profit_units == pytest.approx(-0.5)
    total = settle_market("totals", "over", 100, 1.0, 2, 1, line=2.75)
    assert total.profit_units == pytest.approx(0.5)
