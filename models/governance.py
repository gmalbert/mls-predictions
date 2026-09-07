"""Prediction snapshots, uncertainty, market consistency, staking, and settlement."""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence
from pathlib import Path

import numpy as np


DISABLED_MARKETS = {"player_prop", "first_scorer", "correct_score", "parlay"}


def repository_code_version() -> str:
    """Return the deployment SHA or a deterministic digest of repository Python."""
    deployment_sha = os.getenv("GITHUB_SHA", "").strip()
    if deployment_sha:
        return deployment_sha
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for file_path in sorted(root.rglob("*.py"), key=lambda path: path.as_posix()):
        if any(part.startswith(".") or part in {"venv", "node_modules"} for part in file_path.relative_to(root).parts):
            continue
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(file_path.read_bytes())
    return f"source-{digest.hexdigest()[:16]}"


@dataclass(frozen=True)
class Snapshot:
    fixture_id: str
    prediction_time: datetime
    lineup_scenario: str
    probabilities: dict[str, float]
    data_manifest: str
    model_version: str
    uncertainty: float = 0.0
    abstain: bool = False
    explanations: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["prediction_time"] = self.prediction_time.astimezone(timezone.utc).isoformat()
        record["explanations"] = list(self.explanations)
        return record


@dataclass(frozen=True)
class BetDecision:
    allowed: bool
    state: str
    market: str
    edge: float
    stake_units: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Settlement:
    status: str
    profit_units: float
    returned_units: float
    components: tuple[str, ...] = field(default_factory=tuple)


def data_manifest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_snapshot(
    fixture_id: str,
    probabilities: Mapping[str, float],
    model_version: str,
    manifest_payload: object,
    lineup_scenario: str = "projected",
    uncertainty: float = 0.0,
    abstain: bool = False,
    explanations: Sequence[str] = (),
    prediction_time: datetime | None = None,
) -> Snapshot:
    probs = {key: float(value) for key, value in probabilities.items()}
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Snapshot probabilities must have positive mass.")
    probs = {key: value / total for key, value in probs.items()}
    return Snapshot(
        fixture_id=fixture_id,
        prediction_time=prediction_time or datetime.now(timezone.utc),
        lineup_scenario=lineup_scenario,
        probabilities=probs,
        data_manifest=data_manifest(manifest_payload),
        model_version=model_version,
        uncertainty=float(uncertainty),
        abstain=bool(abstain),
        explanations=tuple(explanations),
    )


def scenario_mixture(snapshots: Sequence[Snapshot], weights: Sequence[float]) -> dict[str, float]:
    if not snapshots or len(snapshots) != len(weights):
        raise ValueError("Scenarios and weights must be non-empty and aligned.")
    normalized = np.asarray(weights, dtype=float)
    if (normalized < 0).any() or normalized.sum() <= 0:
        raise ValueError("Scenario weights must be non-negative with positive mass.")
    normalized /= normalized.sum()
    labels = sorted(set().union(*(snapshot.probabilities for snapshot in snapshots)))
    return {
        label: float(sum(weight * snapshot.probabilities.get(label, 0.0) for snapshot, weight in zip(snapshots, normalized)))
        for label in labels
    }


def probability_change(previous: Snapshot, current: Snapshot) -> list[dict[str, object]]:
    labels = sorted(set(previous.probabilities).union(current.probabilities))
    changes = [
        {
            "outcome": label,
            "previous": previous.probabilities.get(label, 0.0),
            "current": current.probabilities.get(label, 0.0),
            "change_pp": 100.0 * (current.probabilities.get(label, 0.0) - previous.probabilities.get(label, 0.0)),
        }
        for label in labels
    ]
    return sorted(changes, key=lambda item: abs(float(item["change_pp"])), reverse=True)


class ConformalAbstainer:
    """Split-conformal prediction sets for a visible uncertainty/no-bet state."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one.")
        self.alpha = alpha
        self.quantile = 1.0

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "ConformalAbstainer":
        probs = np.asarray(probabilities, dtype=float)
        y = np.asarray(labels, dtype=int)
        if len(y) == 0:
            return self
        scores = 1.0 - probs[np.arange(len(y)), y]
        level = min(math.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores), 1.0)
        self.quantile = float(np.quantile(scores, level, method="higher"))
        return self

    def prediction_set(self, probabilities: Sequence[float]) -> tuple[int, ...]:
        probs = np.asarray(probabilities, dtype=float)
        return tuple(int(index) for index in np.flatnonzero(1.0 - probs <= self.quantile))

    def should_abstain(self, probabilities: Sequence[float]) -> bool:
        return len(self.prediction_set(probabilities)) != 1


def score_distribution_markets(score_grid: np.ndarray, totals_line: float = 2.5) -> dict[str, float]:
    grid = np.asarray(score_grid, dtype=float)
    if grid.ndim != 2 or grid.sum() <= 0:
        raise ValueError("A non-empty two-dimensional score distribution is required.")
    grid = grid / grid.sum()
    home_scores, away_scores = np.indices(grid.shape)
    return {
        "home": float(grid[home_scores > away_scores].sum()),
        "draw": float(grid[home_scores == away_scores].sum()),
        "away": float(grid[home_scores < away_scores].sum()),
        "over": float(grid[(home_scores + away_scores) > totals_line].sum()),
        "under": float(grid[(home_scores + away_scores) < totals_line].sum()),
        "btts_yes": float(grid[(home_scores > 0) & (away_scores > 0)].sum()),
    }


def cross_market_consistency(
    outcome_probabilities: Mapping[str, float],
    score_grid: np.ndarray,
    tolerance: float = 0.04,
) -> tuple[bool, list[str]]:
    derived = score_distribution_markets(score_grid)
    reasons: list[str] = []
    for label in ("home", "draw", "away"):
        if label in outcome_probabilities and abs(float(outcome_probabilities[label]) - derived[label]) > tolerance:
            reasons.append(
                f"{label} probability differs from the joint score model by "
                f"{abs(float(outcome_probabilities[label]) - derived[label]):.1%}."
            )
    return not reasons, reasons


def american_to_decimal(odds: float) -> float:
    value = float(odds)
    if value >= 100:
        return 1.0 + value / 100.0
    if value <= -100:
        return 1.0 + 100.0 / abs(value)
    if value > 1.0:
        return value
    raise ValueError("Odds must be American (absolute value >=100) or decimal (>1).")


def fractional_kelly(probability: float, odds: float, fraction: float = 0.25, cap: float = 0.5) -> float:
    decimal = american_to_decimal(odds)
    b = decimal - 1.0
    full = (b * probability - (1.0 - probability)) / b
    return float(np.clip(full * fraction, 0.0, cap))


def decide_bet(
    market: str,
    model_probability: float,
    de_vigged_market_probability: float,
    odds: float,
    calibrated: bool,
    release_gate_passed: bool,
    conformal_abstain: bool = False,
    consistent: bool = True,
    paper_trade: bool = True,
    kelly_approved: bool = False,
) -> BetDecision:
    normalized_market = market.lower().replace(" ", "_")
    reasons: list[str] = []
    edge = float(model_probability - de_vigged_market_probability)
    if normalized_market in DISABLED_MARKETS:
        reasons.append("This market is disabled because the repository does not have adequate outcome data.")
    if normalized_market in {"1x2", "moneyline"} and not calibrated:
        reasons.append("Moneyline remains no-bet until calibration is measured.")
    if conformal_abstain:
        reasons.append("The conformal prediction set is ambiguous.")
    if not consistent:
        reasons.append("Cross-market probabilities are incoherent.")
    if edge <= 0.03:
        reasons.append("Model edge does not exceed the 3% value threshold.")
    if reasons:
        return BetDecision(False, "NO BET", normalized_market, edge, 0.0, tuple(reasons))
    if paper_trade:
        explanation = (
            "Paper-trade selection; no capital is authorized."
            if release_gate_passed
            else "Release gate pending; selection is frozen for paper evidence only."
        )
        return BetDecision(True, "PAPER", normalized_market, edge, 0.0, (explanation,))
    if not release_gate_passed:
        return BetDecision(
            False,
            "NO BET",
            normalized_market,
            edge,
            0.0,
            ("The release gate has not passed; only paper trading is allowed.",),
        )
    # Release starts at a flat 0.25 units. Kelly requires a second explicit approval
    # after stable calibration and CLV, and remains capped at half a unit.
    stake = fractional_kelly(model_probability, odds, fraction=0.25, cap=0.5) if kelly_approved else 0.25
    return BetDecision(True, "RELEASED", normalized_market, edge, stake, ())


def _single_settlement(value: float, decimal: float, stake: float) -> Settlement:
    if value > 1e-9:
        profit = stake * (decimal - 1.0)
        return Settlement("win", profit, stake + profit)
    if value < -1e-9:
        return Settlement("loss", -stake, 0.0)
    return Settlement("push", 0.0, stake)


def _quarter_lines(line: float) -> tuple[float, ...]:
    quarter = round(line * 4) / 4
    if abs(quarter * 2 - round(quarter * 2)) < 1e-9:
        return (quarter,)
    lower = math.floor(quarter * 2) / 2
    return (lower, lower + 0.5)


def settle_market(
    market: str,
    selection: str,
    odds: float,
    stake: float,
    home_goals: int | None,
    away_goals: int | None,
    line: float | None = None,
    void: bool = False,
) -> Settlement:
    if void or home_goals is None or away_goals is None:
        return Settlement("void", 0.0, float(stake), ("void",))
    decimal = american_to_decimal(odds)
    normalized = market.lower().replace(" ", "_")
    side = selection.lower()
    goal_diff = float(home_goals - away_goals)
    total = float(home_goals + away_goals)

    if normalized in {"1x2", "moneyline"}:
        result = "home" if goal_diff > 0 else ("away" if goal_diff < 0 else "draw")
        return _single_settlement(1.0 if side == result else -1.0, decimal, stake)
    if normalized in {"draw_no_bet", "dnb"}:
        if goal_diff == 0:
            return Settlement("push", 0.0, stake)
        value = goal_diff if side == "home" else -goal_diff
        return _single_settlement(value, decimal, stake)
    if normalized in {"asian_handicap", "handicap"}:
        if line is None or side not in {"home", "away"}:
            raise ValueError("Asian handicap requires a home/away selection and line.")
        components = []
        settlements = []
        for component in _quarter_lines(float(line)):
            value = (goal_diff if side == "home" else -goal_diff) + component
            settlements.append(_single_settlement(value, decimal, stake / len(_quarter_lines(float(line)))))
            components.append(f"{side} {component:+g}")
        profit = sum(result.profit_units for result in settlements)
        returned = sum(result.returned_units for result in settlements)
        statuses = {result.status for result in settlements}
        status = statuses.pop() if len(statuses) == 1 else "/".join(result.status for result in settlements)
        return Settlement(status, profit, returned, tuple(components))
    if normalized in {"total", "totals", "team_total"}:
        if line is None:
            raise ValueError("Totals settlement requires a line.")
        observed = total
        if normalized == "team_total":
            observed = float(home_goals if side.startswith("home") else away_goals)
            side = "over" if side.endswith("over") else "under"
        settlements = []
        components = []
        for component in _quarter_lines(float(line)):
            value = observed - component if side == "over" else component - observed
            settlements.append(_single_settlement(value, decimal, stake / len(_quarter_lines(float(line)))))
            components.append(f"{side} {component:g}")
        profit = sum(result.profit_units for result in settlements)
        returned = sum(result.returned_units for result in settlements)
        statuses = {result.status for result in settlements}
        status = statuses.pop() if len(statuses) == 1 else "/".join(result.status for result in settlements)
        return Settlement(status, profit, returned, tuple(components))
    if normalized == "btts":
        occurred = home_goals > 0 and away_goals > 0
        wins = occurred if side in {"yes", "btts_yes"} else not occurred
        return _single_settlement(1.0 if wins else -1.0, decimal, stake)
    raise ValueError(f"Unsupported market for settlement: {market}")


def new_selection_id() -> str:
    return str(uuid.uuid4())
