"""Point-in-time feed quality checks used by refresh, evaluation, and the UI."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import pandas as pd


def audit_feed_frame(frame: pd.DataFrame, name: str, now: datetime | None = None) -> dict[str, object]:
    """Return an honest quality summary without fabricating missing provider data."""
    now = now or datetime.now(timezone.utc)
    result: dict[str, object] = {"feed": name, "rows": int(len(frame)), "status": "missing"}
    if frame.empty:
        return result
    if "available_at" not in frame.columns:
        return {**result, "status": "invalid", "reason": "available_at column is required"}
    available = pd.to_datetime(frame["available_at"], errors="coerce", utc=True)
    invalid = int(available.isna().sum())
    latest = available.max()
    age_hours = None if pd.isna(latest) else round((pd.Timestamp(now) - latest).total_seconds() / 3600, 2)
    status = "ready" if invalid == 0 else "invalid"
    if status == "ready" and age_hours is not None and age_hours > 168:
        status = "stale"
    return {**result, "status": status, "available_at_invalid": invalid, "duplicate_rows": int(frame.duplicated().sum()), "latest_available_at": None if pd.isna(latest) else latest.isoformat(), "age_hours": age_hours}


def audit_optional_sources(sources: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    feeds = {name: audit_feed_frame(frame, name) for name, frame in sorted(sources.items())}
    statuses = [str(item["status"]) for item in feeds.values()]
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "feeds": feeds, "summary": {status: statuses.count(status) for status in ("ready", "stale", "invalid", "missing")}}


def audit_market_odds(frame: pd.DataFrame) -> dict[str, object]:
    """Assess whether archived odds can support market-relative and CLV evidence."""
    required = {"fixture_id", "snapshot_at", "book", "market", "selection", "available_at"}
    base = audit_feed_frame(frame, "multi_book_odds")
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        return {**base, "status": "invalid", "missing_columns": missing_columns}
    if frame.empty:
        return {**base, "closing_rows": 0, "books": 0, "markets": 0, "fixtures": 0}
    close = pd.to_numeric(frame.get("is_closing", 0), errors="coerce").fillna(0).astype(bool)
    return {**base, "closing_rows": int(close.sum()), "books": int(frame["book"].nunique()), "markets": int(frame["market"].nunique()), "fixtures": int(frame["fixture_id"].nunique())}
