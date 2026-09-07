from __future__ import annotations

import pandas as pd

from models.data_quality import audit_feed_frame, audit_market_odds


def test_feed_audit_requires_available_at() -> None:
    assert audit_feed_frame(pd.DataFrame([{"team": "LAFC"}]), "roster")["status"] == "invalid"


def test_market_audit_reports_closing_coverage() -> None:
    frame = pd.DataFrame([{"fixture_id": "f1", "snapshot_at": "2026-08-01T12:00:00Z", "book": "book", "market": "h2h", "selection": "LAFC", "available_at": "2026-08-01T12:00:00Z", "is_closing": True}])
    result = audit_market_odds(frame)
    assert result["status"] in {"ready", "stale"}
    assert result["closing_rows"] == 1
