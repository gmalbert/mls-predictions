"""Render and optionally send the governed Friday MLS matchday brief."""
from __future__ import annotations

import os
import smtplib
import ssl
from datetime import date
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = ROOT / "data_files" / "picks_today.csv"
PREVIEW_PATH = ROOT / "data_files" / "matchday_email_preview.html"


def render_brief(picks: pd.DataFrame, target: date | None = None) -> str:
    target = target or date.today()
    if picks.empty:
        body = "<p>No MLS selections are available for this slate.</p>"
    else:
        rows = []
        for _, pick in picks.iterrows():
            reasons = str(pick.get("NoBetReasons", ""))
            rows.append(
                "<tr>"
                f"<td>{pick.get('AwayTeam', '')} at {pick.get('HomeTeam', '')}</td>"
                f"<td>{pick.get('Bet', '')}</td><td>{pick.get('State', 'NO BET')}</td>"
                f"<td>{pick.get('Model', 0)}%</td><td>{pick.get('Edge', 0)}%</td>"
                f"<td>{reasons}</td>"
                "</tr>"
            )
        body = (
            "<table><thead><tr><th>Match</th><th>Lean</th><th>State</th>"
            "<th>Model</th><th>Edge</th><th>Notes</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>body{{font-family:Arial,sans-serif;color:#172b4d}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #dfe1e6;padding:8px;text-align:left}}th{{background:#f4f5f7}}
.gate{{padding:10px;background:#fff3cd;border-left:4px solid #ffab00}}</style></head>
<body><h1>MLS Matchday Brief — {target:%B %d, %Y}</h1>
<p class="gate">Selections marked PAPER or NO BET are research only. Only RELEASED rows authorize a stake.</p>
{body}<p>Generated from a frozen, timestamped model snapshot.</p></body></html>"""


def main() -> None:
    load_dotenv(ROOT / ".env")
    try:
        picks = pd.read_csv(PICKS_PATH) if PICKS_PATH.exists() and PICKS_PATH.stat().st_size else pd.DataFrame()
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        picks = pd.DataFrame()
    html = render_brief(picks)
    PREVIEW_PATH.write_text(html, encoding="utf-8")
    recipients = [value.strip() for value in os.getenv("MATCHDAY_EMAIL_TO", "").split(",") if value.strip()]
    host = os.getenv("SMTP_HOST", "").strip()
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("MATCHDAY_EMAIL_FROM", username).strip()
    if not recipients or not host or not sender or not password:
        print(f"Email configuration incomplete; preview written to {PREVIEW_PATH}")
        return
    message = EmailMessage()
    message["Subject"] = f"MLS Matchday Brief — {date.today():%b %d}"
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content("View this message in an HTML-capable email client.")
    message.add_alternative(html, subtype="html")
    port = int(os.getenv("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as smtp:
        smtp.login(username, password)
        smtp.send_message(message)
    print(f"Sent matchday brief to {len(recipients)} recipient(s).")


if __name__ == "__main__":
    main()
