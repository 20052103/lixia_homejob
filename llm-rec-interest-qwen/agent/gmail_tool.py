# -*- coding: utf-8 -*-
"""
Gmail API helper — OAuth2 authentication + email fetching.

First-time use:
  1. Place credentials.json (downloaded from Google Cloud Console) in agent/
  2. Run the agent and ask about emails — a browser window will open for OAuth consent.
  3. The token is saved to gmail_token.json for all future runs.

Scopes:  readonly access to Gmail messages.
"""
from __future__ import annotations

import base64
import email as _email_stdlib
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

# Google API imports are deferred so the rest of the agent works even if the
# google packages are not yet installed.

_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

_DEFAULT_CREDENTIALS = os.path.join(os.path.dirname(__file__), "credentials.json")
_DEFAULT_TOKEN = os.path.join(os.path.dirname(__file__), "gmail_token.json")


@dataclass
class GmailMessage:
    msg_id: str
    subject: str
    sender: str
    date: str
    snippet: str
    body: str = ""
    labels: List[str] = field(default_factory=list)
    important: bool = True        # False = filtered out as ad/promo
    filter_reason: str = ""       # why it was filtered

    def to_text(self, include_body: bool = True, body_max_chars: int = 400) -> str:
        lines = [
            f"Subject : {self.subject}",
            f"From    : {self.sender}",
            f"Date    : {self.date}",
            f"Snippet : {self.snippet}",
        ]
        if include_body and self.body:
            body_preview = self.body[:body_max_chars]
            if len(self.body) > body_max_chars:
                body_preview += "\n...[truncated]"
            lines.append(f"Body    :\n{body_preview}")
        return "\n".join(lines)


# ============================================================
# Importance heuristics
# ============================================================

# Gmail system label IDs that indicate automated/low-priority mail
_UNIMPORTANT_LABELS = {
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
    "SPAM",
}

_SPAM_SENDER_KEYWORDS = [
    "noreply", "no-reply", "do-not-reply", "donotreply",
    "newsletter", "mailer", "notification", "automated",
    "marketing", "promotions", "bounce", "alerts@",
]

_SPAM_SUBJECT_KEYWORDS = [
    "unsubscribe", "newsletter", "% off", "%off", "deal",
    "sale", "offer", "promo", "coupon", "discount",
    "click here", "verify your", "confirm your email",
    "winning", "you won", "limited time",
    "订阅", "优惠", "折扣", "广告", "促销",
]

_SPAM_SNIPPET_KEYWORDS = [
    "unsubscribe", "view in browser", "email preferences",
    "opt out", "manage preferences",
]


def _is_likely_unimportant(msg: "GmailMessage") -> tuple[bool, str]:
    """
    Return (is_unimportant, reason) using label + heuristic checks.
    Gmail's own category labels are the most reliable signal.
    """
    # 1. Gmail category labels (most reliable)
    bad_labels = _UNIMPORTANT_LABELS.intersection(set(msg.labels))
    if bad_labels:
        return True, f"Gmail label: {', '.join(bad_labels)}"

    sender_lower = msg.sender.lower()
    subject_lower = msg.subject.lower()
    snippet_lower = msg.snippet.lower()

    # 2. Sender heuristics
    for kw in _SPAM_SENDER_KEYWORDS:
        if kw in sender_lower:
            return True, f"sender contains '{kw}'"

    # 3. Subject heuristics
    for kw in _SPAM_SUBJECT_KEYWORDS:
        if kw in subject_lower:
            return True, f"subject contains '{kw}'"

    # 4. Snippet heuristics
    for kw in _SPAM_SNIPPET_KEYWORDS:
        if kw in snippet_lower:
            return True, f"snippet contains '{kw}'"

    return False, ""


def _get_credentials(credentials_path: str, token_path: str):
    """Load or refresh OAuth2 credentials, triggering browser flow if needed."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"credentials.json not found at: {credentials_path}\n"
                    "Download it from Google Cloud Console > APIs & Services > Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, _SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return creds


def _decode_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime_type == "text/plain" and body_data:
        raw = base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
        return raw.strip()

    if mime_type == "text/html" and body_data:
        raw = base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
        # Strip HTML tags for a plain-text approximation
        clean = re.sub(r"<[^>]+>", " ", raw)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    # Multipart: recurse into parts
    for part in payload.get("parts", []):
        result = _decode_body(part)
        if result:
            return result

    return ""


def _parse_header(headers: list, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def fetch_emails(
    query: str = "is:unread",
    max_results: int = 10,
    include_body: bool = True,
    credentials_path: str = _DEFAULT_CREDENTIALS,
    token_path: str = _DEFAULT_TOKEN,
) -> List[GmailMessage]:
    """
    Fetch emails from Gmail matching the given query.

    Args:
        query:            Gmail search query (e.g. 'is:unread', 'newer_than:1d', 'from:boss@example.com')
        max_results:      Maximum number of messages to return (capped at 50).
        include_body:     Whether to fetch and decode the full message body.
        credentials_path: Path to credentials.json from Google Cloud Console.
        token_path:       Path where the OAuth token will be cached.

    Returns:
        List of GmailMessage dataclass instances.
    """
    from googleapiclient.discovery import build

    max_results = min(max_results, 50)
    creds = _get_credentials(credentials_path, token_path)
    service = build("gmail", "v1", credentials=creds)

    # List matching message IDs
    list_resp = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )

    messages_meta = list_resp.get("messages", [])
    if not messages_meta:
        return []

    results: List[GmailMessage] = []

    for meta in messages_meta:
        msg_id = meta["id"]

        # Phase 1: fetch metadata only (subject, sender, snippet, labels) — very cheap
        msg_meta = (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="metadata",
                 metadataHeaders=["Subject", "From", "Date"])
            .execute()
        )

        headers = msg_meta.get("payload", {}).get("headers", [])
        subject = _parse_header(headers, "Subject") or "(no subject)"
        sender = _parse_header(headers, "From") or "(unknown)"
        date = _parse_header(headers, "Date") or ""
        snippet = msg_meta.get("snippet", "")
        labels = msg_meta.get("labelIds", [])

        candidate = GmailMessage(
            msg_id=msg_id,
            subject=subject,
            sender=sender,
            date=date,
            snippet=snippet,
            labels=labels,
        )

        # Phase 2: importance check — skip body fetch for ads/promos
        unimportant, reason = _is_likely_unimportant(candidate)
        if unimportant:
            candidate.important = False
            candidate.filter_reason = reason
            results.append(candidate)
            continue

        # Phase 3: important email — fetch body only if requested
        if include_body:
            msg_full = (
                service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
            candidate.body = _decode_body(msg_full.get("payload", {}))

        results.append(candidate)

    return results


def fetch_emails_as_text(
    query: str = "is:unread",
    max_results: int = 10,
    include_body: bool = True,
    credentials_path: str = _DEFAULT_CREDENTIALS,
    token_path: str = _DEFAULT_TOKEN,
    max_total_chars: int = 4000,
) -> str:
    """
    Two-phase fetch: scan all emails for importance first (metadata only),
    then retrieve body only for important ones. Returns a structured summary.
    """
    msgs = fetch_emails(
        query=query,
        max_results=max_results,
        include_body=include_body,
        credentials_path=credentials_path,
        token_path=token_path,
    )

    if not msgs:
        return f"No emails found matching query: {query!r}"

    important = [m for m in msgs if m.important]
    filtered = [m for m in msgs if not m.important]

    lines = [
        f"=== Scanned {len(msgs)} email(s) | Important: {len(important)} | Filtered (ads/promo): {len(filtered)} ===",
        "",
    ]

    if important:
        lines.append("## Important Emails\n")
        total_chars = sum(len(l) for l in lines)
        for i, m in enumerate(important, 1):
            entry = f"[{i}] {m.to_text(include_body=include_body)}\n"
            if total_chars + len(entry) > max_total_chars:
                lines.append(f"[...remaining {len(important)-i+1} important email(s) omitted to fit context]")
                break
            lines.append(entry)
            total_chars += len(entry)
    else:
        lines.append("No important emails found.")

    if filtered:
        lines.append("\n## Filtered Out (ads / promotions / automated)\n")
        for m in filtered:
            lines.append(f"- {m.subject!r}  from {m.sender}  [{m.filter_reason}]")

    return "\n".join(lines)
