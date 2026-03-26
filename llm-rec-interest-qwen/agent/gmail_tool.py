# -*- coding: utf-8 -*-
"""
Gmail API helper — OAuth2 authentication + email fetching + sending.

First-time use:
  1. Place credentials.json (downloaded from Google Cloud Console) in agent/
  2. Run the agent and ask about emails — a browser window will open for OAuth consent.
  3. The token is saved to gmail_token.json for all future runs.

Scopes:  gmail.readonly (fetch) + gmail.send (send)
VIP contacts are loaded from contacts.json and always checked separately.

NOTE: If you had a previous gmail_token.json with readonly scope only,
delete it so the OAuth flow re-runs with the new send scope.
"""
from __future__ import annotations

import base64
import email as _email_stdlib
import os
import re
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from typing import List, Optional

try:
    from .contacts import Contact, build_vip_gmail_query, is_from_vip, load_vip_contacts
except ImportError:
    from contacts import Contact, build_vip_gmail_query, is_from_vip, load_vip_contacts

# Google API imports are deferred so the rest of the agent works even if the
# google packages are not yet installed.

# Both readonly and send scopes — delete gmail_token.json if upgrading from readonly-only
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

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
    thread_id: str = ""            # Gmail thread ID (may differ from msg_id)
    labels: List[str] = field(default_factory=list)
    important: bool = True        # False = filtered out as ad/promo
    filter_reason: str = ""       # why it was filtered
    vip_contact: Optional[Contact] = None  # set if sender is a VIP contact

    def to_text(self, include_body: bool = True, body_max_chars: int = 250) -> str:
        vip_tag = f" ⭐ VIP: {self.vip_contact.name}" if self.vip_contact else ""
        lines = [
            f"Subject : {self.subject}{vip_tag}",
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

    # Always restrict to inbox — never search Sent, Drafts, Spam, etc.
    # But don't override if the caller explicitly specified an `in:` label.
    if not re.search(r'\bin:', query, re.IGNORECASE):
        query = f"in:inbox {query}"

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
            thread_id=msg_meta.get("threadId", msg_id),
            subject=subject,
            sender=sender,
            date=date,
            snippet=snippet,
            labels=labels,
        )

        # Phase 2: importance check — skip body fetch for ads/promos
        # VIP contacts always bypass the spam filter
        vip = is_from_vip(sender, load_vip_contacts())
        if vip:
            candidate.vip_contact = vip
        else:
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
    max_total_chars: int = 2500,
) -> tuple:
    """
    Two-phase fetch: scan all emails for importance first (metadata only),
    then retrieve body only for important ones. Returns a structured summary.

    VIP contacts (from contacts.json) are always checked separately and
    shown at the top of the output regardless of the main query.

    Returns:
        (text: str, email_cache: dict[str, GmailMessage])
        email_cache maps display IDs like "E1", "E2" to GmailMessage objects.
    """
    vip_contacts = load_vip_contacts()
    vip_section_lines: List[str] = []
    email_cache: dict = {}
    counter = [0]  # mutable counter for assigning E1, E2, ...

    def assign_id(msg: GmailMessage) -> str:
        counter[0] += 1
        eid = f"E{counter[0]}"
        email_cache[eid] = msg
        return eid

    # Always run a dedicated VIP query so their emails are never missed
    vip_query = build_vip_gmail_query(vip_contacts)
    if vip_query:
        vip_msgs = fetch_emails(
            query=vip_query + " newer_than:7d",
            max_results=10,
            include_body=include_body,
            credentials_path=credentials_path,
            token_path=token_path,
        )
        if vip_msgs:
            vip_section_lines.append("## ⭐ VIP Contact Emails\n")
            for m in vip_msgs:
                eid = assign_id(m)
                name = m.vip_contact.name if m.vip_contact else m.sender
                vip_section_lines.append(f"[{eid}] [{name}] {m.to_text(include_body=include_body)}\n")
        else:
            vip_section_lines.append("## ⭐ VIP Contact Emails\n")
            vip_section_lines.append("No recent emails from VIP contacts.\n")

    # Main inbox query
    msgs = fetch_emails(
        query=query,
        max_results=max_results,
        include_body=include_body,
        credentials_path=credentials_path,
        token_path=token_path,
    )

    important = [m for m in msgs if m.important]
    filtered = [m for m in msgs if not m.important]

    lines = [
        f"=== Scanned {len(msgs)} email(s) | Important: {len(important)} | Filtered (ads/promo): {len(filtered)} ===",
        "",
    ]

    # VIP section goes first
    lines.extend(vip_section_lines)

    if important:
        lines.append("## Important Emails\n")
        total_chars = sum(len(l) for l in lines)
        for m in important:
            eid = assign_id(m)
            entry = f"[{eid}] {m.to_text(include_body=include_body)}\n"
            if total_chars + len(entry) > max_total_chars:
                lines.append(f"[...remaining emails omitted to fit context]")
                break
            lines.append(entry)
            total_chars += len(entry)
    else:
        lines.append("## Important Emails\n")
        lines.append("No important emails found.")

    if filtered:
        lines.append("\n## Filtered Out (ads / promotions / automated)\n")
        for m in filtered:
            lines.append(f"- {m.subject!r}  from {m.sender}  [{m.filter_reason}]")

    lines.append("\n💡 To reply to an email, say: 回复 E1 / reply to E2")

    return "\n".join(lines), email_cache


# ============================================================
# Send email
# ============================================================

def resolve_recipient(to: str) -> str:
    """
    Resolve recipient to a verified email address.
    - "Name <email>" format: first try to match Name against VIP contacts (LLM hallucinates addresses)
    - Plain email (has @, no name part): return as-is
    - Plain name (no @): look up in VIP contacts
    """
    to = to.strip()

    # Parse "Name <email@...>" format
    name_angle_re = re.compile(r"^(.+?)\s*<([^>]+)>$")
    m = name_angle_re.match(to)
    if m:
        name_part = m.group(1).strip()
        email_part = m.group(2).strip()
        # Try to match name against VIP contacts first (LLM may hallucinate the address)
        contacts = load_vip_contacts()
        name_lower = name_part.lower()
        for c in contacts:
            if name_lower in c.name.lower() or c.name.lower() in name_lower:
                return c.email
        # No VIP match — use the email from the angle-bracket format
        return email_part

    if "@" in to:
        return to

    contacts = load_vip_contacts()
    to_lower = to.lower()
    for c in contacts:
        if to_lower in c.name.lower() or to_lower in c.email.lower():
            return c.email
    raise ValueError(
        f"Could not resolve recipient {to!r} — no '@' found and no matching contact. "
        f"Please provide a full email address."
    )


def send_email(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    credentials_path: str = _DEFAULT_CREDENTIALS,
    token_path: str = _DEFAULT_TOKEN,
) -> str:
    """
    Send an email via the Gmail API.

    Args:
        to:      Recipient email address, or a VIP contact name (resolved via contacts.json).
        subject: Email subject line.
        body:    Plain-text email body.
        cc:      Optional CC addresses (comma-separated).

    Returns:
        Confirmation string with message ID on success.
    """
    from googleapiclient.discovery import build

    to_addr = resolve_recipient(to)
    creds = _get_credentials(credentials_path, token_path)
    service = build("gmail", "v1", credentials=creds)

    # Look up the sender's own address from Gmail profile
    profile = service.users().getProfile(userId="me").execute()
    from_addr = profile.get("emailAddress", "me")
    from_display = f"Li <{from_addr}>"

    # Use simple MIMEText (not MIMEMultipart) for plain-text emails
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = from_display
    msg["To"] = to_addr
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = cc

    # Gmail API requires URL-safe base64, no padding issues
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    msg_id = result.get("id", "unknown")

    return (
        f"✅ Email sent successfully!\n"
        f"From    : {from_display}\n"
        f"To      : {to_addr}\n"
        f"Subject : {subject}\n"
        f"Message ID: {msg_id}"
    )


def reply_email(
    original: "GmailMessage",
    body: str,
    credentials_path: str = _DEFAULT_CREDENTIALS,
    token_path: str = _DEFAULT_TOKEN,
) -> str:
    """
    Send a reply to an existing email, with proper threading headers
    (In-Reply-To, References) so it appears as a thread in Gmail.
    """
    from googleapiclient.discovery import build
    import email.header as _email_header

    creds = _get_credentials(credentials_path, token_path)
    service = build("gmail", "v1", credentials=creds)

    profile = service.users().getProfile(userId="me").execute()
    from_addr = profile.get("emailAddress", "me")
    from_display = f"Li <{from_addr}>"

    # Decode RFC 2047 encoded sender (e.g. =?UTF-8?Q?...?=) and extract bare email
    def _extract_email(raw: str) -> str:
        # Decode any RFC 2047 encoding first
        try:
            parts = _email_header.decode_header(raw)
            decoded = ""
            for part, enc in parts:
                if isinstance(part, bytes):
                    decoded += part.decode(enc or "utf-8", errors="replace")
                else:
                    decoded += part
            raw = decoded
        except Exception:
            pass
        # Extract email from "Name <email>" format
        m = re.search(r"<([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})>", raw)
        if m:
            return m.group(1).strip()
        # Try bare email address
        m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", raw)
        if m:
            return m.group(0).strip()
        return raw.strip()

    to_addr = _extract_email(original.sender)
    print(f"[reply_email] replying to: {to_addr!r} (raw sender: {original.sender!r})")

    subject = original.subject
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    # Build the message with threading headers
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = from_display
    msg["To"] = to_addr
    msg["Subject"] = subject
    # In-Reply-To / References use the original message's RFC 2822 Message-ID header if available,
    # fallback to Gmail message ID format
    thread_id_header = f"<{original.msg_id}@mail.gmail.com>"
    msg["In-Reply-To"] = thread_id_header
    msg["References"] = thread_id_header

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")

    # Use thread_id (not msg_id) to attach reply to the correct Gmail thread
    gmail_thread_id = original.thread_id or original.msg_id
    result = service.users().messages().send(
        userId="me",
        body={"raw": raw, "threadId": gmail_thread_id},
    ).execute()
    sent_id = result.get("id", "unknown")

    return (
        f"✅ Reply sent!\n"
        f"From    : {from_display}\n"
        f"To      : {to_addr}\n"
        f"Subject : {subject}\n"
        f"Message ID: {sent_id}"
    )


# ============================================================
# VIP contact discovery from Sent box
# ============================================================

# Greeting patterns to extract a name from the first line of an email body.
# Returns the captured name group on match.
_GREETING_PATTERNS = [
    # English
    re.compile(r"^(?:Hi|Hello|Hey|Dear|Dearest)\s+([A-Za-z][A-Za-z\s\-\.]{1,40}?)[\s,!:\n]", re.IGNORECASE),
    # Chinese
    re.compile(r"^(?:你好|您好|亲爱的|嗨|Hi)\s*[，,]?\s*([^\s，,！!。.\n]{2,15})\s*[，,！!]", re.UNICODE),
]


def _extract_greeting_name(body: str) -> Optional[str]:
    """Try to extract the recipient name from the greeting line of an email body."""
    first_lines = body.strip().splitlines()[:3]
    for line in first_lines:
        line = line.strip()
        if not line:
            continue
        for pat in _GREETING_PATTERNS:
            m = pat.match(line)
            if m:
                name = m.group(1).strip().rstrip(",!: \t")
                if name:
                    return name
    return None


def discover_contacts_from_sent(
    days: int = 150,
    max_emails: int = 200,
    credentials_path: str = _DEFAULT_CREDENTIALS,
    token_path: str = _DEFAULT_TOKEN,
) -> List[dict]:
    """
    Scan the Sent box for the last `days` days and discover recipient contacts.

    For each unique recipient email:
    - Parse the To: header for their name (falls back to greeting in body)
    - Skip anyone already in contacts.json

    Returns a list of dicts: [{"name": str, "email": str, "source": "header"|"greeting"}]
    """
    from googleapiclient.discovery import build
    import datetime

    creds = _get_credentials(credentials_path, token_path)
    service = build("gmail", "v1", credentials=creds)

    # Get sender's own address so we never add ourselves
    profile = service.users().getProfile(userId="me").execute()
    own_email = profile.get("emailAddress", "").lower()

    existing = {c.email.strip().lower() for c in load_vip_contacts()}
    existing.add(own_email)  # exclude self

    # Use after:YYYY/MM/DD — more reliable than newer_than:Nd for large values
    since_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y/%m/%d")
    query = f"after:{since_date}"
    print(f"[discover_contacts_from_sent] querying SENT label, after={since_date}, max={max_emails}")

    list_resp = (
        service.users()
        .messages()
        .list(userId="me", q=query, labelIds=["SENT"], maxResults=max_emails)
        .execute()
    )
    messages_meta = list_resp.get("messages", [])
    print(f"[discover_contacts_from_sent] found {len(messages_meta)} sent message(s)")
    if not messages_meta:
        return []

    seen_emails: dict = {}  # email_lower → candidate dict

    for meta in messages_meta:
        msg_id = meta["id"]
        try:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
        except Exception as e:
            print(f"[discover_contacts_from_sent] skip {msg_id}: {e}")
            continue

        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
        to_raw = headers.get("to", "")
        if not to_raw:
            continue

        # Parse "Name <email>" or bare "email" (may be comma-separated list)
        for part in to_raw.split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^(.+?)\s*<([^>]+)>$", part)
            if m:
                header_name = m.group(1).strip().strip('"')
                email_addr = m.group(2).strip().lower()
            elif "@" in part:
                header_name = ""
                email_addr = part.strip().lower()
            else:
                continue

            if email_addr in existing or email_addr in seen_emails:
                continue

            # Name priority: To: header name → greeting in body → local part before @
            name = header_name or ""
            source = "header"
            if not name:
                body_text = _decode_body(msg.get("payload", {}))
                greeting = _extract_greeting_name(body_text)
                if greeting:
                    name = greeting
                    source = "greeting"
            if not name:
                name = email_addr.split("@")[0]
                source = "email"

            seen_emails[email_addr] = {"name": name, "email": email_addr, "source": source}
            print(f"[discover_contacts_from_sent]   + {name} <{email_addr}> ({source})")

    return list(seen_emails.values())


def _decode_body(payload: dict) -> str:
    """Decode the plain-text body from a Gmail message payload."""
    def _get_text(part: dict) -> str:
        mime = part.get("mimeType", "")
        if mime == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                try:
                    return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
                except Exception:
                    return ""
        for sub in part.get("parts", []):
            result = _get_text(sub)
            if result:
                return result
        return ""
    return _get_text(payload)
