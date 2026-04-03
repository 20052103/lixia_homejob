# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as _dt
import html
import json
import os
import re
import subprocess
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

try:
    from .config import (
        SERPAPI_API_KEY,
        WEB_SEARCH_MAX_RESULTS,
        WEB_SEARCH_PROVIDER,
        WEB_SEARCH_TIMEOUT_SEC,
        WEB_SEARCH_USER_AGENT,
        GMAIL_CREDENTIALS_PATH,
        GMAIL_TOKEN_PATH,
        GCAL_CREDENTIALS_PATH,
        GCAL_TOKEN_PATH,
        GDRIVE_CREDENTIALS_PATH,
        GDRIVE_TOKEN_PATH,
    )
except ImportError:
    from config import (
        SERPAPI_API_KEY,
        WEB_SEARCH_MAX_RESULTS,
        WEB_SEARCH_PROVIDER,
        WEB_SEARCH_TIMEOUT_SEC,
        WEB_SEARCH_USER_AGENT,
        GMAIL_CREDENTIALS_PATH,
        GMAIL_TOKEN_PATH,
        GCAL_CREDENTIALS_PATH,
        GCAL_TOKEN_PATH,
        GDRIVE_CREDENTIALS_PATH,
        GDRIVE_TOKEN_PATH,
    )


# ============================================================
# Tool result container
# ============================================================

@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: Dict[str, Any]


class ToolError(Exception):
    pass


# ============================================================
# HTTP helpers (requests based)
# ============================================================

def _http_get_json(url: str, timeout_sec: int) -> Dict[str, Any]:
    r = requests.get(
        url,
        headers={"User-Agent": WEB_SEARCH_USER_AGENT},
        timeout=timeout_sec,
    )
    r.raise_for_status()
    return r.json()


def _http_get_text(url: str, timeout_sec: int) -> str:
    r = requests.get(
        url,
        headers={"User-Agent": WEB_SEARCH_USER_AGENT},
        timeout=timeout_sec,
    )
    r.raise_for_status()
    return r.text


# ============================================================
# Web search implementations
# ============================================================

def _search_serpapi_google(query: str, num_results: int) -> Dict[str, Any]:

    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERPAPI_API_KEY,
    }

    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)

    data = _http_get_json(url, WEB_SEARCH_TIMEOUT_SEC)

    results = []

    for r in data.get("organic_results", [])[:num_results]:
        results.append(
            {
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
            }
        )

    return {
        "provider": "serpapi",
        "query": query,
        "results": results,
    }


def _search_duckduckgo_html(query: str, num_results: int) -> Dict[str, Any]:

    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)

    html_text = _http_get_text(url, WEB_SEARCH_TIMEOUT_SEC)

    pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<link>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )

    snippet_pattern = re.compile(
        r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>|'
        r'<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet2>.*?)</div>',
        flags=re.IGNORECASE | re.DOTALL,
    )

    titles = list(pattern.finditer(html_text))
    snippets = list(snippet_pattern.finditer(html_text))

    results = []

    for i, m in enumerate(titles[:num_results]):

        title = re.sub("<[^>]+>", "", m.group("title"))
        link = html.unescape(m.group("link"))

        snippet = ""
        if i < len(snippets):
            snippet = re.sub("<[^>]+>", "", snippets[i].group(0))

        results.append(
            {
                "title": title.strip(),
                "link": link.strip(),
                "snippet": snippet.strip(),
            }
        )

    return {
        "provider": "duckduckgo",
        "query": query,
        "results": results,
    }


def _choose_search_provider(provider: str | None):

    p = (provider or WEB_SEARCH_PROVIDER or "auto").lower()

    if p == "auto":
        return "serpapi" if SERPAPI_API_KEY else "duckduckgo"

    return p


# ============================================================
# Sandbox
# ============================================================

class ToolSandbox:

    def __init__(
        self,
        allowed_roots: List[str],
        allowed_cmd_prefixes: Optional[List[str]] = None,
        max_read_bytes: int = 200000,
        max_output_chars: int = 40000,
        cwd: Optional[str] = None,
    ):

        self.allowed_roots = [os.path.abspath(r) for r in allowed_roots]

        self.allowed_cmd_prefixes = allowed_cmd_prefixes or [
            "python",
            "py",
            "git",
            "dir",
            "ls",
            "pip",
        ]

        self.max_read_bytes = max_read_bytes
        self.max_output_chars = max_output_chars
        self.cwd = cwd
        # Pending email draft — set by draft_gmail, consumed by send_gmail
        self.pending_email: Optional[Dict[str, str]] = None
        # Email cache — maps display IDs (E1, E2, ...) to GmailMessage objects
        self.email_cache: Dict[str, Any] = {}
        # Pending calendar event draft — set by draft_calendar_event, consumed on confirm
        self.pending_cal_event: Optional[Dict[str, Any]] = None
        # Last Drive search results — list of DriveFile objects from most recent search_drive call
        self.last_drive_files: List[Any] = []
        # Lazy-initialised Playwright browser session (None until first use)
        self._browser_session = None

    # ----------------------------------------------------------
    # File tools
    # ----------------------------------------------------------

    def read_file(self, path: str, start: int = 0, limit: int = 50000) -> ToolResult:

        ap = os.path.abspath(path)

        if not os.path.isfile(ap):
            return ToolResult(False, f"Not a file: {ap}", {"path": ap})

        with open(ap, "r", encoding="utf-8", errors="replace") as f:
            f.seek(start)
            text = f.read(limit)

        return ToolResult(True, text, {"path": ap})

    def list_dir(self, path: str, max_items: int = 200) -> ToolResult:

        ap = os.path.abspath(path)

        if not os.path.isdir(ap):
            return ToolResult(False, f"Not a directory: {ap}", {"path": ap})

        items = sorted(os.listdir(ap))

        return ToolResult(True, "\n".join(items[:max_items]), {"path": ap})

    # ----------------------------------------------------------
    # Command tool
    # ----------------------------------------------------------

    def run_cmd(self, cmd: str, timeout_sec: int = 60) -> ToolResult:

        first = cmd.split()[0].lower()

        if first not in [p.lower() for p in self.allowed_cmd_prefixes]:
            return ToolResult(False, "Command not allowed", {"cmd": cmd})

        try:

            r = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=self.cwd,
            )

            out = r.stdout + "\n" + r.stderr

            return ToolResult(True, out[: self.max_output_chars], {"cmd": cmd})

        except Exception as e:

            return ToolResult(False, str(e), {"cmd": cmd})

    # ----------------------------------------------------------
    # Web search
    # ----------------------------------------------------------

    def search_web(self, query: str, max_results: int = 5, provider: str = "auto") -> ToolResult:

        q = query.strip()

        if not q:
            return ToolResult(False, "Empty query", {})

        provider = _choose_search_provider(provider)

        print(f"[search_web] query={q}")
        print(f"[search_web] provider={provider}")

        try:

            if provider == "serpapi":

                payload = _search_serpapi_google(q, max_results)

            else:

                payload = _search_duckduckgo_html(q, max_results)

            out = json.dumps(payload, indent=2, ensure_ascii=False)

            return ToolResult(True, out, {"query": q})

        except Exception as e:

            import traceback

            traceback.print_exc()

            return ToolResult(False, str(e), {"query": q})

    # ----------------------------------------------------------
    # Gmail tool
    # ----------------------------------------------------------

    def fetch_gmail(
        self,
        query: str = "is:unread",
        max_results: int = 10,
        include_body: bool = True,
    ) -> ToolResult:
        """Fetch emails from Gmail using the Gmail API (OAuth2)."""
        try:
            try:
                from .gmail_tool import fetch_emails_as_text
            except ImportError:
                from gmail_tool import fetch_emails_as_text

            print(f"[fetch_gmail] query={query!r} max_results={max_results} include_body={include_body}")

            text = fetch_emails_as_text(
                query=query,
                max_results=max_results,
                include_body=include_body,
                credentials_path=GMAIL_CREDENTIALS_PATH,
                token_path=GMAIL_TOKEN_PATH,
            )

            text, cache = text  # fetch_emails_as_text returns (text, email_cache)
            self.email_cache.update(cache)  # merge new IDs into the persistent cache
            print(f"[fetch_gmail] cached {len(cache)} email(s) with IDs: {list(cache.keys())}")

            return ToolResult(True, text, {"query": query, "max_results": max_results})

        except FileNotFoundError as e:
            return ToolResult(False, str(e), {"query": query})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {"query": query})

    def draft_gmail(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
    ) -> ToolResult:
        """Store a pending email draft and return a formatted preview for user confirmation."""
        # Ensure body always ends with the user's signature "Li"
        body = body.rstrip()
        bad_closings = ["你的助手", "best regards", "assistant", "your assistant", "此致", "敬上"]
        last_line = body.rsplit("\n", 1)[-1].strip().lower()
        if any(c in last_line for c in bad_closings):
            body = body.rsplit("\n", 1)[0].rstrip()
        if not body.endswith("\n\nLi") and not body.endswith("\nLi"):
            body = body + "\n\nLi"
        try:
            try:
                from .gmail_tool import resolve_recipient
            except ImportError:
                from gmail_tool import resolve_recipient

            to_addr = resolve_recipient(to)
            self.pending_email = {"to": to_addr, "subject": subject, "body": body, "cc": cc}

            preview = (
                f"---\n"
                f"To      : {to_addr}\n"
                f"Subject : {subject}\n"
                f"Body    :\n{body}\n"
                f"---\n"
                f"请确认发送，或告诉我需要修改的地方。"
            )
            print(f"[draft_gmail] draft saved → to={to_addr!r} subject={subject!r}")
            return ToolResult(True, preview, {"to": to_addr, "subject": subject})

        except ValueError as e:
            return ToolResult(False, str(e), {"to": to})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {"to": to})

    def send_gmail(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
    ) -> ToolResult:
        """Send an email via Gmail API. Recipient can be an email address or a VIP contact name."""
        try:
            try:
                from .gmail_tool import send_email
            except ImportError:
                from gmail_tool import send_email

            print(f"[send_gmail] to={to!r} subject={subject!r}")

            result = send_email(
                to=to,
                subject=subject,
                body=body,
                cc=cc,
                credentials_path=GMAIL_CREDENTIALS_PATH,
                token_path=GMAIL_TOKEN_PATH,
            )

            return ToolResult(True, result, {"to": to, "subject": subject})

        except ValueError as e:
            return ToolResult(False, str(e), {"to": to})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {"to": to})

    def reply_gmail(self, email_id: str, body: str) -> ToolResult:
        """Draft a reply to a previously fetched email identified by its display ID (e.g. E1)."""
        try:
            email_id = email_id.strip().upper()
            original = self.email_cache.get(email_id)
            if original is None:
                available = list(self.email_cache.keys())
                return ToolResult(
                    False,
                    f"Email ID {email_id!r} not found in cache. Available: {available}. "
                    f"Please fetch emails first.",
                    {"email_id": email_id},
                )

            # Enforce Li signature
            body = body.rstrip()
            bad_closings = ["你的助手", "best regards", "assistant", "your assistant", "此致", "敬上"]
            last_line = body.rsplit("\n", 1)[-1].strip().lower()
            if any(c in last_line for c in bad_closings):
                body = body.rsplit("\n", 1)[0].rstrip()
            if not body.endswith("\n\nLi") and not body.endswith("\nLi"):
                body = body + "\n\nLi"

            import re as _re
            import email.header as _ehdr
            subject = original.subject if original.subject.lower().startswith("re:") else f"Re: {original.subject}"

            # Robustly extract bare email address, handling RFC 2047 encoding
            def _bare_email(raw: str) -> str:
                try:
                    parts = _ehdr.decode_header(raw)
                    decoded = ""
                    for part, enc in parts:
                        decoded += part.decode(enc or "utf-8", errors="replace") if isinstance(part, bytes) else part
                    raw = decoded
                except Exception:
                    pass
                m = _re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", raw)
                return m.group(0).strip() if m else raw.strip()

            to_addr = _bare_email(original.sender)
            self.pending_email = {
                "to": to_addr,
                "subject": subject,
                "body": body,
                "cc": "",
                "_reply_to": original,
            }

            preview = (
                f"---\n"
                f"To      : {to_addr}  (Reply to: {original.subject!r})\n"
                f"Subject : {subject}\n"
                f"Body    :\n{body}\n"
                f"---\n"
                f"请确认发送，或告诉我需要修改的地方。"
            )
            print(f"[reply_gmail] draft saved → reply to {email_id} ({to_addr!r})")
            return ToolResult(True, preview, {"to": to_addr, "subject": subject, "email_id": email_id})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {"email_id": email_id})

    # ----------------------------------------------------------
    # Stock market tool
    # ----------------------------------------------------------

    def fetch_stock(
        self,
        tickers: str,
        include_news: bool = True,
    ) -> ToolResult:
        """
        Fetch near-realtime stock quote(s) and recent news.
        tickers: comma-separated string, e.g. "AAPL,MSFT,TSLA"
        """
        try:
            try:
                from .stock_tool import fetch_stock_summary
            except ImportError:
                from stock_tool import fetch_stock_summary

            ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            if not ticker_list:
                return ToolResult(False, "No tickers provided.", {"tickers": tickers})

            print(f"[fetch_stock] tickers={ticker_list} include_news={include_news}")
            try:
                from .config import FINNHUB_API_KEY, POLYGON_API_KEY
            except ImportError:
                from config import FINNHUB_API_KEY, POLYGON_API_KEY
            print(f"[fetch_stock] finnhub_key={'set' if FINNHUB_API_KEY else 'MISSING'} polygon_key={'set' if POLYGON_API_KEY else 'MISSING'}")
            text = fetch_stock_summary(ticker_list, include_news=include_news)
            return ToolResult(True, text, {"tickers": ticker_list})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {"tickers": tickers})

    def fetch_market(self) -> ToolResult:
        """Fetch a full market overview: indices, sectors, tech mega-caps, top movers."""
        try:
            try:
                from .stock_tool import fetch_market_overview
            except ImportError:
                from stock_tool import fetch_market_overview
            print("[fetch_market] fetching market overview...")
            text = fetch_market_overview()
            return ToolResult(True, text, {})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {})

    def read_image(self, prompt: str = "Describe this image in detail. If it contains text, transcribe it fully.") -> ToolResult:
        """Open GUI dialog to capture an image from clipboard or file, then describe it."""
        try:
            try:
                from .image_tool import read_image_interactive
                from .config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL_NAME
            except ImportError:
                from image_tool import read_image_interactive
                from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL_NAME

            result = read_image_interactive(
                prompt=prompt,
                base_url=LM_STUDIO_BASE_URL,
                api_key=LM_STUDIO_API_KEY,
                model=LM_STUDIO_MODEL_NAME,
            )
            return ToolResult(True, result, {})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {})

    def discover_vip_contacts(self, days: int = 150, auto_add: bool = True) -> ToolResult:
        """
        Scan the Sent box for the last `days` days, extract recipients and greeting names,
        and add new ones to contacts.json as VIP contacts.

        Returns a summary of what was found and added.
        """
        try:
            try:
                from .gmail_tool import discover_contacts_from_sent
                from .contacts import add_vip_contact, deduplicate_contacts
            except ImportError:
                from gmail_tool import discover_contacts_from_sent
                from contacts import add_vip_contact, deduplicate_contacts

            print(f"[discover_vip_contacts] scanning sent box (last {days} days)...")
            candidates = discover_contacts_from_sent(
                days=days,
                credentials_path=GMAIL_CREDENTIALS_PATH,
                token_path=GMAIL_TOKEN_PATH,
            )

            if not candidates:
                return ToolResult(True, f"过去 {days} 天的发件箱里没有找到新联系人。", {"added": []})

            added = []
            skipped = []

            if auto_add:
                for c in candidates:
                    try:
                        add_vip_contact(name=c["name"], email=c["email"])
                        added.append(c)
                        print(f"[discover_vip_contacts] added: {c['name']} <{c['email']}> (via {c['source']})")
                    except ValueError:
                        skipped.append(c)  # already exists

            lines = [f"📬 发件箱扫描完成（过去 {days} 天）\n"]
            if added:
                lines.append(f"✅ 新增 {len(added)} 个VIP联系人：")
                for c in added:
                    src_label = {"header": "邮件头", "greeting": "称呼", "email": "邮箱推断"}.get(c["source"], c["source"])
                    lines.append(f"  • {c['name']} <{c['email']}> （来源：{src_label}）")
            if skipped:
                lines.append(f"\n⏭ 已跳过 {len(skipped)} 个（已存在）：")
                for c in skipped:
                    lines.append(f"  • {c['name']} <{c['email']}>")
            if not added and not skipped:
                lines.append("没有找到新联系人（所有人已在VIP列表中）。")

            return ToolResult(True, "\n".join(lines), {"added": added, "skipped": skipped})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {})

    # ------------------------------------------------------------------
    # Google Calendar
    # ------------------------------------------------------------------

    def fetch_calendar(
        self,
        range: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days_ahead: int = 7,
    ) -> ToolResult:
        """
        Fetch events from Google Calendar for the given range.

        range: today | tomorrow | this_week | next_week | next_7_days | next_30_days
               | YYYY-MM-DD (single day)
               | None  → uses days_ahead starting today
        start / end: explicit YYYY-MM-DD dates (used if range is None)
        days_ahead: default window when range and start are both None
        """
        try:
            try:
                from .gcal_tool import fetch_calendar_as_text
            except ImportError:
                from gcal_tool import fetch_calendar_as_text

            print(f"[fetch_calendar] range={range!r} start={start!r} end={end!r} days_ahead={days_ahead}")
            text = fetch_calendar_as_text(
                range_str=range,
                start_str=start,
                end_str=end,
                days_ahead=days_ahead,
                credentials_path=GCAL_CREDENTIALS_PATH,
                token_path=GCAL_TOKEN_PATH,
            )
            return ToolResult(True, text, {"range": range, "start": start, "end": end})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, f"❌ 日历获取失败: {e}", {})

    def draft_calendar_event(
        self,
        title: str,
        date: str,
        start_time: str,
        duration_minutes: int = 60,
        end_time: str = "",
        location: str = "",
        description: str = "",
    ) -> ToolResult:
        """
        Store a calendar event draft and show a preview for user confirmation.
        Actual creation happens when the user confirms.
        """
        try:
            try:
                from .gcal_tool import format_event_preview
            except ImportError:
                from gcal_tool import format_event_preview

            self.pending_cal_event = {
                "title": title,
                "date": date,
                "start_time": start_time,
                "duration_minutes": duration_minutes,
                "end_time": end_time,
                "location": location,
                "description": description,
            }
            print(f"[draft_calendar_event] draft saved → title={title!r} date={date!r} start={start_time!r}")
            preview = format_event_preview(
                title=title,
                date_str=date,
                start_time=start_time,
                duration_minutes=duration_minutes,
                end_time=end_time or None,
                location=location,
                description=description,
            )
            return ToolResult(True, preview, self.pending_cal_event)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, f"❌ 日程草稿保存失败: {e}", {})

    def confirm_calendar_event(self) -> ToolResult:
        """
        Create the pending calendar event on Google Calendar.
        Called automatically when the user confirms the draft preview.
        """
        if not self.pending_cal_event:
            return ToolResult(False, "没有待确认的日程。", {})
        ev = self.pending_cal_event
        try:
            try:
                from .gcal_tool import create_event
            except ImportError:
                from gcal_tool import create_event

            created = create_event(
                title=ev["title"],
                date_str=ev["date"],
                start_time=ev["start_time"],
                duration_minutes=int(ev.get("duration_minutes", 60)),
                end_time=ev.get("end_time") or None,
                location=ev.get("location", ""),
                description=ev.get("description", ""),
                credentials_path=GCAL_CREDENTIALS_PATH,
                token_path=GCAL_TOKEN_PATH,
            )
            self.pending_cal_event = None
            html_link = created.get("htmlLink", "")
            return ToolResult(
                True,
                f"✅ 日程已添加！\n"
                f"📅 {ev['title']} @ {ev['date']} {ev['start_time']}\n"
                f"🔗 {html_link}",
                {"event_id": created.get("id", ""), "link": html_link},
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, f"❌ 日程创建失败: {e}", {})

    # ------------------------------------------------------------------
    # Google Drive
    # ------------------------------------------------------------------

    def search_drive(self, query: str, max_results: int = 10) -> ToolResult:
        """Search Google Drive files by name or content keywords."""
        try:
            try:
                from .gdrive_tool import search_files, format_search_results
            except ImportError:
                from gdrive_tool import search_files, format_search_results

            print(f"[search_drive] query={query!r} max={max_results}")
            files = search_files(
                query=query,
                max_results=max_results,
                credentials_path=GDRIVE_CREDENTIALS_PATH,
                token_path=GDRIVE_TOKEN_PATH,
            )
            self.last_drive_files = files  # cache for auto-read interceptor
            text = format_search_results(files, query)
            return ToolResult(True, text, {"query": query})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, f"❌ Drive search failed: {e}", {})

    def read_drive_file(self, file_id: str, max_chars: int = 6000) -> ToolResult:
        """Read the text content of a Google Drive file by its file ID."""
        try:
            try:
                from .gdrive_tool import read_file_content
            except ImportError:
                from gdrive_tool import read_file_content

            print(f"[read_drive_file] file_id={file_id!r}")
            text = read_file_content(
                file_id=file_id,
                max_chars=max_chars,
                credentials_path=GDRIVE_CREDENTIALS_PATH,
                token_path=GDRIVE_TOKEN_PATH,
            )
            return ToolResult(True, text, {"file_id": file_id})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, f"❌ Drive read failed: {e}", {})
    # ------------------------------------------------------------------
    # Browser tools (Playwright)
    # ------------------------------------------------------------------

    def _get_browser_session(self):
        """Return the shared BrowserSession, creating it on first use."""
        if self._browser_session is None:
            try:
                from .browser_tool import BrowserSession
            except ImportError:
                from browser_tool import BrowserSession
            self._browser_session = BrowserSession(headless=False)
        return self._browser_session

    def browse_page(self, url: str) -> "ToolResult":
        """
        Open *url* in a browser window, extract the page text and list all
        interactive elements (links, buttons, inputs).
        Returns a human-readable summary suitable for the LLM.
        """
        try:
            session = self._get_browser_session()
            print(f"[browse_page] url={url!r}")
            snap = session.fetch_page(url)
            return ToolResult(True, snap.summary(), {"url": snap.url, "title": snap.title})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {"url": url})

    def browser_act(
        self,
        action: str,
        target: str = "",
        value: str = "",
    ) -> "ToolResult":
        """
        Perform an action in the currently open browser tab.

        action   target / value
        -------- -------------------------------------------------------
        click    CSS selector or visible text of the element to click
        type     selector or text of input field  |  value = text to type
        scroll   "up" / "down" / pixel amount (e.g. "300" or "-400")
        back     –  (navigate back)
        goto     URL to navigate to
        snapshot –  (return a fresh page snapshot)
        screenshot  optional file path for the PNG
        """
        try:
            session = self._get_browser_session()
            print(f"[browser_act] action={action!r} target={target!r} value={value!r}")
            import json as _json
            result = session.act(action=action, target=target, value=value)
            if not result.get("ok"):
                return ToolResult(False, result.get("error", "action failed"), result)
            # For snapshot, the result already has a human-readable string
            if "snapshot" in result:
                return ToolResult(True, result["snapshot"], result)
            return ToolResult(True, _json.dumps(result, ensure_ascii=False, indent=2), result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(False, str(e), {"action": action})
