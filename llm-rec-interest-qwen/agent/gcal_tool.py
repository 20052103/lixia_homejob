# -*- coding: utf-8 -*-
"""
Google Calendar API helper — OAuth2 authentication + event fetching + event creation.

First-time use:
  1. Enable the Google Calendar API in Google Cloud Console (same project as Gmail).
  2. The existing credentials.json already works — just run the agent and ask about
     your calendar. A browser window opens for OAuth consent.
  3. The token is saved to gcal_token.json for future runs.

Scopes:  calendar (full read + write — required for creating events)

NOTE: gcal_token.json is separate from gmail_token.json so Gmail auth is unaffected.
      If you had a previous gcal_token.json with readonly scope, DELETE it so the
      OAuth flow re-runs with the new write scope.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

_SCOPES = [
    "https://www.googleapis.com/auth/calendar",        # read + write events
]

_DEFAULT_CREDENTIALS = os.path.join(os.path.dirname(__file__), "credentials.json")
_DEFAULT_TOKEN = os.path.join(os.path.dirname(__file__), "gcal_token.json")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_service(credentials_path: str = _DEFAULT_CREDENTIALS,
                 token_path: str = _DEFAULT_TOKEN):
    """Return an authenticated Google Calendar API service object."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"credentials.json not found at {credentials_path}. "
                    "Download it from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, _SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as fh:
            fh.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Date range resolution
# ---------------------------------------------------------------------------

def _today_local() -> date:
    return datetime.now().date()


def _week_bounds(d: date) -> Tuple[date, date]:
    """Return (Monday, Sunday) of the week containing d."""
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def resolve_range(
    range_str: Optional[str] = None,
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    days_ahead: int = 7,
    cal_tz=None,
) -> Tuple[datetime, datetime]:
    """
    Resolve a human range keyword or explicit dates to (start_dt, end_dt).
    Boundaries are in cal_tz (calendar's timezone) so day edges match the user's clock.

    range_str options:
        today, tomorrow, this_week, next_week, next_7_days, next_30_days,
        YYYY-MM-DD  (single day)

    start_str / end_str: ISO date strings YYYY-MM-DD for explicit range.
    days_ahead: used when range_str is None and start_str is None.
    """
    if cal_tz is None:
        local_offset = datetime.now(timezone.utc).astimezone().utcoffset()
        cal_tz = timezone(local_offset)
    # Use "today" in the calendar's timezone, not the server's UTC date
    today = datetime.now(cal_tz).date()

    if range_str:
        r = range_str.strip().lower()
        if r == "today":
            d_start = d_end = today
        elif r == "tomorrow":
            d_start = d_end = today + timedelta(days=1)
        elif r == "this_week":
            d_start, d_end = _week_bounds(today)
        elif r == "next_week":
            next_mon = today + timedelta(days=(7 - today.weekday()))
            d_start, d_end = _week_bounds(next_mon)
        elif r in ("next_7_days", "next7days"):
            d_start = today
            d_end = today + timedelta(days=6)
        elif r in ("next_30_days", "next30days"):
            d_start = today
            d_end = today + timedelta(days=29)
        else:
            # Try YYYY-MM-DD single day
            try:
                parsed = date.fromisoformat(r)
                d_start = d_end = parsed
            except ValueError:
                # Try "YYYY-MM-DD to YYYY-MM-DD"
                m = re.match(r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", r)
                if m:
                    d_start = date.fromisoformat(m.group(1))
                    d_end = date.fromisoformat(m.group(2))
                else:
                    # Default fallback
                    d_start = today
                    d_end = today + timedelta(days=days_ahead - 1)
    elif start_str:
        d_start = date.fromisoformat(start_str)
        d_end = date.fromisoformat(end_str) if end_str else d_start + timedelta(days=days_ahead - 1)
    else:
        d_start = today
        d_end = today + timedelta(days=days_ahead - 1)

    # Convert to RFC3339 datetimes in the calendar's own timezone so day
    # boundaries match the user's wall clock, not the server's UTC clock.
    start_dt = datetime(d_start.year, d_start.month, d_start.day, 0, 0, 0, tzinfo=cal_tz)
    end_dt   = datetime(d_end.year,   d_end.month,   d_end.day,   23, 59, 59, tzinfo=cal_tz)
    return start_dt, end_dt


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalEvent:
    event_id: str
    summary: str
    start: str          # ISO string (date or datetime)
    end: str
    location: str = ""
    description: str = ""
    all_day: bool = False
    organizer: str = ""
    attendees: List[str] = field(default_factory=list)


def _get_calendar_timezone(service) -> "datetime.tzinfo":
    """
    Return a tzinfo matching the primary calendar's configured timezone.
    Falls back to the machine's local timezone if lookup fails.
    """
    try:
        cal_meta = service.calendars().get(calendarId="primary").execute()
        tz_name = cal_meta.get("timeZone", "UTC")
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            return ZoneInfo(tz_name)
        except ImportError:
            import pytz
            return pytz.timezone(tz_name)
    except Exception:
        # Fall back to machine local offset
        local_offset = datetime.now(timezone.utc).astimezone().utcoffset()
        return timezone(local_offset)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_events(
    range_str: Optional[str] = None,
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    days_ahead: int = 7,
    max_results: int = 50,
    credentials_path: str = _DEFAULT_CREDENTIALS,
    token_path: str = _DEFAULT_TOKEN,
) -> tuple:
    """
    Fetch Google Calendar events from ALL calendars for the given range.
    Returns (list[CalEvent], cal_tz) ordered by start time.
    """
    service = _get_service(credentials_path, token_path)
    cal_tz = _get_calendar_timezone(service)
    start_dt, end_dt = resolve_range(range_str, start_str, end_str, days_ahead, cal_tz=cal_tz)

    # Enumerate all calendars the user has
    calendars = []
    page_token = None
    while True:
        cal_list = service.calendarList().list(pageToken=page_token).execute()
        calendars.extend(cal_list.get("items", []))
        page_token = cal_list.get("nextPageToken")
        if not page_token:
            break

    result: List[CalEvent] = []

    for cal in calendars:
        cal_id = cal.get("id", "")
        cal_name = cal.get("summary", cal_id)
        # Skip calendars the user has declined or that are hidden
        if cal.get("selected") is False:
            continue

        try:
            events_result = (
                service.events()
                .list(
                    calendarId=cal_id,
                    timeMin=start_dt.isoformat(),
                    timeMax=end_dt.isoformat(),
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
        except Exception:
            continue  # skip calendars we can't read

        for item in events_result.get("items", []):
            start_raw = item.get("start", {})
            end_raw = item.get("end", {})
            all_day = "date" in start_raw and "dateTime" not in start_raw

            start_str_val = start_raw.get("dateTime") or start_raw.get("date", "")
            end_str_val   = end_raw.get("dateTime")   or end_raw.get("date", "")

            organizer_email = item.get("organizer", {}).get("email", "")
            attendees = [
                a.get("email", "") for a in item.get("attendees", [])
                if a.get("email") and a.get("responseStatus") != "declined"
            ]

            result.append(CalEvent(
                event_id=item.get("id", ""),
                summary=item.get("summary", "(No title)"),
                start=start_str_val,
                end=end_str_val,
                location=item.get("location", ""),
                description=(item.get("description") or "")[:200],
                all_day=all_day,
                organizer=organizer_email,
                attendees=attendees,
            ))

    # Sort all events across calendars by start time
    def _sort_key(ev: CalEvent) -> str:
        return ev.start or ""

    result.sort(key=_sort_key)
    return result, cal_tz


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_time(iso_str: str, all_day: bool) -> str:
    """Format ISO datetime/date to a human-readable string."""
    if not iso_str:
        return "?"
    if all_day:
        try:
            d = date.fromisoformat(iso_str)
            return d.strftime("%a %b %d").replace(" 0", " ")
        except Exception:
            return iso_str
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%a %b %d, %I:%M %p").replace(" 0", " ").replace("  ", " ")
    except Exception:
        return iso_str


def format_events_as_text(
    events: List[CalEvent],
    range_label: str = "",
    cal_tz=None,
) -> str:
    """Return a concise, human-readable summary of events."""
    if not events:
        label = f" for {range_label}" if range_label else ""
        return f"📅 No events found{label}."

    label = f" — {range_label}" if range_label else ""
    lines = [f"📅 **Calendar{label}** ({len(events)} event{'s' if len(events)!=1 else ''})\n"]

    current_date_str = ""
    for i, ev in enumerate(events, 1):
        if ev.all_day:
            day_key = ev.start[:10]
            time_str = "All day"
        else:
            try:
                start_dt = datetime.fromisoformat(ev.start)
                end_dt   = datetime.fromisoformat(ev.end)
                # Convert to calendar's timezone for display
                if cal_tz is not None:
                    start_dt = start_dt.astimezone(cal_tz)
                    end_dt   = end_dt.astimezone(cal_tz)
                day_key  = start_dt.strftime("%Y-%m-%d")
                s_time = start_dt.strftime("%I:%M %p").lstrip("0") or "12:00 AM"
                e_time = end_dt.strftime("%I:%M %p").lstrip("0") or "12:00 AM"
                time_str = f"{s_time} – {e_time}"
            except Exception:
                day_key  = ev.start[:10]
                time_str = ev.start

        # Group header by date
        if day_key != current_date_str:
            current_date_str = day_key
            try:
                day_label = date.fromisoformat(day_key).strftime("%A, %B %d").replace(" 0", " ")
            except Exception:
                day_label = day_key
            lines.append(f"\n📆 **{day_label}**")

        line = f"  {i}. **{ev.summary}** | {time_str}"
        if ev.location:
            line += f" | 📍 {ev.location}"
        lines.append(line)

        if ev.description:
            lines.append(f"     _{ev.description[:120]}_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def fetch_calendar_as_text(
    range_str: Optional[str] = None,
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    days_ahead: int = 7,
    credentials_path: str = _DEFAULT_CREDENTIALS,
    token_path: str = _DEFAULT_TOKEN,
) -> str:
    """Fetch calendar events and return formatted text ready for the agent."""
    range_label = range_str or (f"{start_str} to {end_str}" if start_str else f"next {days_ahead} days")
    try:
        events, cal_tz = fetch_events(
            range_str=range_str,
            start_str=start_str,
            end_str=end_str,
            days_ahead=days_ahead,
            credentials_path=credentials_path,
            token_path=token_path,
        )
        return format_events_as_text(events, range_label=range_label, cal_tz=cal_tz)
    except FileNotFoundError as e:
        return f"❌ Calendar setup error: {e}"
    except Exception as e:
        return f"❌ Failed to fetch calendar: {e}"


# ---------------------------------------------------------------------------
# Create event
# ---------------------------------------------------------------------------

def _next_weekday(weekday_name: str) -> date:
    """Return the date of the next occurrence of the named weekday (today counts if it matches)."""
    names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    target = names.index(weekday_name.lower())
    today = date.today()
    days_ahead = (target - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # "next thursday" when today is thursday → next week
    return today + timedelta(days=days_ahead)


def _parse_event_datetime(date_str: str, time_str: str, cal_tz=None) -> datetime:
    """
    Parse a date + time into a timezone-aware datetime using the calendar's timezone.
    date_str: YYYY-MM-DD | 'today' | 'tomorrow' | weekday name ('thursday', 'monday', ...)
    time_str: HH:MM  (24h or 12h like '3pm', '3:30pm', '15:00')
    cal_tz  : tzinfo from _get_calendar_timezone(); falls back to machine-local if None.
    """
    _WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}

    if cal_tz is None:
        local_offset = datetime.now(timezone.utc).astimezone().utcoffset()
        cal_tz = timezone(local_offset)

    today = date.today()
    ds = date_str.strip().lower()
    if ds == "today":
        d = today
    elif ds == "tomorrow":
        d = today + timedelta(days=1)
    elif ds in _WEEKDAYS:
        d = _next_weekday(ds)
    else:
        try:
            d = date.fromisoformat(ds)
        except ValueError:
            raise ValueError(
                f"Cannot parse date: {date_str!r}. "
                "Use YYYY-MM-DD, 'today', 'tomorrow', or a weekday name like 'thursday'."
            )

    # Normalise time string
    ts = time_str.strip().lower().replace(" ", "")
    for fmt in ("%I:%M%p", "%I%p", "%H:%M", "%H%M"):
        try:
            t = datetime.strptime(ts, fmt).time()
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Cannot parse time: {time_str!r}. Use format like '3pm', '14:30', '9:00am'.")

    # Attach the calendar's timezone so the event lands at the right wall-clock time
    naive = datetime(d.year, d.month, d.day, t.hour, t.minute)
    return naive.replace(tzinfo=cal_tz)


def create_event(
    title: str,
    date_str: str,
    start_time: str,
    duration_minutes: int = 60,
    end_time: Optional[str] = None,
    location: str = "",
    description: str = "",
    credentials_path: str = _DEFAULT_CREDENTIALS,
    token_path: str = _DEFAULT_TOKEN,
) -> dict:
    """
    Create a Google Calendar event and return the created event dict.

    title           : event name / summary
    date_str        : YYYY-MM-DD | 'today' | 'tomorrow' | weekday name
    start_time      : '3pm' | '14:30' | '9:00am'
    duration_minutes: length in minutes (default 60); ignored if end_time given
    end_time        : optional explicit end time (same format as start_time)
    location        : optional venue string
    description     : optional notes
    """
    service = _get_service(credentials_path, token_path)
    cal_tz = _get_calendar_timezone(service)

    start_dt = _parse_event_datetime(date_str, start_time, cal_tz)
    if end_time:
        end_dt = _parse_event_datetime(date_str, end_time, cal_tz)
    else:
        end_dt = start_dt + timedelta(minutes=duration_minutes)

    # Get the IANA timezone name for the event body (required for DST-correct recurrence)
    try:
        cal_meta = service.calendars().get(calendarId="primary").execute()
        tz_name = cal_meta.get("timeZone", "UTC")
    except Exception:
        tz_name = str(cal_tz)

    event_body = {
        "summary": title,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": tz_name},
        "end":   {"dateTime": end_dt.isoformat(),   "timeZone": tz_name},
    }
    if location:
        event_body["location"] = location
    if description:
        event_body["description"] = description

    created = service.events().insert(calendarId="primary", body=event_body).execute()
    return created


def format_event_preview(
    title: str,
    date_str: str,
    start_time: str,
    duration_minutes: int = 60,
    end_time: Optional[str] = None,
    location: str = "",
    description: str = "",
) -> str:
    """Return a human-readable preview string for draft confirmation."""
    _WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    today = date.today()
    ds = date_str.strip().lower()
    if ds == "today":
        d = today
    elif ds == "tomorrow":
        d = today + timedelta(days=1)
    elif ds in _WEEKDAYS:
        d = _next_weekday(ds)
    else:
        try:
            d = date.fromisoformat(ds)
        except Exception:
            d = None

    date_label = d.strftime("%A, %B %d").replace(" 0", " ") if d else date_str
    if end_time:
        time_label = f"{start_time} – {end_time}"
    else:
        time_label = f"{start_time} ({duration_minutes} min)"

    lines = [
        "─" * 40,
        f"📅  新日程预览",
        f"标题  : {title}",
        f"日期  : {date_label}",
        f"时间  : {time_label}",
    ]
    if location:
        lines.append(f"地点  : {location}")
    if description:
        lines.append(f"备注  : {description}")
    lines += [
        "─" * 40,
        "请确认添加（输入'确认'/'ok'），或告诉我需要修改的地方。",
    ]
    return "\n".join(lines)
