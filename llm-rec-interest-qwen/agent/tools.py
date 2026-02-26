# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import json

import datetime as _dt

def _start_of_day(dt: _dt.datetime) -> _dt.datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _start_of_week_monday(dt: _dt.datetime) -> _dt.datetime:
    # Monday=0
    return _start_of_day(dt) - _dt.timedelta(days=dt.weekday())

def resolve_time_window(range_name: str | None, start: str | None, days_ahead: int | None):
    """
    Returns (start_dt, end_dt) in local naive datetime.
    Priority: explicit start -> range_name -> default (next 7 days from today)
    """
    now = _dt.datetime.now().astimezone().replace(tzinfo=None)

    if start:
        # "2026-02-17" or "2026-02-17T00:00"
        if "T" in start:
            s = _dt.datetime.fromisoformat(start)
        else:
            s = _dt.datetime.fromisoformat(start + "T00:00")
        s = _start_of_day(s)
        da = int(days_ahead or 7)
        return s, s + _dt.timedelta(days=da)

    rn = (range_name or "").lower().strip()
    if rn in ("today", "今天"):
        s = _start_of_day(now)
        return s, s + _dt.timedelta(days=1)
    if rn in ("tomorrow", "明天"):
        s = _start_of_day(now) + _dt.timedelta(days=1)
        return s, s + _dt.timedelta(days=1)
    if rn in ("this_week", "thisweek", "这周", "本周"):
        s = _start_of_week_monday(now)
        return s, s + _dt.timedelta(days=7)
    if rn in ("next_week", "nextweek", "下周"):
        s = _start_of_week_monday(now) + _dt.timedelta(days=7)
        return s, s + _dt.timedelta(days=7)
    if rn in ("next_7_days", "future_7_days", "未来7天", "未来 7 天"):
        s = _start_of_day(now)
        return s, s + _dt.timedelta(days=7)
    if rn in ("next_14_days", "未来14天", "未来 14 天"):
        s = _start_of_day(now)
        return s, s + _dt.timedelta(days=14)

    # default
    s = _start_of_day(now)
    da = int(days_ahead or 7)
    return s, s + _dt.timedelta(days=da)


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: Dict[str, Any]


class ToolError(Exception):
    pass


def _norm(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


class ToolSandbox:
    """
    Default policy:
    - Only allow filesystem access under allowed_roots.
    - Only allow running commands in allowed_cmd_prefixes (whitelist).
    """

    def __init__(
        self,
        allowed_roots: List[str],
        allowed_cmd_prefixes: Optional[List[str]] = None,
        max_read_bytes: int = 200_000,
        max_output_chars: int = 40_000,
        cwd: Optional[str] = None,
    ) -> None:
        self.allowed_roots = [_norm(r) for r in allowed_roots]
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
        self.cwd = _norm(cwd) if cwd else None

    def _check_path(self, path: str) -> str:
        raw = (path or "").strip()

        # If empty or placeholder, fallback to first allowed root
        placeholders = {"c:\\repo_path", "c:/repo_path", "repo_path", "<repo_path>", "{repo_path}"}
        if not raw or raw.lower() in placeholders:
            return self.allowed_roots[0]

        # If relative path, resolve under allowed root[0]
        if not os.path.isabs(raw):
            ap = _norm(os.path.join(self.allowed_roots[0], raw))
        else:
            ap = _norm(raw)

        for root in self.allowed_roots:
            if ap == root or ap.startswith(root + os.sep):
                return ap

        raise ToolError(
            f"Path not allowed by sandbox: {ap}\n"
            f"Allowed roots: {self.allowed_roots}\n"
            f"Tip: use paths under the allowed root, e.g. '{self.allowed_roots[0]}'"
        )


    def read_file(self, path: str, start: int = 0, limit: int = 50_000) -> ToolResult:
        ap = self._check_path(path)
        if not os.path.isfile(ap):
            return ToolResult(False, f"Not a file: {ap}", {"path": ap})

        # enforce byte limit
        size = os.path.getsize(ap)
        if size > self.max_read_bytes:
            # Still allow partial read, but warn
            warn = f"[WARN] file is large ({size} bytes); reading partial content.\n"
        else:
            warn = ""

        # Read as bytes then decode best-effort
        with open(ap, "rb") as f:
            f.seek(max(0, start))
            data = f.read(min(limit, self.max_read_bytes))

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")

        out = warn + text
        out = out[: self.max_output_chars]
        return ToolResult(True, out, {"path": ap, "start": start, "limit": limit, "size_bytes": size})

    def list_dir(self, path: str, max_items: int = 200) -> ToolResult:
        ap = self._check_path(path)
        if not os.path.isdir(ap):
            return ToolResult(False, f"Not a directory: {ap}", {"path": ap})

        items = []
        try:
            for name in os.listdir(ap):
                items.append(name)
        except Exception as e:
            return ToolResult(False, f"Failed to list dir: {e}", {"path": ap})

        items.sort()
        if len(items) > max_items:
            shown = items[:max_items]
            suffix = f"\n... ({len(items) - max_items} more items truncated)"
        else:
            shown = items
            suffix = ""

        out = "\n".join(shown) + suffix
        out = out[: self.max_output_chars]
        return ToolResult(True, out, {"path": ap, "count": len(items), "shown": min(len(items), max_items)})

    def run_cmd(self, cmd: str, timeout_sec: int = 60) -> ToolResult:
        """
        Runs a command string with whitelist enforcement.
        Windows-friendly: uses shell=True.
        """
        cmd_str = cmd.strip()
        if not cmd_str:
            return ToolResult(False, "Empty cmd", {"cmd": cmd_str})

        # Basic whitelist check on first token
        first = cmd_str.split()[0].lower()
        allowed = [p.lower() for p in self.allowed_cmd_prefixes]
        if first not in allowed:
            return ToolResult(
                False,
                f"Command not allowed by whitelist: '{first}'. Allowed: {self.allowed_cmd_prefixes}",
                {"cmd": cmd_str, "first": first},
            )

        # prevent obviously destructive patterns (extra guard)
        banned_substrings = ["del ", "rd ", "rmdir", "format", "diskpart", "reg delete", "shutdown", "reboot"]
        low = cmd_str.lower()
        for b in banned_substrings:
            if b in low:
                return ToolResult(False, f"Blocked potentially destructive command pattern: {b}", {"cmd": cmd_str})

        try:
            completed = subprocess.run(
                cmd_str,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            code = completed.returncode
            out = (stdout + ("\n[stderr]\n" + stderr if stderr.strip() else "")).strip()
            if len(out) > self.max_output_chars:
                out = out[: self.max_output_chars] + "\n... [output truncated]"
            ok = (code == 0)
            return ToolResult(ok, out if out else "(no output)", {"cmd": cmd_str, "returncode": code, "cwd": self.cwd})
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Command timed out after {timeout_sec}s", {"cmd": cmd_str, "timeout_sec": timeout_sec})
        except Exception as e:
            return ToolResult(False, f"Command failed: {e}", {"cmd": cmd_str})
        
    def analyze_ics(self, path: str, range: str | None = None, start: str | None = None, days_ahead: int = 7) -> ToolResult:
        ap = self._check_path(path)
        if not os.path.isfile(ap):
            return ToolResult(False, f"Not a file: {ap}", {"path": ap})

        with open(ap, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(min(self.max_read_bytes, 2_000_000))

        ws, we = resolve_time_window(range, start, days_ahead)
        summary = analyze_ics_text(text, ws, we)

        out = json.dumps(summary, ensure_ascii=False, indent=2)
        if len(out) > self.max_output_chars:
            out = out[: self.max_output_chars] + "\n... [output truncated]"
        return ToolResult(True, out, {"path": ap, "range": range, "start": start, "days_ahead": int(days_ahead)})


# --- ICS ANALYZER ------------------------------------------------------------
import datetime as _dt
import re as _re
from typing import Iterable as _Iterable

def _parse_dt(val: str) -> _dt.datetime:
    val = val.strip()

    # UTC form
    if val.endswith("Z"):
        core = val[:-1]
        for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
            try:
                dt = _dt.datetime.strptime(core, fmt)
                return dt.replace(tzinfo=_dt.timezone.utc).astimezone().replace(tzinfo=None)
            except ValueError:
                pass
        raise ValueError(f"Unrecognized UTC datetime: {val}")

    # local datetime form
    if "T" in val:
        for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
            try:
                return _dt.datetime.strptime(val, fmt)
            except ValueError:
                pass
        raise ValueError(f"Unrecognized local datetime: {val}")

    # date-only
    return _dt.datetime.strptime(val, "%Y%m%d")


def _unfold_ics_lines(lines: _Iterable[str]) -> list[str]:
    """
    iCalendar line unfolding: lines that start with space/tab continue previous line.
    """
    out = []
    for ln in lines:
        if ln.startswith((" ", "\t")) and out:
            out[-1] += ln[1:].rstrip("\n")
        else:
            out.append(ln.rstrip("\n"))
    return out

def _get_prop(line: str) -> tuple[str, str]:
    """
    Extract property name and value from something like:
    DTSTART;TZID=America/Los_Angeles:20260216T090000
    """
    if ":" not in line:
        return "", ""
    left, val = line.split(":", 1)
    name = left.split(";", 1)[0].strip().upper()
    return name, val.strip()

def analyze_ics_text(ics_text: str, window_start: _dt.datetime, window_end: _dt.datetime) -> dict:
    start = window_start
    end = window_end

    lines = _unfold_ics_lines(ics_text.splitlines())
    events = []
    in_evt = False
    evt = {}

    for ln in lines:
        if ln == "BEGIN:VEVENT":
            in_evt = True
            evt = {}
            continue
        if ln == "END:VEVENT" and in_evt:
            in_evt = False
            # normalize
            dtstart = evt.get("DTSTART")
            dtend = evt.get("DTEND")
            if dtstart:
                try:
                    s = _parse_dt(dtstart)
                except Exception:
                    s = None
            else:
                s = None
            if dtend:
                try:
                    e = _parse_dt(dtend)
                except Exception:
                    e = None
            else:
                e = None

            # filter only those intersecting [start, end)
            if s is not None:
                # all-day events may have dtend = next day 00:00
                if e is None:
                    e = s
                # make naive local (drop tzinfo if any)
                if getattr(s, "tzinfo", None) is not None:
                    s = s.astimezone().replace(tzinfo=None)
                if getattr(e, "tzinfo", None) is not None:
                    e = e.astimezone().replace(tzinfo=None)

                if not (e <= start or s >= end):
                    events.append({
                        "start": s.isoformat(timespec="minutes"),
                        "end": e.isoformat(timespec="minutes"),
                        "summary": evt.get("SUMMARY", "(no title)"),
                        "location": evt.get("LOCATION", ""),
                        "description": (evt.get("DESCRIPTION", "")[:500] + ("..." if len(evt.get("DESCRIPTION","")) > 500 else "")),
                        "uid": evt.get("UID", ""),
                    })
            continue

        if in_evt:
            name, val = _get_prop(ln)
            if not name:
                continue
            # keep first or append
            if name in evt and evt[name]:
                # concat multiline properties (DESCRIPTION sometimes)
                evt[name] = str(evt[name]) + "\n" + val
            else:
                evt[name] = val

    # Aggregate by day
    by_day = {}
    total_minutes = 0
    for ev in events:
        s = _dt.datetime.fromisoformat(ev["start"])
        e = _dt.datetime.fromisoformat(ev["end"])
        dur = max(0, int((e - s).total_seconds() // 60))
        total_minutes += dur
        day = s.date().isoformat()
        by_day.setdefault(day, {"events": [], "minutes": 0})
        by_day[day]["events"].append({**ev, "minutes": dur})
        by_day[day]["minutes"] += dur

    # Sort days/events
    for day in by_day:
        by_day[day]["events"].sort(key=lambda x: x["start"])

    days_sorted = sorted(by_day.keys())
    return {
        "window": {"start": start.isoformat(timespec="minutes"), "end": end.isoformat(timespec="minutes")},
        "days": [{"date": d, **by_day[d]} for d in days_sorted],
        "totals": {"events": len(events), "minutes": total_minutes, "hours": round(total_minutes / 60.0, 2)},
    }
