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
    )
except ImportError:
    from config import (
        SERPAPI_API_KEY,
        WEB_SEARCH_MAX_RESULTS,
        WEB_SEARCH_PROVIDER,
        WEB_SEARCH_TIMEOUT_SEC,
        WEB_SEARCH_USER_AGENT,
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