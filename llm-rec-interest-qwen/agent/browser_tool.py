# -*- coding: utf-8 -*-
"""
Browser tool powered by Playwright.

Provides two high-level operations:
  - fetch_page(url)  → navigate and extract text + interactive elements
  - act(action, target, value) → click / type / scroll / back / goto / screenshot
"""
from __future__ import annotations

import base64
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InteractiveElement:
    kind: str          # "button" | "link" | "input" | "select" | "textarea"
    text: str          # visible label / placeholder / inner text
    selector: str      # CSS selector usable with page.locator()
    href: str = ""     # filled for links
    input_type: str = ""  # filled for <input>

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "text": self.text, "selector": self.selector}
        if self.href:
            d["href"] = self.href
        if self.input_type:
            d["input_type"] = self.input_type
        return d


@dataclass
class PageSnapshot:
    url: str
    title: str
    text: str
    elements: List[InteractiveElement] = field(default_factory=list)

    def summary(self, max_text: int = 1500, max_elements: int = 25, max_total: int = 3000) -> str:
        lines = [
            f"URL   : {self.url}",
            f"Title : {self.title}",
            "",
            "── Page Text ──────────────────────────────────────────",
            textwrap.shorten(self.text, width=max_text, placeholder="… [truncated]"),
            "",
            "── Interactive Elements ────────────────────────────────",
        ]
        for i, el in enumerate(self.elements[:max_elements], 1):
            desc = f"  [{i}] {el.kind.upper()} | {el.text!r}"
            if el.href:
                short_href = el.href[:80] + ("…" if len(el.href) > 80 else "")
                desc += f" → {short_href}"
            if el.input_type:
                desc += f" (type={el.input_type})"
            desc += f"  [sel: {el.selector}]"
            lines.append(desc)
        if len(self.elements) > max_elements:
            lines.append(f"  … and {len(self.elements) - max_elements} more elements (use browser_act snapshot to see more)")
        result = "\n".join(lines)
        if len(result) > max_total:
            result = result[:max_total] + "\n… [output truncated to fit context]"
        return result


# ---------------------------------------------------------------------------
# BrowserSession — wraps a single Playwright chromium window
# ---------------------------------------------------------------------------

class BrowserSession:
    """Lazy-initialized, persistent Playwright browser session."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._page = None

    # ---- lifecycle --------------------------------------------------------

    def _ensure_running(self) -> None:
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright  # type: ignore
        self._playwright = sync_playwright().__enter__()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        self._page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.__exit__(None, None, None)
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._playwright = None

    # ---- core operations --------------------------------------------------

    def fetch_page(self, url: str, timeout_ms: int = 20_000) -> PageSnapshot:
        """Navigate to *url* and return a PageSnapshot with text + elements."""
        self._ensure_running()
        page = self._page

        page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=5_000)
        except Exception:
            pass

        title = page.title()
        current_url = page.url
        text = _extract_text(page)
        elements = _extract_interactive(page)

        return PageSnapshot(url=current_url, title=title, text=text, elements=elements)

    def act(
        self,
        action: str,
        target: str = "",
        value: str = "",
        timeout_ms: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Perform a browser action.

        action  target              value       description
        ------  ------              -----       -----------
        click   selector or text    –           click an element
        type    selector or text    text to type  clear + type into input
        scroll  "up"/"down"/px      –           scroll the page
        back    –                   –           navigate back
        goto    url                 –           navigate to url
        snapshot –                  –           refresh snapshot of current page
        screenshot –                file path   save PNG (optional path)
        """
        self._ensure_running()
        page = self._page
        action = action.lower().strip()

        if action == "goto":
            page.goto(target, timeout=timeout_ms, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:
                pass
            return {"ok": True, "url": page.url, "title": page.title()}

        if action == "back":
            page.go_back(timeout=timeout_ms, wait_until="domcontentloaded")
            return {"ok": True, "url": page.url, "title": page.title()}

        if action == "scroll":
            amount = _parse_scroll(target or value)
            page.evaluate(f"window.scrollBy(0, {amount})")
            return {"ok": True, "scrolled": amount}

        if action == "snapshot":
            snap = PageSnapshot(
                url=page.url,
                title=page.title(),
                text=_extract_text(page),
                elements=_extract_interactive(page),
            )
            return {"ok": True, "snapshot": snap.summary(max_text=1500, max_elements=25, max_total=3000)}

        if action == "screenshot":
            path = value or os.path.join(os.path.dirname(__file__), "..", "outputs", "screenshot.png")
            path = os.path.abspath(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            page.screenshot(path=path, full_page=False)
            return {"ok": True, "path": path}

        if action == "click":
            locator = _resolve_locator(page, target, timeout_ms)
            locator.click(timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:
                pass
            return {"ok": True, "url": page.url, "title": page.title()}

        if action == "type":
            locator = _resolve_locator(page, target, timeout_ms)
            locator.click(timeout=timeout_ms)
            locator.fill(value)
            return {"ok": True, "typed": value}

        return {"ok": False, "error": f"Unknown action: {action!r}"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_text(page) -> str:
    """Extract readable text from the page body, stripping scripts/styles."""
    raw = page.evaluate(r"""
        () => {
            const clone = document.body.cloneNode(true);
            for (const el of clone.querySelectorAll('script,style,noscript,svg,img')) {
                el.remove();
            }
            return (clone.innerText || clone.textContent || '').replace(/\s{3,}/g, '\n\n').trim();
        }
    """)
    return (raw or "").strip()


def _extract_interactive(page) -> List[InteractiveElement]:
    """Extract visible interactive elements from the current page."""
    raw: List[Dict[str, Any]] = page.evaluate(r"""
        () => {
            const results = [];
            const seen = new Set();

            function label(el) {
                const t = (el.innerText || el.textContent || '').trim().replace(/\s+/g, ' ');
                return t.slice(0, 80) || el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('name') || el.getAttribute('placeholder') || '';
            }

            function cssPath(el) {
                if (el.id) return '#' + CSS.escape(el.id);
                const parts = [];
                let cur = el;
                while (cur && cur !== document.body) {
                    let seg = cur.tagName.toLowerCase();
                    if (cur.className) {
                        const cls = [...cur.classList].slice(0, 2).map(c => '.' + CSS.escape(c)).join('');
                        if (cls) seg += cls;
                    }
                    const idx = [...(cur.parentElement?.children || [])].indexOf(cur) + 1;
                    if (idx > 1) seg += `:nth-child(${idx})`;
                    parts.unshift(seg);
                    cur = cur.parentElement;
                    if (parts.length >= 4) break;
                }
                return parts.join(' > ');
            }

            function isVisible(el) {
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
            }

            // Links
            for (const el of document.querySelectorAll('a[href]')) {
                if (!isVisible(el)) continue;
                const t = label(el);
                const href = el.href || '';
                const key = 'link:' + t + href;
                if (seen.has(key)) continue;
                seen.add(key);
                results.push({ kind: 'link', text: t, selector: cssPath(el), href });
            }

            // Buttons
            for (const el of document.querySelectorAll('button,[role="button"]')) {
                if (!isVisible(el)) continue;
                const t = label(el);
                const key = 'btn:' + t;
                if (seen.has(key)) continue;
                seen.add(key);
                results.push({ kind: 'button', text: t, selector: cssPath(el) });
            }

            // Inputs
            for (const el of document.querySelectorAll('input:not([type="hidden"]),textarea')) {
                if (!isVisible(el)) continue;
                const t = label(el) || el.getAttribute('placeholder') || '';
                const key = 'input:' + t;
                if (seen.has(key)) continue;
                seen.add(key);
                results.push({ kind: el.tagName.toLowerCase() === 'textarea' ? 'textarea' : 'input',
                                text: t, selector: cssPath(el), input_type: el.getAttribute('type') || 'text' });
            }

            // Selects
            for (const el of document.querySelectorAll('select')) {
                if (!isVisible(el)) continue;
                const t = label(el);
                const key = 'sel:' + t;
                if (seen.has(key)) continue;
                seen.add(key);
                results.push({ kind: 'select', text: t, selector: cssPath(el) });
            }

            return results;
        }
    """)

    elements = []
    for r in (raw or []):
        elements.append(InteractiveElement(
            kind=r.get("kind", ""),
            text=r.get("text", ""),
            selector=r.get("selector", ""),
            href=r.get("href", ""),
            input_type=r.get("input_type", ""),
        ))
    return elements


def _resolve_locator(page, target: str, timeout_ms: int):
    """
    Try to resolve *target* as a CSS selector first; fall back to visible text match.
    """
    # CSS-like selector heuristic
    if re.match(r"^[#\.\[\(]|^[a-z][\w\-]*(\s*[>+~]\s*|\:)", target) or " > " in target:
        return page.locator(target).first

    # Try as a CSS selector quietly
    try:
        loc = page.locator(target).first
        loc.wait_for(state="visible", timeout=2_000)
        return loc
    except Exception:
        pass

    # Fall back to text match
    return page.get_by_text(target, exact=False).first


def _parse_scroll(s: str) -> int:
    s = s.strip().lower()
    if s in ("", "down"):
        return 600
    if s == "up":
        return -600
    try:
        return int(s)
    except ValueError:
        return 600
