# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

try:
    from .prompts import SYSTEM_PROMPT, ASSISTANT_STYLE, CHAT_SYSTEM_PROMPT
    from .tools import ToolSandbox, ToolResult
    from .config import (
        LM_STUDIO_BASE_URL,
        LM_STUDIO_API_KEY,
        LM_STUDIO_MODEL_NAME,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
    )
except ImportError:
    from prompts import SYSTEM_PROMPT, ASSISTANT_STYLE, CHAT_SYSTEM_PROMPT
    from tools import ToolSandbox, ToolResult
    from config import (
        LM_STUDIO_BASE_URL,
        LM_STUDIO_API_KEY,
        LM_STUDIO_MODEL_NAME,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
    )


_JSON_LINE_RE = re.compile(r"^\s*\{.*\}\s*$")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2}, 20\d{2})\b")


@dataclass
class AgentConfig:
    base_url: str = LM_STUDIO_BASE_URL
    api_key: str = LM_STUDIO_API_KEY
    model_name: str = LM_STUDIO_MODEL_NAME
    max_new_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P


class LocalAgent:
    def __init__(self, cfg: AgentConfig, sandbox: Optional[ToolSandbox] = None) -> None:
        self.cfg = cfg
        self.sandbox = sandbox

        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
        )

        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": ASSISTANT_STYLE},
        ]

        self.skills = {
            "chat": self.chat_simple,
            "tool": self.chat_with_tools,
        }

    def _chat_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.cfg.model_name,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_new_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

    def _history_without_system(self) -> List[Dict[str, str]]:
        return [m for m in self.messages if m.get("role") != "system"]

    def _looks_time_sensitive(self, text: str) -> bool:
        t = (text or "").lower()
        keywords = [
            "news", "latest", "recent", "today", "current", "update", "updates",
            "breaking", "headline", "headlines",
            "market", "stock", "stocks", "earnings", "price", "prices",
            "war", "conflict", "iran", "israel", "ukraine", "tariff", "fed",
        ]
        return any(k in t for k in keywords)

    def _needs_tool_search(self, text: str) -> bool:
        return self._looks_time_sensitive(text)

    def _answer_has_date(self, text: str) -> bool:
        return bool(_DATE_RE.search(text or ""))

    def _tool_result_uses_search_web(self) -> bool:
        for m in self.messages:
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            if not content.startswith("TOOL_RESULT:\n"):
                continue
            if '"tool": "search_web"' in content or '"tool":"search_web"' in content:
                return True
        return False

    def chat_simple(self, user_text: str) -> str:
        temp_messages: List[Dict[str, str]] = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        ]
        temp_messages.extend(self._history_without_system())
        temp_messages.append({"role": "user", "content": user_text})

        response = self.client.chat.completions.create(
            messages=temp_messages,
            **self._chat_kwargs(),
        )
        text = response.choices[0].message.content or ""
        answer = text.strip()

        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def stream_chat_simple(self, user_text: str, stop_event=None) -> Iterator[str]:
        temp_messages: List[Dict[str, str]] = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        ]
        temp_messages.extend(self._history_without_system())
        temp_messages.append({"role": "user", "content": user_text})

        stream = self.client.chat.completions.create(
            messages=temp_messages,
            stream=True,
            **self._chat_kwargs(),
        )

        collected: List[str] = []
        try:
            for event in stream:
                if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                    break

                if not event.choices:
                    continue

                delta = getattr(event.choices[0], "delta", None)
                if delta is None:
                    continue

                token = getattr(delta, "content", None)
                if token:
                    collected.append(token)
                    yield token
        finally:
            try:
                stream.close()
            except Exception:
                pass

        final_text = "".join(collected).strip()
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": final_text})

    def route(self, user_text: str, forced_skill: Optional[str] = None) -> str:
        if forced_skill and forced_skill != "auto":
            if forced_skill not in self.skills:
                return "chat"
            return forced_skill

        t = (user_text or "").lower()

        if t.startswith("tool:"):
            return "tool"
        if t.startswith("chat:"):
            return "chat"

        if self._needs_tool_search(user_text):
            return "tool"

        return "chat"

    def _generate(self) -> str:
        response = self.client.chat.completions.create(
            messages=self.messages,
            **self._chat_kwargs(),
        )
        text = response.choices[0].message.content or ""
        return text.strip()

    def _try_parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None

        if len(lines) == 1 and _JSON_LINE_RE.match(lines[0]):
            try:
                obj = json.loads(lines[0])
                if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                    return obj
            except Exception:
                return None

        for ln in lines:
            if _JSON_LINE_RE.match(ln):
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                        return obj
                except Exception:
                    continue
        return None

    def _run_tool(self, tool: str, args: Dict[str, Any]) -> ToolResult:
        if self.sandbox is None:
            return ToolResult(False, "Tool sandbox is not configured.", {"tool": tool})

        if tool == "read_file":
            return self.sandbox.read_file(
                path=str(args.get("path", "")),
                start=int(args.get("start", 0)),
                limit=int(args.get("limit", 50_000)),
            )

        if tool == "list_dir":
            return self.sandbox.list_dir(
                path=str(args.get("path", "")),
                max_items=int(args.get("max_items", 200)),
            )

        if tool == "run_cmd":
            return self.sandbox.run_cmd(
                cmd=str(args.get("cmd", "")),
                timeout_sec=int(args.get("timeout_sec", 60)),
            )

        if tool == "analyze_ics":
            return self.sandbox.analyze_ics(
                path=str(args.get("path", "")),
                range=args.get("range", None),
                start=args.get("start", None),
                days_ahead=int(args.get("days_ahead", 7)),
            )

        if tool == "search_web":
            if not hasattr(self.sandbox, "search_web"):
                return ToolResult(False, "search_web is not implemented in ToolSandbox.", {"tool": tool})
            return self.sandbox.search_web(
                query=str(args.get("query", "")),
                max_results=int(args.get("max_results", 5)),
            )

        return ToolResult(False, f"Unknown tool: {tool}", {"tool": tool})

    def _append_tool_result(self, tool_name: str, result: ToolResult):
        obs = {
            "tool": tool_name,
            "ok": result.ok,
            "output": result.output,
            "meta": result.meta,
        }

        extra_instruction = ""
        if tool_name == "search_web":
            extra_instruction = (
                "\n\nNow write the final answer or call another tool if needed.\n"
                "Response quality rules:\n"
                "1. For EACH summarized item, include the EXACT source date.\n"
                "2. Prefer concise summaries with a few concrete points.\n"
                "3. If results are weak, mixed, conflicting, or outdated, say so clearly.\n"
                "4. Prefer format: [YYYY-MM-DD] Source - concise summary.\n"
            )

        self.messages.append(
            {
                "role": "user",
                "content": "TOOL_RESULT:\n" + json.dumps(obs, ensure_ascii=False) + extra_instruction,
            }
        )

    def _force_search_web_first(self, user_text: str) -> bool:
        """
        For clearly time-sensitive / news-like questions, inject an initial search_web call
        instead of hoping the model decides to do it.
        """
        if not self._needs_tool_search(user_text):
            return False

        result = self._run_tool("search_web", {"query": user_text, "max_results": 5})
        self._append_tool_result("search_web", result)
        return True

    def _rewrite_if_missing_dates_after_search(self, final_answer: str) -> str:
        if not self._tool_result_uses_search_web():
            return final_answer

        if self._answer_has_date(final_answer):
            return final_answer

        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous answer did not follow the required response quality rules. "
                    "Rewrite it now so that EACH summarized item includes the EXACT source date. "
                    "Keep it concise. If results are weak or mixed, say so explicitly."
                ),
            }
        )
        rewritten = self._generate().strip()
        return rewritten

    def chat(self, user_text: str, max_steps: int = 6, skill: str = "auto") -> str:
        chosen = self.route(user_text, forced_skill=skill)

        if user_text.lower().startswith("chat:"):
            user_text = user_text[5:].strip()
        elif user_text.lower().startswith("tool:"):
            user_text = user_text[5:].strip()

        if chosen == "tool":
            return self.chat_with_tools(user_text=user_text, max_steps=max_steps)

        return self.chat_simple(user_text=user_text)

    def chat_with_tools(self, user_text: str, max_steps: int = 6) -> str:
        if self.sandbox is None:
            return self.chat_simple(user_text)

        # Optional helpful bootstrap for repo questions
        if any(k in user_text.lower() for k in ["repo", "project", "结构", "目录"]):
            forced = self.sandbox.list_dir(self.sandbox.allowed_roots[0])
            self._append_tool_result("list_dir", forced)

        self.messages.append({"role": "user", "content": user_text})

        # Force web search first for time-sensitive/current-event questions
        self._force_search_web_first(user_text)

        final_answer = ""

        for _ in range(max_steps):
            model_text = self._generate()
            tool_call = self._try_parse_tool_call(model_text)

            if not tool_call:
                self.messages.append({"role": "assistant", "content": model_text})
                final_answer = model_text
                break

            tool_name = str(tool_call.get("tool"))
            tool_args = tool_call.get("args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"value": tool_args}

            self.messages.append(
                {"role": "assistant", "content": json.dumps(tool_call, ensure_ascii=False)}
            )

            try:
                result = self._run_tool(tool_name, tool_args)
            except Exception as e:
                result = ToolResult(
                    False,
                    f"Tool exception: {e}",
                    {"tool": tool_name, "args": tool_args},
                )

            self._append_tool_result(tool_name, result)

            if not result.ok:
                continue

        if not final_answer:
            final_answer = "Reached max_steps without a final answer."

        final_answer = self._rewrite_if_missing_dates_after_search(final_answer)

        # keep rewritten final answer in history
        if not self.messages or self.messages[-1].get("role") != "assistant":
            self.messages.append({"role": "assistant", "content": final_answer})
        else:
            self.messages[-1]["content"] = final_answer

        return final_answer