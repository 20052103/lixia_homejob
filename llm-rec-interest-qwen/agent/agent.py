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

        # session memory
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

        t = user_text.lower()
        if t.startswith("tool:"):
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

        return ToolResult(False, f"Unknown tool: {tool}", {"tool": tool})

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

        if any(k in user_text.lower() for k in ["repo", "project", "结构", "目录"]):
            forced = self.sandbox.list_dir(self.sandbox.allowed_roots[0])
            self.messages.append(
                {
                    "role": "user",
                    "content": "TOOL_RESULT:\n"
                    + str(
                        {
                            "tool": "list_dir",
                            "ok": forced.ok,
                            "output": forced.output,
                            "meta": forced.meta,
                        }
                    ),
                }
            )

        self.messages.append({"role": "user", "content": user_text})
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

            obs = {
                "tool": tool_name,
                "ok": result.ok,
                "output": result.output,
                "meta": result.meta,
            }
            self.messages.append(
                {
                    "role": "user",
                    "content": "TOOL_RESULT:\n" + json.dumps(obs, ensure_ascii=False),
                }
            )

            if not result.ok:
                continue

        if not final_answer:
            final_answer = "Reached max_steps without a final answer."

        return final_answer