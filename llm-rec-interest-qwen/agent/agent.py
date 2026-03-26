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
        GMAIL_CREDENTIALS_PATH,
        GMAIL_TOKEN_PATH,
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
        GMAIL_CREDENTIALS_PATH,
        GMAIL_TOKEN_PATH,
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

        # IMPORTANT:
        # Keep ONLY non-system history here.
        self.messages: List[Dict[str, str]] = []

        # Extra runtime system notes injected by run_agent.py / voice pipeline.
        self.extra_system_messages: List[str] = []

        self.skills = {
            "chat": self.chat_simple,
            "tool": self.chat_with_tools,
        }

    # ------------------------------------------------------------
    # Pending email confirmation helpers
    # ------------------------------------------------------------
    _CONFIRM_KEYWORDS = ["确认", "确认发送", "发送", "yes", "ok", "send", "好的", "行", "可以", "没问题"]

    def _is_send_confirmation(self, text: str) -> bool:
        t = text.strip().lower()
        return any(k in t for k in self._CONFIRM_KEYWORDS)

    def _try_auto_send_pending(self, user_text: str) -> Optional[str]:
        """If there's a pending draft and the user confirms, send immediately without LLM."""
        if self.sandbox is None:
            return None
        pending = getattr(self.sandbox, "pending_email", None)
        if not pending:
            return None
        if not self._is_send_confirmation(user_text):
            # User is modifying — clear the pending draft and let LLM handle
            self.sandbox.pending_email = None
            return None

        # Execute send directly
        print(f"[auto_send] confirmed — sending pending email to {pending['to']!r}")

        original = pending.get("_reply_to")  # set when replying to an existing email
        if original is not None:
            # Use reply_email for proper threading
            try:
                from .gmail_tool import reply_email
            except ImportError:
                from gmail_tool import reply_email
            try:
                output = reply_email(
                    original=original,
                    body=pending["body"],
                    credentials_path=GMAIL_CREDENTIALS_PATH,
                    token_path=GMAIL_TOKEN_PATH,
                )
                result_ok = True
                result_output = output
            except Exception as e:
                result_ok = False
                result_output = str(e)
        else:
            result = self.sandbox.send_gmail(
                to=pending["to"],
                subject=pending["subject"],
                body=pending["body"],
                cc=pending.get("cc", ""),
            )
            result_ok = result.ok
            result_output = result.output

        self.sandbox.pending_email = None  # clear after send

        if result_ok:
            reply = (
                f"✅ 邮件已发送！\n"
                f"收件人：{pending['to']}\n"
                f"主题：{pending['subject']}"
            )
        else:
            reply = f"❌ 发送失败：{result_output}"

        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _try_auto_confirm_cal_event(self, user_text: str) -> Optional[str]:
        """If there's a pending calendar draft and the user confirms, create it immediately."""
        if self.sandbox is None:
            return None
        pending = getattr(self.sandbox, "pending_cal_event", None)
        if not pending:
            return None
        if not self._is_send_confirmation(user_text):
            # User wants to modify — clear draft, let LLM handle
            self.sandbox.pending_cal_event = None
            return None

        print(f"[auto_cal] confirmed — creating event {pending.get('title')!r}")
        result = self.sandbox.confirm_calendar_event()
        reply = result.output if result.ok else f"❌ 日程创建失败：{result.output}"
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    _REPLY_ID_RE = re.compile(r"(?:reply|回复)\s*[eE](\d+)", re.IGNORECASE)

    def _try_auto_reply_draft(self, user_text: str) -> Optional[str]:
        """
        If the user says 'reply E1 ...' / '回复E2 ...' and E1 is in the email cache,
        extract the body from their message, call reply_gmail directly, and return the
        draft preview — bypassing the LLM entirely to avoid hallucinated recipients.
        """
        if self.sandbox is None:
            return None
        m = self._REPLY_ID_RE.search(user_text)
        if not m:
            return None
        email_id = f"E{m.group(1)}"
        if email_id not in self.sandbox.email_cache:
            return None  # unknown ID — let LLM handle with error

        # Extract body: everything after "E{N}" in user text, stripped
        body_hint = user_text[m.end():].strip().lstrip("，,：:、 ")
        if not body_hint:
            body_hint = "(请在正文里写回复内容)"

        # Use LLM to polish the body only (simple chat, no tools)
        original_email = self.sandbox.email_cache.get(email_id)
        original_context = ""
        if original_email:
            original_context = (
                f"\n\nOriginal email you are replying to:\n"
                f"From: {original_email.sender}\n"
                f"Subject: {original_email.subject}\n"
                f"Body:\n{original_email.body or original_email.snippet}\n"
            )
        polish_prompt = (
            f"Draft a polite, concise reply email body based on the user's intent. "
            f"Do NOT include To/Subject/From headers — just the body text. "
            f"Base the greeting and context ONLY on the original email below — do NOT invent names or context from elsewhere. "
            f"End with 'Li' on its own line. User intent: {body_hint}"
            f"{original_context}"
        )
        try:
            polished_body = self.chat_simple(polish_prompt)
        except Exception:
            polished_body = body_hint + "\n\nLi"

        result = self.sandbox.reply_gmail(email_id=email_id, body=polished_body)
        reply = result.output
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    # Patterns that suggest user wants to read a specific drive file by index
    _READ_FILE_RE = re.compile(
        r"(?:read|open|show|view|see|看|读取|打开)\s*(?:file\s*|it\s*|the\s*)?(?:number\s*)?(?:#?\s*(\d+))?",
        re.IGNORECASE,
    )

    def _try_auto_read_drive_file(self, user_text: str) -> Optional[str]:
        """
        If the user says 'read file 1' / 'read it' / 'open the first file' and we have
        cached drive search results, call read_drive_file directly without LLM.
        Also intercepts cases where no specific number is given but only 1 result exists.
        """
        if self.sandbox is None:
            return None
        files = getattr(self.sandbox, "last_drive_files", [])
        if not files:
            return None

        t = user_text.lower()
        # Must look like a read/open intent
        read_keywords = ["read", "open", "show me the file", "view file", "read it",
                         "open it", "read the", "open the", "看", "读取", "打开"]
        if not any(k in t for k in read_keywords):
            return None

        # Determine which file to read
        m = re.search(r"(?:file\s*|#\s*|number\s*)(\d+)", t)
        if m:
            idx = int(m.group(1)) - 1  # 1-based → 0-based
        elif re.search(r"\bfirst\b|第一", t):
            idx = 0
        elif re.search(r"\bsecond\b|第二", t):
            idx = 1
        elif re.search(r"\bthird\b|第三", t):
            idx = 2
        elif len(files) == 1:
            idx = 0  # only one result — obviously "it"
        else:
            return None  # ambiguous with multiple results — let LLM decide

        if idx < 0 or idx >= len(files):
            return None

        target = files[idx]
        print(f"[auto_read_drive] reading file {idx+1}: {target.name!r} ({target.file_id})")
        result = self.sandbox.read_drive_file(file_id=target.file_id)
        reply = result.output
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------
    # System-message handling
    # ------------------------------------------------------------
    def add_system_message(self, content: str) -> None:
        content = (content or "").strip()
        if content:
            self.extra_system_messages.append(content)

    def _merge_system_prompt(self, base_prompt: str) -> str:
        parts: List[str] = []

        if base_prompt and base_prompt.strip():
            parts.append(base_prompt.strip())

        # Keep assistant style inside the SAME system message.
        if base_prompt == SYSTEM_PROMPT and ASSISTANT_STYLE.strip():
            parts.append(ASSISTANT_STYLE.strip())

        # Inject VIP contacts so the LLM always knows real email addresses
        if base_prompt == SYSTEM_PROMPT:
            try:
                try:
                    from .contacts import load_vip_contacts
                except ImportError:
                    from contacts import load_vip_contacts
                contacts = load_vip_contacts()
                if contacts:
                    lines = ["## Known Contacts (use these exact email addresses)"]
                    for c in contacts:
                        lines.append(f"- {c.name}: {c.email}")
                    parts.append("\n".join(lines))
            except Exception:
                pass

        for msg in self.extra_system_messages:
            if msg and msg.strip():
                parts.append(msg.strip())

        return "\n\n".join(parts).strip()

    def _normalized_history(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in self.messages:
            role = str(m.get("role", "")).strip()
            if role == "system":
                continue
            if role not in {"user", "assistant"}:
                continue
            content = m.get("content", "")
            if content is None:
                content = ""
            out.append({"role": role, "content": str(content)})
        return out

    def _prune_old_messages(self, max_recent_pairs: int = 8) -> None:
        """
        Keep only the most recent message pairs (user+assistant) to avoid exceeding context limit.
        With max_recent_pairs=8, keeps last 16 messages (8 user + 8 assistant).
        """
        if len(self.messages) <= max_recent_pairs * 2:
            return
        
        # Keep the most recent N pairs
        self.messages = self.messages[-(max_recent_pairs * 2):]

    def _build_messages(
        self,
        *,
        base_system_prompt: str,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        # Prune old messages to prevent context overflow
        self._prune_old_messages(max_recent_pairs=8)
        
        final_messages: List[Dict[str, str]] = []

        merged_system = self._merge_system_prompt(base_system_prompt)
        if merged_system:
            final_messages.append({"role": "system", "content": merged_system})

        final_messages.extend(self._normalized_history())

        if extra_messages:
            for m in extra_messages:
                role = str(m.get("role", "")).strip()
                if role == "system":
                    continue
                if role not in {"user", "assistant"}:
                    continue
                content = m.get("content", "")
                if content is None:
                    content = ""
                final_messages.append({"role": role, "content": str(content)})

        return final_messages

    # ------------------------------------------------------------
    # Model call helpers
    # ------------------------------------------------------------
    def _chat_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.cfg.model_name,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_new_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        }

    def _needs_search(self, text: str) -> bool:
        """Check if web search is explicitly needed."""
        t = (text or "").lower()
        keywords = [
            "search",      # English "search"
            "搜索",        # Chinese "search"
            "news",        # "news"
        ]
        return any(k in t for k in keywords)

    def _needs_file_tools(self, text: str) -> bool:
        """Check if file/directory operations are needed."""
        t = (text or "").lower()
        keywords = [
            "列出",        # Chinese "list"
            "list",        # English "list"
            "show",        # "show"
            "查看",        # Chinese "view/check"
            "找",          # Chinese "find"
            "find",        # English "find"
            "打开",        # Chinese "open"
            "open",        # English "open"
            "读取",        # Chinese "read"
            "read",        # English "read"
            "文件",        # Chinese "file"
            "目录",        # Chinese "directory"
            "结构",        # Chinese "structure"
        ]
        return any(k in t for k in keywords)

    def _looks_time_sensitive(self, text: str) -> bool:
        """Legacy: only for search."""
        return self._needs_search(text)

    def _needs_gmail(self, text: str) -> bool:
        """Check if Gmail fetching or sending is needed."""
        t = (text or "").lower()
        keywords = [
            "邮件", "邮箱", "收件箱", "未读",                  # Chinese email keywords
            "发邮件", "写信", "回复", "回邮件",                  # Chinese send/reply keywords
            "回复e", "reply e",                                   # reply by ID (e.g. 回复E1)
            "vip联系人", "vip 联系人", "联系人", "discover",      # contact discovery
            "gmail", "email", "mail", "inbox", "unread",          # English read
            "send email", "write to", "reply to", "reply e",      # English send/reply
            "总结邮件", "查邮件", "看邮件",                       # Chinese phrases
        ]
        # Also detect "reply E1" / "回复 E2" patterns
        import re as _re
        if _re.search(r"(reply|回复)\s*[eE]\d+", t):
            return True
        return any(k in t for k in keywords)

    def _needs_tool_search(self, text: str) -> bool:
        """Check if tool use is needed (file ops, search, gmail, calendar, or drive)."""
        return (
            self._needs_search(text)
            or self._needs_file_tools(text)
            or self._needs_gmail(text)
            or self._needs_calendar(text)
            or self._needs_drive(text)
        )

    def _needs_drive(self, text: str) -> bool:
        """Check if Google Drive access is needed."""
        t = (text or "").lower()
        keywords = [
            "drive", "google drive", "gdrive",
            "doc", "docs", "google doc", "spreadsheet", "sheet",
            "slides", "presentation",
            "find file", "search file", "find doc", "find my",
            "read file", "open file",
            "文件", "谷歌文档", "云盘",
        ]
        return any(k in t for k in keywords)

    def _needs_calendar(self, text: str) -> bool:
        """Check if Google Calendar fetching is needed."""
        t = (text or "").lower()
        keywords = [
            "calendar", "schedule", "日历", "日程", "行程",
            "today", "tomorrow", "this week", "next week",
            "今天", "明天", "本周", "这周", "下周",
            "what's on", "what do i have", "any meetings",
            "meeting", "event", "appointment",
            "会议", "事件", "安排",
        ]
        return any(k in t for k in keywords)

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

    def _looks_like_search_disclaimer(self, text: str) -> bool:
        t = (text or "").lower()
        bad_patterns = [
            "i apologize",
            "unable to perform web searches",
            "unable to browse",
            "cannot browse",
            "can't browse",
            "cannot access the internet",
            "can't access the internet",
            "do not have access to the internet",
            "i do not have access to the internet",
            "i cannot perform live google searches",
            "i'm unable to perform web searches",
        ]
        return any(p in t for p in bad_patterns)

    def _generate(self) -> str:
        messages = self._build_messages(base_system_prompt=SYSTEM_PROMPT)
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **self._chat_kwargs(),
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                err = str(e)
                if "exceed_context_size" in err or "context size" in err.lower() or "400" in err:
                    # Drop the oldest non-system messages (keep system at index 0)
                    non_sys = [m for m in messages if m.get("role") != "system"]
                    if len(non_sys) <= 2:
                        raise  # can't trim further
                    # Remove 2 oldest non-system messages
                    to_drop = non_sys[:2]
                    for m in to_drop:
                        if m in messages:
                            messages.remove(m)
                    # Also trim from self.messages to keep state consistent
                    for m in to_drop:
                        if m in self.messages:
                            self.messages.remove(m)
                else:
                    raise
        return ""

    # ------------------------------------------------------------
    # Plain chat
    # ------------------------------------------------------------
    def chat_simple(self, user_text: str) -> str:
        response = self.client.chat.completions.create(
            messages=self._build_messages(
                base_system_prompt=CHAT_SYSTEM_PROMPT,
                extra_messages=[{"role": "user", "content": user_text}],
            ),
            **self._chat_kwargs(),
        )
        text = response.choices[0].message.content or ""
        answer = text.strip()

        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def stream_chat_simple(self, user_text: str, stop_event=None) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            messages=self._build_messages(
                base_system_prompt=CHAT_SYSTEM_PROMPT,
                extra_messages=[{"role": "user", "content": user_text}],
            ),
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

    # ------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------
    # Patterns for detecting a plain-text email draft the LLM writes instead of calling draft_gmail
    _EMAIL_TO_RE = re.compile(r"To\s*[:：]\s*(.+)", re.IGNORECASE)
    _EMAIL_SUBJ_RE = re.compile(r"Subject\s*[:：]\s*(.+)", re.IGNORECASE)
    _EMAIL_BODY_RE = re.compile(r"Body\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)

    def _extract_email_draft_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        If the LLM generates a plain-text email draft (To/Subject/Body lines) instead of
        calling draft_gmail, parse it out so we can call draft_gmail ourselves.
        """
        to_m = self._EMAIL_TO_RE.search(text)
        sub_m = self._EMAIL_SUBJ_RE.search(text)
        if not to_m or not sub_m:
            return None

        to_val = to_m.group(1).strip().rstrip("---").strip()
        subject_val = sub_m.group(1).strip()

        # Extract body: everything between "Body :" and the next "---" separator or end
        body_val = ""
        body_m = re.search(r"Body\s*[:：]\s*([\s\S]+?)(?:---|$)", text, re.IGNORECASE)
        if body_m:
            body_val = body_m.group(1).strip()
        else:
            # fallback: everything after Subject line
            after_subj = text[sub_m.end():]
            body_val = after_subj.strip()

        if not body_val:
            return None

        return {"tool": "draft_gmail", "args": {"to": to_val, "subject": subject_val, "body": body_val}}

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

        # Single-line scan (body with \n escaped keeps JSON on one line)
        for ln in lines:
            if _JSON_LINE_RE.match(ln):
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                        return obj
                except Exception:
                    continue

        # Multi-line fallback: LLM sometimes emits JSON with literal newlines in body field.
        # Try to parse the whole block. Find the outermost {...} span and parse it.
        raw = text.strip()
        start = raw.find("{")
        if start != -1:
            # Walk from the end to find the matching closing brace
            depth = 0
            end = -1
            in_str = False
            escape = False
            for i, ch in enumerate(raw[start:], start):
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_str:
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end != -1:
                candidate = raw[start:end]
                # Normalize literal newlines inside JSON string values to \n
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                        return obj
                except Exception:
                    # Try replacing literal newlines that are inside string values
                    try:
                        normalized = re.sub(r'(?<!\\)\n', r'\\n', candidate)
                        obj = json.loads(normalized)
                        if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                            return obj
                    except Exception:
                        pass

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

        if tool in ("fetch_calendar", "analyze_ics"):
            if not hasattr(self.sandbox, "fetch_calendar"):
                return ToolResult(False, "fetch_calendar is not implemented in ToolSandbox.", {"tool": tool})
            return self.sandbox.fetch_calendar(
                range=args.get("range", None),
                start=args.get("start", None),
                end=args.get("end", None),
                days_ahead=int(args.get("days_ahead", 7)),
            )

        if tool in ("draft_calendar_event", "create_calendar_event"):
            # Redirect direct create calls through the draft-confirm flow
            if not hasattr(self.sandbox, "draft_calendar_event"):
                return ToolResult(False, "draft_calendar_event not available.", {"tool": tool})
            return self.sandbox.draft_calendar_event(
                title=str(args.get("title", args.get("summary", ""))),
                date=str(args.get("date", "")),
                start_time=str(args.get("start_time", "")),
                duration_minutes=int(args.get("duration_minutes", 60)),
                end_time=str(args.get("end_time", "")),
                location=str(args.get("location", "")),
                description=str(args.get("description", "")),
            )

        if tool == "search_drive":
            if not hasattr(self.sandbox, "search_drive"):
                return ToolResult(False, "search_drive is not implemented.", {"tool": tool})
            return self.sandbox.search_drive(
                query=str(args.get("query", "")),
                max_results=int(args.get("max_results", 10)),
            )

        if tool == "read_drive_file":
            if not hasattr(self.sandbox, "read_drive_file"):
                return ToolResult(False, "read_drive_file is not implemented.", {"tool": tool})
            return self.sandbox.read_drive_file(
                file_id=str(args.get("file_id", "")),
                max_chars=int(args.get("max_chars", 6000)),
            )

        if tool == "search_web":
            if not hasattr(self.sandbox, "search_web"):
                return ToolResult(False, "search_web is not implemented in ToolSandbox.", {"tool": tool})
            return self.sandbox.search_web(
                query=str(args.get("query", "")),
                max_results=int(args.get("max_results", 5)),
            )

        if tool == "fetch_gmail":
            if not hasattr(self.sandbox, "fetch_gmail"):
                return ToolResult(False, "fetch_gmail is not implemented in ToolSandbox.", {"tool": tool})
            return self.sandbox.fetch_gmail(
                query=str(args.get("query", "is:unread")),
                max_results=int(args.get("max_results", 10)),
                include_body=bool(args.get("include_body", True)),
            )

        if tool == "draft_gmail":
            if not hasattr(self.sandbox, "draft_gmail"):
                return ToolResult(False, "draft_gmail is not implemented in ToolSandbox.", {"tool": tool})
            return self.sandbox.draft_gmail(
                to=str(args.get("to", "")),
                subject=str(args.get("subject", "")),
                body=str(args.get("body", "")),
                cc=str(args.get("cc", "")),
            )

        if tool == "send_gmail":
            # Redirect direct send_gmail calls through draft_gmail so the user always
            # gets a preview and confirmation before anything is actually sent.
            if not hasattr(self.sandbox, "draft_gmail"):
                return ToolResult(False, "draft_gmail is not implemented in ToolSandbox.", {"tool": tool})
            return self.sandbox.draft_gmail(
                to=str(args.get("to", "")),
                subject=str(args.get("subject", "")),
                body=str(args.get("body", "")),
            )

        if tool == "reply_gmail":
            if not hasattr(self.sandbox, "reply_gmail"):
                return ToolResult(False, "reply_gmail is not implemented in ToolSandbox.", {"tool": tool})
            return self.sandbox.reply_gmail(
                email_id=str(args.get("email_id", "")),
                body=str(args.get("body", "")),
            )

        if tool == "discover_vip_contacts":
            if not hasattr(self.sandbox, "discover_vip_contacts"):
                return ToolResult(False, "discover_vip_contacts is not implemented.", {"tool": tool})
            return self.sandbox.discover_vip_contacts(
                days=int(args.get("days", 150)),
            )

        return ToolResult(False, f"Unknown tool: {tool}", {"tool": tool})

    def _append_tool_result(self, tool_name: str, result: ToolResult) -> None:
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
                "5. Do NOT say you cannot browse or search the web, because search results are already provided above.\n"
                "6. Ground your answer only in the TOOL_RESULT content.\n"
            )
        if tool_name == "fetch_gmail":
            extra_instruction = (
                "\n\nSummarize the emails for the user. "
                "CRITICAL: You MUST preserve every email's tag exactly as shown (e.g. [E1], [E2], [E3]) at the start of each email entry. "
                "Never drop or omit these tags — the user needs them to reply (e.g. 'reply E1 with ...'). "
                "Format each email as: [E1] From: ... | Subject: ... | <brief summary>"
            )
        if tool_name == "send_gmail":
            if result.ok:
                extra_instruction = "\n\nTell the user the email was sent successfully. Include the recipient and subject from the tool result."
            else:
                extra_instruction = "\n\nThe send FAILED. Tell the user clearly what went wrong and do NOT claim the email was sent."

        self.messages.append(
            {
                "role": "user",
                "content": "TOOL_RESULT:\n" + json.dumps(obs, ensure_ascii=False) + extra_instruction,
            }
        )

    def _force_search_web_first(self, user_text: str) -> bool:
        if not self._needs_search(user_text):  # Only search when explicitly needed
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
        if rewritten:
            return rewritten
        return final_answer

    def _rewrite_if_search_disclaimer(self, final_answer: str) -> str:
        if not self._tool_result_uses_search_web():
            return final_answer
        if not self._looks_like_search_disclaimer(final_answer):
            return final_answer

        self.messages.append(
            {
                "role": "user",
                "content": (
                    "You already have web search results in the TOOL_RESULT above. "
                    "Do NOT say you cannot browse, search, or access the internet. "
                    "Rewrite the answer now using ONLY the provided TOOL_RESULT. "
                    "Be concise and include exact source dates when available."
                ),
            }
        )
        rewritten = self._generate().strip()
        if rewritten:
            return rewritten
        return final_answer

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def chat(self, user_text: str, max_steps: int = 6, skill: str = "auto") -> str:
        # Intercept pending email confirmation — bypasses LLM entirely
        auto_reply = self._try_auto_send_pending(user_text)
        if auto_reply is not None:
            return auto_reply

        # Intercept pending calendar event confirmation
        auto_cal = self._try_auto_confirm_cal_event(user_text)
        if auto_cal is not None:
            return auto_cal

        # Intercept "reply E1 / 回复 E2" — extract ID and body, call reply_gmail directly
        auto_draft = self._try_auto_reply_draft(user_text)
        if auto_draft is not None:
            return auto_draft

        # Intercept "read file 1" / "read it" when drive search results are cached
        auto_read = self._try_auto_read_drive_file(user_text)
        if auto_read is not None:
            return auto_read

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
            self._append_tool_result("list_dir", forced)

        self.messages.append({"role": "user", "content": user_text})
        self._force_search_web_first(user_text)

        final_answer = ""

        for _ in range(max_steps):
            model_text = self._generate()
            tool_call = self._try_parse_tool_call(model_text)

            if not tool_call:
                # Fallback: if LLM wrote a plain-text email draft instead of calling draft_gmail,
                # extract it and call draft_gmail ourselves.
                if self._needs_gmail(user_text) and self.sandbox is not None:
                    extracted = self._extract_email_draft_from_text(model_text)
                    if extracted:
                        print("[agent] LLM wrote plain-text draft — intercepting and calling draft_gmail")
                        args = extracted.get("args", {})
                        try:
                            result = self.sandbox.draft_gmail(
                                to=str(args.get("to", "")),
                                subject=str(args.get("subject", "")),
                                body=str(args.get("body", "")),
                            )
                            final_answer = result.output
                        except Exception as e:
                            final_answer = f"草稿保存失败: {e}"
                        self.messages.append({"role": "assistant", "content": final_answer})
                        break
                    else:
                        self.messages.append({"role": "assistant", "content": model_text})
                        final_answer = model_text
                        break
                else:
                    self.messages.append({"role": "assistant", "content": model_text})
                    final_answer = model_text
                    break

            tool_name = str(tool_call.get("tool"))
            tool_args = tool_call.get("args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"value": tool_args}

            self.messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(tool_call, ensure_ascii=False),
                }
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

        if not final_answer:
            final_answer = "Reached max_steps without a final answer."

        final_answer = self._rewrite_if_search_disclaimer(final_answer)
        final_answer = self._rewrite_if_missing_dates_after_search(final_answer)

        if not self.messages or self.messages[-1].get("role") != "assistant":
            self.messages.append({"role": "assistant", "content": final_answer})
        else:
            self.messages[-1]["content"] = final_answer

        return final_answer