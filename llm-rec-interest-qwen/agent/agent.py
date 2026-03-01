# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ===== Qwen2.5 (Transformers Local Version) - Deprecated =====
# --------------------------------------------------------------
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# --------------------------------------------------------------

# ===== Qwen3 (LM Studio OpenAI-compatible Version) =====
from openai import OpenAI

from agent.prompts import SYSTEM_PROMPT, ASSISTANT_STYLE, CHAT_SYSTEM_PROMPT
from agent.tools import ToolSandbox, ToolResult


_JSON_LINE_RE = re.compile(r'^\s*\{.*\}\s*$')


@dataclass
class AgentConfig:
    # ===== Qwen2.5 fields (kept for compatibility) =====
    # model_path: str
    # device: str = "cuda"
    # dtype: str = "auto"

    # ===== Qwen3 LM Studio config =====
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model_name: str = "qwen3-coder-30b-a3b-instruct"

    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95


class LocalAgent:
    def __init__(self, cfg: AgentConfig, sandbox: ToolSandbox) -> None:
        self.cfg = cfg
        self.sandbox = sandbox

        # ===== Qwen2.5 Transformers Model Loading (Deprecated) =====
        # -----------------------------------------------------------
        # self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        # torch_dtype = None
        # if cfg.dtype in ("float16", "fp16"):
        #     torch_dtype = torch.float16
        # elif cfg.dtype in ("bfloat16", "bf16"):
        #     torch_dtype = torch.bfloat16
        #
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     cfg.model_path,
        #     device_map="auto" if cfg.device.startswith("cuda") else None,
        #     torch_dtype=torch_dtype,
        #     trust_remote_code=True,
        # )
        # self.model.eval()
        # -----------------------------------------------------------

        # ===== Qwen3 LM Studio Client =====
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
        )

        # conversation messages in ChatML style
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": ASSISTANT_STYLE},
        ]
                # ===========================
        # Skills registry (Qwen3 LM Studio)
        # ===========================
        self.skills = {
            "chat": self.chat_simple,          # pure chat, no tools
            "tool": self.chat_with_tools,      # your existing Plan->Tool->Observe loop
        }

    # ===== Qwen2.5 Render (Deprecated) =====
    # def _render(self) -> str:
    #     return self.tokenizer.apply_chat_template(
    #         self.messages,
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )

    # ===== Qwen3 Generate via LM Studio =====
        # ============================================================
    # Skill: Pure Chat (No Tools, Single LLM Call)
    # ============================================================
    def chat_simple(self, user_text: str) -> str:
        # Use chat-only system prompt (no tool instructions)
        temp_messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        ]
        temp_messages.append({"role": "user", "content": user_text})

        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=temp_messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_new_tokens,
        )
        text = response.choices[0].message.content or ""
        # 纯聊天也要把assistant回复记录进对话历史（不然上下文断）
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": text.strip()})
        return text.strip()
    # ============================================================
    # Router: decide which skill to use
    # ============================================================
    def route(self, user_text: str, forced_skill: Optional[str] = None) -> str:
        """
        Return skill name: 'chat' or 'tool'
        - forced_skill: 'chat'|'tool'|'auto'|None
        
        Default behavior: PURE CHAT (no tools)
        Only use tools if:
          1. forced_skill == 'tool' (explicit flag)
          2. user_text starts with "tool:" (explicit prefix)
        """
        # Explicit forced_skill takes priority
        if forced_skill and forced_skill != "auto":
            if forced_skill not in self.skills:
                return "chat"
            return forced_skill

        # --- Check for explicit tool prefix ---
        t = user_text.lower()
        if t.startswith("tool:"):
            return "tool"

        # DEFAULT: pure chat (no auto-detection of keywords)
        return "chat"
    
    def _generate(self) -> str:
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=self.messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_new_tokens,
        )
        text = response.choices[0].message.content
        return (text or "").strip()

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
        print("[DEBUG] tool requested:", tool)
        return ToolResult(False, f"Unknown tool: {tool}", {"tool": tool})
    
    # ============================================================
    # Unified entry: uses router + skills
    # ============================================================
    def chat(self, user_text: str, max_steps: int = 6, skill: str = "auto") -> str:
        chosen = self.route(user_text, forced_skill=skill)

        # strip explicit prefixes so模型看不到指令痕迹
        if user_text.lower().startswith("chat:"):
            user_text = user_text[5:].strip()
        elif user_text.lower().startswith("tool:"):
            user_text = user_text[5:].strip()

        if chosen == "tool":
            return self.chat_with_tools(user_text=user_text, max_steps=max_steps)
        return self.chat_simple(user_text=user_text)
    
    def chat_with_tools(self, user_text: str, max_steps: int = 6) -> str:
        if any(k in user_text.lower() for k in ["repo", "project", "结构", "目录"]):
            forced = self.sandbox.list_dir(self.sandbox.allowed_roots[0])
            self.messages.append({
                "role": "user",
                "content": "TOOL_RESULT:\n" + str({
                    "tool": "list_dir",
                    "ok": forced.ok,
                    "output": forced.output,
                    "meta": forced.meta,
                })
            })

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

            self.messages.append({"role": "assistant", "content": json.dumps(tool_call, ensure_ascii=False)})

            # print(f"\n[DEBUG] Tool call: {tool_name} with args {tool_args}")  # Disabled for performance

            try:
                result = self._run_tool(tool_name, tool_args)
            except Exception as e:
                import traceback
                # print("\n[TOOL EXCEPTION]", repr(e))  # Disabled for performance
                # traceback.print_exc()  # Disabled for performance
                result = ToolResult(False, f"Tool exception: {e}", {"tool": tool_name, "args": tool_args})

            obs = {
                "tool": tool_name,
                "ok": result.ok,
                "output": result.output,
                "meta": result.meta,
            }
            self.messages.append({"role": "user", "content": "TOOL_RESULT:\n" + json.dumps(obs, ensure_ascii=False)})

            if not result.ok:
                continue

        if not final_answer:
            final_answer = "Reached max_steps without a final answer. Try asking more specifically."
        return final_answer