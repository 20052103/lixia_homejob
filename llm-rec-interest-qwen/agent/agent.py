# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import SYSTEM_PROMPT, ASSISTANT_STYLE
from tools import ToolSandbox, ToolResult


_JSON_LINE_RE = re.compile(r'^\s*\{.*\}\s*$')


@dataclass
class AgentConfig:
    model_path: str
    device: str = "cuda"
    dtype: str = "auto"  # "auto" or "bfloat16"/"float16"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95


class LocalAgent:
    def __init__(self, cfg: AgentConfig, sandbox: ToolSandbox) -> None:
        self.cfg = cfg
        self.sandbox = sandbox

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        torch_dtype = None
        if cfg.dtype in ("float16", "fp16"):
            torch_dtype = torch.float16
        elif cfg.dtype in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map="auto" if cfg.device.startswith("cuda") else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.eval()

        # conversation messages in ChatML style
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": ASSISTANT_STYLE},
        ]

    def _render(self) -> str:
        # Qwen Instruct supports chat template
        return self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def _generate(self) -> str:
        prompt = self._render()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.cfg.device.startswith("cuda"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.cfg.temperature > 0
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=self.cfg.temperature if do_sample else None,
            top_p=self.cfg.top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = out_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text.strip()

    def _try_parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Expect exactly one-line JSON. We accept either:
        - the whole response is a JSON object line
        - or the first JSON-looking line is a tool call (robust)
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None

        # Strict: if single line and is JSON
        if len(lines) == 1 and _JSON_LINE_RE.match(lines[0]):
            try:
                obj = json.loads(lines[0])
                if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                    return obj
            except Exception:
                return None

        # Robust: find first line that looks like JSON dict
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

    def chat(self, user_text: str, max_steps: int = 6) -> str:
        """
        Plan->Tool->Observe loop.
        """
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
                # normal answer
                self.messages.append({"role": "assistant", "content": model_text})
                final_answer = model_text
                break

            tool_name = str(tool_call.get("tool"))
            tool_args = tool_call.get("args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"value": tool_args}

            # record the tool decision (so model "knows" what it asked)
            self.messages.append({"role": "assistant", "content": json.dumps(tool_call, ensure_ascii=False)})

            # execute tool
            # result = self._run_tool(tool_name, tool_args)
            print(f"\n[DEBUG] Tool call: {tool_name} with args {tool_args}")
            # execute tool
            try:
                result = self._run_tool(tool_name, tool_args)
            except Exception as e:
                import traceback
                print("\n[TOOL EXCEPTION]", repr(e))
                traceback.print_exc()
                from tools import ToolResult
                result = ToolResult(False, f"Tool exception: {e}", {"tool": tool_name, "args": tool_args})

            # feed back observation
            obs = {
                "tool": tool_name,
                "ok": result.ok,
                "output": result.output,
                "meta": result.meta,
            }
            self.messages.append({"role": "user", "content": "TOOL_RESULT:\n" + json.dumps(obs, ensure_ascii=False)})

            # if tool failed, let model respond; continue loop
            if not result.ok:
                continue

        if not final_answer:
            final_answer = "Reached max_steps without a final answer. Try asking more specifically."
        return final_answer
