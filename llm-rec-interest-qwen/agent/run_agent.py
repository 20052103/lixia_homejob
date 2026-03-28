# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Load .env from the project root before importing config
_ENV_FILE = Path(__file__).parent.parent / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_FILE)
    except ImportError:
        # dotenv not installed — parse manually
        with open(_ENV_FILE, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

from agent.agent import AgentConfig, LocalAgent
from agent.tools import ToolSandbox
from agent.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_STEPS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--skill", default="auto", choices=["auto", "chat", "tool"])
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\repo",
        help="Allowed filesystem root for tools (sandbox).",
    )
    args = parser.parse_args()

    sandbox = ToolSandbox(
        allowed_roots=[args.root],
        allowed_cmd_prefixes=["python", "py", "git", "dir", "ls", "pip"],
        max_read_bytes=200_000,
        max_output_chars=40_000,
        cwd=args.root,
    )

    cfg = AgentConfig(
        max_new_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
    )

    agent = LocalAgent(cfg=cfg, sandbox=sandbox)

    # IMPORTANT:
    # Do NOT insert raw system messages into agent.messages.
    # Keep all system messages at the front via LocalAgent helper.
    agent.add_system_message(
        f"Allowed filesystem root: {args.root}.\nOnly use paths under this root."
    )

    print("\n[LMStudio Agent Ready] Type your message. Ctrl+C to exit.\n", flush=True)

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        ans = agent.chat(user_text, max_steps=args.max_steps, skill=args.skill)
        print("\nAssistant>\n" + ans + "\n", flush=True)


if __name__ == "__main__":
    main()