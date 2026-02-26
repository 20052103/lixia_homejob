# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os

from agent import AgentConfig, LocalAgent
from tools import ToolSandbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\repo\lixia_homejob\llm-rec-interest-qwen",
        help="Allowed filesystem root for tools (sandbox).",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=r"D:\repo\lixia_homejob\llm-rec-interest-qwen",
        help="Working directory for run_cmd.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16")
    args = parser.parse_args()

    # Sandbox: only allow within your repo root (you can add more roots if needed)
    sandbox = ToolSandbox(
        allowed_roots=[args.root],
        allowed_cmd_prefixes=["python", "py", "git", "dir", "ls", "pip"],
        cwd=args.cwd,
    )

    cfg = AgentConfig(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
    )

    agent = LocalAgent(cfg, sandbox)
    agent.messages.insert(1, {
        "role": "system",
        "content": f"Allowed filesystem root: {args.root}. Only use paths under this root."
    })

    print("Local Agent ready. Type your message. Type 'exit' to quit.")
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        ans = agent.chat(user_text=user, max_steps=6)
        print("\nAgent>\n" + ans)


if __name__ == "__main__":
    main()
