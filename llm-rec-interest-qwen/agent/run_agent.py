# -*- coding: utf-8 -*-
from __future__ import annotations

# ============================================================
# run_agent.py
# - Qwen2.5 (Transformers local) 版本：已保留并注释
# - Qwen3 (LM Studio + GGUF OpenAI-compatible) 版本：当前使用
# ============================================================

import argparse

from agent.agent import AgentConfig, LocalAgent
from agent.tools import ToolSandbox


# ============================================================
# [Version: Qwen2.5-7B Instruct | Transformers Local] (Deprecated)
# ============================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default=r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28",
#     )
#     parser.add_argument(
#         "--root",
#         type=str,
#         default=r"D:\repo\lixia_homejob\llm-rec-interest-qwen",
#         help="Allowed filesystem root for tools (sandbox).",
#     )
#     parser.add_argument(
#         "--cwd",
#         type=str,
#         default=r"D:\repo\lixia_homejob\llm-rec-interest-qwen",
#         help="Working directory for run_cmd.",
#     )
#     parser.add_argument("--max_new_tokens", type=int, default=512)
#     parser.add_argument("--temperature", type=float, default=0.2)
#     parser.add_argument("--top_p", type=float, default=0.95)
#     parser.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16")
#     args = parser.parse_args()
#
#     sandbox = ToolSandbox(
#         allowed_roots=[args.root],
#         allowed_cmd_prefixes=["python", "py", "git", "dir", "ls", "pip"],
#         cwd=args.cwd,
#     )
#
#     cfg = AgentConfig(
#         model_path=args.model_path,
#         max_new_tokens=args.max_new_tokens,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         dtype=args.dtype,
#     )
#
#     agent = LocalAgent(cfg, sandbox)
#     agent.messages.insert(1, {
#         "role": "system",
#         "content": f"Allowed filesystem root: {args.root}. Only use paths under this root."
#     })
#
#     print("Local Agent ready. Type your message. Type 'exit' to quit.")
#     while True:
#         try:
#             user = input("\nYou> ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print("\nBye.")
#             break
#         if not user:
#             continue
#         if user.lower() in ("exit", "quit", "q"):
#             print("Bye.")
#             break
#         ans = agent.chat(user_text=user, max_steps=6)
#         print("\nAgent>\n" + ans)


# ============================================================
# [Version: Qwen3-Coder-30B-A3B | LM Studio + GGUF] (Active)
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--skill", default="auto", choices=["auto", "chat", "tool"])

    # 可选：把 sandbox root 显式设成你的项目根目录（更可控）
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
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model_name="qwen3-coder-30b-a3b-instruct",
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
    )

    agent = LocalAgent(cfg=cfg, sandbox=sandbox)
    agent.messages.insert(1, {
        "role": "system",
        "content": f"Allowed filesystem root: {args.root}. Only use paths under this root."
    })

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