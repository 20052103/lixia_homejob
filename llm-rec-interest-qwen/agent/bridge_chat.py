# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from agent.agent import AgentConfig, LocalAgent
from agent.tools import ToolSandbox
from agent.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_STEPS,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def build_local_agent(root: str) -> LocalAgent:
    sandbox = ToolSandbox(
        allowed_roots=[root],
        allowed_cmd_prefixes=["python", "py", "git", "dir", "ls", "pip"],
        max_read_bytes=200_000,
        max_output_chars=40_000,
        cwd=root,
    )

    cfg = AgentConfig(
        max_new_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
    )

    agent = LocalAgent(cfg=cfg, sandbox=sandbox)
    agent.add_system_message(
        f"Allowed filesystem root: {root}.\n"
        f"Only use paths under this root."
    )
    return agent


def ensure_copilot_exists() -> None:
    if shutil.which("copilot") is None:
        raise RuntimeError(
            "Cannot find 'copilot' in PATH.\n"
            "Install GitHub Copilot CLI first, then restart PowerShell."
        )


def ask_copilot(
    prompt: str,
    cwd: str,
    timeout_sec: int = 180,
) -> str:
    """
    Uses GitHub Copilot CLI non-interactively.

    Official docs support:
      copilot -p "..."
      copilot -sp "..."
    We use -sp to return only the model output.
    """
    ensure_copilot_exists()

    cmd = ["copilot", "-sp", prompt]

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return "[Copilot timeout] No response within timeout."
    except FileNotFoundError:
        return "[Copilot error] 'copilot' command not found."
    except Exception as e:
        return f"[Copilot error] {e}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if result.returncode == 0 and stdout:
        return stdout

    # Useful auth/trust hints
    joined = "\n".join(x for x in [stdout, stderr] if x).strip()
    if not joined:
        joined = f"Copilot exited with code {result.returncode}."

    return (
        "[Copilot CLI returned an error]\n"
        f"{joined}\n\n"
        "Tip:\n"
        "1) In repo root, run: copilot\n"
        "2) trust this folder\n"
        "3) run /login\n"
        "4) exit, then rerun this script"
    )


def ask_qwen(
    agent: LocalAgent,
    prompt: str,
    skill: str,
    max_steps: int,
) -> str:
    try:
        return agent.chat(prompt, max_steps=max_steps, skill=skill).strip()
    except Exception as e:
        return f"[Qwen error] {e}"


def short_block(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def save_markdown(transcript: List[Dict[str, str]], output_path: Path, topic: str) -> None:
    lines = [
        f"# Copilot ↔ Qwen Friendly Chat",
        "",
        f"**Topic:** {topic}",
        f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for item in transcript:
        lines.append(f"## {item['speaker']}")
        lines.append("")
        lines.append(item["text"])
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------

def first_prompt_for_copilot(topic: str) -> str:
    return textwrap.dedent(
        f"""
        You are a friendly senior AI engineering partner.
        You are starting a constructive discussion with another AI called Qwen.

        Topic:
        {topic}

        Rules:
        - Be friendly and collaborative.
        - Focus on practical engineering suggestions.
        - Keep your reply under 180 words.
        - Do not use markdown tables.
        - End with 1 concrete question to Qwen.
        """
    ).strip()


def next_prompt_for_qwen(topic: str, copilot_reply: str) -> str:
    return textwrap.dedent(
        f"""
        你正在和另一个 AI 工程师 Copilot 友好交流，讨论下面的话题：

        话题：
        {topic}

        对方刚才说：
        {copilot_reply}

        请按下面规则回复：
        1. 先肯定对方一个合理点。
        2. 再补充你的不同看法或更适合本地 agent 的实现建议。
        3. 保持友好，不要重复，不要跑题。
        4. 控制在 180 词以内。
        5. 最后问 Copilot 1 个推进讨论的问题。
        """
    ).strip()


def next_prompt_for_copilot(topic: str, qwen_reply: str) -> str:
    return textwrap.dedent(
        f"""
        You are continuing a friendly engineering discussion with Qwen.

        Topic:
        {topic}

        Qwen just replied:
        {qwen_reply}

        Reply with these rules:
        - First acknowledge one useful point from Qwen.
        - Then add one practical implementation idea.
        - Stay collaborative and concise.
        - Keep under 180 words.
        - End with 1 concrete follow-up question.
        """
    ).strip()


def final_summary_prompt(transcript_text: str) -> str:
    return textwrap.dedent(
        f"""
        请根据下面这段 Copilot 和 Qwen 的讨论，输出一个简洁总结。

        要求：
        1. 先给 3 条达成共识的点
        2. 再给 3 条下一步最值得实现的 action items
        3. 尽量贴近本地 agent / Copilot bridge 场景
        4. 不要写太长

        讨论记录：
        {transcript_text}
        """
    ).strip()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Discussion topic for Copilot and Qwen.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="How many back-and-forth rounds. Recommended: 3 to 5.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\repo",
        help="Allowed filesystem root for Qwen tool sandbox.",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=".",
        help="Working directory for Copilot CLI. Use your repo root.",
    )
    parser.add_argument(
        "--skill",
        type=str,
        default="chat",
        choices=["auto", "chat", "tool"],
        help="Qwen skill routing mode.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max tool-reasoning steps for Qwen when skill=auto/tool.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1500,
        help="Safety trim for each side's reply before passing to the other side.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional markdown output path, e.g. logs/copilot_qwen_chat.md",
    )

    args = parser.parse_args()

    repo_cwd = str(Path(args.cwd).resolve())
    agent = build_local_agent(args.root)

    transcript: List[Dict[str, str]] = []

    print("\n=== Topic ===")
    print(args.topic)
    print("")

    # Round 1: Copilot starts
    copilot_prompt = first_prompt_for_copilot(args.topic)

    for i in range(1, args.rounds + 1):
        copilot_reply = ask_copilot(copilot_prompt, cwd=repo_cwd)
        copilot_reply = short_block(copilot_reply, args.max_chars)

        transcript.append({
            "speaker": f"Copilot Round {i}",
            "text": copilot_reply,
        })

        print(f"\n[Copilot Round {i}]")
        print(copilot_reply)

        qwen_prompt = next_prompt_for_qwen(args.topic, copilot_reply)
        qwen_reply = ask_qwen(
            agent=agent,
            prompt=qwen_prompt,
            skill=args.skill,
            max_steps=args.max_steps,
        )
        qwen_reply = short_block(qwen_reply, args.max_chars)

        transcript.append({
            "speaker": f"Qwen Round {i}",
            "text": qwen_reply,
        })

        print(f"\n[Qwen Round {i}]")
        print(qwen_reply)

        copilot_prompt = next_prompt_for_copilot(args.topic, qwen_reply)

    # Final summary from Qwen
    merged = []
    for item in transcript:
        merged.append(f"{item['speaker']}:\n{item['text']}")
    transcript_text = "\n\n".join(merged)

    summary = ask_qwen(
        agent=agent,
        prompt=final_summary_prompt(transcript_text),
        skill="chat",
        max_steps=args.max_steps,
    ).strip()

    transcript.append({
        "speaker": "Qwen Final Summary",
        "text": summary,
    })

    print("\n[Qwen Final Summary]")
    print(summary)

    if args.save:
        output_path = Path(args.save).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_markdown(transcript, output_path, args.topic)
        print(f"\nSaved transcript to:\n{output_path}")


if __name__ == "__main__":
    main()