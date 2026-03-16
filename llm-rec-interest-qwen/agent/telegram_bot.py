from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ------------------------------------------------
# repo root -> import agent package
# ------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
AUTHORIZED_CHAT_ID = os.getenv("AUTHORIZED_CHAT_ID", "").strip()
TELEGRAM_MAX_STEPS = int(os.getenv("TELEGRAM_MAX_STEPS", "8"))
TELEGRAM_SKILL = os.getenv("TELEGRAM_SKILL", "auto").strip()
TELEGRAM_ROOT = os.getenv("TELEGRAM_ROOT", r"D:\repo").strip()

AGENT = None
AGENT_LOCK = asyncio.Lock()


def build_agent():
    """
    Reuse the same construction logic as agent/run_agent.py
    so Telegram and CLI behave consistently.
    """
    from agent.agent import AgentConfig, LocalAgent
    from agent.tools import ToolSandbox
    from agent.config import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
    )

    sandbox = ToolSandbox(
        allowed_roots=[TELEGRAM_ROOT],
        allowed_cmd_prefixes=["python", "py", "git", "dir", "ls", "pip"],
        max_read_bytes=200_000,
        max_output_chars=40_000,
        cwd=TELEGRAM_ROOT,
    )

    cfg = AgentConfig(
        max_new_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
    )

    agent = LocalAgent(cfg=cfg, sandbox=sandbox)
    agent.messages.insert(
        1,
        {
            "role": "system",
            "content": (
                f"Allowed filesystem root: {TELEGRAM_ROOT}.\n"
                "Only use paths under this root."
            ),
        },
    )
    return agent


def split_text(text: str, chunk_size: int = 3500) -> list[str]:
    text = text or ""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or ["(empty)"]


def is_authorized(update: Update) -> bool:
    if not AUTHORIZED_CHAT_ID:
        return True
    if update.effective_chat is None:
        return False
    return str(update.effective_chat.id) == AUTHORIZED_CHAT_ID


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Local agent is online.\n"
        "Send a message and I will run it on your PC."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Examples:\n"
        "1) news about AI\n"
        "2) search news of iran\n"
        "3) stock agent: tsla news\n"
        "4) summarize today's market\n\n"
        "Commands:\n"
        "/start\n"
        "/help\n"
        "/ping\n"
        "/reset"
    )


async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"
    await update.message.reply_text(f"pong\nchat_id={chat_id}")


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global AGENT
    if update.message is None:
        return

    if not is_authorized(update):
        await update.message.reply_text("Unauthorized.")
        return

    AGENT = None
    await update.message.reply_text("Agent session cleared.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global AGENT

    if update.message is None or not update.message.text:
        return

    if not is_authorized(update):
        await update.message.reply_text("Unauthorized.")
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    await update.message.reply_text("Running on local PC...")

    try:
        if AGENT is None:
            AGENT = build_agent()

        async with AGENT_LOCK:
            result = AGENT.chat(
                user_text,
                max_steps=TELEGRAM_MAX_STEPS,
                skill=TELEGRAM_SKILL,
            )

        result = str(result or "(agent returned empty result)")
        for chunk in split_text(result):
            await update.message.reply_text(chunk)

    except Exception:
        err = traceback.format_exc(limit=8)
        for chunk in split_text(f"Agent error:\n{err}"):
            await update.message.reply_text(chunk)


def main():
    if not BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("[Telegram Bot] started. Press Ctrl+C to stop.", flush=True)
    app.run_polling()


if __name__ == "__main__":
    main()