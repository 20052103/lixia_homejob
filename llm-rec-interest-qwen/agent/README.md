# Agent Module

Local AI agent powered by Qwen3.5 with LM Studio support.

## Features

- **Text Agent**: Interactive chat interface with reasoning and tool access
- **Voice Agent**: Real-time voice input → STT → Agent reasoning → Output
- **Tool Support**: File operations, directory listing, command execution, web search, calendar analysis, **Gmail inbox summary**

## Skills (Tools)

| Tool | Trigger keywords | Description |
|---|---|---|
| `read_file` / `list_dir` | 文件、目录、结构、find… | Read local files and directories |
| `run_cmd` | python, git, pip… | Execute sandboxed shell commands |
| `search_web` | 搜索、search、news… | Web search via SerpAPI or DuckDuckGo |
| `analyze_ics` | 日历、日程、calendar… | Parse and summarize `.ics` calendar files |
| `fetch_url` | URL provided | Fetch and read a web page |
| `fetch_gmail` | 邮件、邮箱、Gmail、email… | **Fetch & filter Gmail inbox** (see below) |
| `send_gmail` | 发邮件、写信、send email… | **Compose, confirm, and send Gmail** (see below) |

---

## Gmail Skill

### Overview

The Gmail skill lets you ask the agent natural-language questions about your inbox.
It uses a **two-phase approach** to avoid context overflow:

1. **Scan** — fetch metadata only (subject, sender, snippet, labels) for up to N emails
2. **Filter** — automatically skip ads, promotions, and automated mail using:
   - Gmail's own category labels (`CATEGORY_PROMOTIONS`, `CATEGORY_SOCIAL`, etc.)
   - Sender heuristics (`noreply`, `newsletter`, `marketing`, …)
   - Subject/snippet keyword matching
3. **Retrieve** — fetch full body **only** for emails that pass the filter
4. **Return** — structured summary with important emails + filtered-out list

### One-time Setup

#### 1. Create Google Cloud credentials

1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (e.g. `gmail-agent`)
3. Enable the **Gmail API**: *APIs & Services → Library → Gmail API → Enable*
4. Configure the **OAuth consent screen**: *APIs & Services → OAuth consent screen*
   - User type: **External**
   - Add your Gmail address under **Test users**
5. Create **OAuth credentials**: *APIs & Services → Credentials → + Create Credentials → OAuth client ID*
   - Application type: **Desktop app**
   - Download the JSON → rename to `credentials.json`
6. Place the file at:
   ```
   agent/credentials.json
   ```

#### 2. Install dependencies

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

#### 3. First-time authorization

On first use, a browser window will open for Google OAuth consent.
After you approve, a token is saved to `agent/gmail_token.json` — no further login needed.

> ⚠️ Both `credentials.json` and `gmail_token.json` are in `.gitignore` and will never be committed.

### Usage Examples

Simply chat with the agent:

```
总结今天的邮件
有哪些未读邮件？
帮我看看来自 boss@example.com 的最新邮件
最近一周有没有主题包含 invoice 的邮件？
Check my unread emails
Summarize emails from last 3 days
```

### Gmail Query Syntax

The agent passes queries directly to the Gmail API search engine:

| Query | Meaning |
|---|---|
| `is:unread` | All unread emails |
| `newer_than:1d` | Emails from today |
| `newer_than:7d` | Emails from this week |
| `from:someone@example.com` | From specific sender |
| `subject:invoice newer_than:7d` | Subject keyword + date filter |
| `is:unread newer_than:1d` | Today's unread emails |

### VIP Contacts

VIP contacts are stored in `agent/contacts.json` and get special treatment every time emails are checked:

- Their emails are **always fetched separately** (last 7 days), regardless of the main query
- They are shown at the **top of every email report** under `⭐ VIP Contact Emails`
- They **bypass the spam/promo filter** — their emails are never silently dropped

**Current VIP contacts** (`agent/contacts.json`):

```json
{
  "vip_contacts": [
    { "name": "Lin Yang", "email": "lynn69688@gmail.com" }
  ]
}
```

To add more contacts, edit `contacts.json` directly:

```json
{ "name": "Another Person", "email": "another@example.com" }
```

> ✅ **`send_gmail` is now available** — see the Send Email section below.

```env
GMAIL_CREDENTIALS_PATH=D:\path\to\credentials.json
GMAIL_TOKEN_PATH=D:\path\to\gmail_token.json
```

---

### Send Email

The agent composes, polishes, and sends emails via Gmail. It always follows a **draft → confirm → send** flow — it will never send without your explicit approval.

#### Usage Examples

```
给 Lin Yang 发邮件，说我明天下午不能来了
发邮件给 lynn69688@gmail.com，问一下会议是否改期
Send an email to Lin Yang asking about the project deadline
Write to lynn69688@gmail.com: are we still meeting tomorrow?
```

#### Draft-Confirm Flow

```
You:   给 Lin Yang 发邮件，说周五的会议我要迟到20分钟

Agent: ---
       To      : Lin Yang <lynn69688@gmail.com>
       Subject : 关于周五会议
       Body    :
       Hi Yang, 周五的会议我会晚到大约20分钟，请见谅。
       ---
       请确认发送，或告诉我需要修改的地方。

You:   确认发送

Agent: ✅ Email sent successfully to lynn69688@gmail.com
```

#### Recipient Resolution

| What you type | Resolved to |
|---|---|
| `lynn69688@gmail.com` | used directly |
| `Lin Yang` | looked up in `contacts.json` → `lynn69688@gmail.com` |

> ⚠️ **Re-authorization required on first send**: Sending requires an additional OAuth scope. Delete `agent/gmail_token.json` and re-run the agent once to re-authorize with both read + send permissions.

---

## Setup

### Prerequisites

```powershell
conda activate llmrec
```

### 1. Start LM Studio Server

Start llama.cpp with the Qwen3.5 model:

```powershell
cd llama
.\llama-server.exe -m "C:\Users\xiali\.lmstudio\models\bartowski\Qwen3.5-27B-GGUF\Qwen_Qwen3.5-27B-Q4_K_M.gguf" --host 127.0.0.1 --port 1234 --ctx-size 8192
```

## Usage

### Text Agent

```powershell
cd llm-rec-interest-qwen
python -m agent.run_agent --max_steps 10 --skill auto
```

**Options:**
- `--max_steps`: Maximum reasoning steps (default: 6)
- `--skill`: `auto` | `chat` | `tool` (default: auto)
- `--root`: Sandbox root directory (default: D:\repo)

### Voice Agent

```powershell
python agent/voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 2 --model tiny --lang zh --skill auto
```

**Voice Options:**
- `--device`: Microphone device ID (default: 1)
- `--auto`: Enable voice activity detection (VAD)
- `--rms`: VAD RMS threshold (default: 100.0)
- `--silence`: Stop after N seconds of silence (default: 5.0)

**STT Options:**
- `--model`: Whisper model size: `tiny|base|small|medium|large-v3` (default: small)
- `--lang`: Language code: `zh|en|ja|etc` (default: zh)

**Agent Options:**
- `--max_steps`: Maximum reasoning steps
- `--skill`: `auto|chat|tool`
- `--temperature`: LLM temperature (default: 0.2)
- `--max_tokens`: Max output tokens (default: 512)

## Configuration

Central configuration is in `config.py`:

- `LM_STUDIO_BASE_URL`: LM Studio server URL
- `LM_STUDIO_MODEL_NAME`: Model identifier
- `DEFAULT_MAX_TOKENS`, `DEFAULT_TEMPERATURE`: Default LLM parameters
- `GMAIL_CREDENTIALS_PATH`: Path to OAuth credentials JSON
- `GMAIL_TOKEN_PATH`: Path to cached OAuth token

## Project Structure

```
agent/
├── agent.py                 # Main agent logic & routing
├── config.py               # Central configuration
├── prompts.py              # System prompts & tool schemas
├── tools.py                # Tool sandbox & implementations
├── gmail_tool.py           # Gmail API OAuth2 + smart email filtering
├── contacts.py             # VIP contact list manager
├── contacts.json           # VIP contacts data (name + email)
├── run_agent.py            # Text agent CLI
├── voice_agent_pipeline.py # Voice agent CLI
└── README.md               # This file
```

### Telegram Remote Control

1. Create a `.env` file at repo root:
   - `TELEGRAM_BOT_TOKEN`
   - `AUTHORIZED_CHAT_ID`
   - `TELEGRAM_MAX_STEPS`
   - `TELEGRAM_SKILL`
   - `TELEGRAM_ROOT`

2. Install dependencies:

```bash
pip install python-telegram-bot python-dotenv