# lixia_homejob

A collection of personal home projects by **Lixia**, spanning AI/LLM fine-tuning, local LLM inference, machine-learning from scratch, an AI voice agent, a tax-return assistant, a tennis-court booking bot, and algorithm practice.

---

## Repository Map

```
lixia_homejob/
├── llm-rec-interest-qwen/   # LLM interest-recommendation system (Qwen2.5 + LoRA fine-tuning)
├── ml-playground/           # ML from scratch: MTML → CTR → advanced models (PyTorch)
├── qwen_local_inference/    # Local GPU chat app for Qwen2.5-7B (Tkinter GUI)
├── common_ai/               # Shared audio utilities: TTS, STT, VAD
├── tax_return/              # Web-based tax return assistant (Flask)
├── tennis/                  # Automated tennis-court booking agent (Selenium)
├── robot_game_ui.py         # Robot simulation game – Tkinter desktop UI
├── play_robot_game.py       # Robot simulation game – Streamlit web UI
├── tests/lru_cache.py       # LRU Cache algorithm solution with tests
└── solution.ipynb           # Jupyter notebook scratchpad
```

---

## Projects

### 1. `llm-rec-interest-qwen` — LLM Interest Recommendation + Voice Agent

**What it does**  
Fine-tunes **Qwen2.5-7B-Instruct** with LoRA on YouTube-8M video-topic data to produce a structured user-interest summariser for a recommendation system.  
Instead of verbose plain-text answers, the LoRA-tuned model outputs concise, ranked interest labels:

```
Primary interests: Basketball, NBA, Golden State Warriors.
Secondary interests: Musician, Drum, Association football.
Recommendation direction: more content related to Basketball and NBA…
```

It also ships a full **Voice Agent** (microphone → Faster-Whisper STT → Qwen3 LLM via LM Studio → response) and a **Telegram bot** front-end.

**Key components**

| Path | Description |
|---|---|
| `scripts/01_make_sft_from_yt8m.py` | Build SFT dataset from YouTube-8M labels |
| `scripts/03_train_qwen_lora_sft*.py` | LoRA fine-tuning with HuggingFace PEFT |
| `scripts/07_final_compare_lora_on_off.py` | Side-by-side LoRA OFF vs LoRA ON comparison |
| `agent/agent.py` | LLM agent with tool-routing (file, web-search, calendar) |
| `agent/voice_agent_pipeline.py` | End-to-end voice pipeline (VAD → STT → LLM) |
| `agent/telegram_bot.py` | Telegram bot interface |
| `agent/memory_store.py` | Persistent conversation memory |

**Tech stack:** Python · HuggingFace Transformers / PEFT · PyTorch · Faster-Whisper · LM Studio (OpenAI-compatible API) · python-telegram-bot

---

### 2. `ml-playground` — ML From Scratch (Tabular / CTR / Recommender)

**What it does**  
A self-directed learning project for mastering deep learning on tabular and recommender-system data *without* high-level frameworks.  Everything is written by hand in PyTorch.

**Stages**

| Stage | Dataset | Model | Goal |
|---|---|---|---|
| **Stage 1** | UCI Adult Census Income (~32k rows) | Shared-bottom MTML (income + married) | Embedding + multi-task learning basics |
| **Stage 2** | Criteo CTR Ads (13 dense + 26 sparse features) | DNN baseline → DeepFM → DCN v1 | Industrial CTR, explicit feature crosses |
| **Stage 3** *(in progress)* | Larger datasets | MMoE / PLE | Advanced multi-task CTR models |

**Tech stack:** Python 3.10 · PyTorch 2.9.1+cu128 · numpy · pandas · scikit-learn · NVIDIA RTX 5090

---

### 3. `qwen_local_inference` — Local Qwen2.5-7B Chat App

**What it does**  
A desktop chat application that runs **Qwen2.5-7B-Instruct** entirely locally on a GPU.  It provides a soft-pink Tkinter GUI with multi-turn conversation history, streaming inference, and a `Ctrl+Enter` send shortcut.

**Key files**

| File | Role |
|---|---|
| `main.py` | Application entry point |
| `model_manager.py` | Loads model onto GPU (`device_map="auto"`, `float16`) |
| `inference_engine.py` | Maintains chat history; calls model with `torch.no_grad()` |
| `ui.py` | Tkinter GUI — pastel purple/pink theme, background inference thread |
| `config.py` | Model path, `MAX_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K` |

**Tech stack:** Python 3.11 · PyTorch 2.9.1+cu128 · HuggingFace Transformers 5+ · Tkinter · NVIDIA RTX 5090 (VRAM ~14 GB)

---

### 4. `common_ai` — Shared Audio Utilities

**What it does**  
A small shared library used by `llm-rec-interest-qwen` and other projects for audio I/O.

| Module | Description |
|---|---|
| `tts.py` | Text-to-speech via `pyttsx3` (Windows SAPI5) with configurable voice/rate/volume |
| `stt/stt.py` | Speech-to-text using Faster-Whisper |
| `stt/audio_io.py` | Microphone capture with Voice Activity Detection (VAD) |
| `text_chunker.py` | Splits long text into chunks (for streaming TTS output) |
| `stt_cli.py` | CLI entry point for quick voice transcription |

**Tech stack:** Python · pyttsx3 · faster-whisper · sounddevice · numpy

---

### 5. `tax_return` — Tax Return Assistant (Web App)

**What it does**  
A Flask-based web application that guides users through preparing their US tax return step-by-step:

1. **Material checklist** — which documents are needed (W-2, 1099, receipts, etc.)
2. **Document upload** — categorised file management
3. **Information forms** — personal info, income, deductions, dependents, credits
4. **Tax calculator** — computes taxable income, standard vs. itemised deduction, refund/owed estimate
5. **Summary & export** — printable/downloadable tax summary

**Key files**

| File | Role |
|---|---|
| `app.py` | Flask routes and session management |
| `models.py` | Data models for tax information |
| `calculator.py` | Tax computation logic (brackets, credits, deductions) |
| `materials_checklist.py` | Document requirements by filing status |
| `upload_handler.py` | File upload and categorisation |
| `templates/` | HTML templates (Jinja2) for each wizard step |
| `test_suite.py` | Unit tests for calculator and models |

**Tech stack:** Python 3.8+ · Flask · Jinja2 · SQLite

**Run:**
```bash
cd tax_return
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

### 6. `tennis` — Automated Tennis Court Booking Agent

**What it does**  
A Selenium-based bot that automatically logs into [Bay Club Connect](https://bayclubconnect.com) and books a tennis court at a configured date and time.  It polls the page at up to 4 times per second until a slot opens, then books immediately.

**Key files**

| File | Role |
|---|---|
| `booking_agent.py` | Core Selenium automation logic |
| `config.py` | Target date/time, login credentials, polling interval |
| `diagnose.py` | Helper to inspect the live page structure |
| `inspect_page.py` | Dumps page source for debugging |

**Tech stack:** Python · Selenium 4 · BeautifulSoup4 · python-dotenv · requests

**Run:**
```bash
cd tennis
pip install -r requirements.txt
# Edit config.py to set your login and target slot
python booking_agent.py
```

> ⚠️ **Note:** Avoid hardcoding credentials in `config.py`. Use environment variables (`TENNIS_EMAIL` / `TENNIS_PASSWORD`) instead.

---

### 7. Robot Simulation Game

**What it does**  
Implements the *Walking Robot Simulation* algorithm (LeetCode #874).  A robot walks on an infinite grid following a command list (`-2` = turn left, `-1` = turn right, `N` = walk N steps) while avoiding obstacles, and returns the maximum Euclidean-distance-squared from the origin.

Two front-ends are provided:

| File | UI | How to run |
|---|---|---|
| `robot_game_ui.py` | Tkinter desktop — animated grid with path visualisation | `python robot_game_ui.py` |
| `play_robot_game.py` | Streamlit web UI | `streamlit run play_robot_game.py` |

---

### 8. `tests/lru_cache.py` — LRU Cache Algorithm

**What it does**  
A standalone, well-documented implementation of an **LRU (Least Recently Used) Cache** (LeetCode #146) using a HashMap + Doubly-Linked List, achieving **O(1)** time complexity for both `get` and `put`.  Includes a full test suite with edge cases.

**Run tests:**
```bash
python tests/lru_cache.py
```

---

## Quick-Start Summary

| Project | Entry point | Key command |
|---|---|---|
| Tax Return Web App | `tax_return/app.py` | `python tax_return/app.py` |
| Local Qwen Chat GUI | `qwen_local_inference/main.py` | `python qwen_local_inference/main.py` |
| Voice Agent | `llm-rec-interest-qwen/agent/voice_agent_pipeline.py` | `python voice_agent_pipeline.py --auto --device 1` |
| Telegram Bot | `llm-rec-interest-qwen/agent/telegram_bot.py` | `python telegram_bot.py` |
| Tennis Booking Bot | `tennis/booking_agent.py` | `python tennis/booking_agent.py` |
| Robot Game (desktop) | `robot_game_ui.py` | `python robot_game_ui.py` |
| Robot Game (web) | `play_robot_game.py` | `streamlit run play_robot_game.py` |
| LRU Cache tests | `tests/lru_cache.py` | `python tests/lru_cache.py` |

---

## Author

**Lixia** — personal learning & automation projects  
Last updated: January–March 2026
