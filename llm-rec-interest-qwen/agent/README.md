# Agent Module

Local AI agent powered by Qwen3.5 with LM Studio support.

## Features

- **Text Agent**: Interactive chat interface with reasoning and tool access
- **Voice Agent**: Real-time voice input → STT → Agent reasoning → Output
- **Tool Support**: File operations, directory listing, command execution

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

## Project Structure

```
agent/
├── agent.py                 # Main agent logic
├── config.py               # Central configuration
├── prompts.py              # System prompts
├── tools.py                # Tool sandbox & implementations
├── run_agent.py            # Text agent CLI
├── voice_agent_pipeline.py # Voice agent CLI
└── README.md               # This file
```