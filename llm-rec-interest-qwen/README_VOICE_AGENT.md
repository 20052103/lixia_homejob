# Voice Agent Pipeline

**语音输入 → STT(Faster-Whisper) → Agent推理(Qwen3 LM Studio) → 输出**

A speech-to-text powered conversational AI agent that combines real-time voice input with LLM reasoning.

## Features

- 🎤 **Voice Input**: Auto-detect speech with VAD (Voice Activity Detection)
- 📝 **STT**: Local Faster-Whisper transcription (supports multiple languages)
- 🤖 **Agent**: Qwen3-Coder LLM via LM Studio for reasoning
- 💬 **Pure Chat Mode (Default)**: Conversational responses without tool overhead
- 🔧 **Tools Support (Optional)**: File/command access when explicitly enabled
- ⏱️ **Performance**: Optimized with debug output suppression

## Requirements

- Python 3.8+
- LM Studio with Qwen3-Coder-30B-A3B model loaded
- Microphone input device
- Dependencies: `sounddevice`, `numpy`, `faster-whisper`, `openai`

## Setup

### 1. Environment Setup

Ensure the Python environment is activated:
```powershell
cd d:\repo
& .\.venv\Scripts\Activate.ps1
```

### 2. Navigate to Agent Directory

```powershell
cd d:\repo\lixia_homejob\llm-rec-interest-qwen\agent
```

### 3. Start LM Studio

Ensure LM Studio is running with Qwen3-Coder model loaded at `http://localhost:1234/v1`

## Basic Usage

### Default: Pure Chat Mode (Recommended)

```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 5 --model small --lang zh
```

**Output:**
```
🗣️  You: Can you hear me?

✓ Agent: Yes, I can hear you! How can I help you today?
```

### Common Quick Commands

**English conversation:**
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 5 --model small --lang en
```

**Japanese conversation:**
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 5 --model small --lang ja
```

**Higher quality transcription (slower):**
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 5 --model medium --lang zh
```

**Silent mode (minimal output):**
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 5 --model small --lang zh --quiet
```

## Advanced Usage

### Enable Tools Support

**Option 1: Enable tools globally with `--skill tool`**
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 5 --model small --lang zh --skill tool
```

**Option 2: Enable tools per-message with `tool:` prefix**
```
You: tool: read the contents of /repo/test.py
```

### Microphone Device Selection

Find available microphones:
```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Then specify device ID:
```powershell
python voice_agent_pipeline.py --auto --device 7 --rms 100 --silence 5 --model small --lang zh
```

### VAD (Voice Activity Detection) Tuning

**Lower RMS threshold (more sensitive to quiet speech):**
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 80 --silence 5 --model small --lang zh
```

**Higher RMS threshold (less sensitive, better for noisy environments):**
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 120 --silence 5 --model small --lang zh
```

**Shorter silence timeout (faster end detection):**
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 2 --model small --lang zh
```

### Debug Mode

Show RMS levels during recording:
```powershell
python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 5 --model small --lang zh --debug_rms
```

## Command-Line Arguments

### Voice Input
- `--device INT` - Microphone device ID (default: 1)
- `--auto` - Enable automatic voice detection with VAD
- `--rms FLOAT` - VAD RMS threshold in int16 scale (default: 100.0)
- `--silence FLOAT` - Stop recording after N seconds of silence (default: 5.0)
- `--max_seconds FLOAT` - Maximum recording duration (default: 120.0)
- `--debug_rms` - Print RMS values during recording (verbose, may slow down response)
- `--quiet` - Suppress all debug output for fastest response

### Speech-to-Text
- `--model STR` - Whisper model size: `tiny|base|small|medium|large-v3` (default: `small`)
- `--lang STR` - Language code: `zh|en|ja|etc.` (default: `zh`)
- `--prompt STR` - Optional context prompt for transcription

### Agent Reasoning
- `--skill {auto,chat,tool}` - Mode selection (default: `auto` = chat only)
- `--max_steps INT` - Max tool invocation steps (default: 6)
- `--temperature FLOAT` - LLM temperature 0.0-1.0 (default: 0.2)
- `--max_tokens INT` - Max output tokens per response (default: 512)
- `--root STR` - Filesystem sandbox root for tools (default: `D:\repo`)

## Default Behavior Changes

### Version: 2026-02-28

**Previous behavior:**
- Auto-detected tool intent from keywords (e.g., "file", "read", "directory")
- Could return tool JSON unexpectedly during chat

**Current behavior (Default: Chat Only)**
- Pure conversational mode with no tool invocation
- Tools only activated if:
  1. Message starts with `tool:` prefix, OR
  2. `--skill tool` flag is used on startup
- Cleaner, faster responses for general conversation

## Exit the Application

Press `Ctrl+C` or say:
- `quit`
- `exit`
- `bye`
- `退出` (Chinese)
- `再见` (Chinese)

## Example Sessions

### Example 1: Simple Chat

```powershell
$ python voice_agent_pipeline.py --auto --device 1 --rms 100 --silence 5 --model small --lang zh

[Turn 1] Listening...
🗣️  You: 你好，今天天气怎么样？

✓ Agent: 我无法获取实时天气信息，但我可以按照以下方式帮助您：
1. 如果您告诉我具体位置，我可以讨论该地区的气候类型
2. 您可以查看天气网站或应用程序获取最新信息
有其他问题吗？

[Turn 2] Listening...
🗣️  You: 再见

👋 Bye!
```

### Example 2: With File Access

```powershell
$ python voice_agent_pipeline.py --auto --device 1 --skill tool --model small --lang zh

[Turn 1] Listening...
🗣️  You: 项目结构是什么？

✓ Agent: 这个项目包含以下主要目录：
- agent/: Agent 的核心代码
- stt/: 语音转文本模块
- data/: 数据文件
...

[Turn 2] Listening...
🗣️  You: 退出

👋 Bye!
```

## Troubleshooting

### Issue: No speech detected
- Lower `--rms` value (e.g., `--rms 80`)
- Check microphone is unmuted and enabled
- Use `--debug_rms` to see live RMS levels
- Speak louder or closer to microphone

### Issue: Agent returns tool JSON in chat mode
- This was the old behavior; update code to latest version
- Ensure `agent.py` imports `CHAT_SYSTEM_PROMPT` from prompts
- Restart Python environment

### Issue: Microphone not found
- List devices: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- Find input device (max_input_channels > 0)
- Use correct device ID with `--device`

### Issue: Slow transcription
- Use smaller model: `--model tiny` or `--model base`
- Use `--quiet` to suppress debug output
- Disable `--debug_rms` flag

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Voice Agent Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Microphone Input                                           │
│      ↓                                                       │
│  VAD (Voice Activity Detection)  [audio_io.py]            │
│      ↓                                                       │
│  WAV Output                                                 │
│      ↓                                                       │
│  Faster-Whisper STT            [stt.py]                    │
│      ↓                                                       │
│  Text Transcript                                            │
│      ↓                                                       │
│  Agent Router                  [agent.py]                  │
│  ├─ Chat Mode (default)        → Simple response           │
│  └─ Tool Mode (optional)       → Tool invocation + logic   │
│      ↓                                                       │
│  LLM Response (Qwen3 via LM Studio)                        │
│      ↓                                                       │
│  Console Output                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Performance Tips

1. **Fastest response:** Use `--model tiny` + `--quiet`
2. **Best accuracy:** Use `--model medium` or `--model large-v3`
3. **Balanced:** Use `--model small` (recommended default)
4. **Lower latency:** Reduce `--silence` to 2-3 seconds
5. **Better speech detection:** Lower `--rms` if in quiet environment

## Module Dependencies

- **stt/audio_io.py**: VAD and audio recording
- **stt/stt.py**: Faster-Whisper integration
- **agent/agent.py**: Agent routing and chat logic
- **agent/prompts.py**: System prompts for chat/tool modes
- **agent/tools.py**: Tool sandbox for file/command access

## Notes

- Default behavior is **pure chat mode** for safety and simplicity
- Tools are explicitly opt-in to prevent unintended file/command access
- All voice input is processed locally (no cloud dependency for STT)
- LLM still requires LM Studio running locally
- Microphone audio is stored temporarily in `voice_temp.wav`

---

**Created:** Feb 28, 2026  
**Version:** 1.0
