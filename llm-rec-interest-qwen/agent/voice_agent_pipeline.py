# -*- coding: utf-8 -*-
"""
Voice Agent Pipeline
=====================
语音输入 → STT → Agent推理 → 输出

Flow:
  1. Record audio from microphone with VAD
  2. Transcribe to text using Faster-Whisper
  3. Send to Qwen3 Agent for reasoning
  4. Print agent response
"""

import argparse
import os
import sys
import signal
import threading
import time
from pathlib import Path
from typing import Optional

# ===== Add common_ai (STT module) to path =====
common_ai_dir = Path(__file__).parent.parent.parent / "common_ai"
if common_ai_dir.exists() and str(common_ai_dir) not in sys.path:
    sys.path.insert(0, str(common_ai_dir))

# ===== Add parent directory to path for agent imports =====
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# ===== Import STT (from common_ai) =====
from stt.stt import FasterWhisperEngine
from stt.audio_io import record_wav_auto_vad, VADConfig

# ===== Import Agent (from parent/agent) =====
from agent.agent import AgentConfig, LocalAgent
from agent.tools import ToolSandbox
from agent.config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL_NAME, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P


# ===== TTS (Text-to-Speech) Helpers =====
def init_tts_engine():
    """Initialize pyttsx3 TTS engine for Windows."""
    try:
        import pyttsx3
        engine = pyttsx3.init('sapi5')  # Use SAPI5 for Windows (better support)
        engine.setProperty('rate', 150)  # Speech rate
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        return engine
    except ImportError:
        print("⚠️  pyttsx3 not installed. Install with: pip install pyttsx3")
        return None
    except Exception as e:
        print(f"⚠️  Failed to initialize TTS: {e}")
        return None


def list_available_voices(engine):
    """List all available TTS voices."""
    if engine is None:
        return
    try:
        voices = engine.getProperty('voices')
        print("\n=== Available TTS Voices ===")
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name}")
        print()
    except Exception as e:
        print(f"Error listing voices: {e}")


def speak_text(text: str, voice_name: Optional[str] = None):
    """Speak text using TTS engine in a separate thread (non-blocking)."""
    
    def _speak():
        try:
            import pyttsx3
            # Create a fresh engine for each speech operation
            engine = pyttsx3.init('sapi5')
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            # Set voice if specified
            if voice_name:
                voices = engine.getProperty('voices')
                for voice in voices:
                    if voice_name.lower() in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
            
            engine.say(text)
            engine.runAndWait()
        except ImportError:
            print("⚠️  pyttsx3 not installed")
        except Exception as e:
            print(f"TTS Error: {e}")
    
    # Run TTS in a separate thread to avoid blocking
    tts_thread = threading.Thread(target=_speak, daemon=True)
    tts_thread.start()


def main():
    parser = argparse.ArgumentParser(description="Voice Agent Pipeline: Mic → STT → Agent")

    # ===== Voice input args =====
    parser.add_argument("--device", type=int, default=1, help="Microphone device ID (use --list-devices to find)")
    parser.add_argument("--auto", action="store_true", help="Use auto VAD (speech detection)")
    parser.add_argument("--rms", type=float, default=100.0, help="VAD RMS threshold (int16-equivalent scale)")
    parser.add_argument("--silence", type=float, default=5.0, help="Stop if silent for N seconds")
    parser.add_argument("--max_seconds", type=float, default=120.0, help="Max recording duration (safety cap)")
    parser.add_argument("--debug_rms", action="store_true", help="Print RMS values while recording (SLOW)")
    parser.add_argument("--quiet", action="store_true", help="Suppress all debug output")

    # ===== STT args =====
    parser.add_argument("--model", type=str, default="small", help="Whisper model size: tiny|base|small|medium|large-v3")
    parser.add_argument("--lang", type=str, default="zh", help="Language for transcription: zh, en, ja, etc.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional initial prompt for Whisper")

    # ===== Agent args =====
    parser.add_argument("--max_steps", type=int, default=6, help="Max reasoning steps for agent")
    parser.add_argument("--skill", choices=["auto", "chat", "tool"], default="auto", help="Force skill mode (default=chat, use 'tool' for file/command access)")
    parser.add_argument("--root", type=str, default=r"D:\repo", help="Sandbox root for agent tools")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens")

    # ===== TTS args =====
    parser.add_argument("--speak", action="store_true", help="Speak agent response using text-to-speech")
    parser.add_argument("--voice", type=str, default=None, help="TTS voice name (e.g., 'Zira', 'David'). List available with --list-voices")
    parser.add_argument("--list-voices", action="store_true", help="List available TTS voices and exit")

    args = parser.parse_args()

    # ===== Handle --list-voices =====
    if args.list_voices:
        tts_engine = init_tts_engine()
        list_available_voices(tts_engine)
        return

    # ===== Signal handler for graceful exit =====
    exit_flag = {'pressed': False}
    
    def signal_handler(signum, frame):
        exit_flag['pressed'] = True
        print("\n\n⏹️  Ctrl+C pressed. Shutting down...", flush=True)
    
    signal.signal(signal.SIGINT, signal_handler)

    # ===== Initialize TTS Engine (if --speak enabled) =====
    if args.speak:
        if not args.quiet:
            print("[INIT] TTS Engine enabled (pyttsx3)...", flush=True)

    # ===== Initialize STT Engine =====
    if not args.quiet:
        print("[INIT] Loading Faster-Whisper model ({}): {}...".format(args.model, args.lang), flush=True)
    stt_engine = FasterWhisperEngine(model_size=args.model, device="cuda", compute_type="int8")

    # ===== Initialize Agent =====
    if not args.quiet:
        print("[INIT] Initializing Qwen3 Agent (LM Studio)...", flush=True)
    sandbox = ToolSandbox(
        allowed_roots=[args.root],
        allowed_cmd_prefixes=["python", "py", "git", "dir", "ls", "pip"],
        max_read_bytes=200_000,
        max_output_chars=40_000,
        cwd=args.root,
    )

    cfg = AgentConfig(
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        model_name=LM_STUDIO_MODEL_NAME,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )

    agent = LocalAgent(cfg=cfg, sandbox=sandbox)
    agent.messages.insert(1, {
        "role": "system",
        "content": f"Allowed filesystem root: {args.root}. Only use paths under this root."
    })

    if not args.quiet:
        print("\n" + "=" * 60)
        print("🎤 VOICE AGENT PIPELINE READY")
        print("=" * 60)
        print("Voice Input Settings:")
        print(f"  Device:      {args.device}")
        if args.auto:
            print(f"  VAD RMS:     {args.rms} (int16-equivalent)")
            print(f"  Silence:     {args.silence}s")
        print("\nSTT Settings:")
        print(f"  Model:       {args.model}")
        print(f"  Language:    {args.lang}")
        print("\nAgent Settings:")
        print(f"  Model:       qwen3.5-27b")
        print(f"  Skill:       {args.skill}")
        print(f"  Max Steps:   {args.max_steps}")
        print("\n✓ Ready to accept voice input. Speak into microphone.")
        print("  Type 'quit' or Ctrl+C to exit.")
        print("  (To enable file/command access: start with 'tool: ' or use --skill tool)\n")
        print("=" * 60 + "\n")

    # ===== Main loop =====
    vad_config = VADConfig(
        sample_rate=16000,
        channels=1,
        rms_start_threshold=args.rms,
        rms_stop_threshold=args.rms,
        stop_silence_seconds=args.silence,
        pre_roll_seconds=0.4,
        block_ms=30,
        max_record_seconds=args.max_seconds,
        debug_print=args.debug_rms and not args.quiet,  # Only debug if explicitly requested
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    turn = 0

    while not exit_flag['pressed']:
        try:
            # Check exit flag at start of loop
            if exit_flag['pressed']:
                break

            # ===== 1. Record voice =====
            turn += 1
            print(f"\n[Turn {turn}] Listening... (speak now, or Ctrl+C to quit)")
            mic_path = os.path.join(base_dir, "voice_temp.wav")

            if args.auto:
                try:
                    wav_path = record_wav_auto_vad(
                        out_path=mic_path,
                        device=args.device,
                        cfg=vad_config,
                    )
                except KeyboardInterrupt:
                    exit_flag['pressed'] = True
                    print("\n⏹️  Recording canceled by user")
                    break
            else:
                print("⚠ Auto VAD not enabled. Use --auto flag to enable voice detection.")
                continue

            # Check if exit was pressed during recording
            if exit_flag['pressed']:
                break

            size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
            if size <= 44:  # Empty WAV (header only)
                if not args.quiet:
                    print("⚠ [NO SPEECH DETECTED] Try lowering --rms (e.g., 80) or enable --debug_rms")
                continue

            # ===== 2. Transcribe to text =====

            
            if not args.quiet:
                print(f"📝 Transcribing ({size} bytes)...", flush=True)
            try:
                transcript = stt_engine.transcribe_file(wav_path, language=args.lang, prompt=args.prompt)
            except KeyboardInterrupt:
                exit_flag['pressed'] = True
                print("\n⏹️  Transcription canceled by user")
                break
            user_text = (transcript.text or "").strip()

            if not user_text:
                if not args.quiet:
                    print("⚠ [EMPTY TRANSCRIPT] No speech recognized")
                continue

            print(f"🗣️  You: {user_text}\n")

            # ===== 3. Check for exit commands =====
            if user_text.lower() in ("quit", "exit", "bye", "退出", "再见"):
                print("👋 Bye!")
                break

            # Check if exit was pressed while processing
            if exit_flag['pressed']:
                break

            # ===== 4. Send to agent =====
            if not args.quiet:
                print("🤖 Agent reasoning (press Ctrl+C to stop)...\n", flush=True)
            
            # Run agent.chat() with interrupt support
            response = None
            
            agent_exception = None
            
            def run_agent():
                nonlocal response, agent_exception
                try:
                    response = agent.chat(user_text=user_text, max_steps=args.max_steps, skill=args.skill)
                except Exception as e:
                    agent_exception = e
            
            # Run agent in background thread for interruptibility
            agent_thread = threading.Thread(target=run_agent, daemon=True)
            agent_thread.start()
            
            # Wait for agent response or Ctrl+C
            while agent_thread.is_alive():
                if exit_flag['pressed']:
                    if not args.quiet:
                        print("\n⏹️  Interrupt signal received. Waiting for agent to finish...")
                    # Note: We can't forcefully stop the thread, but user can Ctrl+C again
                    break
                agent_thread.join(timeout=0.5)
            
            # Handle agent errors
            if agent_exception:
                raise agent_exception
            
            if response is None:
                print("⚠ Agent did not provide a response")
                continue
            
            # Check again after agent response
            if exit_flag['pressed']:
                if not args.quiet:
                    print("\n⏹️  Stopping...")
                break

            print(f"\n✓ Agent: {response}\n")
            
            # ===== Speak response if enabled =====
            if args.speak:
                if not args.quiet:
                    print("🔊 Speaking response...", flush=True)
                speak_text(response, args.voice)
                # Give TTS thread a moment to start
                time.sleep(0.5)
            
            if not args.quiet:
                print("-" * 60)

        except KeyboardInterrupt:
            exit_flag['pressed'] = True
            print("\n\n👋 Interrupted. Exiting...")
            break
        except SystemExit as e:
            # Let SystemExit propagate (for sys.exit() calls)
            if "canceled" not in str(e).lower():
                raise
            print(f"⚠ Recording canceled: {e}")
            continue
        except Exception as e:
            import traceback
            print(f"\n❌ ERROR: {e}")
            if args.debug_rms:
                traceback.print_exc()
            print()
    
    print("\n✅ Voice Agent Pipeline closed.\n", flush=True)


if __name__ == "__main__":
    main()