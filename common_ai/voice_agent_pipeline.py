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
from pathlib import Path
from typing import Optional

# ===== Add llm-rec-interest-qwen to path =====
agent_dir = Path(r"D:\repo\lixia_homejob\llm-rec-interest-qwen")
if agent_dir.exists():
    sys.path.insert(0, str(agent_dir))

# ===== Import STT =====
from stt.stt import FasterWhisperEngine
from stt.audio_io import record_wav_auto_vad, VADConfig

# ===== Import Agent =====
from agent.agent import AgentConfig, LocalAgent
from agent.tools import ToolSandbox


def main():
    parser = argparse.ArgumentParser(description="Voice Agent Pipeline: Mic → STT → Agent")

    # ===== Voice input args =====
    parser.add_argument("--device", type=int, default=1, help="Microphone device ID (use --list-devices to find)")
    parser.add_argument("--auto", action="store_true", help="Use auto VAD (speech detection)")
    parser.add_argument("--rms", type=float, default=100.0, help="VAD RMS threshold (int16-equivalent scale)")
    parser.add_argument("--silence", type=float, default=5.0, help="Stop if silent for N seconds")
    parser.add_argument("--max_seconds", type=float, default=120.0, help="Max recording duration (safety cap)")
    parser.add_argument("--debug_rms", action="store_true", help="Print RMS values while recording")

    # ===== STT args =====
    parser.add_argument("--model", type=str, default="small", help="Whisper model size: tiny|base|small|medium|large-v3")
    parser.add_argument("--lang", type=str, default="zh", help="Language for transcription: zh, en, ja, etc.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional initial prompt for Whisper")

    # ===== Agent args =====
    parser.add_argument("--max_steps", type=int, default=6, help="Max reasoning steps for agent")
    parser.add_argument("--skill", choices=["auto", "chat", "tool"], default="auto", help="Force skill selection")
    parser.add_argument("--root", type=str, default=r"D:\repo", help="Sandbox root for agent tools")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max output tokens")

    args = parser.parse_args()

    # ===== Signal handler for graceful exit =====
    exit_flag = {'pressed': False}
    
    def signal_handler(signum, frame):
        exit_flag['pressed'] = True
        print("\n\n⏹️  Ctrl+C pressed. Shutting down...", flush=True)
    
    signal.signal(signal.SIGINT, signal_handler)

    # ===== Initialize STT Engine =====
    print("[INIT] Loading Faster-Whisper model ({}): {}...".format(args.model, args.lang), flush=True)
    stt_engine = FasterWhisperEngine(model_size=args.model, device="cuda", compute_type="int8")

    # ===== Initialize Agent =====
    print("[INIT] Initializing Qwen3 Agent (LM Studio)...", flush=True)
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
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )

    agent = LocalAgent(cfg=cfg, sandbox=sandbox)
    agent.messages.insert(1, {
        "role": "system",
        "content": f"Allowed filesystem root: {args.root}. Only use paths under this root."
    })

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
    print(f"  Model:       qwen3-coder-30b-a3b-instruct")
    print(f"  Skill:       {args.skill}")
    print(f"  Max Steps:   {args.max_steps}")
    print("\n✓ Ready to accept voice input. Speak into microphone.")
    print("  Type 'quit' or Ctrl+C to exit.\n")
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
        debug_print=args.debug_rms,
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
                wav_path = record_wav_auto_vad(
                    out_path=mic_path,
                    device=args.device,
                    cfg=vad_config,
                )
            else:
                print("⚠ Auto VAD not enabled. Use --auto flag to enable voice detection.")
                continue

            # Check if exit was pressed during recording
            if exit_flag['pressed']:
                break

            size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
            if size <= 44:  # Empty WAV (header only)
                print("⚠ [NO SPEECH DETECTED] Try lowering --rms (e.g., 80) or enable --debug_rms")
                continue

            # ===== 2. Transcribe to text =====
            print(f"📝 Transcribing ({size} bytes)...", flush=True)
            transcript = stt_engine.transcribe_file(wav_path, language=args.lang, prompt=args.prompt)
            user_text = (transcript.text or "").strip()

            if not user_text:
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
            print("🤖 Agent reasoning (press Ctrl+C to stop)...\n", flush=True)
            response = agent.chat(user_text=user_text, max_steps=args.max_steps, skill=args.skill)

            # Check again after agent response
            if exit_flag['pressed']:
                print("\n⏹️  Stopping...")
                break

            print(f"\n✓ Agent: {response}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            exit_flag['pressed'] = True
            print("\n\n👋 Interrupted. Bye!")
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
