import argparse
import time

from pathlib import Path

# ===== STT =====
from stt_cli import SpeechToText

# ===== Agent =====
from agent import LocalAgent, AgentConfig

# ===== TTS =====
from tts import Pyttsx3TTS, TTSConfig


def main():

    parser = argparse.ArgumentParser()

    # ===== Mic / VAD =====
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--rms", type=float, default=0.02)
    parser.add_argument("--silence", type=float, default=1.2)
    parser.add_argument("--max_seconds", type=int, default=15)
    parser.add_argument("--debug_rms", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # ===== STT =====
    parser.add_argument("--model", default="small")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--prompt", default="")

    # ===== Agent =====
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--skill", default="auto", choices=["auto", "chat", "tool"])
    parser.add_argument("--root", default=".")

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=512)

    # ===== TTS =====
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--rate", type=int, default=185)

    args = parser.parse_args()

    root = Path(args.root)

    # ===== STT init =====
    stt = SpeechToText(
        model_size=args.model,
        device="cuda",
        compute_type="float16",
        language=args.lang,
        initial_prompt=args.prompt,
    )

    # ===== Agent init =====
    cfg = AgentConfig(
        model_name="Qwen_Qwen3.5-27B-Q4_K_M.gguf",
        base_url="http://127.0.0.1:8080/v1",
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    agent = LocalAgent(cfg)

    # ===== TTS init =====
    tts = None
    if args.speak:
        tts = Pyttsx3TTS(
            TTSConfig(
                voice_contains=args.voice,
                rate=args.rate,
            )
        )

    print("\n🎤 Voice Agent Ready. Ctrl+C to exit.\n")

    while True:

        try:

            print("🎤 Listening...")

            text = stt.record_and_transcribe(
                device=args.device,
                auto=args.auto,
                rms=args.rms,
                silence=args.silence,
                max_seconds=args.max_seconds,
                debug_rms=args.debug_rms,
                quiet=args.quiet,
            )

            if not text.strip():
                continue

            print(f"\n🧑 You: {text}")

            response = agent.chat(
                text,
                max_steps=args.max_steps,
                skill=args.skill,
            )

            print(f"\n🤖 Agent: {response}\n")

            # ===== Speak =====
            if tts:
                tts.speak(response)

        except KeyboardInterrupt:
            print("\n👋 Exit.")
            break

        except Exception as e:
            print("Error:", e)
            time.sleep(1)


if __name__ == "__main__":
    main()