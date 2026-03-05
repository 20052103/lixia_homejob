from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ------------------------------------------------
# add repo root to python path
# ------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


# ------------------------------------------------
# STT imports (from common_ai)
# ------------------------------------------------
from common_ai.stt.stt import FasterWhisperEngine
from common_ai.stt.audio_io import record_wav_auto_vad, VADConfig


# ------------------------------------------------
# Agent
# ------------------------------------------------
from agent.agent import LocalAgent, AgentConfig
from agent.config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, LM_STUDIO_MODEL_NAME, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P


# ------------------------------------------------
# TTS
# ------------------------------------------------
import pyttsx3


class TTS:

    def __init__(self, voice_contains=None, rate=185):

        self.engine = pyttsx3.init()

        self.engine.setProperty("rate", rate)

        if voice_contains:

            want = voice_contains.lower()

            for v in self.engine.getProperty("voices"):

                desc = f"{v.name} {v.id}".lower()

                if want in desc:
                    self.engine.setProperty("voice", v.id)
                    break

    def speak(self, text: str):

        self.engine.say(text)

        self.engine.runAndWait()


# ------------------------------------------------
# main
# ------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    # audio
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--rms", type=float, default=0.02)
    parser.add_argument("--silence", type=float, default=1.2)
    parser.add_argument("--max_seconds", type=int, default=15)

    # STT
    parser.add_argument("--model", default="small")
    parser.add_argument("--lang", default="en")

    # agent
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--skill", default="auto")

    # TTS
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--rate", type=int, default=185)

    args = parser.parse_args()

    # ------------------------------------------------
    # STT
    # ------------------------------------------------
    stt = FasterWhisperEngine(
        model_size=args.model,
        device="cuda",
        compute_type="float16",
    )

    vad_cfg = VADConfig(
        rms_threshold=args.rms,
        silence_seconds=args.silence,
        max_seconds=args.max_seconds,
    )

    # ------------------------------------------------
    # Agent
    # ------------------------------------------------
    cfg = AgentConfig(
        model_name="Qwen_Qwen3.5-27B-Q4_K_M.gguf",
        base_url="http://127.0.0.1:8080/v1",
        temperature=0.2,
        max_new_tokens=512,
    )

    agent = LocalAgent(cfg)

    # ------------------------------------------------
    # TTS
    # ------------------------------------------------
    tts = None
    if args.speak:
        tts = TTS(args.voice, args.rate)

    print("\n🎤 Voice Agent Ready\n")

    while True:

        try:

            print("🎤 listening...")

            wav = record_wav_auto_vad(
                device=args.device,
                vad_cfg=vad_cfg,
            )

            transcript = stt.transcribe_file(wav, language=args.lang)
            text = transcript.text

            if not text.strip():
                continue

            print(f"\n🧑 You: {text}")

            response = agent.chat(
                text,
                max_steps=args.max_steps,
                skill=args.skill,
            )

            print(f"\n🤖 Agent: {response}\n")

            if tts:
                tts.speak(response)

        except KeyboardInterrupt:
            print("\nbye")
            break


if __name__ == "__main__":
    main()