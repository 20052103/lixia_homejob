# stt_cli.py
from __future__ import annotations

import argparse
import os
import wave

from stt.stt import FasterWhisperEngine
from stt.audio_io import record_wav, record_wav_auto_vad, VADConfig


def wav_info(path: str) -> str:
    try:
        with wave.open(path, "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            n = wf.getnframes()
        sec = n / sr if sr else 0.0
        return f"channels={ch} sr={sr} frames={n} seconds={sec:.2f}"
    except Exception:
        return "not a wav or cannot parse wav header"


def list_devices() -> None:
    import sounddevice as sd

    devs = sd.query_devices()
    default_in, default_out = sd.default.device

    print("==== Audio Devices (sounddevice) ====")
    print(f"Default input id:  {default_in}")
    print(f"Default output id: {default_out}")
    print("")
    print(" id  in  out  default_sr   name")
    print("---- --- ---- -----------  ----")
    for i, d in enumerate(devs):
        max_in = int(d.get("max_input_channels", 0))
        max_out = int(d.get("max_output_channels", 0))
        sr = d.get("default_samplerate", None)
        name = d.get("name", "")
        tags = []
        if i == default_in:
            tags.append("DEFAULT_IN")
        if i == default_out:
            tags.append("DEFAULT_OUT")
        tag_str = f" [{' '.join(tags)}]" if tags else ""
        print(f"{i:>3}  {max_in:>2}  {max_out:>3}  {str(sr):>11}  {name}{tag_str}")
    print("====================================")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")

    p.add_argument("--file", type=str, default=None, help="Audio file path (wav/mp3/m4a...)")
    p.add_argument("--mic", action="store_true", help="Record from microphone")

    p.add_argument("--seconds", type=float, default=5.0, help="Mic record duration (core seconds)")
    p.add_argument("--device", type=int, default=None, help="Input device id (use --list-devices)")

    p.add_argument("--model", type=str, default="small", help="tiny|base|small|medium|large-v3 ...")
    p.add_argument("--lang", type=str, default="zh", help="e.g. zh, en, ja")
    p.add_argument("--prompt", type=str, default=None, help="Optional initial prompt")

    # Auto VAD
    p.add_argument("--auto", action="store_true", help="Auto start/stop by RMS threshold (VAD)")
    p.add_argument("--rms", type=float, default=100.0, help="RMS threshold (int16-equivalent)")
    p.add_argument("--silence", type=float, default=5.0, help="Stop if silent for N seconds")
    p.add_argument("--debug_rms", action="store_true", help="Print RMS while arming")

    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    engine = FasterWhisperEngine(model_size=args.model, device="cuda", compute_type="int8")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.mic:
        mic_path = os.path.join(base_dir, "mic.wav")

        if args.auto:
            vad_cfg = VADConfig(
                rms_start_threshold=args.rms,
                rms_stop_threshold=args.rms,
                stop_silence_seconds=args.silence,
                pre_roll_seconds=0.4,
                block_ms=30,
                max_record_seconds=120.0,
                debug_print=args.debug_rms,
            )
            wav = record_wav_auto_vad(
                mic_path,
                cfg=vad_cfg,
                device=args.device,
                wait_enter=False,
            )
        else:
            wav = record_wav(
                mic_path,
                seconds=args.seconds,
                device=args.device,
                pre_roll=0.4,
                post_roll=0.3,
                wait_enter=True,
            )

        print(f"[WAV] {wav}")
        print(f"[WAV] {wav_info(wav)}")
        t = engine.transcribe_file(wav, language=args.lang, prompt=args.prompt)
        print("\n[TRANSCRIPT]\n" + t.text)
        return

    if not args.file:
        raise SystemExit("Need --file <path> or --mic")

    fpath = args.file
    if not os.path.isabs(fpath):
        fpath = os.path.abspath(os.path.join(base_dir, fpath))

    if not os.path.exists(fpath):
        raise SystemExit(f"File not found: {fpath}")

    print(f"[FILE] {fpath}")
    t = engine.transcribe_file(fpath, language=args.lang, prompt=args.prompt)
    print("\n[TRANSCRIPT]\n" + t.text)


if __name__ == "__main__":
    main()