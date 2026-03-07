from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional


# ------------------------------------------------
# repo root -> import common_ai + agent package
# ------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LLM_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_REPO_ROOT))


# ------------------------------------------------
# STT imports (from common_ai)
# ------------------------------------------------
from common_ai.stt.stt import FasterWhisperEngine
from common_ai.stt.audio_io import (
    VADConfig,
    record_wav_auto_vad,
    _rms_int16_equivalent,
)

# ------------------------------------------------
# Agent
# ------------------------------------------------
from agent import LocalAgent, AgentConfig

# ------------------------------------------------
# Audio input monitor for barge-in
# ------------------------------------------------
import sounddevice as sd


_SENTENCE_RE = re.compile(r"(?<=[\.\!\?\。\！\？])\s+")
_SAFE_VOICE_RE = re.compile(r"[^a-zA-Z0-9 _\-]")


def split_ready_sentences(buffer: str) -> tuple[List[str], str]:
    if not buffer.strip():
        return [], buffer

    parts = _SENTENCE_RE.split(buffer)
    if len(parts) <= 1:
        return [], buffer

    ready = [p.strip() for p in parts[:-1] if p and p.strip()]
    tail = parts[-1] if parts else ""
    return ready, tail


def _sanitize_voice_name(voice: Optional[str]) -> str:
    if not voice:
        return ""
    return _SAFE_VOICE_RE.sub("", voice).strip()


class TTSWorker:
    """
    Windows TTS via PowerShell + System.Speech.
    Uses a subprocess per chunk so we can terminate immediately on interrupt.
    """

    def __init__(self, voice_contains: Optional[str] = None, rate: int = 0, volume: int = 100):
        self.voice_contains = _sanitize_voice_name(voice_contains)
        self.rate = max(-10, min(10, int(rate)))
        self.volume = max(0, min(100, int(volume)))

        self.q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stop = threading.Event()
        self._current_proc_lock = threading.Lock()
        self._current_proc: Optional[subprocess.Popen] = None
        self._speaking = threading.Event()

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _build_ps_command(self, text: str) -> str:
        escaped = text.replace("'", "''")
        voice_part = ""
        if self.voice_contains:
            voice_part = (
                f"$want='{self.voice_contains.lower()}'; "
                "$v=$s.GetInstalledVoices() | "
                "ForEach-Object {$_.VoiceInfo.Name} | "
                "Where-Object { $_.ToLower().Contains($want) } | "
                "Select-Object -First 1; "
                "if ($v) { $s.SelectVoice($v) }; "
            )

        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {self.rate}; "
            f"$s.Volume = {self.volume}; "
            f"{voice_part}"
            f"$s.Speak('{escaped}');"
        )
        return ps

    def _speak_blocking(self, text: str):
        cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            self._build_ps_command(text),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with self._current_proc_lock:
            self._current_proc = proc
        self._speaking.set()

        try:
            while True:
                if self._stop.is_set():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

                ret = proc.poll()
                if ret is not None:
                    break

                time.sleep(0.03)
        finally:
            with self._current_proc_lock:
                self._current_proc = None
            self._speaking.clear()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return

            try:
                if not self._stop.is_set():
                    self._speak_blocking(item)
            except Exception as e:
                print(f"[TTS] error: {e}", flush=True)
            finally:
                self.q.task_done()

    def speak_async(self, text: str):
        text = (text or "").strip()
        if text and not self._stop.is_set():
            self.q.put(text)

    def interrupt(self):
        self._stop.set()

        with self._current_proc_lock:
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
                self._current_proc = None

        while True:
            try:
                item = self.q.get_nowait()
                self.q.task_done()
                if item is None:
                    break
            except queue.Empty:
                break

        self._speaking.clear()

    def reset_after_interrupt(self):
        self._stop.clear()

    def is_busy(self) -> bool:
        return self._speaking.is_set() or (not self.q.empty())

    def close(self):
        self.interrupt()
        self.q.put(None)
        try:
            self.worker.join(timeout=2.0)
        except Exception:
            pass


class AdaptiveBargeInMonitor:
    """
    Adaptive interrupt monitor:
    - calibrates ambient baseline before each answer
    - trigger threshold = max(min_abs_rms, baseline * multiplier)
    """

    def __init__(
        self,
        device: Optional[int],
        min_abs_rms: float = 600.0,
        multiplier: float = 1.3,
        hold_ms: int = 30,
        calibrate_ms: int = 120,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        block_ms: int = 30,
        debug: bool = False,
    ):
        self.device = device
        self.min_abs_rms = float(min_abs_rms)
        self.multiplier = float(multiplier)
        self.hold_ms = int(hold_ms)
        self.calibrate_ms = int(calibrate_ms)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.dtype = dtype
        self.block_ms = int(block_ms)
        self.debug = bool(debug)

        self._armed = threading.Event()
        self._stop = threading.Event()
        self._triggered = threading.Event()
        self._baseline_rms = 0.0
        self._trigger_rms = self.min_abs_rms
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def arm(self):
        self._triggered.clear()
        self._armed.set()

    def disarm(self):
        self._armed.clear()

    def triggered(self) -> bool:
        return self._triggered.is_set()

    def clear_trigger(self):
        self._triggered.clear()

    def close(self):
        self._stop.set()
        self._armed.clear()

    def current_threshold(self) -> float:
        with self._lock:
            return self._trigger_rms

    def current_baseline(self) -> float:
        with self._lock:
            return self._baseline_rms

    def _run(self):
        block_frames = int(self.sample_rate * self.block_ms / 1000)
        consecutive_needed = max(1, int(self.hold_ms / self.block_ms))
        calibrate_blocks = max(1, int(self.calibrate_ms / self.block_ms))
        consecutive = 0

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device,
                blocksize=block_frames,
            ) as stream:
                while not self._stop.is_set():
                    if not self._armed.is_set():
                        consecutive = 0
                        time.sleep(0.02)
                        continue

                    rms_values = []
                    for _ in range(calibrate_blocks):
                        if self._stop.is_set() or not self._armed.is_set():
                            break
                        x, _ = stream.read(block_frames)
                        rms_values.append(_rms_int16_equivalent(x))

                    if not rms_values:
                        continue

                    baseline = sum(rms_values) / len(rms_values)
                    trigger = max(self.min_abs_rms, baseline * self.multiplier)

                    with self._lock:
                        self._baseline_rms = baseline
                        self._trigger_rms = trigger

                    if self.debug:
                        print(
                            f"[BARGE-IN] baseline={baseline:.1f}, trigger={trigger:.1f}",
                            flush=True,
                        )

                    consecutive = 0
                    while self._armed.is_set() and not self._stop.is_set():
                        x, _ = stream.read(block_frames)
                        rms = _rms_int16_equivalent(x)

                        if rms >= trigger:
                            consecutive += 1
                        else:
                            consecutive = 0

                        if self.debug:
                            print(
                                f"[BARGE-IN] rms={rms:.1f}, trigger={trigger:.1f}, hit={consecutive}/{consecutive_needed}",
                                flush=True,
                            )

                        if consecutive >= consecutive_needed:
                            self._triggered.set()
                            self._armed.clear()
                            consecutive = 0
                            break

        except Exception as e:
            print(f"[BARGE-IN] disabled: {e}", flush=True)


def should_accept_wake(text: str, wake_words: List[str]) -> bool:
    if not wake_words:
        return True

    t = text.lower()
    return any(w.strip().lower() in t for w in wake_words if w.strip())


def build_vad_cfg(args) -> VADConfig:
    cfg = VADConfig()
    cfg.sample_rate = 16000
    cfg.channels = 1
    cfg.dtype = "int16"

    cfg.rms_start_threshold = float(args.rms_start)
    cfg.rms_stop_threshold = float(args.rms_stop)
    cfg.stop_silence_seconds = float(args.silence)
    cfg.block_ms = int(args.block_ms)
    cfg.pre_roll_seconds = float(args.pre_roll)
    cfg.max_record_seconds = float(args.max_seconds)
    cfg.debug_print = bool(args.debug_rms)
    cfg.debug_print_interval = 0.8
    return cfg


def transcribe_file(stt: FasterWhisperEngine, wav_path: str, lang: Optional[str], prompt: str) -> str:
    result = stt.transcribe_file(
        wav_path,
        language=None if lang in ("", "auto", None) else lang,
        prompt=prompt or None,
    )
    return (result.text or "").strip()


def drain_tts_with_interrupt(tts: Optional[TTSWorker], barge: Optional[AdaptiveBargeInMonitor], enable_barge_in: bool):
    if tts is None:
        return

    while tts.is_busy():
        if enable_barge_in and barge is not None and barge.triggered():
            tts.interrupt()
            print("\n[interrupt] user speech detected during TTS.\n", flush=True)
            return
        time.sleep(0.03)


def stream_to_tts(
    agent: LocalAgent,
    user_text: str,
    tts: Optional[TTSWorker],
    barge: Optional[AdaptiveBargeInMonitor],
    enable_barge_in: bool,
) -> str:
    collected: List[str] = []
    sentence_buf = ""
    stop_event = threading.Event()

    if tts is not None:
        tts.reset_after_interrupt()

    if tts is not None and barge is not None and enable_barge_in:
        barge.clear_trigger()
        barge.arm()
        time.sleep(max(0.03, barge.calibrate_ms / 1000.0 + 0.02))

    try:
        for token in agent.stream_chat_simple(user_text, stop_event=stop_event):
            if enable_barge_in and barge is not None and barge.triggered():
                stop_event.set()
                if tts is not None:
                    tts.interrupt()
                print("\n[interrupt] user speech detected during generation.\n", flush=True)
                break

            print(token, end="", flush=True)
            collected.append(token)

            if tts is not None:
                sentence_buf += token
                ready, sentence_buf = split_ready_sentences(sentence_buf)
                for s in ready:
                    tts.speak_async(s)

        if tts is not None and sentence_buf.strip():
            tts.speak_async(sentence_buf.strip())

        drain_tts_with_interrupt(tts, barge, enable_barge_in)

    finally:
        if barge is not None:
            barge.disarm()

    print("")
    return "".join(collected).strip()


def main():
    parser = argparse.ArgumentParser()

    # audio / VAD
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--auto", action="store_true", default=True, help="Kept for compatibility; auto VAD is always used.")
    parser.add_argument("--rms_start", type=float, default=900.0)
    parser.add_argument("--rms_stop", type=float, default=500.0)
    parser.add_argument("--silence", type=float, default=1.0)
    parser.add_argument("--max_seconds", type=float, default=20.0)
    parser.add_argument("--block_ms", type=int, default=30)
    parser.add_argument("--pre_roll", type=float, default=0.4)
    parser.add_argument("--debug_rms", action="store_true", default=False)

    # STT
    parser.add_argument("--model", default="small")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--prompt", default="")

    # agent
    parser.add_argument("--skill", default="chat", choices=["auto", "chat", "tool"])
    parser.add_argument("--max_steps", type=int, default=4)

    # TTS / voice UX
    speak_group = parser.add_mutually_exclusive_group()
    speak_group.add_argument("--speak", dest="speak", action="store_true", help="Enable spoken audio output.")
    speak_group.add_argument("--no-speak", dest="speak", action="store_false", help="Disable spoken audio output.")
    parser.set_defaults(speak=True)

    parser.add_argument("--voice", default="Zira")
    parser.add_argument("--rate", type=int, default=0, help="System.Speech rate: -10 to 10")
    parser.add_argument("--volume", type=int, default=100, help="0 to 100")

    # interrupt / adaptive barge-in
    parser.add_argument("--barge_in", action="store_true", default=True)
    parser.add_argument("--barge_min_abs_rms", type=float, default=600.0)
    parser.add_argument("--barge_multiplier", type=float, default=1.3)
    parser.add_argument("--barge_hold_ms", type=int, default=30)
    parser.add_argument("--barge_calibrate_ms", type=int, default=120)
    parser.add_argument("--barge_debug", action="store_true", default=False)

    # wake words
    parser.add_argument("--wake_words", default="")

    # debug
    parser.add_argument("--save_wav", action="store_true", default=False)

    args = parser.parse_args()

    stt = FasterWhisperEngine(
        model_size=args.model,
        device="cuda",
        compute_type="float16",
    )

    cfg = AgentConfig()
    agent = LocalAgent(cfg=cfg, sandbox=None)

    tts = TTSWorker(
        voice_contains=args.voice,
        rate=args.rate,
        volume=args.volume,
    ) if args.speak else None

    barge = (
        AdaptiveBargeInMonitor(
            device=args.device,
            min_abs_rms=args.barge_min_abs_rms,
            multiplier=args.barge_multiplier,
            hold_ms=args.barge_hold_ms,
            calibrate_ms=args.barge_calibrate_ms,
            debug=args.barge_debug,
        )
        if args.speak and args.barge_in
        else None
    )

    wake_words = [w.strip() for w in args.wake_words.split(",") if w.strip()]

    print("\n[Voice Agent Ready] Ctrl+C to exit.\n", flush=True)
    print(
        f"[defaults] speak={args.speak}, voice={args.voice}, barge_in={args.barge_in}, "
        f"barge_min_abs_rms={args.barge_min_abs_rms}, barge_multiplier={args.barge_multiplier}, "
        f"barge_hold_ms={args.barge_hold_ms}, barge_calibrate_ms={args.barge_calibrate_ms}\n",
        flush=True,
    )

    try:
        while True:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name

                vad_cfg = build_vad_cfg(args)

                print("🎤 Listening...", flush=True)
                wav_path = record_wav_auto_vad(
                    out_path=wav_path,
                    cfg=vad_cfg,
                    device=args.device,
                    wait_enter=False,
                )

                text = transcribe_file(stt, wav_path, args.lang, args.prompt)

                if not args.save_wav:
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                if not text:
                    continue

                if not should_accept_wake(text, wake_words):
                    print(f"\n[ignored - no wake word] {text}\n", flush=True)
                    continue

                print(f"\nYou> {text}\n", flush=True)

                if args.speak:
                    _ = stream_to_tts(
                        agent=agent,
                        user_text=text,
                        tts=tts,
                        barge=barge,
                        enable_barge_in=bool(args.barge_in),
                    )
                else:
                    answer = agent.chat(text, max_steps=args.max_steps, skill=args.skill)
                    print(f"Agent> {answer}\n", flush=True)

            except SystemExit as e:
                print(f"[recorder] {e}", flush=True)
                continue
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[error] {e}", flush=True)
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[shutdown] stopping...\n", flush=True)
        if barge is not None:
            barge.close()
        if tts is not None:
            tts.close()
        print("Bye.\n", flush=True)


if __name__ == "__main__":
    main()