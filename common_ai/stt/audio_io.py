# stt/audio_io.py
from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class RecordConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"


def record_wav(
    out_path: str,
    seconds: float = 5.0,
    cfg: Optional[RecordConfig] = None,
    device: Optional[int] = None,
    pre_roll: float = 0.4,
    post_roll: float = 0.3,
    wait_enter: bool = True,
) -> str:
    """
    Reliable microphone recorder for Windows:
    - Wait for user ENTER to start (optional)
    - Countdown for timing
    - Pre-roll / Post-roll to avoid cutting start/end
    - Always save to absolute path
    - Print RMS + bytes for sanity check

    Dependencies:
      pip install sounddevice numpy
    """
    cfg = cfg or RecordConfig()
    import numpy as np
    import sounddevice as sd

    out_path = os.path.abspath(out_path)

    if wait_enter:
        input(f"Press ENTER to start recording ({seconds:.1f}s)... ")

    for i in [3, 2, 1]:
        print(f"Recording starts in {i}...")
        time.sleep(0.35)

    total = seconds + pre_roll + post_roll
    frames = int(total * cfg.sample_rate)

    print(
        f"[REC] device={device} sr={cfg.sample_rate} ch={cfg.channels} "
        f"dur={total:.2f}s (core={seconds:.2f}s) -> {out_path}",
        flush=True,
    )

    audio = sd.rec(
        frames,
        samplerate=cfg.sample_rate,
        channels=cfg.channels,
        dtype=cfg.dtype,
        device=device,
    )
    sd.wait()
    print("[REC] done", flush=True)

    # Trim to the core window (remove pre/post roll)
    start = int(pre_roll * cfg.sample_rate)
    end = int((pre_roll + seconds) * cfg.sample_rate)
    audio = audio[start:end]

    # Write wav (assume int16)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(cfg.channels)
        wf.setsampwidth(2)  # int16 => 2 bytes
        wf.setframerate(cfg.sample_rate)
        wf.writeframes(audio.tobytes())

    # RMS (int16 scale)
    xf = audio.astype("float32")
    if xf.dtype.kind == "f":
        # if float stream ever happens, convert to int16-equivalent scale
        xf = xf * 32768.0
    rms = float((xf * xf).mean() ** 0.5)

    size = os.path.getsize(out_path)
    print(f"[REC] saved. frames={len(audio)} rms(int16)={rms:.2f} bytes={size}", flush=True)

    return out_path


# ----------------------------
# Auto VAD (RMS-based) recorder
# ----------------------------

@dataclass
class VADConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"

    # Start recording when chunk RMS > this (INT16-EQUIVALENT scale)
    rms_start_threshold: float = 100.0

    # Consider "silence" when chunk RMS < this (INT16-EQUIVALENT scale)
    rms_stop_threshold: float = 100.0

    # Stop recording when silence lasts this long
    stop_silence_seconds: float = 5.0

    # Audio chunk size for detection
    block_ms: int = 30

    # Keep some audio before speech start to avoid clipping beginning
    pre_roll_seconds: float = 0.4

    # Safety cap
    max_record_seconds: float = 120.0

    # Debug prints while arming
    debug_print: bool = False
    debug_print_interval: float = 0.8  # seconds
    debug_precision: int = 4  # decimals

    # If user Ctrl+C before START: optionally save pre-roll ring to help debug
    save_ring_on_interrupt: bool = True


def _rms_int16_equivalent(x):
    """
    Compute RMS in int16-equivalent scale regardless of stream dtype:
      - if x is float in [-1,1], scale by 32768
      - if x is int16, use as-is
    """
    import numpy as np

    xf = x.astype(np.float32)
    # If the incoming stream is float32/float64 (normalized), scale up
    if np.issubdtype(x.dtype, np.floating):
        xf *= 32768.0
    return float(np.sqrt(np.mean(xf * xf)))


def _to_int16_pcm(x):
    """
    Convert numpy audio to int16 PCM for wav writing.
    Accepts float [-1,1] or int16.
    """
    import numpy as np

    if np.issubdtype(x.dtype, np.integer):
        # assume int16 already
        return x.astype(np.int16, copy=False)

    # float -> int16 with clipping
    y = np.clip(x, -1.0, 1.0) * 32767.0
    return y.astype(np.int16)


def record_wav_auto_vad(
    out_path: str,
    cfg: Optional[VADConfig] = None,
    device: Optional[int] = None,
    wait_enter: bool = False,
) -> str:
    """
    Auto start/stop recording using RMS threshold (INT16-EQUIVALENT scale):

    - Arms immediately (or waits for ENTER if wait_enter=True)
    - Reads microphone in small blocks
    - When RMS > rms_start_threshold => START (prepend pre-roll)
    - After start: if RMS < rms_stop_threshold continuously for stop_silence_seconds => STOP
    - Saves WAV to out_path and returns absolute path

    Dependencies:
      pip install sounddevice numpy
    """
    cfg = cfg or VADConfig()
    import numpy as np
    import sounddevice as sd

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if wait_enter:
        input("Press ENTER to arm auto-recording... ")

    block_frames = int(cfg.sample_rate * cfg.block_ms / 1000)
    if block_frames <= 0:
        raise ValueError("block_ms too small")

    pre_roll_frames = int(cfg.pre_roll_seconds * cfg.sample_rate)
    silent_limit_frames = int(cfg.stop_silence_seconds * cfg.sample_rate)
    max_frames = int(cfg.max_record_seconds * cfg.sample_rate)

    ring: List[np.ndarray] = []
    ring_frames = 0

    recorded: List[np.ndarray] = []
    recorded_frames = 0

    started = False
    silent_frames = 0

    if cfg.debug_print:
        print(
            f"[AUTO-REC] device={device} sr={cfg.sample_rate} ch={cfg.channels} "
            f"block={cfg.block_ms}ms start_rms>{cfg.rms_start_threshold} "
            f"stop_if_rms<{cfg.rms_stop_threshold} for {cfg.stop_silence_seconds}s "
            f"pre_roll={cfg.pre_roll_seconds}s -> {out_path}",
            flush=True,
        )

    last_debug = 0.0
    interrupted = False
    last_seen_dtype = None

    try:
        with sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype=cfg.dtype,          # we request int16, but still handle float just in case
            device=device,
            blocksize=block_frames,
        ) as stream:
            while True:
                x, _ = stream.read(block_frames)
                last_seen_dtype = x.dtype

                rms = _rms_int16_equivalent(x)

                if cfg.debug_print and (time.time() - last_debug) >= cfg.debug_print_interval and not started:
                    last_debug = time.time()
                    # print more precision so "0.0" doesn't hide small values
                    fmt = f"{{:.{cfg.debug_precision}f}}"
                    print(
                        f"[AUTO-REC] arm rms(int16)={fmt.format(rms)} "
                        f"(threshold={cfg.rms_start_threshold}) dtype={x.dtype}",
                        flush=True,
                    )

                # Not started: ring buffer
                if not started:
                    ring.append(x.copy())
                    ring_frames += len(x)

                    while ring_frames > pre_roll_frames and ring:
                        drop = ring.pop(0)
                        ring_frames -= len(drop)

                    if rms > cfg.rms_start_threshold:
                        started = True
                        if cfg.debug_print:
                            print(f"[AUTO-REC] START (chunk_rms(int16)={rms:.2f}) dtype={x.dtype}", flush=True)

                        if ring:
                            recorded.extend(ring)
                            recorded_frames += ring_frames
                        ring.clear()
                        ring_frames = 0
                    continue

                # Started: record
                recorded.append(x.copy())
                recorded_frames += len(x)

                if rms < cfg.rms_stop_threshold:
                    silent_frames += len(x)
                else:
                    silent_frames = 0

                if silent_frames >= silent_limit_frames:
                    if cfg.debug_print:
                        print(f"[AUTO-REC] STOP (silence >= {cfg.stop_silence_seconds}s)", flush=True)
                    break

                if recorded_frames >= max_frames:
                    if cfg.debug_print:
                        print(f"[AUTO-REC] STOP (hit max_record_seconds={cfg.max_record_seconds})", flush=True)
                    break

    except KeyboardInterrupt:
        interrupted = True
        if cfg.debug_print:
            print("[AUTO-REC] Interrupted by user (Ctrl+C)", flush=True)

    if recorded_frames <= 0:
        if interrupted:
            if cfg.save_ring_on_interrupt and ring_frames > 0:
                audio = np.concatenate(ring, axis=0)
                pcm16 = _to_int16_pcm(audio)

                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(cfg.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(cfg.sample_rate)
                    wf.writeframes(pcm16.tobytes())

                size = os.path.getsize(out_path)
                rms_total = _rms_int16_equivalent(audio)
                if cfg.debug_print:
                    print(
                        f"[AUTO-REC] saved ring-only debug wav. frames={len(pcm16)} "
                        f"rms(int16)={rms_total:.2f} bytes={size} dtype={audio.dtype}",
                        flush=True,
                    )
                return out_path

            raise SystemExit(
                f"Auto recording canceled before START (no speech detected). "
                f"(last dtype={last_seen_dtype}) Try lowering --rms or enable --debug_rms."
            )

        raise RuntimeError(
            f"No audio recorded. (Speech never exceeded rms_start_threshold?) last dtype={last_seen_dtype}"
        )

    audio = np.concatenate(recorded, axis=0)
    pcm16 = _to_int16_pcm(audio)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(cfg.channels)
        wf.setsampwidth(2)
        wf.setframerate(cfg.sample_rate)
        wf.writeframes(pcm16.tobytes())

    size = os.path.getsize(out_path)
    rms_total = _rms_int16_equivalent(audio)
    if cfg.debug_print:
        print(
            f"[AUTO-REC] saved. frames={len(pcm16)} rms(int16)={rms_total:.2f} bytes={size} dtype={audio.dtype}",
            flush=True,
        )

    return out_path