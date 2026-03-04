# stt/stt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: Optional[List[Segment]] = None
    meta: Optional[Dict[str, Any]] = None


class SpeechToTextEngine(Protocol):
    def transcribe_file(
        self,
        path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Transcript: ...

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Transcript: ...


class FasterWhisperEngine:
    """
    Local STT using faster-whisper (CTranslate2).
    """
    def __init__(self, model_size: str = "small", device: str = "cuda", compute_type: str = "int8"):
        from faster_whisper import WhisperModel  # pip install faster-whisper
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_file(self, path: str, language: Optional[str] = None, prompt: Optional[str] = None) -> Transcript:
        segments, info = self.model.transcribe(path, language=language, initial_prompt=prompt)
        segs: List[Segment] = []
        texts: List[str] = []
        for s in segments:
            segs.append(Segment(float(s.start), float(s.end), s.text))
            texts.append(s.text)
        return Transcript(
            text="".join(texts).strip(),
            segments=segs,
            meta={"language": getattr(info, "language", None), "duration": getattr(info, "duration", None)},
        )

    def transcribe_bytes(self, audio_bytes: bytes, mime_type: str, language: Optional[str] = None, prompt: Optional[str] = None) -> Transcript:
        import tempfile, os
        suffix = ".wav" if "wav" in (mime_type or "") else ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            return self.transcribe_file(tmp, language=language, prompt=prompt)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass