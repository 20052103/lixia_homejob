# common_ai/tts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class TTSConfig:
    voice_contains: Optional[str] = None   # e.g. "Zira" / "David" / "Chinese"
    rate: int = 190                        # words per minute-ish
    volume: float = 1.0                    # 0.0 ~ 1.0

class Pyttsx3TTS:
    def __init__(self, cfg: Optional[TTSConfig] = None):
        self.cfg = cfg or TTSConfig()

        import pyttsx3
        self.engine = pyttsx3.init()  # SAPI5 on Windows

        # rate / volume
        self.engine.setProperty("rate", int(self.cfg.rate))
        self.engine.setProperty("volume", float(self.cfg.volume))

        # optional voice select
        if self.cfg.voice_contains:
            want = self.cfg.voice_contains.lower()
            for v in self.engine.getProperty("voices"):
                desc = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()
                if want in desc:
                    self.engine.setProperty("voice", v.id)
                    break

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()