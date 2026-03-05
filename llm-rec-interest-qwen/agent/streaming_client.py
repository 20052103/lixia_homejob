from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional
from openai import OpenAI

@dataclass
class LLMConfig:
    base_url: str
    api_key: str = "x"
    model: str = "Qwen_Qwen3.5-27B-Q4_K_M.gguf"
    temperature: float = 0.2
    max_tokens: int = 512

class StreamingLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def stream_chat(self, messages: List[Dict], stop_flag=None) -> Iterable[str]:
        # stop_flag: threading.Event() or callable returning bool
        stream = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            stream=True,
        )
        try:
            for event in stream:
                if stop_flag is not None:
                    if callable(stop_flag) and stop_flag():
                        break
                    if hasattr(stop_flag, "is_set") and stop_flag.is_set():
                        break

                delta = event.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    yield token
        finally:
            # closing stream will stop generation server-side
            try:
                stream.close()
            except Exception:
                pass