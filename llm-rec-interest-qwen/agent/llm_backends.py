from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Protocol


class ChatBackend(Protocol):
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        ...


@dataclass
class LMStudioBackend:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "qwen3-coder-30b-a3b-instruct"

    def __post_init__(self) -> None:
        # Lazy import so repo users without openai installed won't break unless they use this backend
        from openai import OpenAI
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


def make_backend(name: str) -> ChatBackend:
    name = (name or "").lower()
    if name in ("lmstudio", "lm", "local"):
        return LMStudioBackend()
    raise ValueError(f"Unknown llm backend: {name}")