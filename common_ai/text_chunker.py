import re
from typing import Iterable

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\。\！\？])\s+|(\n{2,})")

def chunk_text_stream(token_stream: Iterable[str], min_chars: int = 80) -> Iterable[str]:
    buf = ""
    for t in token_stream:
        buf += t
        # 如果有句末标点，尽量切块
        parts = _SENT_SPLIT.split(buf)
        # parts 会包含 None，把它们拼回去
        if len(parts) > 1:
            # 取前面可输出的部分，留最后一段继续攒
            out = "".join(p for p in parts[:-1] if p)
            tail = parts[-1] if parts[-1] else ""
            if len(out.strip()) >= min_chars:
                yield out.strip()
                buf = tail
        # 兜底：太长也切
        if len(buf) > 400:
            yield buf.strip()
            buf = ""
    if buf.strip():
        yield buf.strip()