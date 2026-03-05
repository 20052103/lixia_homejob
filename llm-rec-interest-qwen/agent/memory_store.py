import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

@dataclass
class MemoryItem:
    role: str
    content: str

class JsonMemoryStore:
    def __init__(self, path: str, max_items: int = 20):
        self.path = Path(path)
        self.max_items = max_items
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self) -> List[Dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, role: str, content: str):
        items = self.load()
        items.append(asdict(MemoryItem(role=role, content=content)))
        items = items[-self.max_items:]
        self.path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        return items