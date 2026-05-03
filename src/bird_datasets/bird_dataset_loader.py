from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class BirdSample:
    question_id: int
    db_id: str
    question: str
    sql: str
    evidence: str = ""
    difficulty: Optional[str] = None


class BirdDatasetLoader:
    """Lightweight loader for BIRD JSON/JSONL files."""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load(self) -> List[BirdSample]:
        if self.data_path.suffix.lower() == ".jsonl":
            return list(self._load_jsonl())
        return list(self._load_json())

    def iter_samples(self) -> Iterator[BirdSample]:
        if self.data_path.suffix.lower() == ".jsonl":
            yield from self._load_jsonl()
        else:
            yield from self._load_json()

    def _load_json(self) -> Iterator[BirdSample]:
        with self.data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            yield self._to_sample(item)

    def _load_jsonl(self) -> Iterator[BirdSample]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                yield self._to_sample(item)

    @staticmethod
    def _to_sample(item: dict) -> BirdSample:
        return BirdSample(
            question_id=int(item.get("question_id", item.get("id", 0))),
            db_id=item["db_id"],
            question=item["question"],
            sql=item.get("SQL", item.get("sql", "")),
            evidence=item.get("evidence", ""),
            difficulty=item.get("difficulty"),
        )

