from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pruning.schema_loader import ColumnRef, DatabaseSchema


@dataclass
class RankedColumn:
    full_name: str
    table: str
    column: str
    score: float


class EmbeddingRetriever:
    """Column retriever using sentence-transformers (hashing only if forced)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        hashing_dim: int = 512,
        force_hashing: bool = False,
    ) -> None:
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.hashing_dim = hashing_dim
        self.force_hashing = force_hashing
        self.backend = "hashing" if force_hashing else "sentence-transformers"
        self._model = None
        self._hash_cache: Dict[str, List[Tuple[ColumnRef, Dict[int, float], float]]] = {}

        if not self.force_hashing:
            from sentence_transformers import SentenceTransformer  # type: ignore

            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load sentence-transformers model. "
                    "Install dependencies and ensure model download is available."
                ) from exc

    def rank_columns(self, question: str, schema: DatabaseSchema) -> List[RankedColumn]:
        if self.backend == "sentence-transformers" and self._model is not None:
            return self._rank_with_transformers(question, schema)
        return self._rank_with_hashing(question, schema)

    def _rank_with_transformers(
        self, question: str, schema: DatabaseSchema
    ) -> List[RankedColumn]:
        column_texts = [self._column_text(col) for col in schema.columns]
        q_emb = self._model.encode([question], normalize_embeddings=True)[0]
        col_embs = self._model.encode(column_texts, normalize_embeddings=True)

        scores: List[RankedColumn] = []
        for col, emb in zip(schema.columns, col_embs):
            score = float(self._dot(emb, q_emb))
            scores.append(
                RankedColumn(
                    full_name=col.full_name,
                    table=col.table,
                    column=col.name,
                    score=score,
                )
            )
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores

    def _rank_with_hashing(
        self, question: str, schema: DatabaseSchema
    ) -> List[RankedColumn]:
        cache_key = schema.db_id
        if cache_key not in self._hash_cache:
            self._hash_cache[cache_key] = [
                (col, *self._hash_vector(self._column_text(col)))
                for col in schema.columns
            ]

        q_vec, q_norm = self._hash_vector(question)
        ranked: List[RankedColumn] = []
        for col, vec, norm in self._hash_cache[cache_key]:
            score = self._cosine_sparse(q_vec, q_norm, vec, norm)
            ranked.append(
                RankedColumn(
                    full_name=col.full_name,
                    table=col.table,
                    column=col.name,
                    score=score,
                )
            )
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    @staticmethod
    def _normalize_name(text: Optional[str]) -> str:
        if not text:
            return ""
        text = text.replace("_", " ")
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        return text
    def _column_text(self, col: ColumnRef) -> str:
        table = self._normalize_name(col.table)
        column = self._normalize_name(col.name)
        col_type = self._normalize_name(col.col_type or "")
        return f"table {table} column {column} type {col_type} {table} {column} {col.name}".lower()

    # def _column_text(self, col: ColumnRef) -> str:
    #     # table = col.table
    #     # column = col.name

    #     # def normalize(text):
    #     #     return text.replace("_", " ")

    #     # return f"""
    #     # table {normalize(table)}
    #     # column {normalize(column)}
    #     # {normalize(table)} {normalize(column)}
    #     # {column}
    #     # {table}
    #     # """.lower()
    #     # print(f"table: {col.table} | column: {col.name} | type: {col.col_type}")
    #     return f"{col.table} | {col.name}".lower()

    def _hash_vector(self, text: str) -> Tuple[Dict[int, float], float]:
        tokens = self._tokenize(text)
        vec: Dict[int, float] = {}
        for token in tokens:
            idx = self._stable_hash(token) % self.hashing_dim
            vec[idx] = vec.get(idx, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return vec, norm

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if t]

    @staticmethod
    def _stable_hash(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def _cosine_sparse(
        vec_a: Dict[int, float],
        norm_a: float,
        vec_b: Dict[int, float],
        norm_b: float,
    ) -> float:
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        dot = 0.0
        for idx, val in vec_a.items():
            dot += val * vec_b.get(idx, 0.0)
        return dot / (norm_a * norm_b)

    @staticmethod
    def _dot(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
        return sum(a * b for a, b in zip(vec_a, vec_b))

