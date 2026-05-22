from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from pruning.schema_loader import ColumnRef, DatabaseSchema


@dataclass
class RankedColumn:
    full_name: str
    table: str
    column: str
    score: float


@dataclass
class LexCache:
    docs: List[Tuple[ColumnRef, Dict[str, int], int]]
    idf: Dict[str, float]
    avg_dl: float


class EmbeddingRetrieverV2:
    """Hybrid column retriever using sentence-transformers + lexical BM25 scoring."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        lexical_weight: float = 0.3,
        use_evidence: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.lexical_weight = max(0.0, min(1.0, lexical_weight))
        self.use_evidence = use_evidence
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.backend = "embedding" if self.lexical_weight == 0.0 else "hybrid"
        self._model = None
        self._lex_cache: Dict[str, LexCache] = {}

        from sentence_transformers import SentenceTransformer  # type: ignore

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load sentence-transformers model. "
                "Install dependencies and ensure model download is available."
            ) from exc

    def rank_columns(
        self, question: str, schema: DatabaseSchema, evidence: Optional[str] = None
    ) -> List[RankedColumn]:
        query_text = self._build_query_text(question, evidence)
        embed_scores = self._embed_scores(query_text, schema)
        embed_vals = [col.score for col in embed_scores]
        embed_norm = self._min_max_normalize(embed_vals)

        # Embedding-only mode uses the same question/evidence query text, but skips BM25.
        if self.lexical_weight == 0.0:
            ranked = [
                RankedColumn(
                    full_name=col.full_name,
                    table=col.table,
                    column=col.column,
                    score=e_score,
                )
                for col, e_score in zip(embed_scores, embed_norm)
            ]
            ranked.sort(key=lambda x: x.score, reverse=True)
            return ranked

        lex_scores = self._lex_scores(query_text, schema)
        lex_vals = [lex_scores.get(col.full_name, 0.0) for col in embed_scores]
        lex_norm = self._min_max_normalize(lex_vals)

        combined: List[RankedColumn] = []
        for col, e_score, l_score in zip(embed_scores, embed_norm, lex_norm):
            score = (1.0 - self.lexical_weight) * e_score + self.lexical_weight * l_score
            combined.append(
                RankedColumn(
                    full_name=col.full_name,
                    table=col.table,
                    column=col.column,
                    score=score,
                )
            )

        combined.sort(key=lambda x: x.score, reverse=True)
        return combined

    def _build_query_text(self, question: str, evidence: Optional[str]) -> str:
        if self.use_evidence and evidence:
            return f"{question} {evidence}"
        return question

    def _embed_scores(self, query_text: str, schema: DatabaseSchema) -> List[RankedColumn]:
        if self._model is None:
            raise RuntimeError("Sentence-transformers model is not loaded.")
        return self._embed_scores_transformers(query_text, schema)

    def _embed_scores_transformers(
        self, query_text: str, schema: DatabaseSchema
    ) -> List[RankedColumn]:
        column_texts = [self._column_text(col) for col in schema.columns]
        q_emb = self._model.encode([query_text], normalize_embeddings=True)[0]
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
        return scores

    def _lex_scores(self, query_text: str, schema: DatabaseSchema) -> Dict[str, float]:
        cache = self._ensure_lex_cache(schema)
        q_tokens = self._tokenize(query_text)
        if not q_tokens:
            return {col.full_name: 0.0 for col in schema.columns}

        q_terms = Counter(q_tokens)
        scores: Dict[str, float] = {}
        avg_dl = cache.avg_dl or 1.0

        for col, term_counts, doc_len in cache.docs:
            score = 0.0
            for term in q_terms:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                idf = cache.idf.get(term, 0.0)
                denom = tf + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * (doc_len / avg_dl))
                if denom == 0.0:
                    continue
                score += idf * (tf * (self.bm25_k1 + 1.0) / denom)
            scores[col.full_name] = score
        return scores

    def _ensure_lex_cache(self, schema: DatabaseSchema) -> LexCache:
        cache_key = schema.db_id
        if cache_key in self._lex_cache:
            return self._lex_cache[cache_key]

        docs: List[Tuple[ColumnRef, Dict[str, int], int]] = []
        df: Counter[str] = Counter()
        total_len = 0

        for col in schema.columns:
            tokens = self._tokenize(self._column_text(col))
            counts = Counter(tokens)
            doc_len = sum(counts.values())
            docs.append((col, dict(counts), doc_len))
            total_len += doc_len
            for token in counts.keys():
                df[token] += 1

        n_docs = max(len(docs), 1)
        idf: Dict[str, float] = {}
        for token, freq in df.items():
            idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        avg_dl = total_len / n_docs if n_docs else 0.0
        cache = LexCache(docs=docs, idf=idf, avg_dl=avg_dl)
        self._lex_cache[cache_key] = cache
        return cache

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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if t]

    @staticmethod
    def _dot(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
        return sum(a * b for a, b in zip(vec_a, vec_b))

    @staticmethod
    def _min_max_normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if math.isclose(min_v, max_v):
            return [0.0 for _ in values]
        return [(v - min_v) / (max_v - min_v) for v in values]
