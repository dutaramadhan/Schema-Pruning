from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from pruning.embedding_retriever import EmbeddingRetriever, RankedColumn
from pruning.schema_graph import SchemaGraph
from pruning.schema_loader import DatabaseSchema, SchemaLoader


@dataclass
class PruningResult:
    ranked_columns: List[RankedColumn]
    top_k_columns: List[str]
    preserved_columns: List[str]
    final_columns: List[str]
    backend: str


class SchemaPruner:
    """Column-centric schema pruning pipeline."""

    def __init__(self, schema_loader: SchemaLoader, retriever: EmbeddingRetriever) -> None:
        self.schema_loader = schema_loader
        self.retriever = retriever

    def prune(self, question: str, db_id: str, top_k: int) -> PruningResult:
        schema = self.schema_loader.get_schema(db_id)
        ranked = self.retriever.rank_columns(question, schema)

        top_k = min(top_k, len(ranked))
        top_k_columns = [col.full_name for col in ranked[:top_k]]
        selected_tables = {col.table for col in ranked[:top_k]}

        graph = SchemaGraph.from_schema(schema)
        preserved = graph.required_join_columns(selected_tables)

        final_set: Set[str] = set(top_k_columns) | preserved
        preserved_only = [c for c in preserved if c not in top_k_columns]
        final_columns = sorted(final_set)

        return PruningResult(
            ranked_columns=ranked,
            top_k_columns=top_k_columns,
            preserved_columns=sorted(preserved_only),
            final_columns=final_columns,
            backend=self.retriever.backend,
        )


