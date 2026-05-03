from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from pruning.schema_loader import ColumnRef, DatabaseSchema
from pruning.utils_join import pairwise_combinations


@dataclass(frozen=True)
class JoinEdge:
    table_a: str
    column_a: str
    table_b: str
    column_b: str

    @property
    def full_a(self) -> str:
        return f"{self.table_a}.{self.column_a}"

    @property
    def full_b(self) -> str:
        return f"{self.table_b}.{self.column_b}"


class SchemaGraph:
    """Undirected schema graph built from foreign key relations."""

    def __init__(self) -> None:
        self.adj: Dict[str, List[Tuple[str, JoinEdge]]] = defaultdict(list)

    @classmethod
    def from_schema(cls, schema: DatabaseSchema) -> "SchemaGraph":
        graph = cls()
        for col_a, col_b in schema.foreign_keys:
            edge = JoinEdge(
                table_a=col_a.table,
                column_a=col_a.name,
                table_b=col_b.table,
                column_b=col_b.name,
            )
            graph.adj[col_a.table].append((col_b.table, edge))
            graph.adj[col_b.table].append((col_a.table, edge))
        return graph

    def shortest_path(self, start: str, goal: str) -> Optional[List[JoinEdge]]:
        if start == goal:
            return []
        visited: Set[str] = set([start])
        queue = deque([(start, [])])
        while queue:
            node, path = queue.popleft()
            for neighbor, edge in self.adj.get(node, []):
                if neighbor in visited:
                    continue
                next_path = path + [edge]
                if neighbor == goal:
                    return next_path
                visited.add(neighbor)
                queue.append((neighbor, next_path))
        return None

    def required_join_columns(self, selected_tables: Iterable[str]) -> Set[str]:
        tables = list(dict.fromkeys(selected_tables))
        required: Set[str] = set()
        for t1, t2 in pairwise_combinations(tables):
            path = self.shortest_path(t1, t2)
            if not path:
                continue
            for edge in path:
                required.add(edge.full_a)
                required.add(edge.full_b)
        return required


