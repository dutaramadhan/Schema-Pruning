from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ColumnRef:
    index: int
    table: str
    name: str
    full_name: str
    col_type: Optional[str] = None


@dataclass
class TableSchema:
    name: str
    columns: List[ColumnRef]
    primary_keys: List[ColumnRef]


@dataclass
class DatabaseSchema:
    db_id: str
    tables: Dict[str, TableSchema]
    columns: List[ColumnRef]
    primary_keys: List[ColumnRef]
    foreign_keys: List[Tuple[ColumnRef, ColumnRef]]
    column_name_to_full: Dict[str, List[str]]
    table_name_to_columns: Dict[str, List[ColumnRef]]

    def all_column_full_names(self) -> List[str]:
        return [col.full_name for col in self.columns]


class SchemaLoader:
    """Load BIRD schema metadata from tables.json and map it to database paths."""

    def __init__(self, tables_json_path: str | Path, db_root: str | Path) -> None:
        self.tables_json_path = Path(tables_json_path)
        self.db_root = Path(db_root)
        self._tables_index = self._load_tables_index()
        self._schema_cache: Dict[str, DatabaseSchema] = {}

    def _load_tables_index(self) -> Dict[str, dict]:
        with self.tables_json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {entry["db_id"]: entry for entry in raw}

    def get_schema(self, db_id: str) -> DatabaseSchema:
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]
        if db_id not in self._tables_index:
            raise KeyError(f"db_id not found in tables.json: {db_id}")
        entry = self._tables_index[db_id]

        table_names = entry["table_names_original"]
        column_names = entry["column_names_original"]
        column_types = entry.get("column_types", [])

        tables: Dict[str, TableSchema] = {
            t: TableSchema(name=t, columns=[], primary_keys=[]) for t in table_names
        }

        columns: List[ColumnRef] = []
        index_to_col: Dict[int, ColumnRef] = {}
        for idx, (table_id, col_name) in enumerate(column_names):
            if table_id == -1:
                continue
            table_name = table_names[table_id]
            col_type = column_types[idx] if idx < len(column_types) else None
            full_name = f"{table_name}.{col_name}"
            col = ColumnRef(
                index=idx,
                table=table_name,
                name=col_name,
                full_name=full_name,
                col_type=col_type,
            )
            columns.append(col)
            index_to_col[idx] = col
            tables[table_name].columns.append(col)

        primary_keys: List[ColumnRef] = []
        for pk_idx in entry.get("primary_keys", []):
            # Some datasets store composite PKs as lists (e.g., [1, 2]).
            if isinstance(pk_idx, (list, tuple)):
                pk_indices = pk_idx
            else:
                pk_indices = [pk_idx]
            for pk in pk_indices:
                try:
                    pk_int = int(pk)
                except (TypeError, ValueError):
                    continue
                pk_col = index_to_col.get(pk_int)
                if pk_col is not None and pk_col not in primary_keys:
                    primary_keys.append(pk_col)
                    if pk_col not in tables[pk_col.table].primary_keys:
                        tables[pk_col.table].primary_keys.append(pk_col)

        foreign_keys: List[Tuple[ColumnRef, ColumnRef]] = []
        for idx_a, idx_b in entry.get("foreign_keys", []):
            col_a = index_to_col.get(idx_a)
            col_b = index_to_col.get(idx_b)
            if col_a is not None and col_b is not None:
                foreign_keys.append((col_a, col_b))

        column_name_to_full: Dict[str, List[str]] = {}
        table_name_to_columns: Dict[str, List[ColumnRef]] = {}
        for col in columns:
            key = col.name.lower()
            column_name_to_full.setdefault(key, []).append(col.full_name)
            table_name_to_columns.setdefault(col.table.lower(), []).append(col)

        schema = DatabaseSchema(
            db_id=db_id,
            tables=tables,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            column_name_to_full=column_name_to_full,
            table_name_to_columns=table_name_to_columns,
        )
        self._schema_cache[db_id] = schema
        return schema

    def get_database_path(self, db_id: str) -> Path:
        return self.db_root / db_id / f"{db_id}.sqlite"

    # Kept as a small helper for custom extensions, currently unused.
    @staticmethod
    def _column_by_index(columns: List[ColumnRef], index: int) -> Optional[ColumnRef]:
        for col in columns:
            if col.index == index:
                return col
        return None
