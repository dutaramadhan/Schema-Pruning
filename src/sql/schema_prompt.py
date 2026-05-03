from __future__ import annotations

from typing import Iterable, List, Sequence

from pruning.schema_loader import ColumnRef, DatabaseSchema


def build_schema_text(
    schema: DatabaseSchema,
    selected_columns: Sequence[str],
    include_types: bool = True,
    include_foreign_keys: bool = True,
) -> str:
    selected_set = set(selected_columns)
    table_lines: List[str] = []

    for table_name in schema.tables.keys():
        cols = [col for col in schema.tables[table_name].columns if col.full_name in selected_set]
        if not cols:
            continue
        col_chunks = [_format_column(col, include_types) for col in cols]
        table_lines.append(f"Table {table_name}: " + ", ".join(col_chunks))

    fk_lines: List[str] = []
    if include_foreign_keys:
        for col_a, col_b in schema.foreign_keys:
            if col_a.full_name in selected_set and col_b.full_name in selected_set:
                fk_lines.append(f"{col_a.full_name} = {col_b.full_name}")

    if fk_lines:
        table_lines.append("Foreign keys:")
        table_lines.extend(f"- {line}" for line in fk_lines)

    return "\n".join(table_lines)


def build_prompt(
    question: str,
    schema_text: str,
    evidence: str | None = None,
    template: str | None = None,
) -> str:
    template = template or DEFAULT_PROMPT_TEMPLATE
    evidence_line = f"Evidence: {evidence}" if evidence else ""
    return template.format(
        schema_text=schema_text,
        question=question,
        evidence_line=evidence_line,
    ).strip()


def _format_column(col: ColumnRef, include_types: bool) -> str:
    if include_types and col.col_type:
        return f"{col.name} ({col.col_type})"
    return col.name


DEFAULT_PROMPT_TEMPLATE = """You are a data analyst. Write a SQLite SQL query to answer the question.

Database schema:
{schema_text}

Question: {question}
{evidence_line}

Return only the SQL query."""

