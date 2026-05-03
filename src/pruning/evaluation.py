from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set, Tuple

from pruning.schema_loader import DatabaseSchema
from pruning.utils_join import safe_divide


SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "on",
    "group",
    "by",
    "order",
    "limit",
    "having",
    "as",
    "and",
    "or",
    "not",
    "in",
    "is",
    "null",
    "distinct",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "case",
    "when",
    "then",
    "else",
    "end",
    "asc",
    "desc",
    "union",
    "all",
    "like",
    "exists",
    "between",
}

DOT_RE = re.compile(
    r"([A-Za-z_][\w$]*|`[^`]+`|\"[^\"]+\"|\[[^\]]+\])\s*\.\s*([A-Za-z_][\w$]*|`[^`]+`|\"[^\"]+\"|\[[^\]]+\])"
)
QUOTED_RE = re.compile(r"`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]")
WORD_RE = re.compile(r"[A-Za-z_][\w$]*")
FROM_JOIN_RE = re.compile(
    r"\b(from|join)\s+([A-Za-z_][\w$]*|`[^`]+`|\"[^\"]+\"|\[[^\]]+\])"
    r"(?:\s+(?:as\s+)?([A-Za-z_][\w$]*|`[^`]+`|\"[^\"]+\"|\[[^\]]+\]))?",
    re.IGNORECASE,
)


def extract_gold_columns(sql: str, schema: DatabaseSchema) -> Set[str]:
    """Heuristic extraction of gold columns from SQL using schema info."""
    sql_lower = sql.lower()
    gold: Set[str] = set()

    table_map = {t.lower(): t for t in schema.tables.keys()}
    col_map = schema.column_name_to_full
    alias_map = _extract_alias_map(sql, table_map)

    # 1) dotted identifiers: table.column or alias.column
    for match in DOT_RE.finditer(sql):
        left_raw, right_raw = match.group(1), match.group(2)
        left = _normalize_identifier(left_raw)
        right = _normalize_identifier(right_raw)

        if left in alias_map:
            table_name = alias_map[left]
            table_cols = schema.table_name_to_columns.get(table_name.lower(), [])
            for col in table_cols:
                if col.name.lower() == right:
                    gold.add(col.full_name)
        else:
            # likely alias.column, fallback to unique column name
            if right in col_map and len(col_map[right]) == 1:
                gold.add(col_map[right][0])

    # 2) quoted identifiers
    for match in QUOTED_RE.finditer(sql):
        ident = _normalize_identifier(match.group(0))
        if ident in col_map and len(col_map[ident]) == 1:
            gold.add(col_map[ident][0])

    # 3) bare tokens (avoid keywords, skip quoted identifiers and string literals)
    # Remove quoted identifiers and string literals to avoid partial token matches
    masked = QUOTED_RE.sub(" ", sql_lower)
    masked = re.sub(r"'[^']*'", " ", masked)
    for token in WORD_RE.findall(masked):
        if token in SQL_KEYWORDS:
            continue
        if token in col_map and len(col_map[token]) == 1:
            gold.add(col_map[token][0])

    return gold


def precision_recall_f1(predicted: Iterable[str], gold: Iterable[str]) -> Tuple[float, float, float]:
    pred_set = set(predicted)
    gold_set = set(gold)
    if not gold_set:
        precision = 1.0 if not pred_set else 0.0
        recall = 1.0
        f1 = precision
        return precision, recall, f1

    tp = len(pred_set & gold_set)
    precision = safe_divide(tp, len(pred_set))
    recall = safe_divide(tp, len(gold_set))
    f1 = _fbeta(precision, recall, beta=1.0)
    return precision, recall, f1


def precision_recall_f1_f6(
    predicted: Iterable[str], gold: Iterable[str]
) -> Tuple[float, float, float, float]:
    precision, recall, f1 = precision_recall_f1(predicted, gold)
    f6 = _fbeta(precision, recall, beta=6.0)
    return precision, recall, f1, f6


def _normalize_identifier(token: str) -> str:
    token = token.strip()
    if token.startswith(("`", "\"", "[")) and token.endswith(("`", "\"", "]")):
        token = token[1:-1]
    return token.lower()


def _extract_alias_map(sql: str, table_map: Dict[str, str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for match in FROM_JOIN_RE.finditer(sql):
        table_raw = match.group(2)
        alias_raw = match.group(3)
        table_norm = _normalize_identifier(table_raw)
        table_name = table_map.get(table_norm)
        if not table_name:
            continue
        alias_map[table_norm] = table_name
        if alias_raw:
            alias_norm = _normalize_identifier(alias_raw)
            alias_map[alias_norm] = table_name
    return alias_map


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom

