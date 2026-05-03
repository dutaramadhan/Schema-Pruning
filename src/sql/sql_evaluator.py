from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


@dataclass
class QueryResult:
    rows: List[Tuple]
    elapsed_s: float
    error: Optional[str] = None


@dataclass
class EvalResult:
    exec_accuracy: float
    ves: float
    pred_time_s: float
    gold_time_s: float
    error: Optional[str] = None


def execute_sql(db_path: Path, sql: str) -> QueryResult:
    if not sql.strip():
        return QueryResult(rows=[], elapsed_s=0.0, error="empty_sql")
    db_uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        cursor = conn.cursor()
        start = time.perf_counter()
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.perf_counter() - start
        return QueryResult(rows=rows, elapsed_s=elapsed)
    except Exception as exc:  # noqa: BLE001 - keep broad for eval robustness
        return QueryResult(rows=[], elapsed_s=0.0, error=str(exc))
    finally:
        conn.close()


def evaluate_prediction(
    db_path: Path,
    gold_sql: str,
    pred_sql: str,
    gold_result: Optional[QueryResult] = None,
    float_round: int = 6,
    ves_cap: Optional[float] = 1.0,
) -> Tuple[EvalResult, QueryResult, QueryResult]:
    if gold_result is None:
        gold_result = execute_sql(db_path, gold_sql)

    pred_result = execute_sql(db_path, pred_sql)

    if pred_result.error:
        return (
            EvalResult(
                exec_accuracy=0.0,
                ves=0.0,
                pred_time_s=pred_result.elapsed_s,
                gold_time_s=gold_result.elapsed_s,
                error=pred_result.error,
            ),
            gold_result,
            pred_result,
        )

    order_sensitive = _is_order_sensitive(pred_sql)
    exec_acc = 1.0 if _results_equal(gold_result.rows, pred_result.rows, order_sensitive, float_round) else 0.0
    ves = 0.0
    if exec_acc == 1.0:
        ves = _ves_score(gold_result.elapsed_s, pred_result.elapsed_s, ves_cap)

    return (
        EvalResult(
            exec_accuracy=exec_acc,
            ves=ves,
            pred_time_s=pred_result.elapsed_s,
            gold_time_s=gold_result.elapsed_s,
            error=None,
        ),
        gold_result,
        pred_result,
    )


def _is_order_sensitive(sql: str) -> bool:
    return bool(ORDER_BY_RE.search(sql))


def _ves_score(gold_time: float, pred_time: float, ves_cap: Optional[float]) -> float:
    if gold_time <= 0.0 or pred_time <= 0.0:
        return 0.0
    score = math.sqrt(gold_time / pred_time)
    if ves_cap is not None:
        return min(score, ves_cap)
    return score


def _results_equal(
    gold_rows: Sequence[Tuple],
    pred_rows: Sequence[Tuple],
    order_sensitive: bool,
    float_round: int,
) -> bool:
    gold_norm = _normalize_rows(gold_rows, float_round, order_sensitive)
    pred_norm = _normalize_rows(pred_rows, float_round, order_sensitive)
    return gold_norm == pred_norm


def _normalize_rows(
    rows: Sequence[Tuple],
    float_round: int,
    order_sensitive: bool,
) -> List[Tuple]:
    normalized = [tuple(_normalize_value(v, float_round) for v in row) for row in rows]
    if order_sensitive:
        return normalized
    return sorted(normalized, key=_row_sort_key)


def _row_sort_key(row: Tuple) -> Tuple:
    return tuple(str(item) for item in row)


def _normalize_value(value: object, float_round: int) -> object:
    if value is None:
        return ("__NULL__", 0)
    if isinstance(value, float):
        return ("__FLOAT__", round(value, float_round))
    if isinstance(value, (int, str)):
        return ("__VAL__", value)
    return ("__VAL__", str(value))
