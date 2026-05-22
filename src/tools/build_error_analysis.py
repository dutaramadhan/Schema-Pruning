from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_ROOT = PROJECT_ROOT / "Dataset" / "BIRD" / "dev" / "dev_databases" / "dev_databases"
DEFAULT_DIFF_JSON = PROJECT_ROOT / "Dataset" / "BIRD" / "dev" / "dev.json"

BIRD_SEPARATOR = "\t----- bird -----\t"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_identifier(value: str) -> str:
    value = value.strip().strip("`").strip('"').strip("'")
    return re.sub(r"\s+", " ", value).lower()


def parse_schema_columns(schema_prompt: str) -> set[tuple[str, str]]:
    columns: set[tuple[str, str]] = set()
    table_pattern = re.compile(
        r"CREATE\s+TABLE\s+`?([^`(\s]+)`?\s*\((.*?)\);",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for table_match in table_pattern.finditer(schema_prompt or ""):
        table = normalize_identifier(table_match.group(1))
        body = table_match.group(2)
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            quoted = re.match(r"`([^`]+)`", line)
            unquoted = re.match(r"([A-Za-z_][A-Za-z0-9_\s%()/-]*)\s+", line)
            if quoted:
                col = quoted.group(1)
            elif unquoted:
                col = unquoted.group(1)
            else:
                continue
            columns.add((table, normalize_identifier(col)))
    return columns


def parse_gold_column(value: str) -> tuple[str, str] | None:
    if "." not in value:
        return None
    table, column = value.split(".", 1)
    return normalize_identifier(table), normalize_identifier(column)


def parse_prediction(value: Any) -> tuple[str, str]:
    if not isinstance(value, str):
        return "", ""
    if BIRD_SEPARATOR not in value:
        return value.strip(), ""
    sql, db_id = value.split(BIRD_SEPARATOR, 1)
    return sql.strip(), db_id.strip()


def execute_pair(predicted_sql: str, gold_sql: str, db_path: Path, timeout_seconds: float) -> tuple[int, str]:
    if not predicted_sql:
        return 0, "empty prediction"
    deadline = time.monotonic() + timeout_seconds
    try:
        conn = sqlite3.connect(str(db_path))
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        cur = conn.cursor()
        cur.execute(predicted_sql)
        predicted_rows = cur.fetchall()
        cur.execute(gold_sql)
        gold_rows = cur.fetchall()
        return int(set(predicted_rows) == set(gold_rows)), ""
    except Exception as exc:  # noqa: BLE001 - the message is useful for case analysis.
        return 0, str(exc).replace("\n", " ")[:500]
    finally:
        try:
            conn.close()
        except Exception:
            pass


def analyze_one(args: tuple[int, dict[str, Any], Any, str, str, float]) -> dict[str, Any]:
    idx, item, prediction_value, difficulty, db_root, timeout_seconds = args
    question_id = int(item.get("question_id", idx))
    db_id = item.get("db_id", "")
    predicted_sql, predicted_db = parse_prediction(prediction_value)
    gold_sql = item.get("gold_sql") or item.get("SQL") or item.get("sql") or ""
    schema_columns = parse_schema_columns(item.get("schema_prompt", ""))
    gold_columns = item.get("gold_columns") or item.get("gold_columns_raw") or []

    missing_gold: list[str] = []
    for gold_col in gold_columns:
        parsed = parse_gold_column(str(gold_col))
        if parsed is not None and parsed not in schema_columns:
            missing_gold.append(str(gold_col))

    db_for_exec = predicted_db or db_id
    db_path = Path(db_root) / db_for_exec / f"{db_for_exec}.sqlite"
    correct, execution_error = execute_pair(predicted_sql, gold_sql, db_path, timeout_seconds)

    if correct:
        error_group = "correct"
    elif missing_gold:
        error_group = "pruning_missing_gold_column"
    elif execution_error:
        error_group = "sql_execution_error"
    else:
        error_group = "generation_error_or_sql_logic"

    return {
        "question_id": question_id,
        "row_index": idx,
        "db_id": db_id,
        "difficulty": difficulty,
        "question": item.get("question", ""),
        "evidence": item.get("evidence", ""),
        "gold_sql": gold_sql,
        "predicted_sql": predicted_sql,
        "exec_correct": correct,
        "execution_error": execution_error,
        "error_group": error_group,
        "gold_columns": "; ".join(map(str, gold_columns)),
        "missing_gold_columns": "; ".join(missing_gold),
        "missing_gold_count": len(missing_gold),
        "gold_column_count": len(gold_columns),
        "schema_column_count": len(schema_columns),
        "full_recall": int(len(missing_gold) == 0),
    }


def build_rows(
    dataset: list[dict[str, Any]],
    predictions: dict[str, Any],
    difficulties: dict[int, str],
    db_root: Path,
    timeout_seconds: float,
    workers: int,
) -> list[dict[str, Any]]:
    tasks = [
        (
            idx,
            item,
            predictions.get(str(idx), ""),
            difficulties.get(int(item.get("question_id", idx)), ""),
            str(db_root),
            timeout_seconds,
        )
        for idx, item in enumerate(dataset)
    ]
    if workers <= 1:
        return [analyze_one(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(analyze_one, tasks, chunksize=16))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(int(row["exec_correct"]) for row in rows)
    full_recall = sum(int(row["full_recall"]) for row in rows)
    by_error_group = Counter(row["error_group"] for row in rows)

    by_difficulty: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["difficulty"]].append(row)

    for difficulty, diff_rows in sorted(grouped.items()):
        diff_total = len(diff_rows)
        diff_correct = sum(int(row["exec_correct"]) for row in diff_rows)
        by_difficulty[difficulty] = {
            "total": diff_total,
            "correct": diff_correct,
            "accuracy": round(diff_correct / diff_total * 100, 2) if diff_total else 0,
            "full_recall": sum(int(row["full_recall"]) for row in diff_rows),
            "error_group_counts": dict(Counter(row["error_group"] for row in diff_rows)),
        }

    missing_counter: Counter[str] = Counter()
    for row in rows:
        if row["missing_gold_columns"]:
            for col in row["missing_gold_columns"].split("; "):
                missing_counter[col] += 1

    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": round(correct / total * 100, 2) if total else 0,
        "full_recall_samples": full_recall,
        "full_recall_rate": round(full_recall / total * 100, 2) if total else 0,
        "correct_with_missing_gold": sum(
            int(row["exec_correct"]) and int(row["missing_gold_count"]) > 0 for row in rows
        ),
        "wrong_with_missing_gold": sum(
            not int(row["exec_correct"]) and int(row["missing_gold_count"]) > 0 for row in rows
        ),
        "wrong_without_missing_gold": sum(
            not int(row["exec_correct"]) and int(row["missing_gold_count"]) == 0 for row in rows
        ),
        "error_group_counts": dict(by_error_group),
        "by_difficulty": by_difficulty,
        "top_missing_gold_columns": missing_counter.most_common(20),
    }


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "row_index",
        "db_id",
        "difficulty",
        "exec_correct",
        "error_group",
        "execution_error",
        "missing_gold_count",
        "gold_column_count",
        "schema_column_count",
        "full_recall",
        "missing_gold_columns",
        "gold_columns",
        "question",
        "evidence",
        "gold_sql",
        "predicted_sql",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-sample error analysis for BIRD Text-to-SQL outputs.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--db-root", default=DEFAULT_DB_ROOT, type=Path)
    parser.add_argument("--diff-json", default=DEFAULT_DIFF_JSON, type=Path)
    parser.add_argument("--timeout-seconds", default=30.0, type=float)
    parser.add_argument("--workers", default=min(8, os.cpu_count() or 1), type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_json(args.dataset)
    predictions = load_json(args.predictions)
    dev_items = load_json(args.diff_json)
    difficulties = {int(item["question_id"]): item.get("difficulty", "") for item in dev_items}

    rows = build_rows(dataset, predictions, difficulties, args.db_root, args.timeout_seconds, args.workers)
    write_csv(rows, args.output_csv)

    summary = summarize(rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
