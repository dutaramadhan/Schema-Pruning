"""Paired significance tests for execution accuracy results.

This script reuses the SQL execution logic from ``execution_accuracy.py`` and
applies McNemar tests to paired correct/incorrect outcomes from the same BIRD
dev questions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.execution_accuracy import (  # noqa: E402
    compute_acc_by_diff,
    execute_model,
    package_sqls,
    sort_results,
)


CONFIG_ORDER = [
    "full_schema",
    "hybrid_mkp",
]

CONFIG_LABELS = {
    "full_schema": r"\textit{Full schema}",
    "embedding_retrieval": r"\textit{Embedding retrieval}",
    "hybrid_retrieval": r"\textit{Hybrid retrieval}",
    "hybrid_mkp": r"\textit{Hybrid retrieval} + MKP",
}

EXPERIMENTS = {
    "llama_3_2_3b": {
        "label": "LLaMA 3.2 3B Instruct",
        "predictions": {
            "full_schema": Path("outputs/inference/llama-3.2-3b/llama_3.2_fullschema.json"),
            "embedding_retrieval": Path(
                "outputs/inference/llama-3.2-3b/llama_3.2_embedding_only_top_30.json"
            ),
            "hybrid_retrieval": Path(
                "outputs/inference/llama-3.2-3b/llama_3.2_embedding_bm25_0.5_top_30.json"
            ),
            "hybrid_mkp": Path(
                "outputs/inference/llama-3.2-3b/llama_3.2_embedding_bm25_0.5_mkp_top_30.json"
            ),
        },
    },
    "gemini_2_5_flash": {
        "label": "Gemini 2.5 Flash",
        "predictions": {
            "full_schema": Path("outputs/inference/gemini-2.5-flash/gemini_2.5_flash_fullschema.json"),
            "embedding_retrieval": Path(
                "outputs/inference/gemini-2.5-flash/gemini_2.5_flash_embedding_only_top_30.json"
            ),
            "hybrid_retrieval": Path(
                "outputs/inference/gemini-2.5-flash/gemini_2.5_flash_embedding_bm25_0.5_top_30.json"
            ),
            "hybrid_mkp": Path(
                "outputs/inference/gemini-2.5-flash/gemini_2.5_flash_embedding_bm25_0.5_mkp_top_30.json"
            ),
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run McNemar significance tests for execution accuracy experiments."
    )
    parser.add_argument(
        "--db-root-path",
        default=Path("Dataset/BIRD/dev/dev_databases/dev_databases"),
        type=Path,
        help="Directory containing BIRD dev SQLite databases.",
    )
    parser.add_argument(
        "--ground-truth-path",
        default=Path("Dataset/BIRD/dev/dev.sql"),
        type=Path,
        help="BIRD dev SQL reference file.",
    )
    parser.add_argument(
        "--diff-json-path",
        default=Path("Dataset/BIRD/dev/dev.json"),
        type=Path,
        help="BIRD dev metadata with difficulty labels.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/significance"),
        type=Path,
        help="Directory for significance test outputs.",
    )
    parser.add_argument("--num-cpus", default=16, type=int)
    parser.add_argument("--meta-time-out", default=30.0, type=float)
    parser.add_argument(
        "--engine",
        choices=["sqlite-progress", "func-timeout"],
        default="sqlite-progress",
        help=(
            "SQL timeout engine. sqlite-progress uses SQLite progress handlers "
            "to interrupt long-running queries; func-timeout calls the original "
            "execution_accuracy.execute_model function."
        ),
    )
    return parser.parse_args()


def as_existing_path(path: Path) -> Path:
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def db_root_for_evaluator(path: Path) -> str:
    value = str(path.resolve())
    if not value.endswith(os.sep):
        value += os.sep
    return value


def execute_model_sqlite_progress(
    predicted_sql: str,
    ground_truth: str,
    db_path: str,
    idx: int,
    timeout_seconds: float,
) -> Dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        res = int(set(predicted_res) == set(ground_truth_res))
    except Exception:
        res = 0
    finally:
        if conn is not None:
            conn.close()
    return {"sql_idx": idx, "res": res}


def evaluate_predictions(
    predicted_sql_path: Path,
    db_root_path: Path,
    ground_truth_path: Path,
    diff_json_path: Path,
    num_cpus: int,
    meta_time_out: float,
    engine: str,
) -> Tuple[List[int], Dict[str, float]]:
    pred_queries, db_paths = package_sqls(
        str(predicted_sql_path),
        db_root_for_evaluator(db_root_path),
        mode="gpt",
        data_mode="dev",
    )
    gt_queries, _ = package_sqls(
        str(ground_truth_path),
        db_root_for_evaluator(db_root_path),
        mode="gt",
        data_mode="dev",
    )
    query_pairs = list(zip(pred_queries, gt_queries))

    worker = execute_model if engine == "func-timeout" else execute_model_sqlite_progress
    with __import__("multiprocessing").Pool(processes=num_cpus) as pool:
        raw_results = pool.starmap(
            worker,
            [
                (predicted_sql, ground_truth, db_paths[idx], idx, meta_time_out)
                for idx, (predicted_sql, ground_truth) in enumerate(query_pairs)
            ],
        )

    exec_results = sort_results(raw_results)
    simple_acc, moderate_acc, challenging_acc, total_acc, count_lists = compute_acc_by_diff(
        exec_results, str(diff_json_path)
    )
    scores = {
        "simple_acc": simple_acc,
        "moderate_acc": moderate_acc,
        "challenging_acc": challenging_acc,
        "total_acc": total_acc,
        "simple_count": count_lists[0],
        "moderate_count": count_lists[1],
        "challenging_count": count_lists[2],
        "total_count": count_lists[3],
        "correct_count": sum(result["res"] for result in exec_results),
    }
    return [int(result["res"]) for result in exec_results], scores


def mcnemar_exact_p(only_a_correct: int, only_b_correct: int) -> float:
    discordant = only_a_correct + only_b_correct
    if discordant == 0:
        return 1.0

    smaller = min(only_a_correct, only_b_correct)
    tail = math.fsum(
        math.comb(discordant, k) / (2**discordant) for k in range(smaller + 1)
    )
    return min(1.0, 2.0 * tail)


def mcnemar_statistic(only_a_correct: int, only_b_correct: int) -> float:
    discordant = only_a_correct + only_b_correct
    if discordant == 0:
        return 0.0
    return (max(abs(only_a_correct - only_b_correct) - 1, 0) ** 2) / discordant


def holm_adjust(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    adjusted_rows = [dict(row) for row in rows]
    ordered = sorted(enumerate(adjusted_rows), key=lambda item: float(item[1]["p_value"]))
    m = len(ordered)
    previous = 0.0

    for rank, (original_index, row) in enumerate(ordered, start=1):
        adjusted = min(1.0, (m - rank + 1) * float(row["p_value"]))
        adjusted = max(previous, adjusted)
        previous = adjusted
        adjusted_rows[original_index]["p_holm"] = adjusted
        adjusted_rows[original_index]["significant_0_05"] = adjusted < 0.05

    return adjusted_rows


def compare_pairs(
    model_id: str,
    model_label: str,
    config_results: Dict[str, List[int]],
    summary: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for config_a, config_b in itertools.combinations(CONFIG_ORDER, 2):
        results_a = config_results[config_a]
        results_b = config_results[config_b]
        only_a_correct = sum(1 for a, b in zip(results_a, results_b) if a == 1 and b == 0)
        only_b_correct = sum(1 for a, b in zip(results_a, results_b) if a == 0 and b == 1)
        p_value = mcnemar_exact_p(only_a_correct, only_b_correct)
        rows.append(
            {
                "model_id": model_id,
                "model": model_label,
                "config_a": config_a,
                "config_a_label": CONFIG_LABELS[config_a],
                "config_b": config_b,
                "config_b_label": CONFIG_LABELS[config_b],
                "accuracy_a": summary[config_a]["total_acc"],
                "accuracy_b": summary[config_b]["total_acc"],
                "delta_b_minus_a": summary[config_b]["total_acc"] - summary[config_a]["total_acc"],
                "only_a_correct": only_a_correct,
                "only_b_correct": only_b_correct,
                "discordant": only_a_correct + only_b_correct,
                "mcnemar_chi2_cc": mcnemar_statistic(only_a_correct, only_b_correct),
                "p_value": p_value,
            }
        )
    return holm_adjust(rows)


def load_difficulties(diff_json_path: Path) -> List[str]:
    with diff_json_path.open("r", encoding="utf-8") as file:
        return [item["difficulty"] for item in json.load(file)]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    db_root_path = as_existing_path(args.db_root_path)
    ground_truth_path = as_existing_path(args.ground_truth_path)
    diff_json_path = as_existing_path(args.diff_json_path)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    difficulties = load_difficulties(diff_json_path)
    all_results: Dict[str, Dict[str, List[int]]] = {}
    all_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_id, model_spec in EXPERIMENTS.items():
        model_label = model_spec["label"]
        all_results[model_id] = {}
        all_summary[model_id] = {}
        for config_id in CONFIG_ORDER:
            predicted_sql_path = as_existing_path(model_spec["predictions"][config_id])
            print(f"Evaluating {model_label} / {config_id} ...")
            results, scores = evaluate_predictions(
                predicted_sql_path=predicted_sql_path,
                db_root_path=db_root_path,
                ground_truth_path=ground_truth_path,
                diff_json_path=diff_json_path,
                num_cpus=args.num_cpus,
                meta_time_out=args.meta_time_out,
                engine=args.engine,
            )
            all_results[model_id][config_id] = results
            all_summary[model_id][config_id] = scores

    summary_rows: List[Dict[str, object]] = []
    for model_id, model_spec in EXPERIMENTS.items():
        for config_id in CONFIG_ORDER:
            row = {
                "model_id": model_id,
                "model": model_spec["label"],
                "config": config_id,
                "config_label": CONFIG_LABELS[config_id],
            }
            row.update(all_summary[model_id][config_id])
            summary_rows.append(row)

    significance_rows: List[Dict[str, object]] = []
    for model_id, model_spec in EXPERIMENTS.items():
        significance_rows.extend(
            compare_pairs(
                model_id=model_id,
                model_label=model_spec["label"],
                config_results=all_results[model_id],
                summary=all_summary[model_id],
            )
        )

    per_sample_rows: List[Dict[str, object]] = []
    total_count = len(difficulties)
    for idx in range(total_count):
        row: Dict[str, object] = {"question_id": idx, "difficulty": difficulties[idx]}
        for model_id in EXPERIMENTS:
            for config_id in CONFIG_ORDER:
                row[f"{model_id}_{config_id}"] = all_results[model_id][config_id][idx]
        per_sample_rows.append(row)

    write_csv(output_dir / "execution_accuracy_summary.csv", summary_rows)
    write_csv(output_dir / "execution_accuracy_mcnemar.csv", significance_rows)
    write_csv(output_dir / "execution_accuracy_per_sample.csv", per_sample_rows)

    with (output_dir / "execution_accuracy_significance.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "test": "Exact two-sided McNemar test",
                "comparison": "Planned comparison between full_schema and hybrid_mkp for each model",
                "correction": "No multiple-comparison correction is needed because there is one planned comparison per model; p_holm equals p_value.",
                "summary": summary_rows,
                "significance": significance_rows,
            },
            file,
            indent=2,
        )

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
