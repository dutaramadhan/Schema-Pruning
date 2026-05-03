from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bird_datasets.bird_dataset_loader import BirdDatasetLoader
from experiments.benchmark_runner import BenchmarkConfig, run_benchmark
from pruning.embedding_retriever import EmbeddingRetriever
from pruning.evaluation import extract_gold_columns, precision_recall_f1_f6
from pruning.pruner import SchemaPruner
from pruning.schema_loader import SchemaLoader
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Dataset" / "BIRD" / "dev"
DEFAULT_DATASET = DEFAULT_DATA_ROOT / "dev.json"
DEFAULT_TABLES = DEFAULT_DATA_ROOT / "dev_tables.json"
DEFAULT_DB_ROOT = DEFAULT_DATA_ROOT / "dev_databases" / "dev_databases"


def run_demo(
    dataset_path: Path,
    tables_path: Path,
    db_root: Path,
    top_k: int,
    question: Optional[str] = None,
    db_id: Optional[str] = None,
    sql: Optional[str] = None,
    force_hashing: bool = False,
    model_name: Optional[str] = None,
) -> None:
    schema_loader = SchemaLoader(tables_path, db_root)
    retriever = EmbeddingRetriever(model_name=model_name, force_hashing=force_hashing)
    pruner = SchemaPruner(schema_loader, retriever)

    if question is None or db_id is None:
        sample = next(iter(BirdDatasetLoader(dataset_path).iter_samples()))
        question = sample.question
        db_id = sample.db_id
        sql = sql or sample.sql

    pruning = pruner.prune(question, db_id, top_k)
    schema = schema_loader.get_schema(db_id)
    gold_columns = extract_gold_columns(sql or "", schema)
    precision, recall, f1, f6 = precision_recall_f1_f6(
        pruning.final_columns, gold_columns
    )

    print("=== Demo Result ===")
    print(f"DB: {db_id}")
    print(f"Question: {question}")
    print(f"Retriever backend: {pruning.backend}")
    print(f"Top-k: {top_k}")
    print("Top-k columns:")
    pprint(pruning.top_k_columns)
    print("Preserved columns:")
    pprint(pruning.preserved_columns)
    print("Final columns:")
    pprint(pruning.final_columns)
    print("Gold columns:")
    pprint(sorted(gold_columns))
    print(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | F6: {f6:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BIRD schema pruning demo/benchmark")
    parser.add_argument("--mode", choices=["demo", "benchmark"], default="demo")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--tables", type=str, default=str(DEFAULT_TABLES))
    parser.add_argument("--db-root", type=str, default=str(DEFAULT_DB_ROOT))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--demo-question", type=str, default=None)
    parser.add_argument("--demo-db", type=str, default=None)
    parser.add_argument("--demo-sql", type=str, default=None)
    parser.add_argument("--force-hashing", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Label metode untuk penamaan file metrik TXT.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    tables_path = Path(args.tables)
    db_root = Path(args.db_root)

    if args.mode == "demo":
        run_demo(
            dataset_path=dataset_path,
            tables_path=tables_path,
            db_root=db_root,
            top_k=args.top_k,
            question=args.demo_question,
            db_id=args.demo_db,
            sql=args.demo_sql,
            force_hashing=args.force_hashing,
            model_name=args.model_name,
        )
    else:
        config = BenchmarkConfig(
            dataset_path=dataset_path,
            tables_path=tables_path,
            db_root=db_root,
            top_k=args.top_k,
            output_dir=Path(args.output_dir),
            max_samples=args.max_samples,
            force_hashing=args.force_hashing,
            model_name=args.model_name,
            method_name=args.method_name,
        )
        _, aggregate = run_benchmark(config)
        print("=== Benchmark Summary ===")
        pprint(aggregate)


if __name__ == "__main__":
    main()


