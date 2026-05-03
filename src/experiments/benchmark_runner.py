from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from bird_datasets.bird_dataset_loader import BirdDatasetLoader
from pruning.embedding_retriever import EmbeddingRetriever
from pruning.evaluation import extract_gold_columns, precision_recall_f1_f6
from pruning.pruner import SchemaPruner
from pruning.schema_loader import SchemaLoader
from utils.results_writer import write_csv, write_json, write_jsonl, write_metrics_txt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "outputs" / "pruning stats"


@dataclass
class BenchmarkConfig:
    dataset_path: Path
    tables_path: Path
    db_root: Path
    top_k: int = 5
    output_dir: Path = Path("outputs")
    max_samples: Optional[int] = None
    force_hashing: bool = False
    model_name: Optional[str] = None
    method_name: Optional[str] = None


def run_benchmark(config: BenchmarkConfig) -> Tuple[List[Dict], Dict]:
    dataset = BirdDatasetLoader(config.dataset_path)
    schema_loader = SchemaLoader(config.tables_path, config.db_root)
    retriever = EmbeddingRetriever(
        model_name=config.model_name,
        force_hashing=config.force_hashing,
    )
    pruner = SchemaPruner(schema_loader, retriever)

    per_sample: List[Dict] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    f6s: List[float] = []
    reductions: List[float] = []

    for idx, sample in enumerate(dataset.iter_samples()):
        if config.max_samples is not None and idx >= config.max_samples:
            break

        schema = schema_loader.get_schema(sample.db_id)
        pruning = pruner.prune(sample.question, sample.db_id, config.top_k)
        gold_columns = extract_gold_columns(sample.sql, schema)
        precision, recall, f1, f6 = precision_recall_f1_f6(
            pruning.final_columns, gold_columns
        )

        total_cols = len(schema.columns)
        pruned_cols = len(pruning.final_columns)
        reduction = 1.0 - (pruned_cols / total_cols) if total_cols else 0.0

        row = {
            "question_id": sample.question_id,
            "db_id": sample.db_id,
            "question": sample.question,
            "gold_sql": sample.sql,
            "retriever_backend": pruning.backend,
            "ranked_columns": [c.full_name for c in pruning.ranked_columns],
            "top_k_columns": pruning.top_k_columns,
            "preserved_columns": pruning.preserved_columns,
            "final_columns": pruning.final_columns,
            "gold_columns": sorted(gold_columns),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f6": f6,
            "total_columns": total_cols,
            "pruned_columns": pruned_cols,
            "reduction_ratio": reduction,
        }
        per_sample.append(row)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        f6s.append(f6)
        reductions.append(reduction)

    aggregate = {
        "num_samples": len(per_sample),
        "top_k": config.top_k,
        "mean_precision": mean(precisions) if precisions else 0.0,
        "mean_recall": mean(recalls) if recalls else 0.0,
        "mean_f1": mean(f1s) if f1s else 0.0,
        "mean_f6": mean(f6s) if f6s else 0.0,
        "mean_reduction_ratio": mean(reductions) if reductions else 0.0,
        "retriever_backend": retriever.backend,
    }

    output_dir = Path(config.output_dir)
    write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    write_json(output_dir / "aggregate.json", aggregate)
    write_csv(output_dir / "per_sample.csv", _flatten_for_csv(per_sample))

    method_name = _resolve_method_name(config)
    txt_name = f"aggregate_{method_name}_top_{config.top_k}.txt"
    write_metrics_txt(
        METRICS_DIR / txt_name,
        aggregate,
        title=f"Benchmark Summary ({method_name}, top_k={config.top_k})",
    )

    return per_sample, aggregate


def _flatten_for_csv(rows: List[Dict]) -> List[Dict]:
    flat_rows: List[Dict] = []
    for row in rows:
        flat = dict(row)
        for key in [
            "ranked_columns",
            "top_k_columns",
            "preserved_columns",
            "final_columns",
            "gold_columns",
        ]:
            if key in flat and isinstance(flat[key], list):
                flat[key] = "; ".join(flat[key])
        flat_rows.append(flat)
    return flat_rows


def _resolve_method_name(config: BenchmarkConfig) -> str:
    if config.method_name:
        return _slugify_method_name(config.method_name)
    return "mkp"


def _slugify_method_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "method"
