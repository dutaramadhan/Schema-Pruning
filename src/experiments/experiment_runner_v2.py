from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bird_datasets.bird_dataset_loader import BirdDatasetLoader, BirdSample
from pruning.embedding_retriever_v2 import EmbeddingRetrieverV2, RankedColumn
from sql.llm_client import LLMConfig, LLMClient, build_llm_client
from utils.results_writer import write_csv, write_json, write_jsonl
from pruning.schema_graph import SchemaGraph
from pruning.schema_loader import DatabaseSchema, SchemaLoader
from sql.schema_prompt import build_prompt, build_schema_text
from sql.sql_evaluator import EvalResult, QueryResult, evaluate_prediction


PRUNING_FULL = "full"
PRUNING_EMBEDDING_ONLY = "embedding_only"
PRUNING_EMBEDDING_MANDATORY = "embedding_mandatory"


@dataclass
class ExperimentConfig:
    dataset_path: Path
    tables_path: Path
    db_root: Path
    output_dir: Path
    max_samples: Optional[int] = None
    top_ks: Sequence[int] = (10, 20, 30)
    pruning_modes: Sequence[str] = (
        PRUNING_FULL,
        PRUNING_EMBEDDING_ONLY,
        PRUNING_EMBEDDING_MANDATORY,
    )
    llm_config_path: Optional[Path] = None
    use_evidence: bool = True
    lexical_weight: float = 0.3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    include_types: bool = True
    include_foreign_keys: bool = True
    prompt_template: Optional[str] = None
    save_prompts: bool = False
    save_schema_text: bool = False
    float_round: int = 6
    ves_cap: Optional[float] = 1.0
    dry_run: bool = False


@dataclass
class LLMRuntime:
    config: LLMConfig
    client: Optional[LLMClient]
    predictions: Optional[Dict[Tuple[int, str, Optional[int]], str]]


def run_experiments(config: ExperimentConfig) -> Tuple[List[Dict], List[Dict]]:
    dataset = BirdDatasetLoader(config.dataset_path)
    schema_loader = SchemaLoader(config.tables_path, config.db_root)

    llm_runtimes = _load_llm_runtimes(config.llm_config_path) if not config.dry_run else []
    if not llm_runtimes and not config.dry_run:
        raise ValueError("LLM config is required unless dry_run is enabled.")

    use_embedding = any(
        mode in (PRUNING_EMBEDDING_ONLY, PRUNING_EMBEDDING_MANDATORY)
        for mode in config.pruning_modes
    )
    retriever = None
    if use_embedding:
        retriever = EmbeddingRetrieverV2(
            model_name=None,
            lexical_weight=config.lexical_weight,
            use_evidence=config.use_evidence,
            bm25_k1=config.bm25_k1,
            bm25_b=config.bm25_b,
        )

    per_sample: List[Dict] = []
    aggregates: Dict[Tuple[str, str, Optional[int]], List[EvalResult]] = {}

    gold_cache: Dict[Tuple[str, int], QueryResult] = {}

    for idx, sample in enumerate(dataset.iter_samples()):
        if config.max_samples is not None and idx >= config.max_samples:
            break

        schema = schema_loader.get_schema(sample.db_id)
        evidence = sample.evidence if config.use_evidence else None

        ranked = _rank_columns(retriever, sample, schema, evidence)
        graph = SchemaGraph.from_schema(schema)

        pruning_variants = _build_pruning_variants(
            sample=sample,
            schema=schema,
            ranked=ranked,
            graph=graph,
            top_ks=config.top_ks,
            pruning_modes=config.pruning_modes,
        )

        for variant in pruning_variants:
            schema_text = build_schema_text(
                schema,
                variant["columns"],
                include_types=config.include_types,
                include_foreign_keys=config.include_foreign_keys,
            )
            prompt = build_prompt(
                question=sample.question,
                schema_text=schema_text,
                evidence=evidence,
                template=config.prompt_template,
            )

            if config.dry_run:
                per_sample.append(
                    _row_base(sample, variant, schema_text, prompt, config, None, None, None)
                )
                continue

            for runtime in llm_runtimes:
                pred_sql = _get_prediction(
                    runtime,
                    sample,
                    variant["pruning_mode"],
                    variant["top_k"],
                    prompt,
                )
                eval_result, gold_result, _ = _evaluate_with_cache(
                    schema_loader,
                    sample,
                    pred_sql,
                    gold_cache,
                    float_round=config.float_round,
                    ves_cap=config.ves_cap,
                )

                key = (runtime.config.name, variant["pruning_mode"], variant["top_k"])
                aggregates.setdefault(key, []).append(eval_result)

                per_sample.append(
                    _row_base(
                        sample,
                        variant,
                        schema_text,
                        prompt,
                        config,
                        runtime,
                        pred_sql,
                        eval_result,
                        gold_result,
                    )
                )

    aggregate_rows = _aggregate_results(aggregates)
    output_dir = Path(config.output_dir)
    write_jsonl(output_dir / "experiment_per_sample.jsonl", per_sample)
    write_csv(output_dir / "experiment_per_sample.csv", _flatten_for_csv(per_sample))
    write_json(output_dir / "experiment_aggregate.json", {"rows": aggregate_rows})

    return per_sample, aggregate_rows


def _rank_columns(
    retriever: Optional[EmbeddingRetrieverV2],
    sample: BirdSample,
    schema: DatabaseSchema,
    evidence: Optional[str],
) -> List[RankedColumn]:
    if retriever is None:
        return []
    return retriever.rank_columns(sample.question, schema, evidence=evidence)


def _build_pruning_variants(
    sample: BirdSample,
    schema: DatabaseSchema,
    ranked: List[RankedColumn],
    graph: SchemaGraph,
    top_ks: Sequence[int],
    pruning_modes: Sequence[str],
) -> List[Dict]:
    variants: List[Dict] = []
    for mode in pruning_modes:
        if mode == PRUNING_FULL:
            variants.append(
                {
                    "pruning_mode": mode,
                    "top_k": None,
                    "columns": schema.all_column_full_names(),
                    "top_k_columns": schema.all_column_full_names(),
                    "preserved_columns": [],
                }
            )
            continue

        if not ranked:
            raise ValueError("Embedding retriever is required for embedding-based pruning.")

        for top_k in top_ks:
            top_k = min(top_k, len(ranked))
            top_k_columns = [col.full_name for col in ranked[:top_k]]
            preserved_columns: List[str] = []
            if mode == PRUNING_EMBEDDING_MANDATORY:
                selected_tables = {col.table for col in ranked[:top_k]}
                preserved = graph.required_join_columns(selected_tables)
                preserved_columns = sorted([c for c in preserved if c not in top_k_columns])
                final_columns = sorted(set(top_k_columns) | preserved)
            else:
                final_columns = sorted(set(top_k_columns))

            variants.append(
                {
                    "pruning_mode": mode,
                    "top_k": top_k,
                    "columns": final_columns,
                    "top_k_columns": top_k_columns,
                    "preserved_columns": preserved_columns,
                }
            )
    return variants


def _load_llm_runtimes(config_path: Optional[Path]) -> List[LLMRuntime]:
    if not config_path:
        return []
    with Path(config_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    runtimes: List[LLMRuntime] = []
    for item in payload.get("llms", []):
        llm_config = LLMConfig(
            name=item["name"],
            backend=item.get("backend", "openai_compatible"),
            model=item.get("model"),
            base_url=item.get("base_url"),
            api_key_env=item.get("api_key_env"),
            temperature=float(item.get("temperature", 0.0)),
            max_tokens=int(item.get("max_tokens", 512)),
            system_prompt=item.get("system_prompt"),
            timeout_s=float(item.get("timeout_s", 60.0)),
            predictions_path=item.get("predictions_path"),
        )
        if llm_config.backend == "predictions_file" and not llm_config.predictions_path:
            raise ValueError(f"predictions_path is required for {llm_config.name}.")
        client = build_llm_client(llm_config)
        predictions = _load_predictions(llm_config.predictions_path)
        runtimes.append(LLMRuntime(config=llm_config, client=client, predictions=predictions))
    return runtimes


def _load_predictions(path: Optional[str]) -> Optional[Dict[Tuple[int, str, Optional[int]], str]]:
    if not path:
        return None
    mapping: Dict[Tuple[int, str, Optional[int]], str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question_id = int(item.get("question_id", item.get("id", 0)))
            pruning_mode = item.get("pruning_mode", PRUNING_FULL)
            top_k_raw = item.get("top_k")
            top_k = int(top_k_raw) if top_k_raw is not None else None
            sql = item.get("pred_sql") or item.get("sql") or ""
            mapping[(question_id, pruning_mode, top_k)] = sql
    return mapping


def _get_prediction(
    runtime: LLMRuntime,
    sample: BirdSample,
    pruning_mode: str,
    top_k: Optional[int],
    prompt: str,
) -> str:
    if runtime.predictions is not None:
        key = (sample.question_id, pruning_mode, top_k)
        if key in runtime.predictions:
            return _sanitize_sql(runtime.predictions[key])
        fallback = (sample.question_id, pruning_mode, None)
        return _sanitize_sql(runtime.predictions.get(fallback, ""))

    if runtime.client is None:
        raise RuntimeError(f"LLM backend '{runtime.config.backend}' is not available.")
    raw = runtime.client.generate(prompt, system_prompt=runtime.config.system_prompt)
    return _sanitize_sql(raw)


def _sanitize_sql(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    if "```" in text:
        first_fence = text.find("```")
        second_fence = text.find("```", first_fence + 3)
        if second_fence != -1:
            text = text[first_fence + 3 : second_fence].strip()
        else:
            text = text[first_fence + 3 :].strip()

    # Some models prefix fenced content with a language label (`sql`/`sqlite`).
    text = re.sub(r"^(?:sql|sqlite)\b[\s:;-]*", "", text, flags=re.IGNORECASE).strip()

    # Handle plain-text labels without code fences, e.g. `SQL: SELECT ...`.
    text = re.sub(r"^(?:sql|sqlite)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    if text.endswith("```"):
        text = text[:-3].strip()
    return text.strip()


def _evaluate_with_cache(
    schema_loader: SchemaLoader,
    sample: BirdSample,
    pred_sql: str,
    gold_cache: Dict[Tuple[str, int], QueryResult],
    float_round: int,
    ves_cap: Optional[float],
) -> Tuple[EvalResult, QueryResult, QueryResult]:
    key = (sample.db_id, sample.question_id)
    gold_result = gold_cache.get(key)
    db_path = schema_loader.get_database_path(sample.db_id)
    eval_result, gold_result, pred_result = evaluate_prediction(
        db_path=db_path,
        gold_sql=sample.sql,
        pred_sql=pred_sql,
        gold_result=gold_result,
        float_round=float_round,
        ves_cap=ves_cap,
    )
    gold_cache[key] = gold_result
    return eval_result, gold_result, pred_result


def _row_base(
    sample: BirdSample,
    variant: Dict,
    schema_text: str,
    prompt: str,
    config: ExperimentConfig,
    runtime: Optional[LLMRuntime],
    pred_sql: Optional[str],
    eval_result: Optional[EvalResult],
    gold_result: Optional[QueryResult] = None,
) -> Dict:
    row: Dict = {
        "question_id": sample.question_id,
        "db_id": sample.db_id,
        "question": sample.question,
        "evidence": sample.evidence,
        "gold_sql": sample.sql,
        "pruning_mode": variant["pruning_mode"],
        "top_k": variant["top_k"],
        "top_k_columns": variant["top_k_columns"],
        "preserved_columns": variant["preserved_columns"],
        "final_columns": variant["columns"],
    }

    if runtime is not None:
        row["llm_name"] = runtime.config.name
        row["llm_backend"] = runtime.config.backend
        row["llm_model"] = runtime.config.model

    if pred_sql is not None:
        row["pred_sql"] = pred_sql

    if eval_result is not None:
        row["execution_accuracy"] = eval_result.exec_accuracy
        row["ves"] = eval_result.ves
        row["pred_time_s"] = eval_result.pred_time_s
        row["gold_time_s"] = eval_result.gold_time_s
        row["error"] = eval_result.error

    if config.save_prompts:
        row["prompt"] = prompt
    if config.save_schema_text:
        row["schema_text"] = schema_text

    if gold_result is not None and eval_result is not None and eval_result.error:
        row["gold_exec_time_s"] = gold_result.elapsed_s

    return row


def _aggregate_results(aggregates: Dict[Tuple[str, str, Optional[int]], List[EvalResult]]) -> List[Dict]:
    rows: List[Dict] = []
    for (llm_name, pruning_mode, top_k), results in aggregates.items():
        exec_accs = [r.exec_accuracy for r in results]
        ves_scores = [r.ves for r in results]
        row = {
            "llm_name": llm_name,
            "pruning_mode": pruning_mode,
            "top_k": top_k,
            "num_samples": len(results),
            "mean_execution_accuracy": mean(exec_accs) if exec_accs else 0.0,
            "mean_ves": mean(ves_scores) if ves_scores else 0.0,
        }
        rows.append(row)
    return rows


def _flatten_for_csv(rows: Iterable[Dict]) -> List[Dict]:
    flat_rows: List[Dict] = []
    for row in rows:
        flat = dict(row)
        for key in ["top_k_columns", "preserved_columns", "final_columns"]:
            if key in flat and isinstance(flat[key], list):
                flat[key] = "; ".join(flat[key])
        flat_rows.append(flat)
    return flat_rows


