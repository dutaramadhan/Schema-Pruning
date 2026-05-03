from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.experiment_runner_v2 import ExperimentConfig, run_experiments
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Dataset" / "BIRD" / "dev"
DEFAULT_DATASET = DEFAULT_DATA_ROOT / "dev.json"
DEFAULT_TABLES = DEFAULT_DATA_ROOT / "dev_tables.json"
DEFAULT_DB_ROOT = DEFAULT_DATA_ROOT / "dev_databases" / "dev_databases"


def _parse_top_ks(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _parse_modes(value: str) -> tuple[str, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="BIRD LLM experiment runner (V2)")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--tables", type=str, default=str(DEFAULT_TABLES))
    parser.add_argument("--db-root", type=str, default=str(DEFAULT_DB_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-ks", type=str, default="10,20,30")
    parser.add_argument(
        "--pruning-modes",
        type=str,
        default="full,embedding_only,embedding_mandatory",
    )
    parser.add_argument("--llm-config", type=str, default=None)
    parser.add_argument("--lexical-weight", type=float, default=0.3)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--no-evidence", action="store_false", dest="use_evidence")
    parser.add_argument("--no-types", action="store_false", dest="include_types")
    parser.add_argument("--no-foreign-keys", action="store_false", dest="include_foreign_keys")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--save-schema-text", action="store_true")
    parser.add_argument("--float-round", type=int, default=6)
    parser.add_argument("--ves-cap", type=float, default=1.0)
    parser.add_argument("--no-ves-cap", action="store_true")
    parser.add_argument("--prompt-template", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(use_evidence=True, include_types=True, include_foreign_keys=True)

    args = parser.parse_args()

    prompt_template = None
    if args.prompt_template:
        prompt_template = Path(args.prompt_template).read_text(encoding="utf-8")

    ves_cap = None if args.no_ves_cap else args.ves_cap

    config = ExperimentConfig(
        dataset_path=Path(args.dataset),
        tables_path=Path(args.tables),
        db_root=Path(args.db_root),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        top_ks=_parse_top_ks(args.top_ks),
        pruning_modes=_parse_modes(args.pruning_modes),
        llm_config_path=Path(args.llm_config) if args.llm_config else None,
        use_evidence=args.use_evidence,
        lexical_weight=args.lexical_weight,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        include_types=args.include_types,
        include_foreign_keys=args.include_foreign_keys,
        prompt_template=prompt_template,
        save_prompts=args.save_prompts,
        save_schema_text=args.save_schema_text,
        float_round=args.float_round,
        ves_cap=ves_cap,
        dry_run=args.dry_run,
    )

    _, aggregate = run_experiments(config)
    print("=== Experiment Summary (V2) ===")
    pprint(aggregate)


if __name__ == "__main__":
    main()
