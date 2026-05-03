from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES = PROJECT_ROOT / "Dataset" / "BIRD" / "dev" / "dev_tables.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hitung jumlah kolom untuk setiap database dari file tables.json "
            "(format Spider/BIRD)."
        )
    )
    parser.add_argument(
        "--tables",
        "-t",
        nargs="+",
        default=[str(DEFAULT_TABLES)],
        help=(
            "Path ke tables.json (bisa lebih dari satu). "
            f"Default: {DEFAULT_TABLES}"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="Path output CSV. Jika kosong, hasil dicetak ke stdout (TSV).",
    )
    parser.add_argument(
        "--include-wildcard",
        action="store_true",
        help="Ikut menghitung kolom wildcard (*) jika ada.",
    )
    return parser.parse_args()


def _iter_table_entries(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        yield from raw
    elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
        yield from raw["data"]
    else:
        raise ValueError(f"Format JSON tidak dikenali: {path}")


def _count_columns(entry: dict, include_wildcard: bool) -> int:
    columns = entry.get("column_names_original") or entry.get("column_names")
    if not isinstance(columns, list):
        return 0
    count = 0
    for item in columns:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        table_id, col_name = item[0], item[1]
        if not include_wildcard:
            if table_id == -1 or str(col_name) == "*":
                continue
        count += 1
    return count


def _load_results(paths: List[Path], include_wildcard: bool) -> List[Tuple[str, str, int]]:
    results: List[Tuple[str, str, int]] = []
    for path in paths:
        for entry in _iter_table_entries(path):
            db_id = str(entry.get("db_id", entry.get("database_id", ""))).strip()
            if not db_id:
                continue
            count = _count_columns(entry, include_wildcard=include_wildcard)
            results.append((str(path), db_id, count))
    return results


def main() -> int:
    args = _parse_args()
    table_paths = [Path(p) for p in args.tables]
    results = _load_results(table_paths, include_wildcard=args.include_wildcard)

    if not results:
        print("Tidak ada database yang ditemukan.", file=sys.stderr)
        return 1

    multi_source = len(table_paths) > 1
    console_mode = not args.output

    if console_mode:
        if multi_source:
            print("source_file\tdb_id\tcolumn_count")
        else:
            print("db_id\tcolumn_count")
        for source_file, db_id, count in results:
            if multi_source:
                print(f"{source_file}\t{db_id}\t{count}")
            else:
                print(f"{db_id}\t{count}")
        return 0

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if multi_source:
            writer.writerow(["source_file", "db_id", "column_count"])
            for row in results:
                writer.writerow(row)
        else:
            writer.writerow(["db_id", "column_count"])
            for _, db_id, count in results:
                writer.writerow([db_id, count])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
