from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hitung jumlah token untuk setiap gold/ground-truth SQL "
            "menggunakan tokenizer Llama 3.1 Instruct 8B."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="Path file/folder dataset (.json/.jsonl/.sql). Bisa lebih dari satu.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="Path output CSV. Jika kosong, akan dicetak ke stdout.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Nama model tokenizer di Hugging Face atau path lokal.",
    )
    parser.add_argument(
        "--fields",
        default="gold_sql,SQL,sql",
        help="Urutan field SQL pada JSON/JSONL (comma-separated).",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Jika diaktifkan, tambahkan special tokens saat tokenisasi.",
    )
    return parser.parse_args()


def _iter_input_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for ext in (".json", ".jsonl", ".sql"):
                files.extend(sorted(path.rglob(f"*{ext}")))
        else:
            files.append(path)
    return files


def _extract_sql(item: dict, fields: List[str]) -> str:
    for key in fields:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _iter_json_items(path: Path) -> Iterator[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            for item in raw:
                yield item
        elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
            for item in raw["data"]:
                yield item
        else:
            raise ValueError(f"Format JSON tidak dikenali: {path}")


def _iter_sql_rows(
    path: Path, fields: List[str]
) -> Iterable[Tuple[int, str, str, str]]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        for idx, item in enumerate(_iter_json_items(path), start=1):
            sql = _extract_sql(item, fields)
            if not sql:
                continue
            record_id = str(item.get("question_id", item.get("id", idx)))
            db_id = str(item.get("db_id", item.get("database_id", "")))
            yield idx, record_id, db_id, sql
    elif path.suffix.lower() == ".sql":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("--"):
                    continue
                parts = line.split("\t", 1)
                sql = parts[0].strip()
                db_id = parts[1].strip() if len(parts) > 1 else ""
                if not sql:
                    continue
                yield idx, str(idx), db_id, sql
    else:
        return


def main() -> int:
    args = _parse_args()
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    input_files = _iter_input_files(args.input)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(
            "Error: transformers belum terpasang. Install dulu dengan:\n"
            "pip install transformers",
            file=sys.stderr,
        )
        raise exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    console_mode = not args.output
    out_stream = sys.stdout if console_mode else open(args.output, "w", newline="", encoding="utf-8")
    close_out = not console_mode

    if console_mode:
        print(
            "source_file\trow_index\trecord_id\tdb_id\ttoken_count\tsql",
            file=out_stream,
        )
    else:
        writer = csv.writer(out_stream)
        writer.writerow(
            ["source_file", "row_index", "record_id", "db_id", "token_count", "sql"]
        )

    total = 0
    total_tokens = 0
    min_tokens = None
    max_tokens = None
    min_row = None
    max_row = None

    for path in input_files:
        for row_index, record_id, db_id, sql in _iter_sql_rows(path, fields):
            token_count = len(
                tokenizer.encode(sql, add_special_tokens=args.add_special_tokens)
            )
            clean_sql = " ".join(sql.split())
            row_payload = [str(path), row_index, record_id, db_id, token_count, clean_sql]
            if console_mode:
                print(
                    f"{row_payload[0]}\t{row_payload[1]}\t{row_payload[2]}\t{row_payload[3]}\t{row_payload[4]}\t{row_payload[5]}",
                    file=out_stream,
                )
            else:
                writer.writerow(row_payload)
            total += 1
            total_tokens += token_count
            if min_tokens is None or token_count < min_tokens:
                min_tokens = token_count
                min_row = row_payload
            if max_tokens is None or token_count > max_tokens:
                max_tokens = token_count
                max_row = row_payload

    if close_out:
        out_stream.close()

    if total == 0:
        print("Tidak ada SQL yang diproses.", file=sys.stderr)
        return 1

    avg = total_tokens / total if total else 0
    summary_lines = [
        f"Processed {total} SQL.",
        f"Min tokens: {min_tokens}",
        f"Mean tokens: {avg:.2f}",
        f"Max tokens: {max_tokens}",
    ]
    if min_row is not None:
        summary_lines.append(
            "Min row: "
            f"file={min_row[0]} row_index={min_row[1]} record_id={min_row[2]} db_id={min_row[3]}"
        )
    if max_row is not None:
        summary_lines.append(
            "Max row: "
            f"file={max_row[0]} row_index={max_row[1]} record_id={max_row[2]} db_id={max_row[3]}"
        )

    if console_mode:
        print("", file=out_stream)
        for line in summary_lines:
            print(line, file=out_stream)
    else:
        for line in summary_lines:
            print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
