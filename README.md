# Schema Pruning (BIRD)

Pipeline sesuai alur column-centric yang Anda jelaskan.

## Setup

Install dependensi:

```bash
pip install -r requirements.txt
```

## Step 0: Metadata Embedding

```bash
python -m schema_pruning.cli build-index \
  --tables-json "Dataset/BIRD/dev/dev_tables.json" \
  --store-dir "artifact_store" \
  --store faiss \
  --model "sentence-transformers/all-MiniLM-L6-v2"
```

Jika ingin hanya beberapa database:

```bash
python -m schema_pruning.cli build-index \
  --tables-json "Dataset/BIRD/dev/dev_tables.json" \
  --store-dir "artifact_store" \
  --db-id california_schools,financial
```

Catatan:
- `--store` bisa `faiss`, `chroma`, atau `numpy`.
- Hasil per-db disimpan di `artifact_store/<db_id>/`.

## Step 1-5: Query (DDL ringkas)

Pakai `question_id` dari `dev.json` (evidence otomatis diambil):

```bash
python -m schema_pruning.cli query \
  --tables-json "Dataset/BIRD/dev/dev_tables.json" \
  --dataset-json "Dataset/BIRD/dev/dev.json" \
  --question-id 0 \
  --store-dir "artifact_store" \
  --top-k 8 \
  --table-scope all \
  --output-ddl "out.sql" \
  --output-json "out.json"
```

Atau input manual:

```bash
python -m schema_pruning.cli query \
  --tables-json "Dataset/BIRD/dev/dev_tables.json" \
  --db-id "california_schools" \
  --question "What is the highest eligible free rate for K-12 students in Alameda County?" \
  --evidence "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`" \
  --store-dir "artifact_store" \
  --top-k 8
```

## Evaluasi Evidence vs Top-K

```bash
python -m schema_pruning.cli evaluate \
  --tables-json "Dataset/BIRD/dev/dev_tables.json" \
  --dataset-json "Dataset/BIRD/dev/dev.json" \
  --store-dir "artifact_store" \
  --top-k 8 \
  --output-json "eval.json"
```

Tambahkan `--per-item` untuk output per pertanyaan, dan `--max-questions` untuk sampling.

## Output

- `out.sql`: DDL ringkas per tabel.
- `out.json`: detail kolom terpilih + sumber (`topk`, `evidence`, `pkfk`) + statistik.
- `eval.json`: ringkasan coverage evidence vs top-k.

## Catatan Desain

- Format teks kolom: `[Table: ...] [Column: ...] [Tipe_Data: ...]`.
- Evidence diambil dari backticks di field `evidence`.
- PK/FK dipaksa masuk untuk menjaga kemampuan JOIN.
- `table-scope=all` mempertahankan semua tabel (sesuai Langkah 1).
