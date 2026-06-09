from __future__ import annotations

import html
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_TITLE = "Demo Text-to-SQL BIRD"
SYSTEM_PROMPT = (
    "You are an expert text-to-SQL assistant for SQLite. Use only the provided schema and question. "
    "Output only the final SQL query as plain text. Do not add explanations or formatting. "
    "Quote identifiers with backticks if they contain spaces or symbols."
)

CONFIGS = [
    {
        "method": "Full Schema",
        "top_k": None,
        "lexical_weight": None,
        "path": Path("outputs/dataset/full_schema/llm_dataset_v2_fullschema.json"),
    },
    {
        "method": "Embedding-only Pruning",
        "top_k": 20,
        "lexical_weight": 0.0,
        "path": Path("outputs/dataset/embedding_only/llm_dataset_v2_embedding_only_top_20.json"),
    },
    {
        "method": "Embedding-only Pruning",
        "top_k": 30,
        "lexical_weight": 0.0,
        "path": Path("outputs/dataset/embedding_only/llm_dataset_v2_embedding_only_top_30.json"),
    },
    {
        "method": "Embedding-only Pruning",
        "top_k": 40,
        "lexical_weight": 0.0,
        "path": Path("outputs/dataset/embedding_only/llm_dataset_v2_embedding_only_top_40.json"),
    },
    {
        "method": "Embedding + BM25",
        "top_k": 20,
        "lexical_weight": 0.3,
        "path": Path("outputs/dataset/embedding_bm25/llm_dataset_v2_embedding_top_20.json"),
    },
    {
        "method": "Embedding + BM25",
        "top_k": 30,
        "lexical_weight": 0.3,
        "path": Path("outputs/dataset/embedding_bm25/llm_dataset_v2_embedding__top_30.json"),
    },
    {
        "method": "Embedding + BM25",
        "top_k": 30,
        "lexical_weight": 0.5,
        "path": Path("outputs/dataset/embedding_bm25/llm_dataset_v2_embedding_bm25_0.5_top_30.json"),
    },
    {
        "method": "Embedding + BM25 + MKP",
        "top_k": 20,
        "lexical_weight": 0.3,
        "path": Path("outputs/dataset/mkp/llm_dataset_v2_mkp_top_20.json"),
    },
    {
        "method": "Embedding + BM25 + MKP",
        "top_k": 30,
        "lexical_weight": 0.3,
        "path": Path("outputs/dataset/mkp/llm_dataset_v2_mkp_top_30.json"),
    },
    {
        "method": "Embedding + BM25 + MKP",
        "top_k": 40,
        "lexical_weight": 0.3,
        "path": Path("outputs/dataset/mkp/llm_dataset_v2_mkp_top_40.json"),
    },
    {
        "method": "Embedding + BM25 + MKP",
        "top_k": 30,
        "lexical_weight": 0.5,
        "path": Path("outputs/dataset/mkp/llm_dataset_v2_embedding_bm25_0.5_mkp_top_30.json"),
    },
]

ERROR_FILES = {
    ("Llama 3.2 3B", "Full Schema"): Path("outputs/error_analysis/llama_3.2_fullschema.csv"),
    ("Llama 3.2 3B", "Embedding-only Pruning"): Path("outputs/error_analysis/llama_3.2_embedding_only_top_30.csv"),
    ("Llama 3.2 3B", "Embedding + BM25"): Path("outputs/error_analysis/llama_3.2_embedding_bm25_0.5_top_30.csv"),
    ("Llama 3.2 3B", "Embedding + BM25 + MKP"): Path("outputs/error_analysis/llama_3.2_embedding_bm25_0.5_mkp_top_30.csv"),
    ("Gemini 2.5 Flash", "Full Schema"): Path("outputs/error_analysis/gemini_2.5_flash_fullschema.csv"),
    ("Gemini 2.5 Flash", "Embedding-only Pruning"): Path("outputs/error_analysis/gemini_2.5_flash_embedding_only_top_30.csv"),
    ("Gemini 2.5 Flash", "Embedding + BM25"): Path("outputs/error_analysis/gemini_2.5_flash_embedding_bm25_0.5_top_30.csv"),
    ("Gemini 2.5 Flash", "Embedding + BM25 + MKP"): Path("outputs/error_analysis/gemini_2.5_flash_embedding_bm25_0.5_mkp_top_30.csv"),
}

MODELS = ["Llama 3.2 3B", "Gemini 2.5 Flash"]
DB_ROOT = Path("Dataset/BIRD/dev/dev_databases/dev_databases")


@st.cache_data(show_spinner=False)
def load_json_dataset(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def existing_configs() -> list[dict[str, Any]]:
    return [config for config in CONFIGS if config["path"].exists()]


def records_by_question(path: Path) -> dict[int, dict[str, Any]]:
    return {int(record["question_id"]): record for record in load_json_dataset(str(path))}


def normalize_column(column: str) -> str:
    column = column.replace("`", "").replace('"', "").strip()
    return re.sub(r"\s*\.\s*", ".", column).lower()


def clean_column(column: str) -> str:
    column = column.replace("`", "").replace('"', "").strip()
    return re.sub(r"\s*\.\s*", ".", column)


def parse_schema_columns(schema_prompt: str) -> list[str]:
    columns = []
    blocks = re.finditer(
        r"CREATE\s+TABLE\s+`?([^`\(\n]+)`?\s*\((.*?)\);",
        schema_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in blocks:
        table = block.group(1).strip()
        body = block.group(2)
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            match = re.match(r"`([^`]+)`\s+.+", line)
            if match:
                columns.append(f"{table}.{match.group(1)}")
    return columns


def schema_index(schema_prompt: str) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for column in parse_schema_columns(schema_prompt):
        table, column_name = clean_column(column).split(".", 1)
        index.setdefault(table, set()).add(column_name)
    return index


def parse_key_columns(schema_prompt: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    blocks = re.finditer(
        r"CREATE\s+TABLE\s+`?([^`\(\n]+)`?\s*\((.*?)\);",
        schema_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )

    for block in blocks:
        table = block.group(1).strip()
        body = block.group(2)
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            primary = re.match(r"PRIMARY\s+KEY\s*\((.*?)\)", line, flags=re.IGNORECASE)
            foreign = re.match(
                r"FOREIGN\s+KEY\s*\((.*?)\)\s+REFERENCES\s+`?([^`\(\s]+)`?\s*\((.*?)\)",
                line,
                flags=re.IGNORECASE,
            )

            if primary:
                for column in re.findall(r"`([^`]+)`", primary.group(1)):
                    rows.append(
                        {
                            "table": table,
                            "column": column,
                            "jenis": "primary key",
                            "reference": "-",
                            "full_column": f"{table}.{column}",
                        }
                    )

            if foreign:
                ref_table = foreign.group(2).strip()
                ref_columns = re.findall(r"`([^`]+)`", foreign.group(3))
                columns = re.findall(r"`([^`]+)`", foreign.group(1))
                for index, column in enumerate(columns):
                    ref_column = ref_columns[index] if index < len(ref_columns) else "-"
                    rows.append(
                        {
                            "table": table,
                            "column": column,
                            "jenis": "foreign key",
                            "reference": f"{ref_table}.{ref_column}",
                            "full_column": f"{table}.{column}",
                        }
                    )

    unique_rows: list[dict[str, str]] = []
    seen = set()
    for row in rows:
        key = (row["table"], row["column"], row["jenis"], row["reference"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    return unique_rows


def parse_fk_edges(schema_prompt: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen = set()
    blocks = re.finditer(
        r"CREATE\s+TABLE\s+`?([^`\(\n]+)`?\s*\((.*?)\);",
        schema_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )

    for block in blocks:
        table = block.group(1).strip()
        body = block.group(2)
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            foreign = re.match(
                r"FOREIGN\s+KEY\s*\((.*?)\)\s+REFERENCES\s+`?([^`\(\s]+)`?\s*\((.*?)\)",
                line,
                flags=re.IGNORECASE,
            )
            if not foreign:
                continue

            ref_table = foreign.group(2).strip()
            columns = re.findall(r"`([^`]+)`", foreign.group(1))
            ref_columns = re.findall(r"`([^`]+)`", foreign.group(3))
            if not columns:
                columns = [item.strip().strip("`") for item in foreign.group(1).split(",")]
            if not ref_columns:
                ref_columns = [item.strip().strip("`") for item in foreign.group(3).split(",")]

            for index, column in enumerate(columns):
                ref_column = ref_columns[index] if index < len(ref_columns) else ref_columns[0]
                row = {
                    "from_table": table,
                    "from_column": column,
                    "to_table": ref_table,
                    "to_column": ref_column,
                    "from_full": f"{table}.{column}",
                    "to_full": f"{ref_table}.{ref_column}",
                    "edge": f"{table}.{column} <-> {ref_table}.{ref_column}",
                }
                key = (row["from_full"], row["to_full"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def selected_tables_from_columns(columns: list[str]) -> list[str]:
    tables: list[str] = []
    seen = set()
    for column in columns:
        if "." not in column:
            continue
        table = clean_column(column).split(".", 1)[0]
        key = table.lower()
        if key not in seen:
            seen.add(key)
            tables.append(table)
    return tables


def build_table_graph(fk_edges: list[dict[str, str]]) -> dict[str, list[tuple[str, dict[str, str]]]]:
    graph: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for edge in fk_edges:
        graph.setdefault(edge["from_table"], []).append((edge["to_table"], edge))
        graph.setdefault(edge["to_table"], []).append((edge["from_table"], edge))
    return graph


def shortest_table_path(
    start: str,
    goal: str,
    graph: dict[str, list[tuple[str, dict[str, str]]]],
) -> list[dict[str, str]] | None:
    if start == goal:
        return []

    queue: list[tuple[str, list[dict[str, str]]]] = [(start, [])]
    visited = {start}
    while queue:
        node, path = queue.pop(0)
        for neighbor, edge in graph.get(node, []):
            if neighbor in visited:
                continue
            step = {
                **edge,
                "traverse_from": node,
                "traverse_to": neighbor,
            }
            next_path = path + [step]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return None


def mkp_graph_walk(
    selected_tables: list[str],
    fk_edges: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    graph = build_table_graph(fk_edges)
    rows: list[dict[str, str]] = []
    required_columns: list[str] = []
    seen_required = set()

    for i, table_a in enumerate(selected_tables):
        for table_b in selected_tables[i + 1 :]:
            path = shortest_table_path(table_a, table_b, graph)
            if path is None:
                rows.append(
                    {
                        "pasangan tabel": f"{table_a} -> {table_b}",
                        "path graph": "tidak terhubung",
                        "edge join": "-",
                        "kolom wajib": "-",
                    }
                )
                continue

            path_tables = [table_a] + [step["traverse_to"] for step in path]
            join_columns: list[str] = []
            edge_labels: list[str] = []
            for step in path:
                edge_labels.append(step["edge"])
                for column in (step["from_full"], step["to_full"]):
                    join_columns.append(column)
                    normalized = normalize_column(column)
                    if normalized not in seen_required:
                        seen_required.add(normalized)
                        required_columns.append(column)

            rows.append(
                {
                    "pasangan tabel": f"{table_a} -> {table_b}",
                    "path graph": " -> ".join(path_tables),
                    "edge join": " | ".join(edge_labels) if edge_labels else "-",
                    "kolom wajib": ", ".join(join_columns) if join_columns else "-",
                }
            )

    return rows, required_columns


def extract_table_aliases(sql: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+`?([A-Za-z_][\w\s-]*)`?(?:\s+(?:AS\s+)?([A-Za-z_][\w]*))?",
        flags=re.IGNORECASE,
    )
    reserved = {"where", "join", "inner", "left", "right", "full", "on", "group", "order", "limit", "having"}
    for match in pattern.finditer(sql):
        table = match.group(1).strip()
        alias = (match.group(2) or table).strip()
        if alias.lower() in reserved:
            alias = table
        aliases[alias] = table
        aliases[table] = table
    return aliases


def extract_column_candidates(sql: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen = set()

    qualified_pattern = re.compile(
        r"(`?[A-Za-z_][\w]*`?)\s*\.\s*(`[^`]+`|[A-Za-z_][\w\s%()/-]*)",
        flags=re.IGNORECASE,
    )
    for match in qualified_pattern.finditer(sql):
        prefix = match.group(1).strip("`")
        column = match.group(2).strip().strip("`")
        key = ("qualified", prefix, column)
        if key not in seen:
            seen.add(key)
            candidates.append({"tipe": "qualified", "prefix": prefix, "kolom": column})

    qualified_columns = {item["kolom"] for item in candidates}
    quoted_pattern = re.compile(r"`([^`]+)`")
    for match in quoted_pattern.finditer(sql):
        token = match.group(1).strip()
        if token in qualified_columns:
            continue
        key = ("quoted", "", token)
        if key not in seen:
            seen.add(key)
            candidates.append({"tipe": "quoted", "prefix": "", "kolom": token})

    return candidates


def resolve_column_candidates(
    candidates: list[dict[str, str]],
    aliases: dict[str, str],
    table_columns: dict[str, set[str]],
) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    for candidate in candidates:
        prefix = candidate["prefix"]
        column = candidate["kolom"]
        table = ""
        status = "unresolved"

        if prefix:
            table = aliases.get(prefix, prefix)
            status = "resolved" if column in table_columns.get(table, set()) else "unknown column"
        else:
            matches = [table_name for table_name, columns in table_columns.items() if column in columns]
            if len(matches) == 1:
                table = matches[0]
                status = "resolved"
            elif len(matches) > 1:
                table = ", ".join(matches)
                status = "ambiguous"

        resolved.append(
            {
                "tipe": candidate["tipe"],
                "prefix": prefix or "-",
                "kolom": column,
                "tabel": table or "-",
                "hasil": f"{table}.{column}" if table and "," not in table and status == "resolved" else "-",
                "status": status,
            }
        )
    return resolved


def compute_metrics(selected: list[str], gold: list[str], full: list[str]) -> dict[str, Any]:
    selected_map = {normalize_column(column): clean_column(column) for column in selected}
    gold_map = {normalize_column(column): clean_column(column) for column in gold}
    selected_set = set(selected_map)
    gold_set = set(gold_map)
    full_set = {normalize_column(column) for column in full}
    hit = selected_set & gold_set

    precision = len(hit) / len(selected_set) if selected_set else 0.0
    recall = len(hit) / len(gold_set) if gold_set else 1.0
    beta2 = 36
    f6 = ((1 + beta2) * precision * recall / (beta2 * precision + recall)) if precision + recall else 0.0
    reduction = 1 - (len(selected_set) / len(full_set)) if full_set else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f6": f6,
        "reduction": reduction,
        "full_recall": gold_set.issubset(selected_set),
        "gold_selected": sorted(gold_map[column] for column in hit),
        "missed_gold": sorted(gold_map[column] for column in gold_set - selected_set),
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def short(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def pills(items: list[str], css_class: str, empty: str = "-") -> None:
    if not items:
        st.caption(empty)
        return
    content = "".join(
        f'<span class="pill {css_class}">{html.escape(item)}</span>' for item in items
    )
    st.markdown(f'<div class="pill-box">{content}</div>', unsafe_allow_html=True)


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if record.get("messages"):
        return record["messages"]

    user_content = f"""Database: {record["db_id"]}

Schema:
{record["schema_prompt"]}

Evidence:
{record.get("evidence", "")}

Question: {record["question"]}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def inference_file_for(model: str, config: dict[str, Any]) -> Path | None:
    method = config["method"]
    if method == "Full Schema":
        return ERROR_FILES.get((model, method))
    if config["top_k"] == 30 and config["lexical_weight"] in (0.0, 0.5):
        return ERROR_FILES.get((model, method))
    return None


def get_inference(model: str, config: dict[str, Any], question_id: int) -> dict[str, Any] | None:
    path = inference_file_for(model, config)
    if path is None or not path.exists():
        return None

    df = load_csv(str(path))
    if df.empty:
        return None

    row = df[df["question_id"] == question_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def db_path_for(db_id: str) -> Path:
    return DB_ROOT / db_id / f"{db_id}.sqlite"


def clean_sql_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def execute_sql(db_path: Path, sql: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    sql = clean_sql_value(sql)
    if not sql:
        return {"rows": [], "columns": [], "elapsed": 0.0, "error": "SQL kosong."}

    normalized = sql.lstrip().lower()
    if not (normalized.startswith("select") or normalized.startswith("with")):
        return {
            "rows": [],
            "columns": [],
            "elapsed": 0.0,
            "error": "SQL tidak diawali SELECT/WITH, sehingga tidak dieksekusi untuk demo read-only.",
        }

    if not db_path.exists():
        return {"rows": [], "columns": [], "elapsed": 0.0, "error": f"Database tidak ditemukan: {db_path}"}

    deadline = time.monotonic() + timeout_seconds
    conn = None
    started = time.monotonic()
    try:
        db_uri = db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description or []]
        return {
            "rows": rows,
            "columns": columns,
            "elapsed": time.monotonic() - started,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - error message is useful for live demo.
        return {
            "rows": [],
            "columns": [],
            "elapsed": time.monotonic() - started,
            "error": str(exc).replace("\n", " ")[:500],
        }
    finally:
        if conn is not None:
            conn.close()


def results_equal(gold_rows: list[tuple[Any, ...]], predicted_rows: list[tuple[Any, ...]]) -> bool:
    return set(gold_rows) == set(predicted_rows)


def result_preview(result: dict[str, Any], max_rows: int = 20) -> pd.DataFrame:
    rows = result["rows"][:max_rows]
    columns = result["columns"]
    if not rows:
        return pd.DataFrame(columns=columns)
    if columns and len(columns) == len(rows[0]):
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows)


def select_config(configs: list[dict[str, Any]]) -> dict[str, Any]:
    methods = []
    for config in configs:
        if config["method"] not in methods:
            methods.append(config["method"])

    method = st.sidebar.selectbox("Metode schema", methods, index=methods.index("Embedding + BM25 + MKP"))
    method_configs = [config for config in configs if config["method"] == method]

    if method == "Full Schema":
        st.sidebar.caption("top_k dan lexical_weight tidak dipakai.")
        return method_configs[0]

    top_options = sorted({config["top_k"] for config in method_configs})
    top_k = st.sidebar.selectbox("top_k", top_options, index=top_options.index(30) if 30 in top_options else 0)

    top_configs = [config for config in method_configs if config["top_k"] == top_k]
    lexical_options = sorted({config["lexical_weight"] for config in top_configs})

    if method == "Embedding-only Pruning":
        st.sidebar.caption("lexical_weight = 0.0")
        lexical_weight = lexical_options[0]
    else:
        lexical_weight = st.sidebar.selectbox(
            "lexical_weight BM25",
            lexical_options,
            index=lexical_options.index(0.5) if 0.5 in lexical_options else 0,
        )

    return next(
        config
        for config in top_configs
        if config["lexical_weight"] == lexical_weight
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 1050px; }
        .pill-box { display:flex; flex-wrap:wrap; gap:.35rem; margin:.2rem 0 .75rem; }
        .pill { border-radius:999px; padding:.24rem .55rem; font-size:.82rem; border:1px solid rgba(0,0,0,.12); }
        .gold { background:#dcfce7; color:#14532d; }
        .missing { background:#fee2e2; color:#7f1d1d; }
        .neutral { background:#eef2f7; color:#273142; }
        .mkp { background:#dbeafe; color:#1e3a8a; }
        .ok { color:#14532d; font-weight:700; }
        .bad { color:#7f1d1d; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_config(
    configs: list[dict[str, Any]],
    method: str,
    top_k: int | None,
    lexical_weight: float | None,
) -> dict[str, Any] | None:
    for config in configs:
        if config["method"] == method and config["top_k"] == top_k and config["lexical_weight"] == lexical_weight:
            return config
    return None


def render_demo(
    config: dict[str, Any],
    record: dict[str, Any],
    full_record: dict[str, Any],
    selected_columns: list[str],
    metrics: dict[str, Any],
    model: str,
    question_id: int,
) -> None:
    st.caption(f"`{config['path'].name}`")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Metode", config["method"])
    col_b.metric("top_k", "-" if config["top_k"] is None else config["top_k"])
    col_c.metric("lexical_weight", "-" if config["lexical_weight"] is None else config["lexical_weight"])
    col_d.metric("Kolom schema", len(selected_columns))

    st.divider()
    st.subheader("1. Input")
    st.write(f"**question_id:** {record['question_id']} | **db_id:** {record['db_id']} | **difficulty:** {record.get('difficulty', '-')}")
    st.markdown(f"**Question:** {record['question']}")
    st.markdown(f"**Evidence:** {record.get('evidence') or '-'}")
    with st.expander("Gold SQL", expanded=True):
        st.code(record["gold_sql"], language="sql")

    st.subheader("2. Pruning")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", percent(metrics["precision"]))
    m2.metric("Recall", percent(metrics["recall"]))
    m3.metric("F6-score", percent(metrics["f6"]))
    m4.metric("Reduction", percent(metrics["reduction"]))

    status = "Full recall: YA" if metrics["full_recall"] else "Full recall: TIDAK"
    status_class = "ok" if metrics["full_recall"] else "bad"
    st.markdown(f'<p class="{status_class}">{status}</p>', unsafe_allow_html=True)

    st.write("**Gold**")
    pills([clean_column(column) for column in record.get("gold_columns", [])], "gold")
    st.write("**Terpilih**")
    pills(metrics["gold_selected"], "gold")
    st.write("**Hilang**")
    pills(metrics["missed_gold"], "missing", "Tidak ada.")

    with st.expander("schema_prompt"):
        st.code(record["schema_prompt"], language="sql")

    st.subheader("3. Prompt")
    messages = build_messages(record)
    with st.expander("messages", expanded=True):
        st.code(json.dumps(messages, indent=2, ensure_ascii=False), language="json")

    st.subheader("4. Inference")
    inference = get_inference(model, config, question_id)
    if inference is None:
        st.info("Hasil inference tidak tersedia.")
    else:
        correct = bool(inference.get("exec_correct", 0))
        st.markdown(
            f'<p class="{"ok" if correct else "bad"}">Execution Accuracy: {"BENAR" if correct else "SALAH"}</p>',
            unsafe_allow_html=True,
        )
        st.write(f"**Model:** {model} | **Error:** {inference.get('error_group', '-')}")
        st.write("**Predicted SQL**")
        st.code(str(inference.get("predicted_sql", "")), language="sql")

    st.subheader("5. Execution Check")
    db_path = db_path_for(record["db_id"])
    st.caption(f"`{db_path}`")

    gold_result = execute_sql(db_path, record["gold_sql"])
    if gold_result["error"]:
        st.error(f"Gold error: {gold_result['error']}")
    else:
        st.success(f"Gold: {len(gold_result['rows'])} rows ({gold_result['elapsed']:.3f}s)")

    if inference is None:
        st.info("Predicted SQL tidak tersedia.")
        with st.expander("Gold result"):
            st.dataframe(result_preview(gold_result), use_container_width=True)
        return

    predicted_sql = clean_sql_value(inference.get("predicted_sql", ""))
    predicted_result = execute_sql(db_path, predicted_sql)
    if predicted_result["error"]:
        st.error(f"Predicted error: {predicted_result['error']}")
    else:
        st.success(f"Predicted: {len(predicted_result['rows'])} rows ({predicted_result['elapsed']:.3f}s)")

    if not gold_result["error"] and not predicted_result["error"]:
        same_result = results_equal(gold_result["rows"], predicted_result["rows"])
        if same_result:
            st.markdown('<p class="ok">Result: SAMA</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="bad">Result: BERBEDA</p>', unsafe_allow_html=True)

    col_gold, col_pred = st.columns(2)
    with col_gold:
        with st.expander("Gold result", expanded=True):
            st.dataframe(result_preview(gold_result), use_container_width=True)
    with col_pred:
        with st.expander("Predicted result", expanded=True):
            st.dataframe(result_preview(predicted_result), use_container_width=True)


def render_simulation(
    configs: list[dict[str, Any]],
    record: dict[str, Any],
    full_record: dict[str, Any],
    active_config: dict[str, Any],
    question_id: int,
) -> None:
    st.subheader("A. Gold Column")

    table_columns = schema_index(full_record["schema_prompt"])
    aliases = extract_table_aliases(record["gold_sql"])
    candidates = extract_column_candidates(record["gold_sql"])
    resolved = resolve_column_candidates(candidates, aliases, table_columns)
    resolved_columns = [row["hasil"] for row in resolved if row["hasil"] != "-"]
    gold_columns = [clean_column(column) for column in record.get("gold_columns", [])]

    st.markdown("**1. SQL referensi masuk**")
    st.code(record["gold_sql"], language="sql")

    st.markdown("**2. Baca schema sebagai ruang validasi**")
    schema_rows = [{"table": table, "jumlah kolom": len(columns)} for table, columns in table_columns.items()]
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    st.markdown("**3. Deteksi tabel dan alias dari FROM/JOIN**")
    alias_rows = [{"alias": alias, "table": table} for alias, table in aliases.items() if alias != table]
    if alias_rows:
        st.dataframe(pd.DataFrame(alias_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Tidak ada alias eksplisit.")

    st.markdown("**4. Ambil kandidat kolom dari SQL**")
    if candidates:
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
    else:
        st.caption("Tidak ada kandidat terdeteksi.")

    st.markdown("**5. Resolusi alias ke table.column**")
    if resolved:
        st.dataframe(pd.DataFrame(resolved), use_container_width=True, hide_index=True)
    else:
        st.caption("Tidak ada kolom yang bisa diresolusi.")

    st.markdown("**6. Gold columns final dari preprocessing**")
    pills(gold_columns, "gold")

    with st.expander("Kandidat hasil resolusi"):
        pills(resolved_columns, "neutral", "Tidak ada.")

    full_columns = parse_schema_columns(full_record["schema_prompt"])

    st.divider()
    st.subheader("B. Mandatory Key Preservation")

    top_k = active_config["top_k"] if active_config["top_k"] is not None else 30
    lexical_weight = active_config["lexical_weight"] if active_config["lexical_weight"] is not None else 0.5
    before_config = find_config(configs, "Embedding + BM25", top_k, lexical_weight)
    after_config = find_config(configs, "Embedding + BM25 + MKP", top_k, lexical_weight)

    if before_config is None or after_config is None:
        before_config = find_config(configs, "Embedding + BM25", 30, 0.5)
        after_config = find_config(configs, "Embedding + BM25 + MKP", 30, 0.5)

    if before_config is None or after_config is None:
        st.info("Pasangan hybrid dan MKP tidak ditemukan.")
        return

    before_record = records_by_question(before_config["path"])[question_id]
    after_record = records_by_question(after_config["path"])[question_id]
    before_columns = parse_schema_columns(before_record["schema_prompt"])
    after_columns = parse_schema_columns(after_record["schema_prompt"])
    before_norm = {normalize_column(column) for column in before_columns}
    added_columns = [column for column in after_columns if normalize_column(column) not in before_norm]

    before_metrics = compute_metrics(before_columns, record.get("gold_columns", []), full_columns)
    after_metrics = compute_metrics(after_columns, record.get("gold_columns", []), full_columns)
    before_norm = {normalize_column(column) for column in before_columns}
    after_norm = {normalize_column(column) for column in after_columns}
    key_rows = parse_key_columns(full_record["schema_prompt"])
    fk_edges = parse_fk_edges(full_record["schema_prompt"])
    selected_tables = selected_tables_from_columns(before_columns)
    path_rows, required_join_columns = mkp_graph_walk(selected_tables, fk_edges)
    required_norm = {normalize_column(column) for column in required_join_columns}
    required_missing = [column for column in required_join_columns if normalize_column(column) not in before_norm]
    mkp_added_from_graph = [column for column in added_columns if normalize_column(column) in required_norm]
    mkp_added_other = [column for column in added_columns if normalize_column(column) not in required_norm]

    key_type_map: dict[str, list[str]] = {}
    for row in key_rows:
        key_type_map.setdefault(normalize_column(row["full_column"]), []).append(row["jenis"])

    st.markdown("**1. Ambil schema hasil hybrid retrieval**")
    c_before_1, c_before_2, c_before_3 = st.columns(3)
    c_before_1.metric("top_k", before_config["top_k"])
    c_before_2.metric("lexical_weight", before_config["lexical_weight"])
    c_before_3.metric("kolom top-k", len(before_columns))
    with st.expander("kolom top-k sebelum MKP"):
        pills(before_columns, "neutral")

    st.markdown("**2. Ubah kolom top-k menjadi selected tables**")
    table_rows = []
    for table in selected_tables:
        table_rows.append(
            {
                "selected table": table,
                "kolom terpilih": len(
                    [column for column in before_columns if clean_column(column).split(".", 1)[0] == table]
                ),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("**3. Bentuk schema graph dari foreign key**")
    graph_nodes = sorted({edge["from_table"] for edge in fk_edges} | {edge["to_table"] for edge in fk_edges})
    g1, g2, g3 = st.columns(3)
    g1.metric("node table", len(graph_nodes))
    g2.metric("edge FK", len(fk_edges))
    g3.metric("selected table", len(selected_tables))
    graph_rows = [
        {
            "node A": edge["from_table"],
            "node B": edge["to_table"],
            "edge column": edge["edge"],
        }
        for edge in fk_edges
    ]
    st.dataframe(pd.DataFrame(graph_rows), use_container_width=True, hide_index=True)

    st.markdown("**4. Cari shortest path antar selected tables**")
    if path_rows:
        st.dataframe(pd.DataFrame(path_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Hanya satu tabel terpilih, jadi tidak perlu path antar tabel.")

    st.markdown("**5. Ambil mandatory join columns dari edge path**")
    required_rows = []
    for column in required_join_columns:
        normalized = normalize_column(column)
        required_rows.append(
            {
                "mandatory column": column,
                "jenis": ", ".join(sorted(set(key_type_map.get(normalized, ["bridge column"])))),
                "sebelum MKP": "sudah ada" if normalized in before_norm else "belum ada",
                "sesudah MKP": "ada" if normalized in after_norm else "tidak ada",
            }
        )
    if required_rows:
        st.dataframe(pd.DataFrame(required_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Tidak ada mandatory join column tambahan.")

    st.markdown("**6. Union schema: top-k columns + mandatory columns**")
    u1, u2, u3 = st.columns(3)
    u1.metric("top-k columns", len(before_columns))
    u2.metric("mandatory belum ada", len(required_missing))
    u3.metric("final columns", len(after_columns))
    st.write("**Ditambahkan oleh MKP dari graph**")
    pills(mkp_added_from_graph, "mkp", "Tidak ada kolom graph yang perlu ditambahkan.")
    if mkp_added_other:
        st.write("**Tambahan lain di file MKP**")
        pills(mkp_added_other, "mkp")

    st.markdown("**7. Schema final setelah MKP**")
    with st.expander("kolom setelah MKP", expanded=True):
        pills(after_columns, "neutral")

    st.markdown("**8. Dampak ke metrik pruning**")
    compare = pd.DataFrame(
        [
            {
                "Tahap": "Sebelum MKP",
                "Kolom": len(before_columns),
                "Recall": round(before_metrics["recall"] * 100, 2),
                "Reduction": round(before_metrics["reduction"] * 100, 2),
                "Full recall": "YA" if before_metrics["full_recall"] else "TIDAK",
            },
            {
                "Tahap": "Sesudah MKP",
                "Kolom": len(after_columns),
                "Recall": round(after_metrics["recall"] * 100, 2),
                "Reduction": round(after_metrics["reduction"] * 100, 2),
                "Full recall": "YA" if after_metrics["full_recall"] else "TIDAK",
            },
        ]
    )
    st.dataframe(compare, use_container_width=True, hide_index=True)

    with st.expander("schema sebelum MKP"):
        st.code(before_record["schema_prompt"], language="sql")
    with st.expander("schema sesudah MKP"):
        st.code(after_record["schema_prompt"], language="sql")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()

    configs = existing_configs()
    if not configs:
        st.error("Tidak ada file dataset preprocess yang ditemukan di outputs/dataset.")
        st.stop()

    st.sidebar.header("Pengaturan")
    config = select_config(configs)
    records = records_by_question(config["path"])

    base_records = records_by_question(Path("outputs/dataset/full_schema/llm_dataset_v2_fullschema.json"))
    db_ids = sorted({record["db_id"] for record in base_records.values()})
    db_id = st.sidebar.selectbox(
        "db_id",
        db_ids,
        index=db_ids.index("california_schools") if "california_schools" in db_ids else 0,
    )

    candidates = [
        record
        for question_id, record in base_records.items()
        if record["db_id"] == db_id and question_id in records
    ]
    labels = {
        f'{record["question_id"]} | {record.get("difficulty", "-")} | {short(record["question"], 68)}': int(record["question_id"])
        for record in candidates
    }
    question_id = st.sidebar.selectbox("Sample", list(labels.values()), format_func={v: k for k, v in labels.items()}.get)
    model = st.sidebar.selectbox("Model inference", MODELS, index=1)

    record = records[question_id]
    full_record = base_records[question_id]
    selected_columns = parse_schema_columns(record["schema_prompt"])
    full_columns = parse_schema_columns(full_record["schema_prompt"])
    metrics = compute_metrics(selected_columns, record.get("gold_columns", []), full_columns)

    st.title(APP_TITLE)
    tab_demo, tab_simulation = st.tabs(["Demo", "Simulasi"])

    with tab_demo:
        render_demo(config, record, full_record, selected_columns, metrics, model, question_id)

    with tab_simulation:
        render_simulation(configs, record, full_record, config, question_id)


if __name__ == "__main__":
    main()
