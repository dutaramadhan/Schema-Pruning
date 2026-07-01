from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber


PDF_PATH = Path(r"D:\Kuliah\Skripsi\Berkas\Pendadaran\Skripsi-Diestra Pradana Duta Ramadhan.pdf")
OUT_DIR = Path(r"D:\Kuliah\Skripsi\Schema Pruning\tmp\pdfs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


pages: list[dict[str, object]] = []
with pdfplumber.open(str(PDF_PATH)) as pdf:
    for idx, page in enumerate(pdf.pages, start=1):
        text = clean(page.extract_text(x_tolerance=1.5, y_tolerance=3) or "")
        pages.append({"page": idx, "text": text})

full_text = "\n\n".join(f"--- PAGE {p['page']} ---\n{p['text']}" for p in pages)
(OUT_DIR / "skripsi_text.txt").write_text(full_text, encoding="utf-8")
(OUT_DIR / "skripsi_pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")

heading_re = re.compile(
    r"^(BAB\s+[IVXLC]+|[0-9]+(?:\.[0-9]+)+\s+.+|ABSTRAK|ABSTRACT|DAFTAR\s+ISI|DAFTAR\s+PUSTAKA|LAMPIRAN)\b",
    re.IGNORECASE,
)
headings: list[dict[str, object]] = []
for page in pages:
    for line in str(page["text"]).splitlines():
        line = line.strip()
        if heading_re.match(line):
            headings.append({"page": page["page"], "heading": line})

(OUT_DIR / "skripsi_headings.json").write_text(json.dumps(headings, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({"pages": len(pages), "chars": len(full_text), "headings": headings[:80]}, ensure_ascii=False, indent=2))
