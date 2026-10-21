"""Build eval/golden_set.jsonl from eval/golden_questions.yaml.

Each question carries verbatim evidence quotes. A chunk is relevant iff its (whitespace/quote
normalised) text contains one of the quotes. A quote that matches no chunk is a hard error, so
every answerable question is verified against the real source text on each build.

Usage: python eval/build_golden_set.py [--check]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
QUESTIONS = ROOT / "eval" / "golden_questions.yaml"
OUT = ROOT / "eval" / "golden_set.jsonl"
VALID_TYPES = {"fact", "paraphrase", "number_date", "comparison", "unanswerable"}
_DASHES = dict.fromkeys(map(ord, "‐‑‒–"), "-")
_QUOTES = {ord("’"): "'", ord("‘"): "'", ord("“"): '"', ord("”"): '"'}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_DASHES).translate(_QUOTES)).strip().lower()


def build() -> list[dict]:
    chunks = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]
    normed = [(c, norm(c["text"])) for c in chunks]
    items = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    rows, errors = [], []
    for i, q in enumerate(items, 1):
        qid = f"q{i:03d}"
        if q["type"] not in VALID_TYPES:
            errors.append(f"{qid}: bad type {q['type']}")
        answerable = q["type"] != "unanswerable"
        hits: list[dict] = []
        for quote in q.get("quotes", []):
            found = [c for c, t in normed if norm(quote) in t]
            if not found:
                errors.append(f"{qid}: quote not found in any chunk: {quote!r}")
            hits += found
        if answerable and not hits:
            errors.append(f"{qid}: answerable but no evidence")
        if not answerable and q.get("quotes"):
            errors.append(f"{qid}: unanswerable must not have quotes")
        ids = sorted({c["chunk_id"] for c in hits})
        rows.append(
            {
                "id": qid,
                "question": q["question"],
                "lang": q["lang"],
                "type": q["type"],
                "relevant_chunk_ids": ids,
                "relevant_doc_ids": sorted({c["doc_id"] for c in hits}),
                "answerable": answerable,
                "reference_answer": q.get("answer"),
                "provenance": "llm_suggested_verified",
            }
        )
    if errors:
        sys.exit("\n".join(errors))
    return rows


if __name__ == "__main__":
    rows = build()
    if "--check" not in sys.argv:
        OUT.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    for r in rows:
        print(r["id"], r["type"], r["lang"], len(r["relevant_chunk_ids"]), r["question"][:70])
    print(f"{len(rows)} questions, {sum(not r['answerable'] for r in rows)} unanswerable")
