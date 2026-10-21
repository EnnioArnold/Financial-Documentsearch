"""Measure the RAG evidence gate on the golden set (extractive provider, real index).

Writes reports/rag_gate.md. Thresholds were tuned on this same set (8 negatives), so the numbers
are optimistic; see docs/DECISIONS.md ADR-007.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from finsearch.rag.answer import answer_question
from finsearch.rag.providers.extractive import ExtractiveClient
from finsearch.search.service import get_search_service

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.disable(logging.CRITICAL)
    rows = [json.loads(x) for x in (ROOT / "eval" / "golden_set.jsonl").read_text().splitlines()]
    svc, llm = get_search_service(), ExtractiveClient()
    acc = rej_ok = n_ans = n_un = 0
    wrong: list[str] = []
    for r in rows:
        a = answer_question(r["question"], svc, llm)
        if r["answerable"]:
            n_ans += 1
            acc += a.found
            if not a.found:
                wrong.append(f"- `{r['id']}` fälschlich abgelehnt: {r['question']}")
        else:
            n_un += 1
            rej_ok += not a.found
            if a.found:
                wrong.append(f"- `{r['id']}` fälschlich beantwortet: {r['question']}")
    out = [
        "# RAG-Evidence-Gate (Extractive, Hybrid, echter Index)",
        "",
        f"- Beantwortbare Fragen mit Antwort: **{acc}/{n_ans}**",
        f"- Unbeantwortbare Fragen korrekt mit «In den Quellen nicht gefunden» abgelehnt: **{rej_ok}/{n_un}**",
        "",
        "Schwellen auf demselben Golden Set getunt (nur 8 negative Fragen) – Zahlen sind optimistisch.",
        "",
        *wrong,
    ]
    (ROOT / "reports" / "rag_gate.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
