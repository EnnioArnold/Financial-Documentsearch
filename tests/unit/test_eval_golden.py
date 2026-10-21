import json
from pathlib import Path

import pytest

from finsearch.eval.golden import GoldenSetError, load_golden

SAMPLE = Path(__file__).parents[1] / "fixtures" / "eval_golden_sample.jsonl"


def _valid(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "q1", "question": "Q?", "lang": "de", "type": "fact",
        "relevant_chunk_ids": ["d-0001"], "relevant_doc_ids": ["d"], "answerable": True,
        "reference_answer": None, "provenance": "human",
    }  # fmt: skip
    return {**base, **over}


def _write(tmp_path: Path, *rows: object) -> Path:
    p = tmp_path / "g.jsonl"
    p.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8"
    )
    return p


def test_load_sample() -> None:
    qs = load_golden(SAMPLE)
    assert len(qs) == 6
    assert sum(q.answerable for q in qs) == 5
    doc_level = next(q for q in qs if q.id == "syn004")
    assert doc_level.doc_level and doc_level.relevant_keys == {"syn-d"}
    assert next(q for q in qs if q.id == "syn001").relevant_keys == {"syn-a-0001"}


@pytest.mark.parametrize(
    ("row", "msg"),
    [
        ({"lang": "fr"}, "'lang'"),
        ({"type": "weird"}, "'type'"),
        ({"provenance": "bot"}, "'provenance'"),
        ({"answerable": "yes"}, "'answerable'"),
        ({"question": " "}, "'question'"),
        ({"relevant_chunk_ids": "d-0001"}, "'relevant_chunk_ids'"),
        ({"reference_answer": 3}, "'reference_answer'"),
        ({"relevant_chunk_ids": [], "relevant_doc_ids": []}, "needs relevant"),
        ({"type": "unanswerable"}, "unanswerable"),
        ({"answerable": False}, "unanswerable"),
    ],
)
def test_schema_errors_name_line(tmp_path: Path, row: dict[str, object], msg: str) -> None:
    path = _write(tmp_path, _valid(id="ok"), _valid(**row))
    with pytest.raises(GoldenSetError, match=rf"g\.jsonl:2: .*{msg}"):
        load_golden(path)


def test_other_errors(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="not found"):
        load_golden(tmp_path / "missing.jsonl")
    with pytest.raises(GoldenSetError, match="empty"):
        load_golden(_write(tmp_path, "", "  "))
    with pytest.raises(GoldenSetError, match=":1:"):
        load_golden(_write(tmp_path, "{not json"))
    with pytest.raises(GoldenSetError, match="JSON object"):
        load_golden(_write(tmp_path, "[1]"))
    with pytest.raises(GoldenSetError, match="duplicate id"):
        load_golden(_write(tmp_path, _valid(), _valid()))
