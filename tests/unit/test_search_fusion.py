"""RRF with hand-calculated expectations (k = 60 unless stated)."""

import pytest

from finsearch.search.fusion import rrf_fuse


def test_same_doc_first_in_both_lists():
    # A: rank 1 in both  -> 1/61 + 1/61 = 2/61
    # B: rank 2 in both  -> 1/62 + 1/62 = 2/62
    fused = rrf_fuse({"bm25": ["A", "B"], "knn": ["A", "B"]})
    assert [d for d, _ in fused] == ["A", "B"]
    assert fused[0][1] == pytest.approx(2 / 61)
    assert fused[1][1] == pytest.approx(2 / 62)


def test_doc_in_only_one_list_scores_one_term():
    # X: rank 1 in bm25 only -> 1/61 ; Y: rank 2 in knn only -> 1/62 ; Z: rank 1 in knn only -> 1/61
    fused = dict(rrf_fuse({"bm25": ["X"], "knn": ["Z", "Y"]}))
    assert fused == {
        "X": pytest.approx(1 / 61),
        "Z": pytest.approx(1 / 61),
        "Y": pytest.approx(1 / 62),
    }


def test_agreement_beats_a_single_first_place():
    # A: rank 2 + rank 2 -> 2/62 = 0.032258 ; B: rank 1 in one list -> 1/61 = 0.016393
    fused = rrf_fuse({"bm25": ["B", "A"], "knn": ["C", "A"]})
    assert fused[0][0] == "A"
    assert fused[0][1] == pytest.approx(2 / 62)


def test_custom_k():
    # k = 1: A rank 1 -> 1/2 ; B rank 2 -> 1/3
    assert rrf_fuse({"m": ["A", "B"]}, k=1) == [
        ("A", pytest.approx(1 / 2)),
        ("B", pytest.approx(1 / 3)),
    ]


def test_tie_prefers_better_best_rank_then_id():
    # A: bm25 rank 1 + knn rank 2 = 1/61 + 1/62 ; B: bm25 rank 2 + knn rank 1 = 1/62 + 1/61  => equal score.
    # Best rank of both is 1 => tie remains => id order decides: A before B.
    fused = rrf_fuse({"bm25": ["B", "A"], "knn": ["A", "B"]})
    assert fused[0][1] == fused[1][1]
    assert [d for d, _ in fused] == ["A", "B"]
    # Same score through different routes: C = 1/61 (rank 1, one list) vs D = 1/61 (rank 1, other list).
    # Reverse ids must not matter, only the id order does.
    assert [d for d, _ in rrf_fuse({"x": ["D"], "y": ["C"]})] == ["C", "D"]


def test_tie_break_uses_best_rank_before_id():
    # k = 2 makes an exact tie possible:
    # Z: rank 1 in l1 (1/3) + rank 4 in l2 (1/6) = 1/2 ; B: rank 2 in l1 (1/4) + rank 2 in l2 (1/4) = 1/2.
    # Equal score, Z has the better best rank (1 vs 2) => Z first, although "B" < "Z" alphabetically.
    fused = rrf_fuse({"l1": ["Z", "B"], "l2": ["w", "B", "y", "Z"]}, k=2)
    assert [(d, round(s, 6)) for d, s in fused[:2]] == [("Z", 0.5), ("B", 0.5)]


def test_empty_inputs():
    assert rrf_fuse({}) == []
    assert rrf_fuse({"bm25": [], "knn": []}) == []


def test_one_list_only_keeps_its_order():
    assert [d for d, _ in rrf_fuse({"bm25": ["c", "a", "b"]})] == ["c", "a", "b"]


def test_duplicate_in_a_list_counts_once_at_best_position():
    # "A" listed twice: only rank 1 counts, "B" is then rank 2 (duplicate does not take a rank).
    fused = dict(rrf_fuse({"bm25": ["A", "A", "B"]}))
    assert fused["A"] == pytest.approx(1 / 61)
    assert fused["B"] == pytest.approx(1 / 62)


def test_deterministic():
    rankings = {"bm25": ["a", "b", "c", "d"], "knn": ["d", "c", "b", "a"]}
    assert rrf_fuse(rankings) == rrf_fuse(dict(reversed(list(rankings.items()))))
