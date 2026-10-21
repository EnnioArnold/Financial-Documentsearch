"""Metric tests with hand-calculated expectations."""

import math

import pytest

from finsearch.eval import metrics as m


def test_recall_at_k_hand_calculated() -> None:
    # relevant = {a, b, c}; top-5 = [x, a, y, b, z] -> 2 of 3 found at k=5, 1 of 3 at k=2
    ranked = ["x", "a", "y", "b", "z"]
    rel = {"a", "b", "c"}
    assert m.recall_at_k(ranked, rel, 5) == pytest.approx(2 / 3)
    assert m.recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)
    assert m.recall_at_k(ranked, rel, 1) == 0.0


def test_recall_edge_cases() -> None:
    assert m.recall_at_k(["a"], set(), 5) == 0.0  # no relevant docs
    assert m.recall_at_k([], {"a"}, 5) == 0.0  # empty results
    assert m.recall_at_k(["a"], {"a"}, 100) == 1.0  # k larger than list
    assert m.recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == 0.5  # duplicates credited once


def test_reciprocal_rank() -> None:
    assert m.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(["a"], {"a"}) == 1.0
    assert m.reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0  # beyond cutoff
    assert m.reciprocal_rank([], {"a"}) == 0.0
    assert m.reciprocal_rank(["a"], set()) == 0.0


def test_first_relevant_rank() -> None:
    assert m.first_relevant_rank(["x", "a"], {"a"}) == 2
    assert m.first_relevant_rank(["x"], {"a"}) is None


def test_ndcg_hand_calculated() -> None:
    # relevant = {a, b}; ranking [a, x, b]:
    # DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    ndcg = m.ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3)
    assert ndcg == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    # perfect ranking -> 1.0
    assert m.ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)
    # single relevant at rank 3: DCG = 1/log2(4) = 0.5, IDCG = 1 -> 0.5
    assert m.ndcg_at_k(["x", "y", "a"], {"a"}, 10) == pytest.approx(0.5)


def test_ndcg_edge_cases() -> None:
    assert m.ndcg_at_k(["a"], set(), 10) == 0.0
    assert m.ndcg_at_k([], {"a"}, 10) == 0.0
    # duplicate of a relevant key must not inflate the score above 1
    assert m.ndcg_at_k(["a", "a"], {"a"}, 10) == pytest.approx(1.0)
    # cutoff: relevant item at rank 3 is invisible at k=2
    assert m.ndcg_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_percentile_linear_interpolation() -> None:
    vals = [10.0, 20.0, 30.0, 40.0]
    # median: pos = 3 * 0.5 = 1.5 -> 20 + 0.5 * (30 - 20) = 25
    assert m.median(vals) == 25.0
    # p95: pos = 3 * 0.95 = 2.85 -> 30 + 0.85 * 10 = 38.5
    assert m.percentile(vals, 95) == pytest.approx(38.5)
    assert m.percentile([7.0], 95) == 7.0
    assert m.percentile([], 50) == 0.0
    assert m.percentile([3.0, 1.0, 2.0], 0) == 1.0  # unsorted input
    with pytest.raises(ValueError):
        m.percentile(vals, 101)


def test_mean_and_bootstrap_ci() -> None:
    assert m.mean([]) == 0.0
    assert m.mean([1.0, 2.0]) == 1.5
    assert m.bootstrap_ci([]) == (0.0, 0.0)
    assert m.bootstrap_ci([0.5, 0.5, 0.5]) == (0.5, 0.5)  # no variance -> degenerate interval
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    lo, hi = m.bootstrap_ci(scores)
    assert 0.0 <= lo <= m.mean(scores) <= hi <= 1.0
    assert (lo, hi) == m.bootstrap_ci(scores)  # seeded -> deterministic
    assert m.bootstrap_ci(scores, seed=1) != (lo, hi)
