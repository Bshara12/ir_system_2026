import pytest
from math import log2

from services.ranking_evaluation.metrics import (
	average_precision_at_k,
	mean_average_precision,
	ndcg_at_k,
	precision_at_k,
	recall_at_k,
)


def test_perfect_ranking_binary_metrics():
	retrieved = ["d1", "d2", "d3"]
	relevant = {"d1", "d2", "d3"}

	assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)
	assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)
	assert average_precision_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_partial_relevant_ranking():
	retrieved = ["d5", "d3", "d1", "d8", "d2"]
	relevant = {"d5", "d1", "d2"}

	assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(3 / 5)
	assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)
	assert average_precision_at_k(retrieved, relevant, k=5) == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)


def test_no_relevant_documents_returns_zero():
	retrieved = ["d1", "d2", "d3"]
	relevant = set()

	assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)
	assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)
	assert average_precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)
	assert ndcg_at_k(retrieved, {}, k=3) == pytest.approx(0.0)


def test_empty_retrieved_list_returns_zero():
	relevant = {"d1", "d2"}
	qrels = {"d1": 3, "d2": 2}

	assert precision_at_k([], relevant, k=5) == pytest.approx(0.0)
	assert recall_at_k([], relevant, k=5) == pytest.approx(0.0)
	assert average_precision_at_k([], relevant, k=5) == pytest.approx(0.0)
	assert ndcg_at_k([], qrels, k=5) == pytest.approx(0.0)


def test_ndcg_at_k_with_graded_qrels():
	retrieved = ["d5", "d3", "d1", "d8", "d2"]
	qrels = {"d5": 3, "d3": 1, "d1": 2, "d8": 0, "d2": 3}

	dcg = 7 / log2(2) + 1 / log2(3) + 3 / log2(4) + 0 / log2(5) + 7 / log2(6)
	idcg = 7 / log2(2) + 7 / log2(3) + 3 / log2(4) + 1 / log2(5) + 0 / log2(6)
	expected = dcg / idcg

	assert ndcg_at_k(retrieved, qrels, k=5) == pytest.approx(expected)


def test_mean_average_precision_across_multiple_queries():
	retrieved_lists = [
		["d1", "d2", "d3"],
		["d4", "d5", "d6"],
		["d7", "d8"],
	]
	relevant_sets = [
		{"d1", "d3"},
		{"d5"},
		{"d9"},
	]

	# APs: (1 + 2/3)/2, 1/2, 0
	expected = (((1 + 2 / 3) / 2) + 0.5 + 0.0) / 3

	assert mean_average_precision(retrieved_lists, relevant_sets, k=3) == pytest.approx(expected)
