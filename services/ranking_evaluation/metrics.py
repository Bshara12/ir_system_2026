"""services/ranking_evaluation/metrics.py
=====================================
Core ranking & evaluation metric implementations (pure Python).

This module implements precision, recall, average precision (AP), mean
average precision (MAP), DCG and nDCG. The functions are written to be
safe for empty inputs and to use binary relevance for precision/recall/AP
and graded relevance for DCG/nDCG.

السلوك المتوقع:
- `retrieved_doc_ids` قائمة مرتبة من المعرّفات (الأفضل أولاً)
- `relevant_doc_ids` مجموعة أو قائمة من المعرّفات ذات الصلة (binary)
- `qrels` قاموس من معرّف -> درجة الصلة (graded relevance)

لا نعتمد على مكتبات خارجية هنا — كل شيء محلي وبسيط.
"""

from math import log2
from typing import List, Sequence, Set, Dict, Union


def precision_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Union[Sequence[str], Set[str]], k: int = 10) -> float:
	"""
	Precision@k = (# relevant docs in top-k) / k

	Uses binary relevance: an item is relevant if its id is in `relevant_doc_ids`.
	Returns 0.0 for empty inputs or when k <= 0.
	"""
	if k <= 0:
		return 0.0
	if not retrieved_doc_ids:
		return 0.0

	relevant_set = set(relevant_doc_ids or [])
	top_k = list(retrieved_doc_ids)[:k]
	if not top_k:
		return 0.0
	num_rel = sum(1 for doc in top_k if doc in relevant_set)
	return num_rel / float(k)


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Union[Sequence[str], Set[str]], k: int = 10) -> float:
	"""
	Recall@k = (# relevant docs in top-k) / (# relevant docs total)

	If there are no relevant documents, returns 0.0 to avoid division-by-zero.
	"""
	relevant_set = set(relevant_doc_ids or [])
	if not relevant_set:
		return 0.0
	top_k = list(retrieved_doc_ids)[:k]
	if not top_k:
		return 0.0
	num_rel = sum(1 for doc in top_k if doc in relevant_set)
	return num_rel / float(len(relevant_set))


def average_precision_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Union[Sequence[str], Set[str]], k: int = 10) -> float:
	"""
	Average Precision@k (AP@k).

	AP@k = (1 / R_k) * sum_{i=1..k} P@i * rel_i
	where rel_i = 1 if the i-th retrieved doc is relevant, otherwise 0,
	and R_k = min(number_of_relevant_documents, k) to normalise when using @k.

	Returns 0.0 when there are no relevant documents.
	"""
	if k <= 0:
		return 0.0
	retrieved = list(retrieved_doc_ids or [])[:k]
	relevant_set = set(relevant_doc_ids or [])
	if not relevant_set:
		return 0.0

	num_rel_total = len(relevant_set)
	# normaliser: use min(num_rel_total, k) so AP@k is in [0,1]
	normaliser = min(num_rel_total, k)
	if normaliser == 0:
		return 0.0

	score = 0.0
	num_rel_found = 0
	for idx, doc_id in enumerate(retrieved, start=1):
		if doc_id in relevant_set:
			num_rel_found += 1
			precision_i = num_rel_found / idx
			score += precision_i

	return score / float(normaliser)


def mean_average_precision(list_of_retrieved_lists: List[Sequence[str]], list_of_relevant_sets: List[Union[Sequence[str], Set[str]]], k: int = 10) -> float:
	"""
	Mean Average Precision (MAP) over multiple queries.

	Computes AP@k for each pair and returns the mean. If inputs are empty
	or lengths mismatch, handles gracefully and returns 0.0 for no valid pairs.
	"""
	if not list_of_retrieved_lists or not list_of_relevant_sets:
		return 0.0
	n = min(len(list_of_retrieved_lists), len(list_of_relevant_sets))
	if n == 0:
		return 0.0
	ap_sum = 0.0
	valid = 0
	for i in range(n):
		ap = average_precision_at_k(list_of_retrieved_lists[i], list_of_relevant_sets[i], k=k)
		ap_sum += ap
		valid += 1
	if valid == 0:
		return 0.0
	return ap_sum / float(valid)


def dcg_at_k(relevance_scores: Sequence[float], k: int = 10) -> float:
	"""
	Discounted Cumulative Gain (DCG) for a list of relevance scores.

	We use the common formulation with exponential gains:
		DCG = sum_{i=1..k} (2^{rel_i} - 1) / log2(i+1)

	Returns 0.0 for empty input or k <= 0.
	"""
	if k <= 0:
		return 0.0
	if not relevance_scores:
		return 0.0
	scores = list(relevance_scores)[:k]
	dcg = 0.0
	for idx, rel in enumerate(scores, start=1):
		numerator = (2 ** rel) - 1
		denom = log2(idx + 1)
		dcg += numerator / denom
	return dcg


def ndcg_at_k(retrieved_doc_ids: Sequence[str], qrels: Dict[str, float], k: int = 10) -> float:
	"""
	Normalized Discounted Cumulative Gain (nDCG)@k using graded relevance.

	Steps:
	1. Build relevance scores for retrieved docs using `qrels` (default 0 if missing).
	2. Compute DCG@k on retrieved list.
	3. Compute IDCG@k by sorting all qrels by relevance descending and taking top-k.
	4. Return DCG / IDCG (or 0.0 if IDCG == 0).
	"""
	if k <= 0:
		return 0.0
	if not retrieved_doc_ids:
		return 0.0

	# relevance scores in the order of retrieved documents
	retrieved_scores = [float(qrels.get(doc_id, 0.0)) for doc_id in list(retrieved_doc_ids)[:k]]
	dcg = dcg_at_k(retrieved_scores, k=k)

	# ideal DCG: take top-k highest qrel values
	ideal_scores = sorted([float(v) for v in qrels.values()], reverse=True)[:k]
	idcg = dcg_at_k(ideal_scores, k=k)
	if idcg == 0.0:
		return 0.0
	return dcg / idcg

