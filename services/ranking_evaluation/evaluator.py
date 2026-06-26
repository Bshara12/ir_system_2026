"""Evaluator wrapper for the Ranking Evaluation service.

This module contains both the demo evaluation helper and the real dataset
evaluation workflow that loads queries and qrels from disk and calls the
Retrieval Service for each query.
"""
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import httpx

from shared.constants import DATASETS_DIR, RETRIEVAL_URL
from shared.models import RetrievalModel

from . import metrics


QUERY_ID_FIELDS = ["query_id", "qid", "id", "query-id"]
QUERY_TEXT_FIELDS = ["text", "query", "title"]
QREL_DOC_ID_FIELDS = ["doc_id", "docid", "document_id"]
QREL_RELEVANCE_FIELDS = ["relevance", "score", "rel"]


def _as_value(value: Any) -> str:
	"""Return enum.value if value is an Enum-like object, otherwise str(value)."""
	return str(getattr(value, "value", value))


def _parse_jsonl_file(path: str) -> List[Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as fh:
		lines = []
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				lines.append(json.loads(line))
			except json.JSONDecodeError:
				continue
		return lines


def _extract_query_item(raw: Dict[str, Any]) -> Optional[Tuple[str, str]]:
	query_id = None
	for field in QUERY_ID_FIELDS:
		if field in raw:
			query_id = raw[field]
			break
	if query_id is None:
		return None

	query_text = None
	for field in QUERY_TEXT_FIELDS:
		if field in raw:
			query_text = raw[field]
			break
	if query_text is None:
		return None

	return str(query_id), str(query_text)


def _extract_qrel_item(raw: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
	query_id = None
	for field in QUERY_ID_FIELDS:
		if field in raw:
			query_id = raw[field]
			break
	if query_id is None:
		return None

	doc_id = None
	for field in QREL_DOC_ID_FIELDS:
		if field in raw:
			doc_id = raw[field]
			break
	if doc_id is None:
		return None

	relevance = None
	for field in QREL_RELEVANCE_FIELDS:
		if field in raw:
			relevance = raw[field]
			break
	if relevance is None:
		return None

	try:
		relevance_value = float(relevance)
	except (TypeError, ValueError):
		relevance_value = 0.0

	return str(query_id), str(doc_id), relevance_value


def _load_queries(dataset_name: str) -> List[Dict[str, str]]:
	dataset_value = _as_value(dataset_name)
	path = os.path.join(DATASETS_DIR, dataset_value, "queries.jsonl")
	if not os.path.exists(path):
		raise FileNotFoundError(f"Queries file not found for dataset '{dataset_value}'")

	queries = []
	for raw in _parse_jsonl_file(path):
		item = _extract_query_item(raw)
		if item is None:
			continue
		query_id, query_text = item
		queries.append({"query_id": query_id, "query": query_text})
	return queries


def _load_qrels(dataset_name: str) -> Dict[str, Dict[str, float]]:
	dataset_value = _as_value(dataset_name)
	path = os.path.join(DATASETS_DIR, dataset_value, "qrels.jsonl")
	if not os.path.exists(path):
		raise FileNotFoundError(f"Qrels file not found for dataset '{dataset_value}'")

	qrels_by_query: Dict[str, Dict[str, float]] = {}
	for raw in _parse_jsonl_file(path):
		item = _extract_qrel_item(raw)
		if item is None:
			continue
		query_id, doc_id, relevance = item
		qrels_by_query.setdefault(query_id, {})[doc_id] = relevance
	return qrels_by_query


async def evaluate_dataset(
	dataset_name: str,
	model: RetrievalModel,
	top_k: int,
	max_queries: int,
	bm25_k1: float,
	bm25_b: float,
	apply_refinement: bool,
) -> Dict[str, Any]:
	dataset_value = _as_value(dataset_name)
	model_value = _as_value(model)

	queries = _load_queries(dataset_value)
	qrels_by_query = _load_qrels(dataset_value)

	valid_queries = []
	for query in queries:
		if query["query_id"] in qrels_by_query:
			valid_queries.append(query)
			if len(valid_queries) >= max_queries:
				break

	if not valid_queries:
		raise ValueError(
			f"No valid queries with qrels found for dataset '{dataset_value}'. "
			"Ensure the dataset has both queries and qrels files."
		)

	per_query = []
	retrieved_lists: List[List[str]] = []
	relevant_sets: List[List[str]] = []

	async with httpx.AsyncClient(timeout=120.0) as client:
		for query in valid_queries:
			payload = {
				"query": query["query"],
				"dataset": dataset_value,
				"model": model_value,
				"top_k": top_k,
				"bm25_k1": bm25_k1,
				"bm25_b": bm25_b,
				"apply_refinement": apply_refinement,
			}

			response = await client.post(f"{RETRIEVAL_URL}/search", json=payload)
			response.raise_for_status()
			search_data = response.json()

			retrieved_doc_ids = [
				str(item.get("doc_id"))
				for item in search_data.get("results", [])
				if isinstance(item, dict) and item.get("doc_id") is not None
			]

			qrels = qrels_by_query[query["query_id"]]
			relevant_doc_ids = [doc_id for doc_id, score in qrels.items() if score > 0]

			precision = metrics.precision_at_k(retrieved_doc_ids, relevant_doc_ids, k=top_k)
			recall = metrics.recall_at_k(retrieved_doc_ids, relevant_doc_ids, k=top_k)
			average_precision = metrics.average_precision_at_k(
				retrieved_doc_ids,
				relevant_doc_ids,
				k=top_k,
			)
			ndcg = metrics.ndcg_at_k(retrieved_doc_ids, qrels, k=top_k)

			per_query.append({
				"query_id": query["query_id"],
				"query": query["query"],
				"retrieved_doc_ids": retrieved_doc_ids,
				"num_relevant": len(relevant_doc_ids),
				"precision_at_k": precision,
				"recall_at_k": recall,
				"average_precision_at_k": average_precision,
				"ndcg_at_k": ndcg,
			})

			retrieved_lists.append(retrieved_doc_ids)
			relevant_sets.append(relevant_doc_ids)

	num_queries = len(per_query)
	map_score = metrics.mean_average_precision(retrieved_lists, relevant_sets, k=top_k)
	mean_precision = sum(item["precision_at_k"] for item in per_query) / num_queries
	mean_recall = sum(item["recall_at_k"] for item in per_query) / num_queries
	mean_ndcg = sum(item["ndcg_at_k"] for item in per_query) / num_queries
	notes = (
		"Real evaluation uses qrels from the selected dataset and a fixed retrieval model. "
		"The old trec-covid max_docs=10000 setup was only for local testing; final evaluation "
		"should use the full quora dataset or another complete manageable dataset with qrels."
	)
	result_summary = [{
		"dataset": dataset_value,
		"model": model_value,
		"top_k": top_k,
		"max_queries": max_queries,
		"MAP": map_score,
		"Precision@K": mean_precision,
		"Recall@K": mean_recall,
		"nDCG@K": mean_ndcg,
		"notes": notes,
	}]

	return {
		"dataset_name": dataset_value,
		"model": model_value,
		"top_k": top_k,
		"max_queries": max_queries,
		"evaluated_queries": num_queries,
		"metrics": {
			"MAP": map_score,
			"mean_precision_at_k": mean_precision,
			"mean_recall_at_k": mean_recall,
			"mean_ndcg_at_k": mean_ndcg,
		},
		"result_summary": result_summary,
		"per_query": per_query,
		"notes": notes,
	}


def evaluate_demo() -> Dict[str, Any]:
	"""
	Run the demo evaluation on a small in-memory example and return results.

	Sample data (as requested in the task):
		retrieved = ["d5", "d3", "d1", "d8", "d2"]
		relevant = {"d5", "d1", "d2"}
		qrels = {"d5":3, "d3":1, "d1":2, "d8":0, "d2":3}
		k = 5
	"""
	retrieved = ["d5", "d3", "d1", "d8", "d2"]
	relevant = {"d5", "d1", "d2"}
	qrels = {"d5": 3, "d3": 1, "d1": 2, "d8": 0, "d2": 3}
	k = 5

	precision = metrics.precision_at_k(retrieved, relevant, k=k)
	recall = metrics.recall_at_k(retrieved, relevant, k=k)
	ap = metrics.average_precision_at_k(retrieved, relevant, k=k)
	ndcg = metrics.ndcg_at_k(retrieved, qrels, k=k)

	return {
		"precision_at_k": precision,
		"recall_at_k": recall,
		"average_precision_at_k": ap,
		"ndcg_at_k": ndcg,
		"retrieved": retrieved,
		"relevant": sorted(list(relevant)),
		"qrels": qrels,
		"k": k,
	}
