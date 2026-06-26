"""Rule-based query router for the Gateway Service."""

import re


ENGLISH_QUESTION_WORDS = {
	"what",
	"how",
	"why",
	"explain",
	"describe",
	"when",
	"where",
}

ARABIC_QUESTION_WORDS = {
	"ما",
	"ماذا",
	"كيف",
	"لماذا",
	"اشرح",
	"متى",
	"أين",
	"اين",
}


def _tokenize(query: str) -> list[str]:
	"""Return simple word tokens for English and Arabic routing rules."""
	return re.findall(r"[\w\u0600-\u06FF]+", query.lower(), flags=re.UNICODE)


def choose_retrieval_strategy(query: str) -> dict:
	"""Choose a retrieval model from a query using deterministic rules."""
	clean_query = (query or "").strip()
	if not clean_query:
		return {
			"strategy": "bm25",
			"reason": "Empty query; defaulting to BM25 as the safest lexical fallback.",
		}

	tokens = _tokenize(clean_query)
	token_set = set(tokens)

	if token_set.intersection(ENGLISH_QUESTION_WORDS):
		return {
			"strategy": "embedding",
			"reason": "Natural language question detected; embedding search is preferred for semantic matching.",
		}

	if token_set.intersection(ARABIC_QUESTION_WORDS):
		return {
			"strategy": "embedding",
			"reason": "Arabic question wording detected; embedding search is preferred for semantic matching.",
		}

	if len(tokens) > 8:
		return {
			"strategy": "embedding",
			"reason": "Long natural language query; embedding search is preferred for semantic matching.",
		}

	if 1 <= len(tokens) <= 3:
		return {
			"strategy": "bm25",
			"reason": "Short keyword query; BM25 is preferred for exact lexical matching.",
		}

	return {
		"strategy": "hybrid_parallel",
		"reason": "Medium-length query; hybrid parallel retrieval balances lexical and semantic matching.",
	}
