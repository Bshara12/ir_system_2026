"""
services/retrieval/embedding_retriever.py
==========================================
محرك البحث الدلالي باستخدام Sentence Embeddings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
كيف يتكامل مع المطور الأول؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
المطور الأول يحفظ:
    data/indexes/{dataset}/embedding/
        embedding_index.faiss     ← FAISS index
        embedding_vectors.npy     ← المتجهات الأصلية
        embedding_documents.json  ← قائمة IndexedDocument
        embedding_metadata.json   ← model_name, dim, normalize...
        embedding_docid_map.json  ← doc_id → index

المطور الأول يوفر هذه الدوال الجاهزة:
    encode_query(query_text) → np.ndarray شكله (1, dim)
    get_top_k(query_embedding, k) → List[(IndexedDocument, float)]

نستخدمهما مباشرة — لا نحتاج FAISS أو SentenceTransformer بأنفسنا.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا نمرر النص الأصلي (غير المعالج)؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
موديلات Sentence Transformers مدرّبة على جمل طبيعية كاملة.
"running dogs in the park" → embedding جيد
"run dog park" (بعد stemming) → embedding أقل جودة

لذلك: TF-IDF/BM25 تستخدم tokens المعالجة
       Embedding تستخدم النص الأصلي مباشرة
"""

import sys
import os
import logging
from typing import List

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from shared.models import DocumentResult, DatasetName
from services.indexing.embedding_indexer import get_embedding_indexer, EmbeddingIndexer

logger = logging.getLogger(__name__)


class EmbeddingRetriever:
    """
    محرك البحث الدلالي.

    يستخدم EmbeddingIndexer من المطور الأول الذي يوفر:
        - encode_query(text) → متجه الاستعلام
        - get_top_k(query_emb, k) → [(IndexedDocument, score)]
        - faiss_index: جاهز للبحث بدقة عالية
        - metadata: model_name, embedding_dim
    """

    def __init__(self, dataset: DatasetName) -> None:
        self.dataset = dataset
        self._indexer: EmbeddingIndexer = get_embedding_indexer(dataset.value)

    @property
    def is_loaded(self) -> bool:
        return self._indexer.is_built()

    def search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[DocumentResult]:
        """
        يبحث دلالياً باستخدام النص الأصلي (غير المعالج).

        المعاملات:
            query_text: النص الأصلي كما كتبه المستخدم
                        مثال: "running dogs in the park"
            top_k: عدد النتائج

        الخطوات:
            1. encode_query() → يحوّل النص لمتجه embedding
            2. get_top_k()   → يبحث في FAISS ويُرجع أقرب الوثائق
        """
        if not self.is_loaded:
            logger.warning(f"[EmbeddingRetriever] فهرس {self.dataset.value} غير محمّل")
            return []

        if not query_text.strip():
            return []

        # الخطوة 1: تحويل الاستعلام إلى embedding
        # encode_query() من المطور الأول تُرجع np.ndarray (1, dim) أو None
        query_embedding = self._indexer.encode_query(query_text)
        if query_embedding is None:
            logger.warning("[EmbeddingRetriever] encode_query أرجع None")
            return []

        # الخطوة 2: البحث في FAISS
        # get_top_k() من المطور الأول تُرجع List[(IndexedDocument, float)]
        raw_results = self._indexer.get_top_k(query_embedding, k=top_k)

        # تحويل النتائج إلى DocumentResult
        results: List[DocumentResult] = []
        for rank, (doc, score) in enumerate(raw_results, start=1):
            if doc is None:
                continue
            results.append(
                DocumentResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.original_text,
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def get_stats(self) -> dict:
        if not self.is_loaded:
            return {"loaded": False, "dataset": self.dataset.value}
        meta = self._indexer.metadata
        return {
            "loaded": True,
            "dataset": self.dataset.value,
            "num_documents": meta.num_documents if meta else 0,
            "model_name": meta.model_name if meta else "unknown",
            "embedding_dim": meta.embedding_dim if meta else 0,
        }


# =============================================================
# Singleton
# =============================================================

_embedding_retrievers: dict = {}


def get_embedding_retriever(dataset: DatasetName) -> EmbeddingRetriever:
    """يُرجع EmbeddingRetriever لـ dataset معين (Singleton)."""
    key = dataset.value
    if key not in _embedding_retrievers:
        _embedding_retrievers[key] = EmbeddingRetriever(dataset)
    return _embedding_retrievers[key]
