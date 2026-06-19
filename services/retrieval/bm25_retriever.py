"""
services/retrieval/bm25_retriever.py
======================================
محرك البحث بطريقة BM25 (Best Match 25).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
كيف يتكامل مع المطور الأول؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
المطور الأول يحفظ:
    data/indexes/{dataset}/bm25/
        bm25_model.pkl      ← BM25Okapi object كامل
        bm25_tokens.pkl     ← tokens لكل وثيقة
        bm25_documents.json ← قائمة IndexedDocument
        bm25_metadata.json  ← k1, b, avgdl...
        bm25_docid_map.json ← doc_id → index

نستخدم get_bm25_indexer() من المطور الأول.
هو يُرجع BM25Indexer مع self.bm25 جاهزاً للاستخدام.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM25 مقابل TF-IDF (من المحاضرة الثالثة):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TF-IDF: وزن كلمة = TF × IDF (يزداد خطياً مع التكرار)
BM25:   وزن كلمة = IDF × [TF×(k1+1)] / [TF + k1×(1−b+b×|d|/avgdl)]

المعاملات التي يجب توفيرها للمستخدم حسب المتطلبات:
    k1 = 1.5 (افتراضي) — يتحكم في سقف تأثير التكرار
    b  = 0.75 (افتراضي) — يتحكم في تطبيع الطول
"""

import sys
import os
import logging
from typing import List

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from shared.models import DocumentResult, DatasetName

# نحاول استيراد DocumentStore من المطور الأول
# إذا لم يكن متاحاً بعد (لم يُدمج) نستمر بدونه
try:
    from services.indexing.document_store import get_document_store

    _DOCUMENT_STORE_AVAILABLE = True
except ImportError:
    _DOCUMENT_STORE_AVAILABLE = False
    get_document_store = None
from shared.constants import BM25_DEFAULT_K1, BM25_DEFAULT_B
from services.indexing.bm25_indexer import get_bm25_indexer, BM25Indexer

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    محرك البحث بطريقة BM25.

    يستخدم BM25Indexer من المطور الأول الذي يوفر:
        - self.bm25: BM25Okapi جاهز مع get_scores()
        - self.documents: قائمة IndexedDocument
        - get_top_n(): دالة جاهزة تُرجع (doc, score)
        - metadata: يحتوي k1, b المُستخدمَين في البناء
    """

    def __init__(self, dataset: DatasetName) -> None:
        self.dataset = dataset
        # DocumentStore من المطور الأول — يُوفّر النص الكامل بسرعة O(log N)
        # يُحمَّل عند أول استخدام (Lazy) لأنه قد لا يكون جاهزاً بعد
        self._store = None
        self._indexer: BM25Indexer = get_bm25_indexer(dataset.value)

    @property
    def is_loaded(self) -> bool:
        return self._indexer.is_built()

    def search(
        self,
        query_tokens: List[str],
        top_k: int = 10,
        k1: float = BM25_DEFAULT_K1,
        b: float = BM25_DEFAULT_B,
    ) -> List[DocumentResult]:
        """
        يبحث في الفهرس بطريقة BM25.

        المعاملات:
            query_tokens: tokens المعالجة (نفس الـ stemming المُستخدم في الفهرسة)
            top_k: عدد النتائج
            k1: معامل إشباع التكرار (يُسمح للمستخدم بتغييره من الواجهة)
            b:  معامل تطبيع الطول  (يُسمح للمستخدم بتغييره من الواجهة)

        ملاحظة مهمة حول k1 و b:
            BM25Okapi من مكتبة rank_bm25 يُحسب IDF عند البناء.
            لتغيير k1/b، المطور الأول وفّر دالة get_scores() التي
            تستخدم المعاملات المُستخدمة عند بناء الفهرس.

            إذا أراد المستخدم قيماً مختلفة، نستخدم bm25_indexer.get_scores()
            مع BM25Okapi جديد مؤقت. هذا سلوك مقبول حسب المتطلبات.
        """
        if not self.is_loaded:
            logger.warning(f"[BM25Retriever] فهرس {self.dataset.value} غير محمّل")
            return []

        if not query_tokens:
            return []

        # هل المستخدم طلب نفس قيم k1/b المُستخدمة في البناء؟
        built_k1 = getattr(self._indexer.metadata, "k1", BM25_DEFAULT_K1)
        built_b = getattr(self._indexer.metadata, "b", BM25_DEFAULT_B)

        if abs(k1 - built_k1) < 1e-6 and abs(b - built_b) < 1e-6:
            # القيم متطابقة — نستخدم get_top_n() الجاهزة مباشرة
            raw_results = self._indexer.get_top_n(query_tokens, n=top_k)
        else:
            # قيم مختلفة — نبني BM25 مؤقت بالقيم الجديدة
            logger.info(f"[BM25Retriever] k1/b مختلفان عن البناء، إعادة الحساب")
            from rank_bm25 import BM25Okapi
            import numpy as np

            bm25_temp = BM25Okapi(self._indexer.tokenized_docs, k1=k1, b=b)
            scores = bm25_temp.get_scores(query_tokens)
            top_idx = np.argsort(scores)[::-1][:top_k]
            raw_results = [
                (self._indexer.get_document_by_index(int(i)), float(scores[i]))
                for i in top_idx
                if scores[i] > 0 and self._indexer.get_document_by_index(int(i))
            ]

        # تحويل (IndexedDocument, float) → DocumentResult
        results: List[DocumentResult] = []
        for rank, (doc, score) in enumerate(raw_results, start=1):
            if doc is None or score <= 0.0:
                continue
            results.append(
                DocumentResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=self._get_full_text(doc.doc_id, doc.original_text),
                    score=score,
                    rank=rank,
                )
            )

        return results

    def _get_full_text(self, doc_id: str, fallback_text: str) -> str:
        """
        يجلب النص الكامل من DocumentStore إذا كان متاحاً.
        إذا لم يكن متاحاً → يُرجع النص الاحتياطي من IndexedDocument.
        """
        if not _DOCUMENT_STORE_AVAILABLE or get_document_store is None:
            return fallback_text
        try:
            if self._store is None:
                self._store = get_document_store(self.dataset.value)
            stored = self._store.get(doc_id)
            if stored and stored.get("raw_text"):
                return stored["raw_text"]
        except Exception:
            pass
        return fallback_text

    def get_stats(self) -> dict:
        if not self.is_loaded:
            return {"loaded": False, "dataset": self.dataset.value}
        meta = self._indexer.metadata
        return {
            "loaded": True,
            "dataset": self.dataset.value,
            "num_documents": meta.num_documents if meta else 0,
            "k1": meta.k1 if meta else BM25_DEFAULT_K1,
            "b": meta.b if meta else BM25_DEFAULT_B,
            "avg_doc_length": meta.avg_document_length if meta else 0,
        }


# =============================================================
# Singleton
# =============================================================

_bm25_retrievers: dict = {}


def get_bm25_retriever(dataset: DatasetName) -> BM25Retriever:
    """يُرجع BM25Retriever لـ dataset معين (Singleton)."""
    key = dataset.value
    if key not in _bm25_retrievers:
        _bm25_retrievers[key] = BM25Retriever(dataset)
    return _bm25_retrievers[key]
