"""
services/retrieval/tfidf_retriever.py
======================================
محرك البحث بطريقة TF-IDF + Cosine Similarity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
كيف يتكامل هذا الملف مع عمل المطور الأول؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
المطور الأول بنى TFIDFIndexer الذي يحفظ على القرص:
    data/indexes/{dataset}/tfidf/
        tfidf_vectorizer.pkl   ← TfidfVectorizer مُدرَّب
        tfidf_matrix.npz       ← مصفوفة (N_docs × vocab)
        tfidf_documents.json   ← قائمة IndexedDocument
        tfidf_metadata.json    ← إعدادات الفهرس
        tfidf_docid_map.json   ← doc_id → row_index

نحن نستخدم get_tfidf_indexer() من المطور الأول مباشرة.
لا نُعيد بناء الفهرس — فقط نحمّله ونبحث فيه.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
خطوات البحث (من المحاضرة الثالثة — VSM):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. يأتي الاستعلام محمولاً كـ tokens من Preprocessing Service
   مثال: ["run", "dog", "jump"]

2. نجمع tokens في جملة: "run dog jump"

3. نستخدم vectorizer.transform() لتحويل الجملة إلى متجه TF-IDF
   النتيجة: sparse matrix شكلها (1, vocab_size)

4. نحسب Cosine Similarity بين متجه الاستعلام وكل الوثائق:
   cosine_similarity(query_vec, tfidf_matrix)
   النتيجة: array شكلها (N_docs,) — درجة لكل وثيقة

5. نرتب تنازلياً ونأخذ أفضل top_k وثيقة
"""

import sys
import os
import logging
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
from services.indexing.tfidf_indexer import get_tfidf_indexer, TFIDFIndexer

logger = logging.getLogger(__name__)


class TFIDFRetriever:
    """
    محرك البحث بطريقة TF-IDF.

    يعتمد كلياً على TFIDFIndexer الذي بناه المطور الأول.
    مسؤوليتنا فقط: تحميل الفهرس + تنفيذ البحث.
    """

    def __init__(self, dataset: DatasetName) -> None:
        self.dataset = dataset
        # DocumentStore من المطور الأول — يُوفّر النص الكامل بسرعة O(log N)
        # يُحمَّل عند أول استخدام (Lazy) لأنه قد لا يكون جاهزاً بعد
        self._store = None
        # نستخدم Singleton من المطور الأول — يحمّل الفهرس إذا كان موجوداً
        self._indexer: TFIDFIndexer = get_tfidf_indexer(dataset.value)

    @property
    def is_loaded(self) -> bool:
        """هل الفهرس محمّل وجاهز للبحث؟"""
        return self._indexer.is_built()

    def search(
        self,
        query_tokens: List[str],
        top_k: int = 10,
    ) -> List[DocumentResult]:
        """
        يبحث في الفهرس ويُرجع أفضل top_k وثيقة.

        المعاملات:
            query_tokens: قائمة tokens معالجة من Preprocessing Service
                          مثال: ["cloud", "storag", "sync"]
            top_k: عدد النتائج

        الإرجاع:
            List[DocumentResult] مرتبة تنازلياً حسب درجة التشابه
        """
        if not self.is_loaded:
            logger.warning(f"[TFIDFRetriever] فهرس {self.dataset.value} غير محمّل")
            return []

        if not query_tokens:
            return []

        # الخطوة 1: دمج tokens في جملة
        # المطور الأول استخدم " ".join(tokens) في الفهرسة، نفس الشيء هنا
        processed_query = " ".join(query_tokens)

        # الخطوة 2: تحويل الاستعلام إلى متجه TF-IDF
        # transform_query() من المطور الأول تُرجع sparse matrix (1, vocab_size)
        query_vec = self._indexer.transform_query(processed_query)
        if query_vec is None:
            return []

        # الخطوة 3: Cosine Similarity مع كل الوثائق
        # tfidf_matrix شكله (N_docs, vocab_size)
        # النتيجة scores شكلها (N_docs,)
        # flatten() يحول النتيجة من مصفوفة ثنائية الأبعاد إلى مصفوفة أحادية الأبعاد (قائمة أرقام بسيطة) تحتوي على "درجة التشابه" لكل وثيقة.
        scores = cosine_similarity(query_vec, self._indexer.tfidf_matrix).flatten()

        # الخطوة 4: أفضل top_k وثيقة
        # argsort تُرجع indices مرتبة تصاعدياً → نعكسها
        top_indices = np.argsort(scores)[::-1][:top_k]

        # الخطوة 5: بناء النتائج
        results: List[DocumentResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                break

            doc = self._indexer.get_document_by_index(int(idx))
            if doc is None:
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

        لماذا هذا مهم؟
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        IndexedDocument.original_text قد يكون مقطوعاً أو ناقصاً
        لأن بعض الفهارس لا تحفظ النص كاملاً لتوفير الذاكرة.

        DocumentStore (المطور الأول) يحفظ النص الأصلي الكامل
        في SQLite بسرعة O(log N) للوصول بالـ doc_id.

        إذا DocumentStore غير متاح (لم يُدمج بعد):
            نستخدم fallback_text من IndexedDocument مباشرة.
            النظام يعمل — فقط النص قد يكون مختصراً.
        """
        if not _DOCUMENT_STORE_AVAILABLE or get_document_store is None:
            return fallback_text

        try:
            # تحميل الـ store عند أول استخدام (Lazy Loading)
            if self._store is None:
                self._store = get_document_store(self.dataset.value)
            stored = self._store.get(doc_id)
            if stored and stored.get("raw_text"):
                return stored["raw_text"]
        except Exception:
            pass  # أي خطأ → نستخدم الـ fallback

        return fallback_text

    def get_stats(self) -> dict:
        """إحصائيات الفهرس للـ /health endpoint."""
        if not self.is_loaded:
            return {"loaded": False, "dataset": self.dataset.value}
        return {
            "loaded": True,
            "dataset": self.dataset.value,
            "num_documents": len(self._indexer.documents),
            "vocabulary_size": (
                self._indexer.tfidf_matrix.shape[1]
                if self._indexer.tfidf_matrix is not None
                else 0
            ),
        }


# =============================================================
# Singleton — نسخة واحدة لكل dataset لتجنب إعادة التحميل
# =============================================================

_tfidf_retrievers: dict = {}


def get_tfidf_retriever(dataset: DatasetName) -> TFIDFRetriever:
    """يُرجع TFIDFRetriever لـ dataset معين (Singleton)."""
    key = dataset.value
    if key not in _tfidf_retrievers:
        _tfidf_retrievers[key] = TFIDFRetriever(dataset)
    return _tfidf_retrievers[key]
