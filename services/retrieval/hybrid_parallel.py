"""
services/retrieval/hybrid_parallel.py
=======================================
التمثيل الهجين المتوازي (Parallel Hybrid) بخوارزمية RRF.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
من المحاضرة الثالثة — Hybrid Systems:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"عند استخدام التمثيل الهجين بطريقة متوازية (Parallel)،
يجب استخدام طرق دمج النتائج (Fusion Methods) لاحتساب
الدرجات النهائية للوثائق."

التدفق:
    الاستعلام
        │
    ┌───┼───┐
    ↓   ↓   ↓
  BM25 TFIDF Embedding  ← تعمل كلها معاً
    │   │   │
    └───┼───┘
        ↓
    RRF Fusion           ← ندمج نتائجها
        ↓
    النتائج النهائية

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما هو RRF؟ (Reciprocal Rank Fusion)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
خوارزمية تدمج قوائم مرتبة من مصادر مختلفة.

المعادلة:
    RRF_score(d) = Σ  1 / (k + rank_i(d))
                  i

حيث:
    d      = الوثيقة
    rank_i = ترتيب الوثيقة في النظام i (1 = الأول)
    k      = 60 (ثابت يمنع التركيز الزائد على الأوائل)

مثال عملي:
    وثيقة "A":
        BM25:      rank=1 → 1/(60+1) = 0.01639
        TF-IDF:    rank=3 → 1/(60+3) = 0.01587
        Embedding: rank=2 → 1/(60+2) = 0.01613
        المجموع: 0.04839  ← تفوز

    وثيقة "B":
        BM25:      rank=2 → 0.01613
        TF-IDF:    غائبة → 0
        Embedding: rank=5 → 0.01538
        المجموع: 0.03151  ← تخسر

لماذا RRF يتفوق على المتوسط البسيط للدرجات؟
    درجة BM25 قد تكون 15.3 ودرجة Embedding قد تكون 0.85
    لا يمكن جمعهما مباشرة — الوحدات مختلفة!
    لكن الترتيب "الأول" يعني نفس الشيء في كلا النظامين.
"""

import sys
import os
import logging
from typing import List, Dict
from collections import defaultdict

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from shared.models import DocumentResult, DatasetName
from shared.constants import BM25_DEFAULT_K1, BM25_DEFAULT_B

logger = logging.getLogger(__name__)

# الثابت الموصى به في أبحاث IR
RRF_K = 60


class HybridParallelRetriever:
    """
    دمج موازي لعدة محركات بحث باستخدام RRF.

    يستقبل مراجع للـ retrievers من app.py (Dependency Injection)
    لتجنب إعادة تحميل الفهارس.
    """

    def __init__(
        self,
        tfidf_retriever=None,
        bm25_retriever=None,
        embedding_retriever=None,
    ) -> None:
        self.tfidf = tfidf_retriever
        self.bm25 = bm25_retriever
        self.embedding = embedding_retriever

    def search(
        self,
        query_tokens: List[str],
        query_text: str,
        top_k: int = 10,
        bm25_k1: float = BM25_DEFAULT_K1,
        bm25_b: float = BM25_DEFAULT_B,
        use_tfidf: bool = True,
        use_bm25: bool = True,
        use_embedding: bool = True,
    ) -> List[DocumentResult]:
        """
        يجري البحث الهجين المتوازي.

        المعاملات:
            query_tokens: tokens المعالجة (لـ TF-IDF و BM25)
            query_text:   النص الأصلي (لـ Embedding)
            top_k:        عدد النتائج النهائية
            bm25_k1, bm25_b: معاملات BM25
            use_*:        تشغيل/إيقاف كل محرك

        الخوارزمية:
            1. نشغّل كل محرك مُفعَّل ونجمع نتائجه
            2. نطبق RRF على المجموع
            3. نُرجع أفضل top_k
        """
        # نجمع نتائج كل محرك (كل قائمة = نتائج محرك واحد)
        all_lists: List[List[DocumentResult]] = []

        # نطلب أكثر من top_k من كل محرك لضمان تغطية كافية
        candidates_k = top_k * 3

        if use_tfidf and self.tfidf and self.tfidf.is_loaded:
            res = self.tfidf.search(query_tokens, top_k=candidates_k)
            if res:
                all_lists.append(res)
                logger.debug(f"[HybridParallel] TF-IDF: {len(res)} نتيجة")

        if use_bm25 and self.bm25 and self.bm25.is_loaded:
            res = self.bm25.search(
                query_tokens, top_k=candidates_k, k1=bm25_k1, b=bm25_b
            )
            if res:
                all_lists.append(res)
                logger.debug(f"[HybridParallel] BM25: {len(res)} نتيجة")

        if use_embedding and self.embedding and self.embedding.is_loaded:
            res = self.embedding.search(query_text, top_k=candidates_k)
            if res:
                all_lists.append(res)
                logger.debug(f"[HybridParallel] Embedding: {len(res)} نتيجة")

        if not all_lists:
            return []

        return self._rrf_fusion(all_lists, top_k)

    def _rrf_fusion(
        self,
        ranked_lists: List[List[DocumentResult]],
        top_k: int,
    ) -> List[DocumentResult]:
        """
        يُطبّق Reciprocal Rank Fusion.

        لكل وثيقة في كل قائمة:
            rrf_score[doc_id] += 1 / (RRF_K + rank)

        ثم نرتب تنازلياً ونأخذ أفضل top_k.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_store: Dict[str, DocumentResult] = {}

        for ranked_list in ranked_lists:
            for result in ranked_list:
                doc_id = result.doc_id
                # rank يبدأ من 1 في DocumentResult
                rrf_scores[doc_id] += 1.0 / (RRF_K + result.rank)

                # نحفظ بيانات الوثيقة (أول مرة تظهر)
                if doc_id not in doc_store:
                    doc_store[doc_id] = result

        # ترتيب تنازلي حسب درجة RRF
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        results: List[DocumentResult] = []
        for final_rank, (doc_id, rrf_score) in enumerate(sorted_docs, start=1):
            original = doc_store[doc_id]
            results.append(
                DocumentResult(
                    doc_id=doc_id,
                    title=original.title,
                    text=original.text,
                    score=round(rrf_score, 6),
                    rank=final_rank,
                )
            )

        return results
