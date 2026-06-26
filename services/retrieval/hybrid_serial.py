"""
services/retrieval/hybrid_serial.py
=====================================
التمثيل الهجين التسلسلي (Serial Hybrid) — تصفية ثم إعادة ترتيب.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
من المحاضرة الثالثة — Hybrid Systems:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"يجب تطبيق التمثيل الهجين مرتين: تفرعي (Parallel) مرة
ومرة تسلسلي (Serial) مع توفير خيار بواجهة المستخدم."

الـ Serial يعمل كـ pipeline:
    مرحلة 1 → تصفية سريعة (BM25: سريع، يُرجع 100 مرشح)
    مرحلة 2 → إعادة ترتيب دقيقة (Embedding: أبطأ لكن أدق)

لماذا هذا النهج أذكى؟
    Embedding على 200,000 وثيقة = بطيء جداً
    BM25 يُصفّي لـ 100 مرشح، ثم Embedding على 100 فقط = سريع وأدق!

هذا بالضبط ما تفعله أنظمة البحث الحديثة.

التدفق:
    الاستعلام
        │
    BM25 (المرحلة 1)
    يُرجع 100 مرشح
        │
    Embedding Re-ranker (المرحلة 2)
    يُعيد ترتيب الـ 100 ويُرجع أفضل 10
        │
    النتائج النهائية (دقة عالية + سرعة معقولة)
"""

import sys
import os
import logging
from typing import List, Optional

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from shared.models import DocumentResult, DatasetName
from shared.constants import BM25_DEFAULT_K1, BM25_DEFAULT_B

logger = logging.getLogger(__name__)

# عدد المرشحين من المرحلة الأولى
FIRST_STAGE_K = 100


class HybridSerialRetriever:
    """
    دمج تسلسلي: BM25 (تصفية) → Embedding (إعادة ترتيب).

    لماذا BM25 في المرحلة الأولى؟
    BM25 أسرع بكثير من Embedding ويُرجع مرشحين جيدين.
    Embedding يأخذ هؤلاء المرشحين ويُرتبهم بدقة أعلى.
    """

    def __init__(
        self,
        first_stage_retriever=None,  # BM25Retriever أو TFIDFRetriever
        second_stage_retriever=None,  # EmbeddingRetriever
        first_stage_k: int = FIRST_STAGE_K,
    ) -> None:
        self.first_stage = first_stage_retriever
        self.second_stage = second_stage_retriever
        self.first_stage_k = first_stage_k

    def search(
        self,
        query_tokens: List[str],
        query_text: str,
        top_k: int = 10,
        bm25_k1: float = BM25_DEFAULT_K1,
        bm25_b: float = BM25_DEFAULT_B,
    ) -> List[DocumentResult]:
        """
        يجري البحث التسلسلي بمرحلتين.

        المعاملات:
            query_tokens: tokens المعالجة (للمرحلة الأولى BM25/TF-IDF)
            query_text:   النص الأصلي (للمرحلة الثانية Embedding)
            top_k:        عدد النتائج النهائية
            bm25_k1, bm25_b: معاملات BM25
        """
        if self.first_stage is None:
            logger.error("[HybridSerial] لا يوجد محرك للمرحلة الأولى")
            return []

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # المرحلة الأولى: تصفية سريعة
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        class_name = type(self.first_stage).__name__

        if "BM25" in class_name:
            candidates = self.first_stage.search(
                query_tokens,
                top_k=self.first_stage_k,
                k1=bm25_k1,
                b=bm25_b,
            )
        else:
            # TF-IDF أو أي محرك آخر
            candidates = self.first_stage.search(query_tokens, top_k=self.first_stage_k)

        if not candidates:
            logger.warning("[HybridSerial] المرحلة الأولى لم تُرجع نتائج")
            return []

        logger.debug(f"[HybridSerial] المرحلة الأولى: {len(candidates)} مرشح")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # المرحلة الثانية: إعادة ترتيب بـ Embedding
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if (
            self.second_stage is None
            or not self.second_stage.is_loaded
            or not query_text.strip()
        ):
            # لا يوجد محرك ثانٍ أو نص فارغ → نُرجع المرحلة الأولى مباشرة
            logger.warning("[HybridSerial] لا يوجد محرك ثانٍ، نُرجع المرحلة الأولى")
            return candidates[:top_k]

        reranked = self._rerank_with_embedding(
            candidates=candidates,
            query_text=query_text,
            top_k=top_k,
        )

        return reranked

    def _rerank_with_embedding(
        self,
        candidates: List[DocumentResult],
        query_text: str,
        top_k: int,
    ) -> List[DocumentResult]:
        """
        يُعيد ترتيب المرشحين باستخدام Embedding.

        كيف يعمل؟
        ━━━━━━━━━━
        1. نُحوّل الاستعلام إلى embedding (باستخدام موديل المطور الأول)
        2. نُحوّل نصوص المرشحين إلى embeddings (batch)
        3. نحسب Cosine Similarity
        4. نُرجع أفضل top_k مُرتّبة

        لماذا نحسب embeddings المرشحين هنا وليس من الفهرس؟
        لأن هدفنا إعادة ترتيب مجموعة محددة (100 مرشح)،
        وليس البحث في FAISS كله. الحساب المباشر أبسط وأسرع.
        """
        import numpy as np

        indexer = self.second_stage._indexer
        model = indexer._get_model()

        if model is None:
            logger.warning("[HybridSerial] موديل Embedding غير متاح")
            return candidates[:top_k]

        # تحويل الاستعلام
        query_emb = model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False,
            # تفعيل normalize_embeddings=True يقوم بجعل طول المتجه في الفضاء الرياضي يساوي 1 (Unit Vector). هذا يسهل حساب الشبه اللغوي لاحقاً بعملية ضرب بسيطة جداً.
            normalize_embeddings=True,
        ).astype(
            np.float32
        )  # shape: (1, dim)

        # تحويل نصوص المرشحين دفعةً واحدة (batch أسرع من واحد واحد)
        # نقوم باستخراج النصوص الكاملة للـ 100 وثيقة مرشحة، ونمررها للموديل ليقوم بعملية الـ Encoding لها كدُفعة واحدة (Batch Processing). معالجة المصفوفات دفعة واحدة تستغل ميزات المعالجة المتوازية في المعالج (أو كارت الشاشة GPU إن وجد) وتكون أسرع بعشرات المرات من عمل حلقة تكرارية وتمريرها وثيقة وثيقة. هذه أيضاً تم عمل هندسة تسوية لها
        candidate_texts = [c.text for c in candidates]
        candidate_embs = model.encode(
            candidate_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(
            np.float32
        )  # shape: (N_candidates, dim)

        # Cosine Similarity (بعد normalize = dot product)
        # لحساب درجة تشابه جيب التمام (Cosine Similarity) بين الاستعلام والـ 100 وثيقة، لا نحتاج لكتابة معادلات أو حلقات معقدة. بما أن المتجهات تم عمل تسوية لها مسبقاً، فإن تشابه جيب التمام يساوي رياضياً الضرب النقطي (Dot Product) مباشرة
        scores = (candidate_embs @ query_emb.T).flatten()  # shape: (N_candidates,)

        # ترتيب تنازلي
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: List[DocumentResult] = []
        for new_rank, idx in enumerate(top_idx, start=1):
            c = candidates[idx]
            results.append(
                DocumentResult(
                    doc_id=c.doc_id,
                    title=c.title,
                    text=c.text,
                    score=float(scores[idx]),
                    rank=new_rank,
                )
            )

        return results
