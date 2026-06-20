"""
services/indexing/hybrid_indexer.py
=====================================
Thin Orchestrator فوق BM25Indexer و EmbeddingIndexer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا هذا الملف موجود؟ (المبرر الهندسي)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

بدونه على Developer 2 أن يكتب:

    bm25 = BM25Indexer()
    bm25.build_index("msmarco")
    bm25.save_index("msmarco")

    emb = EmbeddingIndexer()
    emb.build_index("msmarco")
    emb.save_index("msmarco")
    # ثم يُدير الكائنَين يدوياً في كل مكان

مع هذا الملف يكتب فقط:

    hybrid = HybridIndexer()
    hybrid.build_indexes("msmarco")
    # كلا الفهرسين مبنيان ومحفوظان

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما الذي لا يفعله هذا الملف؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✗ لا يكرر منطق BM25     (موجود في bm25_indexer.py)
  ✗ لا يكرر منطق Embedding (موجود في embedding_indexer.py)
  ✗ لا يقوم بالبحث        (مسؤولية hybrid_serial/parallel.py)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
العلاقة مع باقي الملفات
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HybridIndexer          ← يُعدّ الفهارس (هذا الملف)
  HybridSerialRetriever  ← يبحث تسلسلياً  (hybrid_serial.py)
  HybridParallelRetriever← يبحث متوازياً  (hybrid_parallel.py)

  الترتيب الصحيح للاستخدام:
    1. HybridIndexer.build_indexes()  ← مرة واحدة فقط
    2. hybrid.bm25 → يُمرَّر لـ BM25Retriever
    3. hybrid.emb  → يُمرَّر لـ EmbeddingRetriever
    4. Retriever يقوم بالبحث
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.indexing.bm25_indexer import (
    BM25Indexer,
    get_bm25_indexer,
)
from services.indexing.embedding_indexer import (
    EmbeddingIndexer,
    get_embedding_indexer,
    DEFAULT_MODEL_NAME,
)
from shared.constants import (
    BM25_DEFAULT_K1,
    BM25_DEFAULT_B,
    DEFAULT_APPLY_STEMMING,
    DEFAULT_REMOVE_STOPWORDS,
    DEFAULT_LANGUAGE,
    INDEXES_DIR,
)


class HybridIndexer:
    """
    Orchestrator يجمع BM25Indexer و EmbeddingIndexer تحت واجهة واحدة.

    الاستخدام الأساسي (Developer 2):
    ──────────────────────────────────
        # البناء لأول مرة (يأخذ وقتاً)
        hybrid = HybridIndexer()
        hybrid.build_indexes("msmarco-passage")

        # في كل إعادة تشغيل للخادم (سريع)
        hybrid = HybridIndexer.from_saved("msmarco-passage")

        # الوصول للفهارس
        bm25_indexer  = hybrid.bm25   # → BM25Indexer جاهز
        emb_indexer   = hybrid.emb    # → EmbeddingIndexer جاهز

        # فحص الحالة قبل البحث
        if hybrid.is_built():
            results = retriever.search(...)
    """

    def __init__(
        self,
        bm25_indexer:      Optional[BM25Indexer]      = None,
        embedding_indexer: Optional[EmbeddingIndexer] = None,
        indexes_dir:       str                         = INDEXES_DIR,
        model_name:        str                         = DEFAULT_MODEL_NAME,
    ) -> None:
        """
        المعاملات:
            bm25_indexer      : حقن BM25Indexer جاهز (للاختبارات)
            embedding_indexer : حقن EmbeddingIndexer جاهز (للاختبارات)
            indexes_dir       : مجلد حفظ/تحميل الفهارس
            model_name        : نموذج SentenceTransformer

        لماذا نقبل Dependency Injection؟
        ───────────────────────────────────
        في الاختبارات، نُمرّر Mock objects خفيفة بدلاً من
        بناء فهارس حقيقية تأخذ دقائق.

        مثال في الاختبار:
            mock_bm25 = MagicMock(spec=BM25Indexer)
            mock_bm25.is_built.return_value = True
            hybrid = HybridIndexer(bm25_indexer=mock_bm25)
        """
        self.bm25: BM25Indexer = bm25_indexer or BM25Indexer(
            indexes_dir=indexes_dir
        )
        self.emb: EmbeddingIndexer = embedding_indexer or EmbeddingIndexer(
            indexes_dir=indexes_dir,
            model_name=model_name,
        )

    # ─────────────────────────────────────────────────────────
    # البناء
    # ─────────────────────────────────────────────────────────

    def build_indexes(
        self,
        dataset_name:     str,
        # معاملات BM25
        k1:               float = BM25_DEFAULT_K1,
        b:                float = BM25_DEFAULT_B,
        apply_stemming:   bool  = DEFAULT_APPLY_STEMMING,
        remove_stopwords: bool  = DEFAULT_REMOVE_STOPWORDS,
        language:         str   = DEFAULT_LANGUAGE,
        # معاملات Embedding
        batch_size:       int   = 64,
        normalize:        bool  = True,
        # مشترك
        max_docs:         Optional[int] = None,
        # تحكم في ما يُبنى
        build_bm25:       bool  = True,
        build_embedding:  bool  = True,
    ) -> None:
        """
        يبني فهرسَي BM25 و Embedding ويحفظهما على القرص.

        لماذا نوفر build_bm25 و build_embedding منفصلَين؟
        ───────────────────────────────────────────────────
        أحياناً نريد إعادة بناء فهرس واحد فقط:
          - غيّرنا نموذج Embedding → نعيد Embedding فقط
          - غيّرنا k1/b لـ BM25 → نعيد BM25 فقط

        المعاملات:
            dataset_name      : اسم مجموعة البيانات (مثل "msmarco-passage")
            k1, b             : معاملات BM25 — راجع bm25_indexer.py
            apply_stemming    : تطبيق stemming على الوثائق
            remove_stopwords  : حذف stopwords
            language          : لغة الوثائق
            batch_size        : عدد الوثائق في كل دفعة للـ Embedding
            normalize         : L2 normalization للـ Embedding (True دائماً)
            max_docs          : للاختبار فقط — يحدد عدد الوثائق
            build_bm25        : هل نبني فهرس BM25؟
            build_embedding   : هل نبني فهرس Embedding؟
        """
        print(f"\n{'━'*55}")
        print(f"[HybridIndexer] بدء بناء الفهارس الهجينة")
        print(f"[HybridIndexer] المجموعة: '{dataset_name}'")
        print(f"[HybridIndexer] BM25={build_bm25} | Embedding={build_embedding}")
        if max_docs:
            print(f"[HybridIndexer] ⚠️  max_docs={max_docs} (وضع الاختبار)")
        print(f"{'━'*55}")

        # ── بناء BM25 ─────────────────────────────────────────
        if build_bm25:
            print("\n[HybridIndexer] ◆ بناء فهرس BM25...")
            self.bm25.build_index(
                dataset_name=dataset_name,
                k1=k1,
                b=b,
                apply_stemming=apply_stemming,
                remove_stopwords=remove_stopwords,
                language=language,
                max_docs=max_docs,
            )
            # build_index لا يحفظ تلقائياً — نستدعي save_index صراحةً
            self.bm25.save_index(dataset_name)
            print("[HybridIndexer] ✅ فهرس BM25 مكتمل ومحفوظ")
        else:
            print("[HybridIndexer] ⏭  تخطي BM25 (build_bm25=False)")

        # ── بناء Embedding ────────────────────────────────────
        if build_embedding:
            print("\n[HybridIndexer] ◆ بناء فهرس Embedding...")
            self.emb.build_index(
                dataset_name=dataset_name,
                batch_size=batch_size,
                max_docs=max_docs,
                normalize=normalize,
            )
            # نفس المنطق — save_index منفصلة
            self.emb.save_index(dataset_name)
            print("[HybridIndexer] ✅ فهرس Embedding مكتمل ومحفوظ")
        else:
            print("[HybridIndexer] ⏭  تخطي Embedding (build_embedding=False)")

        print(f"\n{'━'*55}")
        print(f"[HybridIndexer] 🎉 البناء اكتمل")
        print(f"{'━'*55}\n")

    # ─────────────────────────────────────────────────────────
    # التحميل
    # ─────────────────────────────────────────────────────────

    def load_indexes(
        self,
        dataset_name:   str,
        load_bm25:      bool = True,
        load_embedding: bool = True,
    ) -> None:
        """
        يحمّل الفهارس من القرص إلى الذاكرة.

        متى تستخدم هذه الدالة؟
        ────────────────────────
        في كل إعادة تشغيل للخادم.
        البناء → دقائق.
        التحميل → ثوانٍ.

        تحذير:
        ───────
        إذا لم يكن الفهرس موجوداً على القرص، ستحصل على
        FileNotFoundError من BM25Indexer/EmbeddingIndexer.
        استخدم is_saved() قبل load_indexes().
        """
        print(f"\n[HybridIndexer] تحميل فهارس: '{dataset_name}'")

        if load_bm25:
            self.bm25.load_index(dataset_name)

        if load_embedding:
            self.emb.load_index(dataset_name)

        print(f"[HybridIndexer] ✅ التحميل مكتمل\n")

    # ─────────────────────────────────────────────────────────
    # فحص الحالة
    # ─────────────────────────────────────────────────────────

    def is_built(self) -> bool:
        """
        هل كلا الفهرسين في الذاكرة وجاهزان للبحث؟

        استخدامها الصحيح:
            if not hybrid.is_built():
                raise RuntimeError("الفهارس غير محمّلة")
        """
        return self.bm25.is_built() and self.emb.is_built()

    def is_saved(self, dataset_name: str) -> bool:
        """
        هل الفهارس محفوظة على القرص؟

        النمط الأمثل عند بداية تشغيل الخادم:
        ────────────────────────────────────────
            hybrid = HybridIndexer()
            if hybrid.is_saved("msmarco-passage"):
                hybrid.load_indexes("msmarco-passage")   # سريع
            else:
                hybrid.build_indexes("msmarco-passage")  # بطيء — مرة واحدة
        """
        return (
            self.bm25.is_saved(dataset_name)
            and self.emb.is_saved(dataset_name)
        )

    def get_status(self) -> dict:
        """
        يُرجع حالة تفصيلية للفهرسين.

        مفيد لـ API endpoint /status أو /health.

        مثال الإخراج:
        {
            "hybrid_ready": True,
            "bm25": {
                "is_built": True,
                "num_documents": 6980,
                "k1": 1.5,
                "b": 0.75,
                "vocab_size": 45000
            },
            "embedding": {
                "is_built": True,
                "model_name": "all-MiniLM-L6-v2",
                "num_documents": 6980,
                "embedding_dim": 384
            }
        }
        """
        bm25_status: dict = {"is_built": self.bm25.is_built()}
        if self.bm25.is_built() and self.bm25.metadata:
            bm25_status.update({
                "num_documents":  self.bm25.metadata.num_documents,
                "k1":             self.bm25.metadata.k1,
                "b":              self.bm25.metadata.b,
                "vocab_size":     self.bm25.metadata.vocab_size,
                "avg_doc_length": self.bm25.metadata.avg_document_length,
            })

        emb_status: dict = {"is_built": self.emb.is_built()}
        if self.emb.is_built() and self.emb.metadata:
            emb_status.update({
                "model_name":    self.emb.metadata.model_name,
                "num_documents": self.emb.metadata.num_documents,
                "embedding_dim": self.emb.metadata.embedding_dim,
                "index_type":    self.emb.metadata.index_type,
            })

        return {
            "hybrid_ready": self.is_built(),
            "bm25":         bm25_status,
            "embedding":    emb_status,
        }

    # ─────────────────────────────────────────────────────────
    # Class Method — اختصار مريح لتحميل من القرص
    # ─────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        dataset_name: str,
        indexes_dir:  str = INDEXES_DIR,
        model_name:   str = DEFAULT_MODEL_NAME,
    ) -> "HybridIndexer":
        """
        ينشئ HybridIndexer ويحمّل الفهارس مباشرةً في سطر واحد.

        هذه هي الطريقة المُفضَّلة في app.py:

            # في بداية تشغيل الخادم
            hybrid = HybridIndexer.from_saved("msmarco-passage")
            # الآن hybrid.bm25 و hybrid.emb جاهزان

        المعاملات:
            dataset_name : اسم مجموعة البيانات
            indexes_dir  : مجلد الفهارس
            model_name   : يجب أن يطابق النموذج المستخدم في البناء

        الإرجاع:
            HybridIndexer مع فهارس محمّلة وجاهزة للبحث فوراً
        """
        instance = cls(indexes_dir=indexes_dir, model_name=model_name)
        instance.load_indexes(dataset_name)
        return instance


# =============================================================
# Singleton Factory — نفس نمط BM25Indexer و EmbeddingIndexer
# =============================================================

_hybrid_instances: dict = {}


def get_hybrid_indexer(
    dataset_name: Optional[str] = None,
    model_name:   str           = DEFAULT_MODEL_NAME,
    indexes_dir:  str           = INDEXES_DIR,
) -> HybridIndexer:
    """
    يُرجع HybridIndexer — نسخة واحدة لكل dataset (Singleton).

    لماذا Singleton؟
    ─────────────────
    فهارس BM25 + Embedding تشغل عشرات الـ MB في الذاكرة.
    لو أنشأنا كائناً جديداً في كل request → الذاكرة تنفد.

    مثال:
        # في أي مكان في الكود — دائماً نفس الكائن
        hybrid = get_hybrid_indexer("msmarco-passage")

    ملاحظة:
        يحمّل الفهارس تلقائياً إذا كانت موجودة على القرص.
    """
    global _hybrid_instances

    key = f"{dataset_name or '__default__'}::{model_name}"

    if key not in _hybrid_instances:
        indexer = HybridIndexer(
            indexes_dir=indexes_dir,
            model_name=model_name,
        )
        if dataset_name and indexer.is_saved(dataset_name):
            try:
                indexer.load_indexes(dataset_name)
            except Exception as exc:
                # نسجّل التحذير لكن لا نوقف التطبيق
                print(f"[HybridIndexer] ⚠️  فشل التحميل التلقائي: {exc}")

        _hybrid_instances[key] = indexer

    return _hybrid_instances[key]