"""
services/indexing/vector_store.py
===================================
Facade بسيط فوق EmbeddingIndexer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا هذا الملف موجود؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EmbeddingIndexer ممتاز لكن واجهته مصممة لـ Developer 1
(بناء فهارس، إدارة FAISS، batch encoding...).

Developer 2 يحتاج شيئاً أبسط:
  store.search("fever treatment", k=10)
  store.add(doc_id, text)
  store.save()
  store.load()

هذا كل ما يفعله هذا الملف.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Facade Pattern — ما معناه؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

تخيل محرك سيارة معقد جداً.
المستخدم لا يحتاج أن يفهم كيف يعمل الإنجن.
يضغط على دواسة الوقود فقط.

دواسة الوقود = Facade
الإنجن = EmbeddingIndexer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما الذي لا يفعله هذا الملف؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✗ لا يكرر منطق FAISS
  ✗ لا يكرر منطق encoding
  ✗ لا ينشئ Vector Database حقيقية
  ✗ لا يستخدم Chroma/Pinecone/Qdrant
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.indexing.embedding_indexer import (
    EmbeddingIndexer,
    IndexedDocument,
    DEFAULT_MODEL_NAME,
    get_embedding_indexer,
)
from shared.constants import INDEXES_DIR


# =============================================================
# نوع النتيجة — بسيط وواضح
# =============================================================

# كل نتيجة بحث: (doc_id, score, text, title)
SearchResult = Tuple[str, float, str, Optional[str]]


class VectorStore:
    """
    Facade بسيط فوق EmbeddingIndexer.

    الاستخدام (Developer 2):
    ────────────────────────
        # تحميل
        store = VectorStore("msmarco-passage")
        store.load()

        # بحث
        results = store.search("fever treatment", k=5)
        for doc_id, score, text, title in results:
            print(f"[{score:.3f}] {title}: {text[:100]}")

        # إضافة وثيقة جديدة
        store.add("new_doc", "Some new text about AI")
        store.save()
    """

    def __init__(
        self,
        dataset_name: str,
        indexes_dir:  str = INDEXES_DIR,
        model_name:   str = DEFAULT_MODEL_NAME,
    ) -> None:
        """
        المعاملات:
            dataset_name : اسم مجموعة البيانات
            indexes_dir  : مجلد الفهارس
            model_name   : نموذج SentenceTransformer
        """
        self._dataset_name = dataset_name
        self._indexes_dir  = indexes_dir

        # الـ EmbeddingIndexer هو المحرك الفعلي — نحن فقط wrapper فوقه
        self._indexer = EmbeddingIndexer(
            indexes_dir=indexes_dir,
            model_name=model_name,
        )

    # ──────────────────────────────────────────────────────────
    # الواجهة الأساسية — ما يحتاجه Developer 2
    # ──────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        k:          int = 10,
    ) -> List[SearchResult]:
        """
        يبحث بالنص ويُرجع أقرب K وثائق معنىً.

        كيف يعمل داخلياً؟
        ──────────────────
        1. يُحوّل query_text لمتجه embedding (384 رقم)
        2. يجد أقرب K متجهات في FAISS (cosine similarity)
        3. يُرجع الوثائق المقابلة مع درجاتها

        لماذا نمرّر النص الخام وليس tokens معالجة؟
        ────────────────────────────────────────────
        Sentence Transformer يفهم النص الطبيعي كاملاً.
        Stemming أو stopword removal يُضعف جودة الـ embedding.
        "running dogs" أفضل من "run dog" للنموذج.

        المعاملات:
            query_text : النص الأصلي (بدون معالجة)
            k          : عدد النتائج

        الإرجاع:
            List of (doc_id, score, text, title)
            مرتبة تنازلياً حسب درجة التشابه (1.0 = متطابق)

        مثال:
            results = store.search("cloud backup sync", k=3)
            # → [("d1", 0.89, "Cloud storage is useful...", "Cloud Storage"),
            #    ("d8", 0.76, "Some apps store preferences...", None),
            #    ...]
        """
        if not self.is_ready():
            return []

        if not query_text.strip():
            return []

        # الخطوة 1: نص → متجه
        query_vec = self._indexer.encode_query(query_text)
        if query_vec is None:
            return []

        # الخطوة 2: بحث في FAISS
        raw_results: List[Tuple[IndexedDocument, float]] = (
            self._indexer.get_top_k(query_vec, k=k)
        )

        # الخطوة 3: تحويل للشكل البسيط
        return [
            (doc.doc_id, float(score), doc.original_text, doc.title)
            for doc, score in raw_results
        ]

    def add(
        self,
        doc_id:   str,
        text:     str,
        title:    Optional[str] = None,
    ) -> bool:
        """
        يضيف وثيقة واحدة للـ store.

        ⚠️ ملاحظة هندسية مهمة:
        ─────────────────────────
        FAISS IndexFlatIP لا يدعم إضافة وثائق بكفاءة
        بعد البناء (لا يوجد "real-time update").

        ما الذي يحدث هنا فعلاً؟
        1. نُشفّر النص لمتجه
        2. نُضيف المتجه لـ FAISS (يدعم add() لكن بدون حذف)
        3. نُضيف الوثيقة لقائمة self._indexer.documents

        متى تستخدم هذه الدالة؟
        - عند إضافة وثائق جديدة بعد البناء الأولي
        - للاختبارات
        - للتحديثات الصغيرة

        للإضافات الكبيرة: أعد build_index() من الصفر.

        الإرجاع:
            True إذا نجحت الإضافة، False إذا فشلت
        """
        if not self.is_ready():
            return False

        if not text.strip():
            return False

        try:
            import numpy as np

            # تشفير النص الجديد
            new_vec = self._indexer.encode_query(text)
            if new_vec is None:
                return False

            # إضافة المتجه لـ FAISS
            import faiss as _faiss
            self._indexer.faiss_index.add(
                np.ascontiguousarray(new_vec.astype(np.float32))
            )

            # تحديث مصفوفة الـ embeddings
            self._indexer.embeddings = np.vstack([
                self._indexer.embeddings,
                new_vec,
            ])

            # إضافة الوثيقة للقائمة
            new_doc = IndexedDocument(
                doc_id=doc_id,
                original_text=text,
                processed_text=text,
                title=title,
            )
            idx = len(self._indexer.documents)
            self._indexer.documents.append(new_doc)
            self._indexer.doc_id_to_idx[doc_id] = idx

            return True

        except Exception as exc:
            print(f"[VectorStore] ⚠️  فشل add(): {exc}")
            return False

    def delete(self, doc_id: str) -> bool:
        """
        يحذف وثيقة من الـ store.

        ⚠️ قيد FAISS المهم:
        ──────────────────────
        FAISS IndexFlatIP لا يدعم الحذف الحقيقي.
        لا توجد دالة "remove vector" في FAISS Flat index.

        الحل المُطبَّق هنا: Soft Delete
        نُزيل الوثيقة من قوائمنا فقط.
        المتجه يبقى في FAISS لكن لن يُعاد في النتائج
        لأننا نتحقق من doc_id قبل إرجاع النتيجة.

        للحذف الحقيقي: أعد build_index() بدون هذه الوثيقة.

        الإرجاع:
            True إذا وُجد وحُذف، False إذا لم يُوجَد
        """
        if doc_id not in self._indexer.doc_id_to_idx:
            return False

        # Soft delete: نُزيل من القوائم فقط
        idx = self._indexer.doc_id_to_idx.pop(doc_id)

        # نضع placeholder بدلاً من الحذف الفعلي
        # (لأن الحذف يُغيّر indices ويُفسد doc_id_to_idx)
        if 0 <= idx < len(self._indexer.documents):
            self._indexer.documents[idx] = IndexedDocument(
                doc_id=f"__deleted__{doc_id}",
                original_text="",
                processed_text="",
                title=None,
            )

        return True

    def save(self) -> bool:
        """
        يحفظ الـ store على القرص.

        يستدعي EmbeddingIndexer.save_index() مباشرة.

        الإرجاع:
            True إذا نجح الحفظ، False إذا فشل
        """
        if not self.is_ready():
            return False

        try:
            self._indexer.save_index(self._dataset_name)
            print(f"[VectorStore] ✅ محفوظ: '{self._dataset_name}'")
            return True
        except Exception as exc:
            print(f"[VectorStore] ❌ فشل الحفظ: {exc}")
            return False

    def load(self) -> bool:
        """
        يحمّل الـ store من القرص.

        يستدعي EmbeddingIndexer.load_index() مباشرة.

        الإرجاع:
            True إذا نجح التحميل، False إذا فشل
        """
        try:
            self._indexer.load_index(self._dataset_name)
            print(f"[VectorStore] ✅ محمّل: '{self._dataset_name}'")
            return True
        except FileNotFoundError:
            print(
                f"[VectorStore] ⚠️  الفهرس غير موجود: '{self._dataset_name}'\n"
                f"   شغّل build_index() أولاً."
            )
            return False
        except Exception as exc:
            print(f"[VectorStore] ❌ فشل التحميل: {exc}")
            return False

    # ──────────────────────────────────────────────────────────
    # دوال الحالة
    # ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """
        هل الـ store جاهز للبحث؟

        يتحقق من: FAISS index موجود + وثائق محمّلة.
        استخدم قبل أي search() أو add().

        مثال:
            if not store.is_ready():
                store.load()
        """
        return self._indexer.is_built()

    def is_persisted(self) -> bool:
        """هل الـ store محفوظ على القرص؟"""
        return self._indexer.is_saved(self._dataset_name)

    def size(self) -> int:
        """عدد الوثائق في الـ store."""
        return len(self._indexer.documents) if self.is_ready() else 0

    def get_status(self) -> Dict:
        """
        حالة الـ store — مفيد لـ /health endpoints.

        مثال الإخراج:
        {
            "dataset_name": "msmarco-passage",
            "is_ready": True,
            "is_persisted": True,
            "num_documents": 6980,
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dim": 384
        }
        """
        status: Dict = {
            "dataset_name": self._dataset_name,
            "is_ready":     self.is_ready(),
            "is_persisted": self.is_persisted(),
            "num_documents": self.size(),
        }

        if self.is_ready() and self._indexer.metadata:
            status.update({
                "model_name":    self._indexer.metadata.model_name,
                "embedding_dim": self._indexer.metadata.embedding_dim,
                "index_type":    self._indexer.metadata.index_type,
            })

        return status

    # ──────────────────────────────────────────────────────────
    # وصول مباشر للـ indexer (لـ Developer 2 المتقدم)
    # ──────────────────────────────────────────────────────────

    @property
    def indexer(self) -> EmbeddingIndexer:
        """
        وصول مباشر للـ EmbeddingIndexer.

        متى تستخدم هذا؟
        عندما تحتاج شيئاً متقدماً لا توفّره واجهة VectorStore.
        مثال: encode_query() مباشرة للاستخدام في HybridRetriever.

        مثال:
            vec = store.indexer.encode_query("fever treatment")
        """
        return self._indexer


# =============================================================
# Singleton Factory
# =============================================================

_vector_store_instances: Dict[str, VectorStore] = {}


def get_vector_store(
    dataset_name: str,
    indexes_dir:  str = INDEXES_DIR,
    model_name:   str = DEFAULT_MODEL_NAME,
) -> VectorStore:
    """
    يُرجع VectorStore — نسخة واحدة لكل dataset (Singleton).

    يحمّل تلقائياً إذا كان الفهرس موجوداً على القرص.

    مثال:
        store = get_vector_store("msmarco-passage")
        results = store.search("fever treatment", k=10)
    """
    global _vector_store_instances
    key = f"{dataset_name}::{model_name}::{indexes_dir}"

    if key not in _vector_store_instances:
        vs = VectorStore(
            dataset_name=dataset_name,
            indexes_dir=indexes_dir,
            model_name=model_name,
        )
        # تحميل تلقائي إذا كان محفوظاً
        if vs.is_persisted():
            try:
                vs.load()
            except Exception as exc:
                print(f"[VectorStore] ⚠️  فشل التحميل التلقائي: {exc}")

        _vector_store_instances[key] = vs

    return _vector_store_instances[key]