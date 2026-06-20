"""
services/indexing/embedding_indexer.py
========================================
بناء فهرس Semantic Embeddings باستخدام SentenceTransformers + FAISS.

═══════════════════════════════════════════════════════════
الفرق الجوهري عن TF-IDF و BM25
═══════════════════════════════════════════════════════════

TF-IDF / BM25:
  يبحثان عن كلمات مشتركة بين الاستعلام والوثيقة.
  "fever treatment" لا يجد "antipyretic medication" ← فشل.

Embeddings:
  يُحوّل النموذج كل نص لمتجه رقمي في فضاء 384 بُعد.
  النصوص ذات المعنى المتشابه → متجهات قريبة في هذا الفضاء.
  "fever treatment" ≈ "antipyretic medication" ← نجاح ✅

═══════════════════════════════════════════════════════════
لماذا لا نحتاج Preprocessing؟
═══════════════════════════════════════════════════════════

Sentence Transformer تدرّب على ملايين الجمل كاملة.
النموذج يفهم "Running", "run", "runs" بنفس الطريقة.
Stemming أو stopword removal يُشوّه المعنى الكامل للجملة.

مثال:
  "AI helps people" → بعد stemming → "ai help peopl"
  النموذج تدرّب على الجملة الكاملة الطبيعية → الجملة المعالجة
  قد تُنتج embedding أقل جودة.

لهذا السبب: encode_query() يأخذ النص الخام مباشرة.

═══════════════════════════════════════════════════════════
لماذا FAISS وليس Cosine مباشرة؟
═══════════════════════════════════════════════════════════

200,000 وثيقة × 384 بُعد:
  cosine_similarity كاملة: 200,000 × 384 = 76.8M عملية/استعلام
  FAISS IndexFlatIP: نفس الحساب لكن بـ BLAS optimized C++
  → 10-50× أسرع من numpy على CPU

لمشروعنا (≤ 200k وثيقة): IndexFlatIP = دقة 100% + سرعة كافية
في الإنتاج الحقيقي (> 1M): نرتقي لـ HNSW أو IVF

═══════════════════════════════════════════════════════════
هيكل الملفات المحفوظة
═══════════════════════════════════════════════════════════

data/indexes/{dataset_name}/embedding/
  embedding_index.faiss     ← FAISS index (الأسرع للتحميل)
  embedding_vectors.npy     ← المتجهات الأصلية (للفحص والبناء)
  embedding_documents.json  ← بيانات الوثائق
  embedding_metadata.json   ← اسم النموذج، البُعد، الإعدادات
  embedding_docid_map.json  ← خريطة doc_id → index
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.constants import INDEXES_DIR
from services.indexing.dataset_loader import DatasetLoader, Document, get_dataset_loader
from services.indexing.tfidf_indexer import IndexedDocument  # نُعيد استخدامه


# =============================================================
# اسم النموذج الافتراضي
# =============================================================

# all-MiniLM-L6-v2: التوازن الأمثل للمشاريع الجامعية
#   - 384 بُعد (أصغر من 768 → أسرع وأخف)
#   - دقة عالية على المهام الإنجليزية
#   - حجم ~90MB (يُحمَّل مرة واحدة)
#   - سرعة encoding: ~14,000 جملة/ثانية على CPU
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# البديل الأقوى للإنتاج الجاد (يحتاج ~420MB):
# DEFAULT_MODEL_NAME = "all-mpnet-base-v2"  # → 768 بُعد، دقة أعلى


# =============================================================
# Metadata Dataclass
# =============================================================

@dataclass
class EmbeddingIndexMetadata:
    """
    بيانات وصفية لفهرس Embedding.

    ما الذي نحتاج حفظه هنا ولماذا؟

    model_name:
      Developer 2 يجب أن يستخدم نفس النموذج لتحويل الاستعلام.
      إذا بُني الفهرس بـ MiniLM وحُوِّل الاستعلام بـ MPNet →
      المتجهات في فضاءات مختلفة → نتائج عشوائية تماماً.

    embedding_dim:
      للتحقق عند التحميل: هل الفهرس المحمّل يطابق النموذج الحالي؟

    index_type:
      "flat_ip" في مشروعنا — يُساعد Developer 2 يفهم نوع البحث.

    normalize_embeddings:
      إذا طبّقنا L2 normalization → dot product = cosine similarity.
      يجب أن يطبّقها Developer 2 على الاستعلام أيضاً.
    """
    dataset_name: str
    model_name: str
    embedding_dim: int
    num_documents: int
    index_type: str              # "flat_ip" دائماً في مشروعنا
    normalize_embeddings: bool   # True دائماً لـ cosine similarity
    build_time_seconds: float
    build_timestamp: str
    batch_size: int              # حجم الدفعة المستخدمة في الـ encoding

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "EmbeddingIndexMetadata":
        return cls(**data)


# =============================================================
# EmbeddingIndexer — الكلاس الرئيسي
# =============================================================

class EmbeddingIndexer:
    """
    يبني فهرس Semantic Embeddings باستخدام FAISS.

    ما يُنتجه للـ Developer 2:
      self.faiss_index     → FAISS index جاهز للبحث
      self.embeddings      → numpy array شكله (N, dim) للفحص
      self.documents       → قائمة IndexedDocument
      self.doc_id_to_idx   → خريطة سريعة O(1)
      self.metadata        → اسم النموذج والإعدادات

    كيف يبحث Developer 2؟
      query_vec = indexer.encode_query("fever treatment")
      results = indexer.get_top_k(query_vec, k=10)
      for doc, score in results:
          print(f"[{score:.3f}] {doc.title}")

    ⚠️ مهم: encode_query يأخذ النص الخام — لا preprocessing!
    """

    # أسماء الملفات — ثوابت لمنع أخطاء الإملاء
    _FAISS_FILE     = "embedding_index.faiss"
    _VECTORS_FILE   = "embedding_vectors.npy"
    _DOCUMENTS_FILE = "embedding_documents.json"
    _METADATA_FILE  = "embedding_metadata.json"
    _DOCID_MAP_FILE = "embedding_docid_map.json"

    def __init__(
        self,
        indexes_dir: str = INDEXES_DIR,
        dataset_loader: Optional[DatasetLoader] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """
        المعاملات:
            indexes_dir   : مجلد الفهارس
            dataset_loader: DatasetLoader (Dependency Injection للاختبارات)
            model_name    : اسم نموذج SentenceTransformer
            
            
            شو هو SentenceTransformer؟

هو نموذج ذكاء اصطناعي جاهز لتحويل النص إلى أرقام.

يعني بدل ما يكون عندك:

"I love machine learning"

بيحولها إلى متجه (Vector):

[0.12, -0.55, 0.81, 0.04, ...]

مثلاً 384 رقم.
        """
        self.indexes_dir = Path(indexes_dir)
        self._loader     = dataset_loader or get_dataset_loader()
        self.model_name  = model_name

        # ════════════════════════════════════════
        # حالة الفهرس — تُملأ عند البناء أو التحميل
        # ════════════════════════════════════════

        # النموذج نفسه — يُحمَّل مرة واحدة فقط (lazy loading)
        # لأن تحميله يأخذ ~2-5 ثوانٍ
        self._model = None  # SentenceTransformer instance

        # FAISS index — الهيكل الرئيسي للبحث
        self.faiss_index = None

        # المتجهات الأصلية للوثائق: shape = (num_docs, embedding_dim)
        # نحفظها منفصلة عن FAISS لأغراض الفحص وإعادة البناء
        self.embeddings: Optional[np.ndarray] = None

        # قائمة الوثائق — نفس نمط TF-IDF و BM25
        self.documents: List[IndexedDocument] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self.metadata: Optional[EmbeddingIndexMetadata] = None

    # ----------------------------------------------------------
    # تحميل النموذج (Lazy Loading)
    # ----------------------------------------------------------

    def _get_model(self):
        """
        يُحمَّل النموذج عند أول استخدام فقط (Lazy Loading).

        لماذا Lazy وليس في __init__؟
        - تحميل النموذج يأخذ 2-5 ثوانٍ
        - إذا أنشأنا EmbeddingIndexer فقط للتحقق من is_saved()،
          لا نريد أن ننتظر تحميل النموذج

        لماذا نحفظه في self._model؟
        - Singleton داخلي — يُحمَّل مرة واحدة طوال حياة الكائن
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers غير مثبتة.\n"
                    "شغّل: pip install sentence-transformers"
                )
            print(f"[EmbeddingIndexer] تحميل النموذج: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            dim = self._model.get_sentence_embedding_dimension()
            print(f"[EmbeddingIndexer]   ✓ بُعد الـ embedding: {dim}")
        return self._model

    # ----------------------------------------------------------
    # بناء الفهرس
    # ----------------------------------------------------------

    def build_index(
        self,
        dataset_name: str,
        batch_size: int = 64,
        max_docs: Optional[int] = None,
        normalize: bool = True,
    ) -> EmbeddingIndexMetadata:
        """
        يبني فهرس Embedding كاملاً.

        الخطوات:
          1. تحميل الوثائق (DatasetLoader)
          2. encoding كل وثيقة → متجه (SentenceTransformer)
          3. L2 Normalization (لجعل dot product = cosine)
          4. بناء FAISS IndexFlatIP
          5. حفظ النتائج في self.*

        لماذا نُرسل batch_size=64 وليس كل الوثائق دفعة واحدة؟
          64 وثيقة × 384 بُعد = 24,576 قيمة float32 = ~100KB
          هذا يتناسب مع cache المعالج → أسرع من وثيقة واحدة أو
          أكثر مما يتسع للـ cache.
          قيمة 32-128 هي الأمثل على CPU. على GPU: 256-512.

        المعاملات:
            dataset_name : اسم مجموعة البيانات
            batch_size   : عدد الوثائق المُعالجة دفعةً واحدة
            max_docs     : للاختبار فقط
            normalize    : تطبيق L2 normalization (True دائماً لـ cosine)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu غير مثبتة.\n"
                "شغّل: pip install faiss-cpu"
            )

        print(f"\n{'='*55}")
        print(f"[EmbeddingIndexer] بدء بناء الفهرس: '{dataset_name}'")
        print(f"[EmbeddingIndexer] النموذج: {self.model_name}")
        print(f"{'='*55}")
        start_time = time.time()

        # ── الخطوة 1: تحميل الوثائق ──────────────────────────
        print("[EmbeddingIndexer] الخطوة 1/4: تحميل الوثائق...")
        raw_docs = self._loader.load_all(dataset_name, max_docs=max_docs)

        if not raw_docs:
            raise ValueError(f"مجموعة البيانات '{dataset_name}' فارغة.")

        print(f"[EmbeddingIndexer]   ✓ {len(raw_docs):,} وثيقة")

        # ── الخطوة 2: تجهيز النصوص ──────────────────────────
        # نستخدم get_full_text() لدمج العنوان مع النص
        # هذا يُحسّن جودة الـ embedding لأن العنوان غالباً
        # يحمل الكلمات المفتاحية الأهم
        texts = [doc.get_full_text() for doc in raw_docs]

        # ── الخطوة 3: Encoding ───────────────────────────────
        print("[EmbeddingIndexer] الخطوة 2/4: Encoding الوثائق...")
        print(f"[EmbeddingIndexer]   batch_size={batch_size}")

        model = self._get_model()

        # encode_texts يُشغّل النموذج على كل الوثائق دفعات
        # show_progress_bar=True يُظهر شريط تقدم جميل
        embeddings_raw = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,    # نريد numpy وليس tensor
            normalize_embeddings=normalize,  # L2 normalization تلقائي
        )

        # embeddings_raw.shape = (num_docs, embedding_dim)
        # مثال: (5, 384) لـ 5 وثائق مع MiniLM
        self.embeddings = embeddings_raw.astype(np.float32)
        embedding_dim   = self.embeddings.shape[1]

        print(f"[EmbeddingIndexer]   ✓ شكل المصفوفة: "
              f"{self.embeddings.shape[0]:,} × {self.embeddings.shape[1]}")

        # ── الخطوة 4: بناء FAISS Index ───────────────────────
        print("[EmbeddingIndexer] الخطوة 3/4: بناء FAISS Index...")

        # IndexFlatIP: Inner Product (= Cosine بعد L2 Norm)
        # "Flat" يعني: لا تقريب، مقارنة كاملة مع كل الوثائق
        # "IP" = Inner Product = dot product
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)

        # نُضيف كل المتجهات للفهرس
        # FAISS يتوقع float32 array بشكل C-contiguous
        self.faiss_index.add(np.ascontiguousarray(self.embeddings))

        print(f"[EmbeddingIndexer]   ✓ FAISS index يحتوي "
              f"{self.faiss_index.ntotal:,} متجه")

        # ── الخطوة 5: بناء هياكل البيانات ────────────────────
        print("[EmbeddingIndexer] الخطوة 4/4: بناء هياكل البيانات...")

        self.documents = [
            IndexedDocument(
                doc_id=doc.doc_id,
                original_text=doc.text,
                processed_text=doc.get_full_text(),  # النص الكامل بدون stemming
                title=doc.title,
            )
            for doc in raw_docs
        ]
        self.doc_id_to_idx = {
            doc.doc_id: idx for idx, doc in enumerate(self.documents)
        }

        build_time = time.time() - start_time

        import datetime
        self.metadata = EmbeddingIndexMetadata(
            dataset_name=dataset_name,
            model_name=self.model_name,
            embedding_dim=embedding_dim,
            num_documents=len(self.documents),
            index_type="flat_ip",
            normalize_embeddings=normalize,
            build_time_seconds=round(build_time, 2),
            build_timestamp=datetime.datetime.now().isoformat(),
            batch_size=batch_size,
        )

        print(f"\n[EmbeddingIndexer] ✅ اكتمل البناء في {build_time:.2f} ثانية")
        print(f"{'='*55}\n")
        return self.metadata

    # ----------------------------------------------------------
    # حفظ الفهرس
    # ----------------------------------------------------------

    def save_index(self, dataset_name: str) -> Path:
        """
        يحفظ الفهرس في 5 ملفات:

          embedding_index.faiss     ← FAISS index (صيغة خاصة بـ FAISS)
          embedding_vectors.npy     ← المتجهات الأصلية (numpy binary)
          embedding_documents.json  ← بيانات الوثائق
          embedding_metadata.json   ← اسم النموذج والإعدادات
          embedding_docid_map.json  ← خريطة doc_id → index

        لماذا نحفظ المتجهات منفصلة عن FAISS؟
          - FAISS index: للبحث السريع
          - vectors.npy: للفحص، إعادة البناء بنوع index مختلف،
            أو حساب إحصائيات بدون بحث كامل

        لماذا نستخدم faiss.write_index وليس pickle؟
          FAISS index يحتوي بنى C++ داخلية — pickle لا يعمل معها.
          faiss.write_index/read_index هي الطريقة الصحيحة الوحيدة.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu غير مثبتة.")

        self._check_index_built()

        index_dir = self.indexes_dir / dataset_name / "embedding"
        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"[EmbeddingIndexer] حفظ الفهرس في: {index_dir}")

        # 1. حفظ FAISS index
        faiss_path = index_dir / self._FAISS_FILE
        faiss.write_index(self.faiss_index, str(faiss_path))
        print(f"[EmbeddingIndexer]   ✓ faiss index: "
              f"{self._get_file_size_mb(faiss_path):.2f} MB")

        # 2. حفظ المتجهات
        vectors_path = index_dir / self._VECTORS_FILE
        np.save(str(vectors_path), self.embeddings)
        print(f"[EmbeddingIndexer]   ✓ vectors: "
              f"{self._get_file_size_mb(vectors_path):.2f} MB")

        # 3. حفظ الوثائق
        docs_path = index_dir / self._DOCUMENTS_FILE
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(
                [doc.to_dict() for doc in self.documents],
                f, ensure_ascii=False,
            )

        # 4. حفظ metadata
        meta_path = index_dir / self._METADATA_FILE
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, ensure_ascii=False, indent=2)

        # 5. حفظ خريطة doc_id
        map_path = index_dir / self._DOCID_MAP_FILE
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_id_to_idx, f, ensure_ascii=False)

        print(f"[EmbeddingIndexer] ✅ حُفظ الفهرس بنجاح")
        return index_dir

    # ----------------------------------------------------------
    # تحميل الفهرس
    # ----------------------------------------------------------

    def load_index(self, dataset_name: str) -> EmbeddingIndexMetadata:
        """
        يحمّل فهرس Embedding من القرص.

        ⚠️ تحقق من model_name في metadata:
          إذا كان مختلفاً عن self.model_name → الفهرس غير متوافق!
          يجب إعادة بنائه بنفس النموذج.

        بعد التحميل:
          encode_query() جاهز للاستخدام
          get_top_k() جاهز للاستخدام
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu غير مثبتة.")

        index_dir = self.indexes_dir / dataset_name / "embedding"
        if not index_dir.exists():
            raise FileNotFoundError(
                f"فهرس Embedding غير موجود: {index_dir}\n"
                f"شغّل build_index('{dataset_name}') أولاً."
            )

        print(f"[EmbeddingIndexer] تحميل الفهرس من: {index_dir}")
        start_time = time.time()

        # 1. تحميل FAISS index
        self.faiss_index = faiss.read_index(
            str(index_dir / self._FAISS_FILE)
        )
        print(f"[EmbeddingIndexer]   ✓ FAISS index: "
              f"{self.faiss_index.ntotal:,} متجه")

        # 2. تحميل المتجهات
        self.embeddings = np.load(str(index_dir / self._VECTORS_FILE))
        print(f"[EmbeddingIndexer]   ✓ vectors: {self.embeddings.shape}")

        # 3. تحميل الوثائق
        with open(index_dir / self._DOCUMENTS_FILE, encoding="utf-8") as f:
            self.documents = [IndexedDocument.from_dict(d) for d in json.load(f)]

        # 4. تحميل metadata
        with open(index_dir / self._METADATA_FILE, encoding="utf-8") as f:
            self.metadata = EmbeddingIndexMetadata.from_dict(json.load(f))

        # 5. تحميل خريطة
        with open(index_dir / self._DOCID_MAP_FILE, encoding="utf-8") as f:
            self.doc_id_to_idx = json.load(f)

        # ⚠️ تحقق من توافق النموذج
        if self.metadata.model_name != self.model_name:
            print(
                f"[EmbeddingIndexer] ⚠️  تحذير: النموذج الحالي '{self.model_name}' "
                f"يختلف عن نموذج الفهرس '{self.metadata.model_name}'.\n"
                f"   الفهرس سيعمل لكن النتائج قد تكون غير دقيقة.\n"
                f"   الحل: أعد البناء بـ model_name='{self.metadata.model_name}'"
            )
            # نحدّث اسم النموذج ليتطابق مع الفهرس المحمّل
            self.model_name = self.metadata.model_name

        load_time = time.time() - start_time
        print(f"[EmbeddingIndexer]   ✓ النموذج: {self.metadata.model_name}")
        print(f"[EmbeddingIndexer]   ✓ البُعد: {self.metadata.embedding_dim}")
        print(f"[EmbeddingIndexer] ✅ تحميل مكتمل في {load_time:.3f} ثانية")
        return self.metadata

    # ----------------------------------------------------------
    # دوال البحث (لـ Developer 2)
    # ----------------------------------------------------------

    def encode_query(self, query_text: str) -> Optional[np.ndarray]:
        """
        يُحوّل استعلاماً نصياً إلى embedding vector.

        ════════════════════════════════════════
        لماذا لا نطبق Preprocessing هنا؟
        ════════════════════════════════════════

        Sentence Transformer تدرّب على جمل طبيعية كاملة.
        النموذج يفهم "running" و"run" وأنهما نفس الشيء.
        Stemming يُحوّل "running" → "run" لكن أحياناً:
          "bank running" → "bank run" (معنى مختلف تماماً!)
        النموذج يفهم السياق — Stemming يُفقده السياق.

        المعاملات:
            query_text: النص الخام كما كتبه المستخدم

        الإرجاع:
            numpy array شكله (1, embedding_dim) أو None إذا فارغ
        """
        if not query_text.strip():
            return None

        model = self._get_model()
        query_vec = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,  # نفس normalize عند البناء
        )
        return query_vec.astype(np.float32)

    def get_top_k(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[IndexedDocument, float]]:
        """
        يُرجع أفضل K وثائق باستخدام FAISS search.

        ════════════════════════════════════════════════════
        كيف يعمل FAISS search داخلياً؟
        ════════════════════════════════════════════════════

        faiss_index.search(query_vec, k) يفعل:
          1. يحسب dot product بين query_vec وكل المتجهات
             (= cosine similarity بعد L2 normalization)
          2. يُرجع أعلى K قيمة مع indices الوثائق

        يُرجع:
          scores: array شكله (1, k) — درجات التشابه
          indices: array شكله (1, k) — أرقام الوثائق

        مثال:
          scores  = [[0.89, 0.76, 0.71, ...]]
          indices = [[3, 0, 7, ...]]
          → الوثيقة رقم 3 هي الأكثر تشابهاً

        المعاملات:
            query_embedding: numpy array شكله (1, dim)
            k: عدد النتائج

        الإرجاع:
            List of (IndexedDocument, score) مرتبة تنازلياً
        """
        self._check_index_built()

        if query_embedding is None:
            return []

        # تأكد من شكل المتجه: يجب أن يكون (1, dim)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # ضمان float32 و C-contiguous (شرط FAISS)
        query_vec = np.ascontiguousarray(
            query_embedding.astype(np.float32)
        )

        # البحث في FAISS
        # scores: (1, k) — درجات التشابه
        # indices: (1, k) — أرقام الوثائق
        actual_k = min(k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vec, actual_k)

        # تحويل النتائج لقائمة (Document, score)
        results: List[Tuple[IndexedDocument, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS يُرجع -1 إذا لم تكن هناك نتائج كافية
                continue
            doc = self.get_document_by_index(int(idx))
            if doc is not None:
                results.append((doc, float(score)))

        return results

    def get_stats(self) -> Dict:
        """
        يُرجع إحصائيات الفهرس.
        مفيد للـ UI ولـ Developer 3 في عرض حالة النظام.
        """
        if not self.is_built():
            return {"status": "not_built"}

        return {
            "status": "ready",
            "model_name": self.metadata.model_name,
            "num_documents": self.metadata.num_documents,
            "embedding_dim": self.metadata.embedding_dim,
            "index_type": self.metadata.index_type,
            "build_time_seconds": self.metadata.build_time_seconds,
            "build_timestamp": self.metadata.build_timestamp,
            "faiss_total": self.faiss_index.ntotal if self.faiss_index else 0,
        }

    # ----------------------------------------------------------
    # دوال الوصول للوثائق (نفس interface TF-IDF وBM25)
    # ----------------------------------------------------------

    def is_built(self) -> bool:
        return (
            self.faiss_index is not None
            and self.embeddings is not None
            and len(self.documents) > 0
        )

    def is_saved(self, dataset_name: str) -> bool:
        index_dir = self.indexes_dir / dataset_name / "embedding"
        required = [
            self._FAISS_FILE,
            self._VECTORS_FILE,
            self._DOCUMENTS_FILE,
            self._METADATA_FILE,
        ]
        return all((index_dir / f).exists() for f in required)

    def get_document_by_id(self, doc_id: str) -> Optional[IndexedDocument]:
        idx = self.doc_id_to_idx.get(doc_id)
        if idx is None:
            return None
        return self.documents[idx]

    def get_document_by_index(self, idx: int) -> Optional[IndexedDocument]:
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        return None

    # ----------------------------------------------------------
    # دوال مساعدة
    # ----------------------------------------------------------

    def _check_index_built(self) -> None:
        if not self.is_built():
            raise RuntimeError(
                "فهرس Embedding غير مبني. "
                "شغّل build_index() أو load_index() أولاً."
            )

    @staticmethod
    def _get_file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)


# =============================================================
# Singleton Factory
# =============================================================

_embedding_instances: Dict[str, EmbeddingIndexer] = {}


def get_embedding_indexer(
    dataset_name: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> EmbeddingIndexer:
    """
    يُرجع EmbeddingIndexer — نسخة واحدة لكل dataset.

    ⚠️ النموذج لا يُحمَّل حتى أول استدعاء لـ encode_query أو build_index.
    هذا يجعل استدعاء get_embedding_indexer() سريعاً جداً.

    يحمّل الفهرس تلقائياً إذا كان محفوظاً على القرص.
    """
    global _embedding_instances
    key = f"{dataset_name or '__default__'}_{model_name}"

    if key not in _embedding_instances:
        indexer = EmbeddingIndexer(model_name=model_name)
        if dataset_name and indexer.is_saved(dataset_name):
            try:
                indexer.load_index(dataset_name)
            except Exception as e:
                print(f"[EmbeddingIndexer] تحذير: فشل التحميل التلقائي: {e}")
        _embedding_instances[key] = indexer

    return _embedding_instances[key]