"""
services/indexing/bm25_indexer.py
==================================
بناء فهرس BM25 وحفظه على القرص لاسترجاعه لاحقاً.

═══════════════════════════════════════════════════
لماذا BM25 وليس TF-IDF فقط؟
═══════════════════════════════════════════════════

TF-IDF مشكلتان أساسيتان:
  1. Term Saturation: كلمة تظهر 10 مرات تحصل على وزن 10×
     رغم أن الفرق بين 5 و 10 مرات لا يعني كثيراً معرفياً.
  2. Document Length Bias: وثيقة قصيرة فيها "cloud" مرة واحدة
     قد تحصل على score أعلى من وثيقة طويلة غنية فيها "cloud" 5 مرات.

BM25 يحل المشكلتين بمعادلة واحدة:

                    TF(t,d) × (k₁ + 1)
BM25(t,d) = IDF(t) × ─────────────────────────────────
                    TF(t,d) + k₁×(1 - b + b×|d|/avgdl)

  k₁ = 1.5 (قياسي): يتحكم في سقف تأثير التكرار
  b  = 0.75 (قياسي): يتحكم في تطبيع طول الوثيقة

═══════════════════════════════════════════════════
الفرق الهندسي عن TFIDFIndexer
═══════════════════════════════════════════════════

TFIDFIndexer يخزّن: sparse matrix (وثائق × مصطلحات)
BM25Indexer يخزّن: قائمة tokens لكل وثيقة

السبب: BM25Okapi (من rank_bm25) يحتاج tokens الأصلية
        ليحسب TF و|d| و avgdl عند وقت البحث.
        لا يعمل مع matrices جاهزة.
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
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.constants import (
    INDEXES_DIR,
    BM25_DEFAULT_K1,
    BM25_DEFAULT_B,
    DEFAULT_APPLY_STEMMING,
    DEFAULT_REMOVE_STOPWORDS,
    DEFAULT_LANGUAGE,
)
from services.indexing.dataset_loader import DatasetLoader, Document, get_dataset_loader
from services.indexing.tfidf_indexer import IndexedDocument   # نُعيد استخدامه


# =============================================================
# نموذج Metadata خاص بـ BM25
# =============================================================

@dataclass
class BM25IndexMetadata:
    """
    بيانات وصفية لفهرس BM25.

    لماذا نحفظ k1 و b في الـ metadata؟
    لأن Developer 2 يحتاج معرفة هذه القيم عند وقت البحث
    ليستخدم نفس الـ BM25Okapi object أو يبنيه بنفس الإعدادات.
    """
    dataset_name: str
    num_documents: int
    avg_document_length: float   # avgdl — يُحسب مرة واحدة عند البناء
    k1: float
    b: float
    apply_stemming: bool
    remove_stopwords: bool
    language: str
    build_time_seconds: float
    build_timestamp: str
    vocab_size: int              # عدد المصطلحات الفريدة في المجموعة

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BM25IndexMetadata":
        return cls(**data)


# =============================================================
# BM25Indexer — الكلاس الرئيسي
# =============================================================

class BM25Indexer:
    """
    يبني فهرس BM25 ويحفظه ويحمّله.

    ما يُنتجه هذا الكلاس للـ Developer 2:
      self.bm25          → BM25Okapi object جاهز للبحث
      self.documents     → قائمة IndexedDocument للوصول بالـ index
      self.doc_id_to_idx → خريطة doc_id → index
      self.metadata      → إعدادات الفهرس

    كيف يبحث Developer 2؟
      scores = indexer.bm25.get_scores(query_tokens)
      top_indices = np.argsort(scores)[::-1][:10]
      for idx in top_indices:
          doc = indexer.get_document_by_index(idx)
    """

    # أسماء ملفات الحفظ
    _BM25_FILE       = "bm25_model.pkl"
    _DOCUMENTS_FILE  = "bm25_documents.json"
    _METADATA_FILE   = "bm25_metadata.json"
    _DOCID_MAP_FILE  = "bm25_docid_map.json"
    _TOKENS_FILE     = "bm25_tokens.pkl"   # tokens لكل وثيقة (لإعادة البناء)

    def __init__(
        self,
        indexes_dir: str = INDEXES_DIR,
        dataset_loader: Optional[DatasetLoader] = None,
    ) -> None:
        self.indexes_dir = Path(indexes_dir)
        self._loader     = dataset_loader or get_dataset_loader()

        # حالة الفهرس
        self.bm25: Optional[BM25Okapi] = None

        # قائمة tokens لكل وثيقة بنفس ترتيب self.documents
        # مثال: [["cloud", "storag"], ["ai", "assist"], ...]
        # BM25Okapi يحتاج هذه للبحث
        self.tokenized_docs: List[List[str]] = []

        self.documents: List[IndexedDocument] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self.metadata: Optional[BM25IndexMetadata] = None

    # ----------------------------------------------------------
    # بناء الفهرس
    # ----------------------------------------------------------

    def build_index(
        self,
        dataset_name: str,
        k1: float = BM25_DEFAULT_K1,
        b: float = BM25_DEFAULT_B,
        apply_stemming: bool = DEFAULT_APPLY_STEMMING,
        remove_stopwords: bool = DEFAULT_REMOVE_STOPWORDS,
        language: str = DEFAULT_LANGUAGE,
        max_docs: Optional[int] = None,
    ) -> BM25IndexMetadata:
        """
        يبني فهرس BM25 من مجموعة بيانات.

        الخطوات:
          1. تحميل الوثائق (DatasetLoader)
          2. معالجة كل وثيقة → قائمة tokens
          3. بناء BM25Okapi من كل قوائم الـ tokens
          4. حفظ النتائج في self.*

        لماذا نمرّر k1 و b هنا وليس عند البحث؟
          BM25Okapi يحتاج k1 و b عند البناء لأنه يحسب
          إحصائيات IDF و avgdl مرة واحدة.
          لا يمكن تغييرهما بعد البناء بدون إعادة بناء الفهرس.

        المعاملات:
            dataset_name  : اسم مجموعة البيانات
            k1            : معامل تشبع TF (1.2-2.0 موصى به)
            b             : معامل تطبيع الطول (0.75 قياسي)
            apply_stemming: تطبيق stemming
            remove_stopwords: حذف stopwords
            language      : لغة الوثائق
            max_docs      : للاختبار
        """
        print(f"\n{'='*55}")
        print(f"[BM25Indexer] بدء بناء الفهرس: '{dataset_name}'")
        print(f"[BM25Indexer] الإعدادات: k1={k1}, b={b}")
        print(f"{'='*55}")
        start_time = time.time()

        # ── الخطوة 1: تحميل الوثائق ──────────────────────────
        print("[BM25Indexer] الخطوة 1/3: تحميل الوثائق...")
        raw_documents = self._loader.load_all(dataset_name, max_docs=max_docs)

        if not raw_documents:
            raise ValueError(
                f"مجموعة البيانات '{dataset_name}' فارغة أو غير موجودة."
            )
        print(f"[BM25Indexer]   ✓ حُمِّل {len(raw_documents):,} وثيقة")

        # ── الخطوة 2: المعالجة المسبقة → tokens ──────────────
        print("[BM25Indexer] الخطوة 2/3: المعالجة المسبقة...")

        tokenized_docs, indexed_docs = self._preprocess_to_tokens(
            raw_documents,
            apply_stemming=apply_stemming,
            remove_stopwords=remove_stopwords,
            language=language,
        )

        # احسب متوسط طول الوثيقة (avgdl)
        # BM25Okapi يحسبه داخلياً أيضاً لكننا نحفظه في الـ metadata
        avg_doc_length = (
            sum(len(tokens) for tokens in tokenized_docs) / len(tokenized_docs)
            if tokenized_docs else 0.0
        )
        print(f"[BM25Indexer]   ✓ متوسط طول الوثيقة: {avg_doc_length:.1f} token")

        # ── الخطوة 3: بناء BM25Okapi ──────────────────────────
        print("[BM25Indexer] الخطوة 3/3: بناء BM25Okapi...")

        # BM25Okapi يأخذ قائمة من قوائم الـ tokens
        # [[token1, token2, ...], [token1, token2, ...], ...]
        # يحسب داخلياً:
        #   - IDF لكل مصطلح عبر كل الوثائق
        #   - avgdl (متوسط طول الوثيقة)
        #   - TF لكل مصطلح في كل وثيقة
        self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
        self.tokenized_docs = tokenized_docs
        self.documents = indexed_docs
        self.doc_id_to_idx = {
            doc.doc_id: idx for idx, doc in enumerate(self.documents)
        }

        # حجم المفردة = عدد المصطلحات الفريدة
        all_tokens = {token for tokens in tokenized_docs for token in tokens}
        vocab_size = len(all_tokens)

        print(f"[BM25Indexer]   ✓ BM25Okapi مبني")
        print(f"[BM25Indexer]   ✓ حجم المفردة: {vocab_size:,} مصطلح")

        build_time = time.time() - start_time

        import datetime
        self.metadata = BM25IndexMetadata(
            dataset_name=dataset_name,
            num_documents=len(self.documents),
            avg_document_length=round(avg_doc_length, 2),
            k1=k1,
            b=b,
            apply_stemming=apply_stemming,
            remove_stopwords=remove_stopwords,
            language=language,
            build_time_seconds=round(build_time, 2),
            build_timestamp=datetime.datetime.now().isoformat(),
            vocab_size=vocab_size,
        )

        print(f"\n[BM25Indexer] ✅ اكتمل البناء في {build_time:.2f} ثانية")
        print(f"{'='*55}\n")
        return self.metadata

    # ----------------------------------------------------------
    # حفظ الفهرس
    # ----------------------------------------------------------

    def save_index(self, dataset_name: str) -> Path:
        """
        يحفظ فهرس BM25 على القرص في 4 ملفات:

          bm25_model.pkl      ← BM25Okapi object (pickle)
          bm25_tokens.pkl     ← tokens لكل وثيقة (pickle)
          bm25_documents.json ← بيانات الوثائق
          bm25_metadata.json  ← معلومات الفهرس
          bm25_docid_map.json ← خريطة doc_id → index

        لماذا نحفظ bm25_tokens.pkl؟
          حتى لا نحتاج إعادة معالجة الوثائق إذا أردنا
          إعادة بناء BM25Okapi بإعدادات k1/b مختلفة.
        """
        self._check_index_built()

        index_dir = self.indexes_dir / dataset_name / "bm25"
        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"[BM25Indexer] حفظ الفهرس في: {index_dir}")

        # 1. حفظ BM25Okapi model
        bm25_path = index_dir / self._BM25_FILE
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        print(f"[BM25Indexer]   ✓ bm25_model: {self._get_file_size_mb(bm25_path):.2f} MB")

        # 2. حفظ tokens (للإعادة البناء لاحقاً بإعدادات مختلفة)
        tokens_path = index_dir / self._TOKENS_FILE
        with open(tokens_path, "wb") as f:
            pickle.dump(self.tokenized_docs, f)
        print(f"[BM25Indexer]   ✓ tokens: {self._get_file_size_mb(tokens_path):.2f} MB")

        # 3. حفظ الوثائق
        docs_path = index_dir / self._DOCUMENTS_FILE
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(
                [doc.to_dict() for doc in self.documents],
                f,
                ensure_ascii=False,
            )
        print(f"[BM25Indexer]   ✓ documents: {self._get_file_size_mb(docs_path):.2f} MB")

        # 4. حفظ metadata
        meta_path = index_dir / self._METADATA_FILE
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, ensure_ascii=False, indent=2)

        # 5. حفظ خريطة doc_id → index
        map_path = index_dir / self._DOCID_MAP_FILE
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_id_to_idx, f, ensure_ascii=False)

        print(f"[BM25Indexer] ✅ حُفظ الفهرس بنجاح")
        return index_dir

    # ----------------------------------------------------------
    # تحميل الفهرس
    # ----------------------------------------------------------

    def load_index(self, dataset_name: str) -> BM25IndexMetadata:
        """
        يحمّل فهرس BM25 من القرص.

        بعد التحميل يكون self.bm25 جاهزاً للبحث مباشرةً:
          scores = self.bm25.get_scores(query_tokens)
        """
        index_dir = self.indexes_dir / dataset_name / "bm25"

        if not index_dir.exists():
            raise FileNotFoundError(
                f"فهرس BM25 غير موجود: {index_dir}\n"
                f"شغّل build_index('{dataset_name}') أولاً."
            )

        print(f"[BM25Indexer] تحميل الفهرس من: {index_dir}")
        start_time = time.time()

        # 1. تحميل BM25Okapi
        with open(index_dir / self._BM25_FILE, "rb") as f:
            self.bm25 = pickle.load(f)

        # 2. تحميل tokens
        with open(index_dir / self._TOKENS_FILE, "rb") as f:
            self.tokenized_docs = pickle.load(f)
        print(f"[BM25Indexer]   ✓ BM25Okapi محمّل ({len(self.tokenized_docs):,} وثيقة)")

        # 3. تحميل الوثائق
        with open(index_dir / self._DOCUMENTS_FILE, encoding="utf-8") as f:
            docs_data = json.load(f)
        self.documents = [IndexedDocument.from_dict(d) for d in docs_data]

        # 4. تحميل metadata
        with open(index_dir / self._METADATA_FILE, encoding="utf-8") as f:
            self.metadata = BM25IndexMetadata.from_dict(json.load(f))

        # 5. تحميل خريطة
        with open(index_dir / self._DOCID_MAP_FILE, encoding="utf-8") as f:
            self.doc_id_to_idx = json.load(f)

        load_time = time.time() - start_time
        print(f"[BM25Indexer]   ✓ k1={self.metadata.k1}, b={self.metadata.b}")
        print(f"[BM25Indexer] ✅ تحميل مكتمل في {load_time:.3f} ثانية")
        return self.metadata

    # ----------------------------------------------------------
    # دوال البحث المساعدة (لـ Developer 2)
    # ----------------------------------------------------------

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """
        يحسب درجة BM25 لكل وثيقة بالنسبة للاستعلام.

        هذه هي الدالة الأساسية التي سيستخدمها Developer 2.

        المعاملات:
            query_tokens: قائمة tokens بعد المعالجة المسبقة
                          مثال: ["cloud", "storag", "sync"]

        الإرجاع:
            numpy array شكله (num_docs,) يحتوي score كل وثيقة

        مثال الاستخدام (Developer 2):
            tokens = ["cloud", "storag"]
            scores = indexer.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:10]
        """
        self._check_index_built()

        if not query_tokens:
            # استعلام فارغ → أصفار لكل الوثائق
            return np.zeros(len(self.documents))

        return self.bm25.get_scores(query_tokens)

    def get_top_n(
        self,
        query_tokens: List[str],
        n: int = 10,
    ) -> List[Tuple[IndexedDocument, float]]:
        """
        يُرجع أفضل N وثيقة مع درجاتها.

        دالة مساعدة جاهزة لـ Developer 2 — تجمع
        get_scores + argsort + get_document_by_index.

        الإرجاع:
            قائمة من (IndexedDocument, score) مرتبة تنازلياً
        """
        scores = self.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:n]

        results = []
        for idx in top_indices:
            doc = self.get_document_by_index(int(idx))
            if doc is not None and scores[idx] > 0:
                results.append((doc, float(scores[idx])))

        return results

    # ----------------------------------------------------------
    # دوال الوصول للوثائق (مماثلة لـ TFIDFIndexer)
    # ----------------------------------------------------------

    def is_built(self) -> bool:
        return self.bm25 is not None and len(self.documents) > 0

    def is_saved(self, dataset_name: str) -> bool:
        index_dir = self.indexes_dir / dataset_name / "bm25"
        required = [
            self._BM25_FILE,
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
    # دوال مساعدة خاصة
    # ----------------------------------------------------------

    def _preprocess_to_tokens(
        self,
        documents: List[Document],
        apply_stemming: bool,
        remove_stopwords: bool,
        language: str,
    ) -> Tuple[List[List[str]], List[IndexedDocument]]:
        """
        يعالج الوثائق ويُرجع قائمة tokens لكل وثيقة.

        الفرق عن TFIDFIndexer._preprocess_documents:
          - هنا نُرجع List[List[str]] (tokens منفصلة)
          - هناك نُرجع List[str] (نص واحد بعد join)

        BM25Okapi يحتاج tokens منفصلة ليحسب TF بدقة.
        """
        from services.preprocessing.preprocessor import get_preprocessor
        preprocessor = get_preprocessor()

        tokenized_docs: List[List[str]] = []
        indexed_docs: List[IndexedDocument] = []

        total = len(documents)
        report_every = max(1, total // 10)

        for i, doc in enumerate(documents):
            full_text = doc.get_full_text()
            tokens, _ = preprocessor.process(
                text=full_text,
                language=language,
                apply_stemming=apply_stemming,
                remove_stopwords=remove_stopwords,
            )

            # BM25Okapi تحتاج قائمة غير فارغة — نضع placeholder إذا فرغت
            if not tokens:
                tokens = ["__empty__"]

            tokenized_docs.append(tokens)
            indexed_docs.append(IndexedDocument(
                doc_id=doc.doc_id,
                original_text=doc.text,
                processed_text=" ".join(tokens),
                title=doc.title,
            ))

            if (i + 1) % report_every == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                print(f"[BM25Indexer]   المعالجة: {i+1:,}/{total:,} ({pct:.0f}%)", end="\r")

        print()
        return tokenized_docs, indexed_docs

    def _check_index_built(self) -> None:
        if not self.is_built():
            raise RuntimeError(
                "فهرس BM25 غير مبني. شغّل build_index() أو load_index() أولاً."
            )

    @staticmethod
    def _get_file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)


# =============================================================
# Singleton
# =============================================================

_bm25_instances: Dict[str, BM25Indexer] = {}


def get_bm25_indexer(dataset_name: Optional[str] = None) -> BM25Indexer:
    """
    يُرجع BM25Indexer — نسخة واحدة لكل dataset.
    يحمّل الفهرس تلقائياً إذا كان محفوظاً على القرص.
    """
    global _bm25_instances
    key = dataset_name or "__default__"

    if key not in _bm25_instances:
        indexer = BM25Indexer()
        if dataset_name and indexer.is_saved(dataset_name):
            indexer.load_index(dataset_name)
        _bm25_instances[key] = indexer

    return _bm25_instances[key]
