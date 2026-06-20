"""
services/indexing/tfidf_indexer.py
===================================
بناء فهرس TF-IDF وحفظه على القرص لاسترجاعه لاحقاً.

═══════════════════════════════════════════════════
المبدأ الأساسي: Offline Indexing
═══════════════════════════════════════════════════

هذا الملف يعمل في وضعين:

  وضع البناء (Offline - مرة واحدة):
    200,000 وثيقة
        ↓  DatasetLoader يقرأها
        ↓  PreprocessingService ينظّفها
        ↓  TfidfVectorizer يبني المصفوفة
        ↓  نحفظ على القرص
    النتيجة: ملفات جاهزة للبحث

  وضع البحث (Online - عند كل استعلام):
    Developer 2 يحمّل الفهرس من القرص
        ↓  يحوّل الاستعلام لمتجه
        ↓  يحسب Cosine Similarity
        ↓  يُرجع أفضل N نتيجة
    الوقت: أجزاء من الثانية

═══════════════════════════════════════════════════
الرياضيات: TF-IDF + Cosine Similarity
═══════════════════════════════════════════════════

TF (Term Frequency):
  TF(t,d) = count(t in d) / total_terms(d)
  كلما ظهرت الكلمة أكثر في الوثيقة، كلما زاد وزنها

IDF (Inverse Document Frequency):
  IDF(t) = log( (1 + N) / (1 + df(t)) ) + 1   ← صيغة sklearn
  حيث N = عدد الوثائق، df(t) = عدد الوثائق التي تحتوي t
  الكلمات النادرة = IDF عالي = أهم للتمييز

TF-IDF:
  weight(t,d) = TF(t,d) × IDF(t)

L2 Normalization (sklearn يطبّقها تلقائياً):
  vector_d = weight_vector / ‖weight_vector‖₂
  يجعل كل المتجهات بنفس الطول → Cosine = dot product فقط

Cosine Similarity:
  sim(q,d) = q⃗ · d⃗   (بعد التطبيع)
  القيمة بين 0 و 1، كلما اقتربت من 1 كلما كانت الوثيقة أكثر صلة
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import spmatrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# نضيف جذر المشروع لـ Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.constants import (
    INDEXES_DIR,
    DEFAULT_APPLY_STEMMING,
    DEFAULT_REMOVE_STOPWORDS,
    DEFAULT_LANGUAGE,
)
from services.indexing.dataset_loader import DatasetLoader, Document, get_dataset_loader


# =============================================================
# نماذج البيانات الداخلية
# =============================================================

@dataclass
class IndexedDocument:
    """
    وثيقة واحدة بعد معالجتها وتجهيزها للفهرس.

    نخزّن:
    - doc_id     : للإرجاع في نتائج البحث
    - original   : النص الأصلي للعرض في الواجهة
    - processed  : النص المعالج الذي بُني منه الفهرس
    - title      : للعرض في نتائج البحث

    لماذا نحفظ original وprocessed معاً؟
    الفهرس يُبنى من processed لكن المستخدم يرى original.
    """
    doc_id: str
    original_text: str
    processed_text: str
    title: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "IndexedDocument":
        return cls(**data)


@dataclass
class TFIDFIndexMetadata:
    """
    بيانات وصفية تُحفظ مع الفهرس.

    لماذا نحفظ metadata؟
    عند تحميل الفهرس، نحتاج أن نعرف:
    - بأي إعدادات بُني؟ (للتطابق مع إعدادات معالجة الاستعلام)
    - كم وثيقة فيه؟ (للتحقق من صحة التحميل)
    - متى بُني؟ (لمعرفة إذا يحتاج إعادة بناء)
    """
    dataset_name: str
    num_documents: int
    vocab_size: int
    build_time_seconds: float
    apply_stemming: bool
    remove_stopwords: bool
    language: str
    build_timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TFIDFIndexMetadata":
        return cls(**data)


# =============================================================
# TFIDFIndexer — الكلاس الرئيسي
# =============================================================

class TFIDFIndexer:
    """
    يبني فهرس TF-IDF ويحفظه ويحمّله.

    مسؤوليته:
      ✅ بناء الفهرس (build_index)
      ✅ حفظه على القرص (save_index)
      ✅ تحميله من القرص (load_index)
      ✅ تجهيز هياكل البيانات للبحث

    ليس مسؤوليته:
      ❌ تنفيذ البحث الفعلي (هذا عمل Developer 2)
      ❌ استقبال HTTP requests (هذا عمل app.py)
      ❌ تحميل البيانات الخام (هذا عمل DatasetLoader)

    كيف يستخدمه Developer 2؟
      indexer = TFIDFIndexer()
      indexer.load_index("dataset1")
      # ثم يستخدم:
      indexer.vectorizer      → لتحويل الاستعلام لمتجه
      indexer.tfidf_matrix    → للمقارنة بالوثائق
      indexer.documents       → للحصول على نص الوثيقة بالـ ID
    """

    # أسماء ملفات الحفظ — ثوابت لمنع الأخطاء الإملائية
    _VECTORIZER_FILE  = "tfidf_vectorizer.pkl"
    _MATRIX_FILE      = "tfidf_matrix.npz"
    _DOCUMENTS_FILE   = "tfidf_documents.json"
    _METADATA_FILE    = "tfidf_metadata.json"
    _DOCID_MAP_FILE   = "tfidf_docid_map.json"

    def __init__(
        self,
        indexes_dir: str = INDEXES_DIR,
        dataset_loader: Optional[DatasetLoader] = None,
    ) -> None:
        """
        المعاملات:
            indexes_dir    : المجلد الذي تُحفظ فيه الفهارس
            dataset_loader : اختياري — يُمرَّر في الاختبارات (Dependency Injection)
        """
        self.indexes_dir    = Path(indexes_dir)
        self._loader        = dataset_loader or get_dataset_loader()

        # =================================================
        # حالة الفهرس (تُملأ عند البناء أو التحميل)
        # =================================================

        # TfidfVectorizer من sklearn — يحفظ المفردة (vocabulary) وأوزان IDF
        self.vectorizer: Optional[TfidfVectorizer] = None

        # مصفوفة TF-IDF الرئيسية: شكلها (num_docs × vocab_size)
        # sparse matrix لأن معظم القيم = 0 (كل وثيقة تحتوي جزءاً صغيراً من المفردة)
        # مثال: 200,000 وثيقة × 50,000 مصطلح = 10 مليار خلية
        # منها 99%+ أصفار → sparse تختصرها لأقل من 1% من الحجم الكامل
        self.tfidf_matrix: Optional[spmatrix] = None

        # قائمة الوثائق بترتيب صفوف المصفوفة
        # الصف 0 في المصفوفة = documents[0]
        self.documents: List[IndexedDocument] = []

        # خريطة من doc_id → رقم الصف في المصفوفة
        # مثال: {"d1": 0, "d2": 1, ...}
        # تُستخدم للبحث السريع O(1) بدلاً من O(n)
        self.doc_id_to_idx: Dict[str, int] = {}

        # بيانات وصفية عن الفهرس
        self.metadata: Optional[TFIDFIndexMetadata] = None

    # ----------------------------------------------------------
    # الدالة الرئيسية: بناء الفهرس
    # ----------------------------------------------------------

    def build_index(
        self,
        dataset_name: str,
        apply_stemming: bool = DEFAULT_APPLY_STEMMING,
        remove_stopwords: bool = DEFAULT_REMOVE_STOPWORDS,
        language: str = DEFAULT_LANGUAGE,
        max_docs: Optional[int] = None,
        max_features: int = 100_000,
    ) -> TFIDFIndexMetadata:
        """
        يبني فهرس TF-IDF كاملاً من مجموعة بيانات.

        الخطوات داخلياً:
          1. تحميل الوثائق من القرص (DatasetLoader)
          2. معالجة كل وثيقة (Preprocessing مُدمج)
          3. تدريب TfidfVectorizer على كل الوثائق
          4. بناء مصفوفة TF-IDF
          5. حفظ كل شيء في self.*

        المعاملات:
            dataset_name   : اسم مجموعة البيانات في data/datasets/
            apply_stemming : تطبيق stemming على الوثائق
            remove_stopwords: حذف stopwords
            language       : لغة الوثائق
            max_docs       : للاختبار — يحدد الحد الأقصى
            max_features   : أقصى حجم للمفردة (vocabulary)
                             يمنع memory overflow على الـ datasets الكبيرة

        الإرجاع:
            TFIDFIndexMetadata : معلومات عن الفهرس المبني
        """
        print(f"\n{'='*55}")
        print(f"[TFIDFIndexer] بدء بناء الفهرس: '{dataset_name}'")
        print(f"{'='*55}")
        start_time = time.time()

        # ── الخطوة 1: تحميل الوثائق ──────────────────────────
        print("[TFIDFIndexer] الخطوة 1/4: تحميل الوثائق...")
        raw_documents = self._loader.load_all(dataset_name, max_docs=max_docs)

        if not raw_documents:
            raise ValueError(
                f"مجموعة البيانات '{dataset_name}' فارغة أو غير موجودة."
            )
        print(f"[TFIDFIndexer]   ✓ حُمِّل {len(raw_documents):,} وثيقة")

        # ── الخطوة 2: المعالجة المسبقة ───────────────────────
        print("[TFIDFIndexer] الخطوة 2/4: المعالجة المسبقة...")

        # نستخدم PreprocessingService محلياً (بدون HTTP)
        # لأن الفهرسة Offline ولا تحتاج network overhead
        preprocessed_texts, indexed_docs = self._preprocess_documents(
            raw_documents,
            apply_stemming=apply_stemming,
            remove_stopwords=remove_stopwords,
            language=language,
        )
        print(f"[TFIDFIndexer]   ✓ عُولجت {len(preprocessed_texts):,} وثيقة")

        # ── الخطوة 3: بناء TF-IDF Matrix ──────────────────────
        print("[TFIDFIndexer] الخطوة 3/4: بناء TF-IDF Matrix...")
        print(f"[TFIDFIndexer]   max_features = {max_features:,}")

        # TfidfVectorizer إعدادات مدروسة:
        self.vectorizer = TfidfVectorizer(
            # max_features: نأخذ أكثر N كلمة شيوعاً فقط
            # يمنع memory overflow ويُسرّع الحساب
            max_features=max_features,

            # sublinear_tf=True: يستخدم log(1+TF) بدلاً من TF
            # يُقلّل تأثير التكرار المبالغ فيه
            # مثال: كلمة ظهرت 100 مرة ليست أهم بـ100× من كلمة ظهرت مرة
            sublinear_tf=True,

            # norm='l2': يُطبّق L2 normalization على كل متجه وثيقة
            # يجعل Cosine Similarity = dot product فقط (أسرع حسابياً)
            norm="l2",

            # analyzer='word': يعمل على مستوى الكلمات
            analyzer="word",

            # token_pattern: regex لاستخراج التوكنز
            # r"(?u)\b\w+\b" يستخرج كلمات من حرف أو أكثر
            # ⚠️ مهم: نحن سبق أن عالجنا النص بـ preprocessing
            # هذا الـ pattern يستخرج التوكنز فقط دون معالجة إضافية
            token_pattern=r"(?u)\b\w+\b",

            # min_df=1: نأخذ كل كلمة ظهرت في وثيقة واحدة على الأقل
            # في المشاريع الكبيرة نرفع هذا الرقم لحذف الأخطاء الإملائية
            min_df=1,

            # max_df=0.95: نتجاهل الكلمات الموجودة في أكثر من 95% من الوثائق
            # هذه الكلمات شائعة جداً ولا تفيد في التمييز
            max_df=0.95,
        )

        # fit_transform: خطوتان في واحدة
        # fit      : يتعلم المفردة وأوزان IDF من كل الوثائق
        # transform: يُحوّل كل وثيقة لمتجه TF-IDF
        # النتيجة: sparse matrix شكلها (num_docs, vocab_size)
        self.tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)

        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"[TFIDFIndexer]   ✓ حجم المصفوفة: "
              f"{self.tfidf_matrix.shape[0]:,} × {self.tfidf_matrix.shape[1]:,}")
        print(f"[TFIDFIndexer]   ✓ حجم المفردة: {vocab_size:,} مصطلح")

        # نحسب الكثافة: نسبة القيم غير الصفرية
        # في IR عادةً أقل من 1% — لهذا sparse matrix مفيدة جداً
        total_cells = self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]
        nonzero = self.tfidf_matrix.nnz
        density = (nonzero / total_cells) * 100 if total_cells > 0 else 0
        print(f"[TFIDFIndexer]   ✓ الكثافة: {density:.4f}% "
              f"(sparse - معظمها أصفار كما هو متوقع)")

        # ── الخطوة 4: بناء هياكل البيانات المساعدة ──────────
        print("[TFIDFIndexer] الخطوة 4/4: بناء هياكل البيانات...")

        self.documents = indexed_docs

        # خريطة doc_id → index للبحث O(1)
        self.doc_id_to_idx = {
            doc.doc_id: idx for idx, doc in enumerate(self.documents)
        }

        # ── حساب وقت البناء ──────────────────────────────────
        build_time = time.time() - start_time

        import datetime
        self.metadata = TFIDFIndexMetadata(
            dataset_name=dataset_name,
            num_documents=len(self.documents),
            vocab_size=vocab_size,
            build_time_seconds=round(build_time, 2),
            apply_stemming=apply_stemming,
            remove_stopwords=remove_stopwords,
            language=language,
            build_timestamp=datetime.datetime.now().isoformat(),
        )

        print(f"\n[TFIDFIndexer] ✅ اكتمل البناء في {build_time:.2f} ثانية")
        print(f"{'='*55}\n")

        return self.metadata

    # ----------------------------------------------------------
    # حفظ الفهرس على القرص
    # ----------------------------------------------------------

    def save_index(self, dataset_name: str) -> Path:
        """
        يحفظ الفهرس على القرص في 4 ملفات:

          tfidf_vectorizer.pkl  ← الـ vectorizer (المفردة + أوزان IDF)
          tfidf_matrix.npz      ← المصفوفة sparse (بصيغة numpy compressed)
          tfidf_documents.json  ← بيانات الوثائق
          tfidf_metadata.json   ← بيانات وصفية
          tfidf_docid_map.json  ← خريطة doc_id → index

        لماذا ملفات منفصلة وليس ملف واحد؟
          - المصفوفة .npz لها صيغة خاصة محسّنة للـ sparse matrices
          - الـ vectorizer يحتاج pickle لأنه كائن Python معقد
          - الوثائق JSON لأنها قابلة للقراءة ومستقلة

        لماذا لا نستخدم pickle لكل شيء؟
          pickle غير آمن من الناحية الأمنية ويُنتج ملفات كبيرة.
          نستخدمه فقط للـ vectorizer لأنه لا يوجد بديل أفضل.

        الإرجاع:
            Path: مسار مجلد الفهرس
        """
        self._check_index_built()

        # إنشاء مجلد خاص بهذا الـ dataset
        index_dir = self.indexes_dir / dataset_name / "tfidf"
        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"[TFIDFIndexer] حفظ الفهرس في: {index_dir}")

        # 1. حفظ الـ vectorizer
        vectorizer_path = index_dir / self._VECTORIZER_FILE
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"[TFIDFIndexer]   ✓ vectorizer: {self._get_file_size_mb(vectorizer_path):.2f} MB")

        # 2. حفظ المصفوفة (صيغة npz لـ sparse matrices)
        matrix_path = index_dir / self._MATRIX_FILE
        save_npz(str(matrix_path), self.tfidf_matrix)
        print(f"[TFIDFIndexer]   ✓ matrix: {self._get_file_size_mb(matrix_path):.2f} MB")

        # 3. حفظ الوثائق (JSON)
        docs_path = index_dir / self._DOCUMENTS_FILE
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(
                [doc.to_dict() for doc in self.documents],
                f,
                ensure_ascii=False,
                indent=None,   # بدون indent لتوفير المساحة
            )
        print(f"[TFIDFIndexer]   ✓ documents: {self._get_file_size_mb(docs_path):.2f} MB")

        # 4. حفظ الـ metadata
        meta_path = index_dir / self._METADATA_FILE
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, ensure_ascii=False, indent=2)

        # 5. حفظ خريطة doc_id → index
        map_path = index_dir / self._DOCID_MAP_FILE
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_id_to_idx, f, ensure_ascii=False)

        print(f"[TFIDFIndexer] ✅ حُفظ الفهرس بنجاح في: {index_dir}")
        return index_dir

    # ----------------------------------------------------------
    # تحميل الفهرس من القرص
    # ----------------------------------------------------------

    def load_index(self, dataset_name: str) -> TFIDFIndexMetadata:
        """
        يحمّل الفهرس المحفوظ من القرص.

        يُستخدم من:
          - هذا الكلاس نفسه قبل البحث
          - Developer 2 في Retrieval Service

        الإرجاع:
            TFIDFIndexMetadata: معلومات الفهرس المحمّل
        """
        index_dir = self.indexes_dir / dataset_name / "tfidf"

        if not index_dir.exists():
            raise FileNotFoundError(
                f"الفهرس غير موجود: {index_dir}\n"
                f"شغّل build_index('{dataset_name}') أولاً."
            )

        print(f"[TFIDFIndexer] تحميل الفهرس من: {index_dir}")
        start_time = time.time()

        # 1. تحميل الـ vectorizer
        with open(index_dir / self._VECTORIZER_FILE, "rb") as f:
            self.vectorizer = pickle.load(f)
        print(f"[TFIDFIndexer]   ✓ vectorizer محمّل "
              f"(مفردة: {len(self.vectorizer.vocabulary_):,} مصطلح)")

        # 2. تحميل المصفوفة
        self.tfidf_matrix = load_npz(str(index_dir / self._MATRIX_FILE))
        print(f"[TFIDFIndexer]   ✓ matrix محمّلة "
              f"{self.tfidf_matrix.shape[0]:,} × {self.tfidf_matrix.shape[1]:,}")

        # 3. تحميل الوثائق
        with open(index_dir / self._DOCUMENTS_FILE, encoding="utf-8") as f:
            docs_data = json.load(f)
        self.documents = [IndexedDocument.from_dict(d) for d in docs_data]
        print(f"[TFIDFIndexer]   ✓ {len(self.documents):,} وثيقة محمّلة")

        # 4. تحميل الـ metadata
        with open(index_dir / self._METADATA_FILE, encoding="utf-8") as f:
            self.metadata = TFIDFIndexMetadata.from_dict(json.load(f))

        # 5. تحميل خريطة doc_id → index
        with open(index_dir / self._DOCID_MAP_FILE, encoding="utf-8") as f:
            self.doc_id_to_idx = json.load(f)

        load_time = time.time() - start_time
        print(f"[TFIDFIndexer] ✅ تحميل مكتمل في {load_time:.3f} ثانية")

        return self.metadata

    # ----------------------------------------------------------
    # التحقق من حالة الفهرس
    # ----------------------------------------------------------

    def is_built(self) -> bool:
        """هل الفهرس مبني في الذاكرة؟"""
        return (
            self.vectorizer is not None
            and self.tfidf_matrix is not None
            and len(self.documents) > 0
        )

    def is_saved(self, dataset_name: str) -> bool:
        """هل الفهرس محفوظ على القرص؟"""
        index_dir = self.indexes_dir / dataset_name / "tfidf"
        required_files = [
            self._VECTORIZER_FILE,
            self._MATRIX_FILE,
            self._DOCUMENTS_FILE,
            self._METADATA_FILE,
        ]
        return all((index_dir / f).exists() for f in required_files)

    def get_document_by_id(self, doc_id: str) -> Optional[IndexedDocument]:
        """
        يُرجع وثيقة بالـ ID.
        يُستخدم من Developer 2 بعد تحديد أفضل النتائج.
        """
        idx = self.doc_id_to_idx.get(doc_id)
        if idx is None:
            return None
        return self.documents[idx]

    def get_document_by_index(self, idx: int) -> Optional[IndexedDocument]:
        """
        يُرجع وثيقة برقم صفها في المصفوفة.
        يُستخدم من Developer 2 بعد حساب Cosine Similarity.
        """
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        return None

    def transform_query(self, processed_query: str) -> Optional[object]:
        """
        يُحوّل استعلاماً معالجاً إلى متجه TF-IDF.

        ⚠️ مهم جداً: processed_query يجب أن يكون معالجاً
        بنفس الإعدادات التي بُني بها الفهرس.

        هذه الدالة مُقدَّمة لـ Developer 2 ليستخدمها مباشرة.

        المعاملات:
            processed_query: نص الاستعلام بعد المعالجة المسبقة

        الإرجاع:
            sparse matrix شكلها (1, vocab_size)
        """
        self._check_index_built()
        if not processed_query.strip():
            return None
        # transform فقط (ليس fit_transform) — نستخدم المفردة المبنية مسبقاً
        return self.vectorizer.transform([processed_query])

    # ----------------------------------------------------------
    # دوال مساعدة خاصة
    # ----------------------------------------------------------

    def _preprocess_documents(
        self,
        documents: List[Document],
        apply_stemming: bool,
        remove_stopwords: bool,
        language: str,
    ) -> Tuple[List[str], List[IndexedDocument]]:
        """
        يعالج الوثائق مسبقاً باستخدام TextPreprocessor محلياً.

        لماذا محلياً وليس عبر HTTP؟
        الفهرسة تحدث Offline — لا يوجد HTTP server يعمل في هذا الوقت.
        نستخدم نفس كود preprocessing مباشرةً لتجنب network overhead.

        ⚠️ يجب استخدام نفس الإعدادات عند معالجة الاستعلامات لاحقاً.
        الإعدادات محفوظة في metadata لهذا الغرض بالضبط.
        """
        # نستورد هنا لتجنب circular imports
        from services.preprocessing.preprocessor import get_preprocessor

        preprocessor = get_preprocessor()
        preprocessed_texts: List[str] = []
        indexed_docs: List[IndexedDocument] = []

        total = len(documents)
        report_every = max(1, total // 10)  # نطبع تقرير كل 10%

        for i, doc in enumerate(documents):

            # نحصل على النص الكامل (عنوان + محتوى)
            full_text = doc.get_full_text()

            # نطبّق المعالجة المسبقة
            tokens, _ = preprocessor.process(
                text=full_text,
                language=language,
                apply_stemming=apply_stemming,
                remove_stopwords=remove_stopwords,
            )

            processed_text = " ".join(tokens)
            preprocessed_texts.append(processed_text)

            indexed_docs.append(IndexedDocument(
                doc_id=doc.doc_id,
                original_text=doc.text,
                processed_text=processed_text,
                title=doc.title,
            ))

            # طباعة التقدم
            if (i + 1) % report_every == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                print(f"[TFIDFIndexer]   المعالجة: {i+1:,}/{total:,} "
                      f"({pct:.0f}%)", end="\r")

        print()  # سطر جديد بعد \r
        return preprocessed_texts, indexed_docs

    def _check_index_built(self) -> None:
        """يتحقق من أن الفهرس مبني قبل أي عملية تعتمد عليه."""
        if not self.is_built():
            raise RuntimeError(
                "الفهرس غير مبني. شغّل build_index() أو load_index() أولاً."
            )

    @staticmethod
    def _get_file_size_mb(path: Path) -> float:
        """يُرجع حجم ملف بالميغابايت."""
        return path.stat().st_size / (1024 * 1024)


# =============================================================
# Singleton + Factory
# =============================================================

_indexer_instances: Dict[str, TFIDFIndexer] = {}


def get_tfidf_indexer(dataset_name: Optional[str] = None) -> TFIDFIndexer:
    """
    يُرجع TFIDFIndexer — نسخة واحدة لكل dataset.

    إذا مرّرت dataset_name وكان الفهرس محفوظاً على القرص،
    يُحمَّل تلقائياً.

    يُستخدم كـ Dependency في FastAPI وفي Retrieval Service.
    """
    global _indexer_instances

    key = dataset_name or "__default__"

    if key not in _indexer_instances:
        indexer = TFIDFIndexer()
        if dataset_name and indexer.is_saved(dataset_name):
            indexer.load_index(dataset_name)
        _indexer_instances[key] = indexer

    return _indexer_instances[key]
