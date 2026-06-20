"""
services/indexing/tests/test_embedding_indexer.py
==================================================
اختبارات وحدة شاملة لـ EmbeddingIndexer.

════════════════════════════════════════════════════
استراتيجية الاختبار: لماذا Mock؟
════════════════════════════════════════════════════

تحميل SentenceTransformer يأخذ ~5 ثوانٍ ويحتاج ~400MB.
في بيئة CI/CD أو على جهاز بموارد محدودة هذا مشكلة.

الحل: نستبدل النموذج الحقيقي بـ Mock يُرجع
متجهات عشوائية ثابتة. هذا يختبر:
  ✅ منطق بناء الفهرس
  ✅ حفظ وتحميل FAISS
  ✅ get_top_k و get_document_by_id
  ✅ encode_query interface

وهو لا يختبر (لأنه Mock):
  ❌ جودة الـ embeddings الفعلية
  ❌ هل النموذج يفهم المعنى حقاً

الاختبار الأخير (TestIntegration) يُعلّق على نفسه
ويُوضح كيف تختبر مع النموذج الحقيقي.

تشغيل على Windows PowerShell:
    cd ir_system_2026
    python -m pytest services/indexing/tests/test_embedding_indexer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.indexing.dataset_loader import DatasetLoader, Document
from services.indexing.embedding_indexer import (
    EmbeddingIndexer,
    EmbeddingIndexMetadata,
    DEFAULT_MODEL_NAME,
)
from services.indexing.tfidf_indexer import IndexedDocument


# =============================================================
# بيانات تجريبية
# =============================================================

SAMPLE_DOCUMENTS = [
    Document("d1", "Cloud Storage",
             "Cloud storage is useful for syncing files across devices."),
    Document("d2", "AI Assistants",
             "AI assistants like Siri use voice recognition."),
    Document("d3", "Information Retrieval",
             "Information retrieval finds relevant documents using queries."),
    Document("d4", "Python Programming",
             "Python is used for machine learning and data science."),
    Document("d5", "BM25 Ranking",
             "BM25 ranks documents using term frequency normalization."),
]

EMBEDDING_DIM = 384  # نفس بُعد all-MiniLM-L6-v2


def _make_mock_loader(docs: List[Document] | None = None) -> MagicMock:
    mock = MagicMock(spec=DatasetLoader)
    mock.load_all.return_value = docs if docs is not None else SAMPLE_DOCUMENTS
    return mock


def _make_deterministic_embeddings(n: int, dim: int) -> np.ndarray:
    """
    يُنشئ متجهات ثابتة وقابلة للتكرار للاختبارات.

    لماذا ثابتة؟ الاختبارات يجب أن تُنتج نفس النتيجة
    في كل تشغيل — np.random.seed يضمن ذلك.

    لماذا مُطبَّعة؟ لأن FAISS IndexFlatIP يعمل صحيحاً
    فقط مع متجهات unit length (بعد L2 normalization).
    """
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # L2 normalization
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32)


def _make_mock_model(n_docs: int = 5, dim: int = EMBEDDING_DIM) -> MagicMock:
    """
    يُنشئ SentenceTransformer mock يُرجع متجهات ثابتة.

    get_sentence_embedding_dimension() يُرجع البُعد.
    encode() يُرجع متجهات ثابتة بغض النظر عن المدخلات.
    """
    embeddings = _make_deterministic_embeddings(n_docs, dim)
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = dim
    # encode يُرجع أول n_docs متجهاً بغض النظر عن المدخلات
    mock.encode.return_value = embeddings
    return mock


# =============================================================
# Fixtures
# =============================================================

@pytest.fixture
def tmp_indexes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "indexes"
    d.mkdir()
    return d


@pytest.fixture
def indexer(tmp_indexes_dir: Path) -> EmbeddingIndexer:
    """EmbeddingIndexer بدون تحميل نموذج حقيقي."""
    idx = EmbeddingIndexer(
        indexes_dir=str(tmp_indexes_dir),
        dataset_loader=_make_mock_loader(),
    )
    # نُعيد تعريف _get_model لتُرجع Mock بدلاً من النموذج الحقيقي
    idx._get_model = lambda: _make_mock_model(len(SAMPLE_DOCUMENTS))
    return idx


@pytest.fixture
def built_indexer(tmp_indexes_dir: Path) -> EmbeddingIndexer:
    """EmbeddingIndexer مع FAISS index مبني."""
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss-cpu غير مثبتة — تخطي الاختبارات")

    idx = EmbeddingIndexer(
        indexes_dir=str(tmp_indexes_dir),
        dataset_loader=_make_mock_loader(),
    )
    idx._get_model = lambda: _make_mock_model(len(SAMPLE_DOCUMENTS))
    idx.build_index("test_dataset")
    return idx


@pytest.fixture
def saved_indexer(built_indexer: EmbeddingIndexer) -> EmbeddingIndexer:
    """EmbeddingIndexer مبني ومحفوظ."""
    built_indexer.save_index("test_dataset")
    return built_indexer


# =============================================================
# اختبارات build_index
# =============================================================

class TestBuildIndex:
    """
    يختبر: هل build_index يبني الفهرس بشكل صحيح؟
    لماذا مهم: هو العملية الأطول والأكثر تعقيداً.
    """

    def test_requires_faiss(self, tmp_indexes_dir: Path) -> None:
        """
        ماذا يختبر: عند غياب faiss يُطرح ImportError واضح.
        لماذا مهم: يمنع أخطاء غامضة عند نسيان التثبيت.
        """
        idx = EmbeddingIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        idx._get_model = lambda: _make_mock_model(len(SAMPLE_DOCUMENTS))
        with patch.dict("sys.modules", {"faiss": None}):
            with pytest.raises((ImportError, TypeError)):
                idx.build_index("test")

    def test_returns_metadata(self, built_indexer: EmbeddingIndexer) -> None:
        """
        ماذا يختبر: build_index يُرجع EmbeddingIndexMetadata.
        لماذا مهم: Developer 2 يعتمد على هذا الكائن.
        """
        assert isinstance(built_indexer.metadata, EmbeddingIndexMetadata)

    def test_correct_document_count(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: عدد الوثائق في الفهرس يطابق المدخلات.
        لماذا مهم: أي فقدان في البيانات يُفسد نتائج البحث.
        """
        assert built_indexer.metadata.num_documents == len(SAMPLE_DOCUMENTS)

    def test_faiss_index_populated(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: FAISS index يحتوي نفس عدد الوثائق.
        لماذا مهم: إذا FAISS.ntotal ≠ len(docs) → بحث خاطئ.
        """
        assert built_indexer.faiss_index.ntotal == len(SAMPLE_DOCUMENTS)

    def test_embeddings_shape(self, built_indexer: EmbeddingIndexer) -> None:
        """
        ماذا يختبر: شكل مصفوفة المتجهات (N, dim).
        لماذا مهم: شكل خاطئ → FAISS search يفشل.
        """
        shape = built_indexer.embeddings.shape
        assert shape[0] == len(SAMPLE_DOCUMENTS)
        assert shape[1] == EMBEDDING_DIM

    def test_documents_list_populated(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: self.documents تُملأ بعد البناء.
        لماذا مهم: get_document_by_index تعتمد عليها.
        """
        assert len(built_indexer.documents) == len(SAMPLE_DOCUMENTS)
        assert all(isinstance(d, IndexedDocument) for d in built_indexer.documents)

    def test_docid_map_complete(self, built_indexer: EmbeddingIndexer) -> None:
        """
        ماذا يختبر: كل doc_id موجود في الخريطة.
        لماذا مهم: get_document_by_id يعتمد على هذه الخريطة.
        """
        for doc in SAMPLE_DOCUMENTS:
            assert doc.doc_id in built_indexer.doc_id_to_idx

    def test_metadata_stores_model_name(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: اسم النموذج محفوظ في metadata.
        لماذا مهم: Developer 2 يحتاجه لتحميل نفس النموذج.
        """
        assert built_indexer.metadata.model_name == DEFAULT_MODEL_NAME

    def test_metadata_stores_dim(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: البُعد محفوظ في metadata.
        لماذا مهم: للتحقق من توافق النموذج عند التحميل.
        """
        assert built_indexer.metadata.embedding_dim == EMBEDDING_DIM

    def test_is_built_true_after_build(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        assert built_indexer.is_built() is True

    def test_is_built_false_before_build(
        self, indexer: EmbeddingIndexer
    ) -> None:
        assert indexer.is_built() is False

    def test_empty_dataset_raises(self, tmp_indexes_dir: Path) -> None:
        """
        ماذا يختبر: dataset فارغ يُطرح ValueError.
        لماذا مهم: يمنع بناء فهرس فارغ بدون تحذير.
        """
        idx = EmbeddingIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader([]),
        )
        idx._get_model = lambda: _make_mock_model(0)
        with pytest.raises(ValueError, match="فارغة"):
            idx.build_index("empty")


# =============================================================
# اختبارات save و load
# =============================================================

class TestSaveAndLoad:
    """
    يختبر: دورة الحفظ والتحميل الكاملة.
    لماذا مهم: هذا هو السيناريو الفعلي — Dev1 يبني، Dev2 يحمّل.
    """

    def test_save_creates_required_files(
        self, saved_indexer: EmbeddingIndexer, tmp_indexes_dir: Path
    ) -> None:
        """
        ماذا يختبر: الحفظ يُنشئ كل الملفات المطلوبة.
        ما الخطأ الذي يمنعه: Developer 2 يحمّل فيجد ملف مفقود.
        """
        index_dir = tmp_indexes_dir / "test_dataset" / "embedding"
        for fname in [
            "embedding_index.faiss",
            "embedding_vectors.npy",
            "embedding_documents.json",
            "embedding_metadata.json",
            "embedding_docid_map.json",
        ]:
            assert (index_dir / fname).exists(), f"مفقود: {fname}"

    def test_is_saved_true_after_save(
        self, saved_indexer: EmbeddingIndexer
    ) -> None:
        assert saved_indexer.is_saved("test_dataset") is True

    def test_is_saved_false_before_save(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        assert built_indexer.is_saved("test_dataset") is False

    def test_load_restores_faiss_count(
        self, saved_indexer: EmbeddingIndexer, tmp_indexes_dir: Path
    ) -> None:
        """
        ماذا يختبر: FAISS index بعد التحميل يحتوي نفس عدد المتجهات.
        ما الخطأ الذي يمنعه: فهرس تالف يُرجع نتائج خاطئة.
        """
        new_idx = EmbeddingIndexer(indexes_dir=str(tmp_indexes_dir))
        new_idx.load_index("test_dataset")
        assert new_idx.faiss_index.ntotal == len(SAMPLE_DOCUMENTS)

    def test_load_restores_embeddings_shape(
        self, saved_indexer: EmbeddingIndexer, tmp_indexes_dir: Path
    ) -> None:
        """
        ماذا يختبر: شكل المتجهات بعد التحميل.
        ما الخطأ الذي يمنعه: dim مختلف → FAISS search يفشل.
        """
        original_shape = saved_indexer.embeddings.shape
        new_idx = EmbeddingIndexer(indexes_dir=str(tmp_indexes_dir))
        new_idx.load_index("test_dataset")
        assert new_idx.embeddings.shape == original_shape

    def test_load_restores_documents(
        self, saved_indexer: EmbeddingIndexer, tmp_indexes_dir: Path
    ) -> None:
        new_idx = EmbeddingIndexer(indexes_dir=str(tmp_indexes_dir))
        new_idx.load_index("test_dataset")
        assert len(new_idx.documents) == len(SAMPLE_DOCUMENTS)

    def test_load_restores_metadata(
        self, saved_indexer: EmbeddingIndexer, tmp_indexes_dir: Path
    ) -> None:
        """
        ماذا يختبر: metadata محفوظة بدقة.
        ما الخطأ الذي يمنعه: Developer 2 لا يعرف أي نموذج يستخدم.
        """
        original = saved_indexer.metadata
        new_idx = EmbeddingIndexer(indexes_dir=str(tmp_indexes_dir))
        loaded = new_idx.load_index("test_dataset")
        assert loaded.model_name    == original.model_name
        assert loaded.embedding_dim == original.embedding_dim
        assert loaded.num_documents == original.num_documents

    def test_load_nonexistent_raises(self, tmp_indexes_dir: Path) -> None:
        """
        ماذا يختبر: تحميل فهرس غير موجود → FileNotFoundError.
        ما الخطأ الذي يمنعه: أخطاء غامضة عند غياب البيانات.
        """
        idx = EmbeddingIndexer(indexes_dir=str(tmp_indexes_dir))
        with pytest.raises(FileNotFoundError):
            idx.load_index("nonexistent")

    def test_vectors_preserved_exactly(
        self, saved_indexer: EmbeddingIndexer, tmp_indexes_dir: Path
    ) -> None:
        """
        ماذا يختبر: المتجهات بعد الحفظ والتحميل متطابقة رياضياً.
        ما الخطأ الذي يمنعه: فقدان دقة float32 عند الحفظ.
        """
        original_vecs = saved_indexer.embeddings.copy()
        new_idx = EmbeddingIndexer(indexes_dir=str(tmp_indexes_dir))
        new_idx.load_index("test_dataset")
        np.testing.assert_array_almost_equal(
            original_vecs, new_idx.embeddings, decimal=6,
        )


# =============================================================
# اختبارات encode_query
# =============================================================

class TestEncodeQuery:
    """
    يختبر: encode_query يُنتج متجهات صحيحة.
    لماذا مهم: هذه هي نقطة البداية لكل عملية بحث.
    """

    def test_returns_numpy_array(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: encode_query يُرجع numpy array.
        ما الخطأ الذي يمنعه: FAISS يتوقع numpy — tensor يُفشله.
        """
        mock_vec = _make_deterministic_embeddings(1, EMBEDDING_DIM)
        built_indexer._get_model = lambda: MagicMock(
            encode=lambda *a, **kw: mock_vec,
            get_sentence_embedding_dimension=lambda: EMBEDDING_DIM,
        )
        result = built_indexer.encode_query("cloud storage")
        assert isinstance(result, np.ndarray)

    def test_returns_correct_shape(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: شكل متجه الاستعلام (1, dim).
        ما الخطأ الذي يمنعه: FAISS يحتاج (1, dim) وليس (dim,).
        """
        mock_vec = _make_deterministic_embeddings(1, EMBEDDING_DIM)
        built_indexer._get_model = lambda: MagicMock(
            encode=lambda *a, **kw: mock_vec,
            get_sentence_embedding_dimension=lambda: EMBEDDING_DIM,
        )
        result = built_indexer.encode_query("test query")
        assert result is not None
        assert result.ndim == 2
        assert result.shape[1] == EMBEDDING_DIM

    def test_empty_query_returns_none(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: استعلام فارغ يُرجع None.
        ما الخطأ الذي يمنعه: FAISS يتعطل مع متجه فارغ.
        """
        result = built_indexer.encode_query("   ")
        assert result is None

    def test_before_build_raises(self, indexer: EmbeddingIndexer) -> None:
        """
        ماذا يختبر: encode_query قبل البناء يحتاج النموذج.
        ما الخطأ الذي يمنعه: استدعاء عشوائي قبل التهيئة.
        """
        # encode_query لا تحتاج الفهرس — تحتاج فقط النموذج
        # هذا الاختبار يتأكد أن الاستدعاء لا يكسر شيئاً
        mock_vec = _make_deterministic_embeddings(1, EMBEDDING_DIM)
        indexer._get_model = lambda: MagicMock(
            encode=lambda *a, **kw: mock_vec,
        )
        result = indexer.encode_query("hello")
        assert result is not None


# =============================================================
# اختبارات get_top_k
# =============================================================

class TestGetTopK:
    """
    يختبر: get_top_k يُرجع نتائج صحيحة ومرتبة.
    لماذا مهم: هذا ما يرى المستخدم في النهاية.
    """

    def _make_query_vec(self) -> np.ndarray:
        return _make_deterministic_embeddings(1, EMBEDDING_DIM)

    def test_returns_list_of_tuples(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: get_top_k يُرجع List[(IndexedDocument, float)].
        ما الخطأ الذي يمنعه: Developer 2 يتوقع هذا الشكل بالضبط.
        """
        results = built_indexer.get_top_k(self._make_query_vec(), k=3)
        assert isinstance(results, list)
        for doc, score in results:
            assert isinstance(doc, IndexedDocument)
            assert isinstance(score, float)

    def test_returns_correct_count(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: عدد النتائج لا يتجاوز k.
        ما الخطأ الذي يمنعه: عرض نتائج أكثر مما طُلب.
        """
        results = built_indexer.get_top_k(self._make_query_vec(), k=3)
        assert len(results) <= 3

    def test_sorted_by_score_descending(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: النتائج مرتبة تنازلياً (الأكثر صلة أولاً).
        ما الخطأ الذي يمنعه: عرض نتائج بترتيب عشوائي.
        """
        results = built_indexer.get_top_k(self._make_query_vec(), k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_none_query_returns_empty(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: query_embedding = None → قائمة فارغة.
        ما الخطأ الذي يمنعه: FAISS يتعطل مع None.
        """
        results = built_indexer.get_top_k(None, k=5)
        assert results == []

    def test_k_larger_than_corpus(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: k أكبر من عدد الوثائق → يُرجع ما هو موجود.
        ما الخطأ الذي يمنعه: FAISS يتعطل أو يُرجع -1 indices.
        """
        results = built_indexer.get_top_k(self._make_query_vec(), k=1000)
        assert len(results) <= len(SAMPLE_DOCUMENTS)

    def test_1d_query_handled(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: متجه 1D يُحوَّل لـ 2D تلقائياً.
        ما الخطأ الذي يمنعه: FAISS يتوقع (1,dim) وليس (dim,).
        """
        flat_vec = _make_deterministic_embeddings(1, EMBEDDING_DIM)[0]
        assert flat_vec.ndim == 1
        results = built_indexer.get_top_k(flat_vec, k=3)
        assert isinstance(results, list)

    def test_before_build_raises(
        self, indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: get_top_k قبل البناء يُطرح RuntimeError.
        ما الخطأ الذي يمنعه: استخدام الفهرس قبل تهيئته.
        """
        with pytest.raises(RuntimeError, match="غير مبني"):
            indexer.get_top_k(self._make_query_vec())


# =============================================================
# اختبارات get_document_by_id و get_document_by_index
# =============================================================

class TestDocumentAccess:
    """
    يختبر: الوصول للوثائق بالمعرّف أو الرقم.
    لماذا مهم: هذا ما يستخدمه Developer 2 لعرض النتائج.
    """

    def test_get_by_id_returns_correct_doc(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: get_document_by_id يُرجع الوثيقة الصحيحة.

        ملاحظة هندسية مهمة:
        EmbeddingIndexer يخزّن الوثائق بدون Preprocessing.
        processed_text = get_full_text() = "title + text" كاملاً.
        title يبقى كما هو من المصدر الأصلي.
        نتحقق من doc_id وجزء من النص الأصلي.
        """
        doc = built_indexer.get_document_by_id("d1")
        assert doc is not None
        assert doc.doc_id == "d1"
        # النص الأصلي يحتوي "cloud storage"
        assert "cloud" in doc.original_text.lower()
        # processed_text = full_text (عنوان + نص)
        assert "cloud" in doc.processed_text.lower()

    def test_get_by_id_unknown_returns_none(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: معرّف غير موجود → None وليس خطأ.
        ما الخطأ الذي يمنعه: KeyError يكسر Retrieval Service.
        """
        assert built_indexer.get_document_by_id("nonexistent") is None

    def test_get_by_index_returns_doc(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        doc = built_indexer.get_document_by_index(0)
        assert doc is not None
        assert doc.doc_id == "d1"

    def test_get_by_index_out_of_range_returns_none(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: رقم خارج النطاق → None.
        ما الخطأ الذي يمنعه: IndexError من FAISS يكسر الـ response.
        """
        assert built_indexer.get_document_by_index(9999) is None

    def test_all_documents_accessible(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        for doc in SAMPLE_DOCUMENTS:
            result = built_indexer.get_document_by_id(doc.doc_id)
            assert result is not None, f"مفقود: {doc.doc_id}"


# =============================================================
# اختبارات get_stats
# =============================================================

class TestGetStats:

    def test_stats_before_build(self, indexer: EmbeddingIndexer) -> None:
        """
        ماذا يختبر: get_stats قبل البناء يُرجع status=not_built.
        ما الخطأ الذي يمنعه: Developer 3 يعرض حالة خاطئة في UI.
        """
        stats = indexer.get_stats()
        assert stats["status"] == "not_built"

    def test_stats_after_build(
        self, built_indexer: EmbeddingIndexer
    ) -> None:
        """
        ماذا يختبر: get_stats بعد البناء يُرجع معلومات كاملة.
        """
        stats = built_indexer.get_stats()
        assert stats["status"]        == "ready"
        assert stats["num_documents"] == len(SAMPLE_DOCUMENTS)
        assert stats["embedding_dim"] == EMBEDDING_DIM
        assert stats["model_name"]    == DEFAULT_MODEL_NAME


# =============================================================
# اختبار التكامل: Developer 2 Workflow
# =============================================================

class TestIntegration:
    """
    يحاكي بالضبط ما سيفعله Developer 2 في Retrieval Service.
    """

    def test_full_cycle_with_mock_model(
        self, tmp_indexes_dir: Path
    ) -> None:
        """
        ماذا يختبر: دورة كاملة build→save→load→search تعمل.
        لماذا مهم: يتحقق من أن كل الأجزاء تعمل معاً.

        ════════════════════════════════════════════════
        ملاحظة حول الـ Mock في هذا الاختبار:
        ════════════════════════════════════════════════
        نستخدم متجهات ثابتة — النموذج الحقيقي سيُنتج
        متجهات مختلفة بطبيعة الحال. الهدف هنا هو اختبار
        أن الـ pipeline يعمل، وليس جودة الـ embeddings.

        لاختبار الجودة الحقيقية، شغّل:
          indexer = EmbeddingIndexer()  # بدون mock
          indexer.build_index("dataset1")
          query = indexer.encode_query("cloud storage")
          results = indexer.get_top_k(query, k=3)
          assert results[0][0].doc_id == "d1"
        ════════════════════════════════════════════════
        """
        try:
            import faiss
        except ImportError:
            pytest.skip("faiss-cpu غير مثبتة")

        # === Developer 1 ===
        dev1 = EmbeddingIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        dev1._get_model = lambda: _make_mock_model(len(SAMPLE_DOCUMENTS))
        dev1.build_index("dataset1")
        dev1.save_index("dataset1")

        # === Developer 2 ===
        dev2 = EmbeddingIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        dev2.load_index("dataset1")

        # Developer 2 يُشفّر الاستعلام
        mock_query_vec = _make_deterministic_embeddings(1, EMBEDDING_DIM)
        results = dev2.get_top_k(mock_query_vec, k=3)

        assert len(results) > 0
        assert len(results) <= 3
        doc, score = results[0]
        assert isinstance(doc, IndexedDocument)
        assert 0.0 <= score <= 1.1  # cosine بعد normalization بين -1 و 1

    def test_scores_consistent_after_save_load(
        self, saved_indexer: EmbeddingIndexer, tmp_indexes_dir: Path
    ) -> None:
        """
        ماذا يختبر: نفس الاستعلام يُنتج نفس الـ scores قبل وبعد التحميل.
        ما الخطأ الذي يمنعه: تغيّر النتائج بعد restart الـ server.
        """
        query_vec = _make_deterministic_embeddings(1, EMBEDDING_DIM)

        # scores من الفهرس الأصلي
        original_results = saved_indexer.get_top_k(query_vec, k=5)
        original_scores = [s for _, s in original_results]

        # تحميل وبحث من جديد
        new_idx = EmbeddingIndexer(indexes_dir=str(tmp_indexes_dir))
        new_idx.load_index("test_dataset")
        loaded_results = new_idx.get_top_k(query_vec, k=5)
        loaded_scores = [s for _, s in loaded_results]

        np.testing.assert_array_almost_equal(
            original_scores, loaded_scores, decimal=5,
            err_msg="الـ scores يجب أن تتطابق بعد الحفظ والتحميل"
        )
