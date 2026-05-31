"""
services/indexing/tests/test_tfidf_indexer.py
==============================================
اختبارات وحدة شاملة لـ TFIDFIndexer.

تشغيل على Windows PowerShell:
    cd ir_system_2026
    python -m pytest services/indexing/tests/test_tfidf_indexer.py -v

ما الذي نختبره؟
  1. بناء الفهرس من بيانات حقيقية
  2. صحة شكل المصفوفة
  3. حفظ وتحميل الفهرس (دورة كاملة)
  4. transform_query يعمل بعد التحميل
  5. get_document_by_id و get_document_by_index
  6. معالجة حالات الخطأ
  7. الاتساق: نفس الاستعلام ينتج نفس المتجه
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

# ── إعداد Python path ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.indexing.dataset_loader import DatasetLoader, Document
from services.indexing.tfidf_indexer import (
    IndexedDocument,
    TFIDFIndexer,
    TFIDFIndexMetadata,
)


# =============================================================
# بيانات تجريبية ثابتة تُستخدم في كل الاختبارات
# =============================================================

SAMPLE_DOCUMENTS = [
    Document(
        doc_id="d1",
        title="Cloud Storage",
        text="Cloud storage is useful for syncing files across devices.",
    ),
    Document(
        doc_id="d2",
        title="AI Assistants",
        text=(
            "AI assistants like Siri use voice recognition "
            "and natural language processing."
        ),
    ),
    Document(
        doc_id="d3",
        title="BM25 Ranking",
        text=(
            "BM25 is a ranking function used in information retrieval. "
            "It improves on TF-IDF by normalizing term frequency."
        ),
    ),
    Document(
        doc_id="d4",
        title="Python Programming",
        text="Python is used for machine learning and data science.",
    ),
    Document(
        doc_id="d5",
        title="Information Retrieval",
        text=(
            "Information retrieval systems help users find relevant "
            "documents using queries and ranking algorithms."
        ),
    ),
]


def _make_mock_loader(documents: List[Document] | None = None) -> MagicMock:
    """
    ينشئ DatasetLoader وهمي يُرجع SAMPLE_DOCUMENTS.
    نستخدمه لعزل الاختبارات عن القرص الفعلي.
    هذا هو مبدأ Dependency Injection في الاختبارات.
    """
    docs = documents if documents is not None else SAMPLE_DOCUMENTS
    mock = MagicMock(spec=DatasetLoader)
    mock.load_all.return_value = docs
    return mock


# =============================================================
# Fixtures
# =============================================================

@pytest.fixture
def tmp_indexes_dir(tmp_path: Path) -> Path:
    """مجلد مؤقت للفهارس يُحذف بعد كل اختبار."""
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    return indexes


@pytest.fixture
def indexer(tmp_indexes_dir: Path) -> TFIDFIndexer:
    """TFIDFIndexer جديد يستخدم مجلداً مؤقتاً."""
    return TFIDFIndexer(
        indexes_dir=str(tmp_indexes_dir),
        dataset_loader=_make_mock_loader(),
    )


@pytest.fixture
def built_indexer(indexer: TFIDFIndexer) -> TFIDFIndexer:
    """TFIDFIndexer مع فهرس مبني جاهز للاستخدام."""
    indexer.build_index(
        dataset_name="test_dataset",
        apply_stemming=True,
        remove_stopwords=True,
    )
    return indexer


@pytest.fixture
def saved_indexer(built_indexer: TFIDFIndexer) -> TFIDFIndexer:
    """TFIDFIndexer مع فهرس مبني ومحفوظ على القرص."""
    built_indexer.save_index("test_dataset")
    return built_indexer


# =============================================================
# اختبارات build_index
# =============================================================

class TestBuildIndex:
    """اختبارات بناء الفهرس."""

    def test_build_returns_metadata(self, indexer: TFIDFIndexer) -> None:
        """build_index يجب أن يُرجع TFIDFIndexMetadata."""
        meta = indexer.build_index("test_dataset")
        assert isinstance(meta, TFIDFIndexMetadata)

    def test_metadata_has_correct_document_count(
        self, indexer: TFIDFIndexer
    ) -> None:
        """عدد الوثائق في metadata يطابق عدد الوثائق الفعلي."""
        meta = indexer.build_index("test_dataset")
        assert meta.num_documents == len(SAMPLE_DOCUMENTS)

    def test_tfidf_matrix_shape_is_correct(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """
        شكل المصفوفة يجب أن يكون (num_docs, vocab_size).

        هذا الاختبار يتحقق من المبدأ الأساسي:
        كل صف = وثيقة، كل عمود = مصطلح
        """
        matrix = built_indexer.tfidf_matrix
        assert matrix is not None
        # عدد الصفوف = عدد الوثائق
        assert matrix.shape[0] == len(SAMPLE_DOCUMENTS)
        # عدد الأعمدة = حجم المفردة (variable)
        assert matrix.shape[1] > 0

    def test_documents_list_populated(self, built_indexer: TFIDFIndexer) -> None:
        """قائمة الوثائق تُملأ بعد البناء."""
        assert len(built_indexer.documents) == len(SAMPLE_DOCUMENTS)

    def test_docid_map_populated(self, built_indexer: TFIDFIndexer) -> None:
        """خريطة doc_id → index تُملأ بعد البناء."""
        assert len(built_indexer.doc_id_to_idx) == len(SAMPLE_DOCUMENTS)
        # تحقق من وجود كل doc_id
        for doc in SAMPLE_DOCUMENTS:
            assert doc.doc_id in built_indexer.doc_id_to_idx

    def test_is_built_returns_true_after_build(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """is_built() يُرجع True بعد البناء."""
        assert built_indexer.is_built() is True

    def test_is_built_returns_false_before_build(
        self, indexer: TFIDFIndexer
    ) -> None:
        """is_built() يُرجع False قبل البناء."""
        assert indexer.is_built() is False

    def test_vectorizer_is_fitted(self, built_indexer: TFIDFIndexer) -> None:
        """الـ vectorizer مدرَّب ويمتلك مفردة."""
        assert built_indexer.vectorizer is not None
        assert len(built_indexer.vectorizer.vocabulary_) > 0

    def test_metadata_records_settings(self, indexer: TFIDFIndexer) -> None:
        """
        metadata تحفظ الإعدادات المستخدمة.
        هذا ضروري لضمان تطابق معالجة الاستعلام لاحقاً.
        """
        meta = indexer.build_index(
            "test_dataset",
            apply_stemming=True,
            remove_stopwords=False,
        )
        assert meta.apply_stemming is True
        assert meta.remove_stopwords is False

    def test_max_docs_limits_documents(
        self, tmp_indexes_dir: Path
    ) -> None:
        """max_docs يحدد الحد الأقصى للوثائق المُفهرَسة."""
        mock_loader = _make_mock_loader()
        mock_loader.load_all.return_value = SAMPLE_DOCUMENTS[:3]

        indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=mock_loader,
        )
        meta = indexer.build_index("test_dataset", max_docs=3)
        assert meta.num_documents == 3

    def test_empty_dataset_raises_error(
        self, tmp_indexes_dir: Path
    ) -> None:
        """مجموعة بيانات فارغة يجب أن تُطرح ValueError."""
        mock_loader = _make_mock_loader(documents=[])
        indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=mock_loader,
        )
        with pytest.raises(ValueError, match="فارغة"):
            indexer.build_index("empty_dataset")


# =============================================================
# اختبارات save_index و load_index
# =============================================================

class TestSaveAndLoadIndex:
    """اختبارات دورة الحفظ والتحميل الكاملة."""

    def test_save_creates_required_files(
        self, saved_indexer: TFIDFIndexer, tmp_indexes_dir: Path
    ) -> None:
        """الحفظ ينشئ كل الملفات المطلوبة."""
        index_dir = tmp_indexes_dir / "test_dataset" / "tfidf"
        required_files = [
            "tfidf_vectorizer.pkl",
            "tfidf_matrix.npz",
            "tfidf_documents.json",
            "tfidf_metadata.json",
            "tfidf_docid_map.json",
        ]
        for filename in required_files:
            assert (index_dir / filename).exists(), (
                f"الملف المطلوب غير موجود: {filename}"
            )

    def test_is_saved_returns_true_after_save(
        self, saved_indexer: TFIDFIndexer
    ) -> None:
        """is_saved() يُرجع True بعد الحفظ."""
        assert saved_indexer.is_saved("test_dataset") is True

    def test_load_restores_matrix_shape(
        self, saved_indexer: TFIDFIndexer, tmp_indexes_dir: Path
    ) -> None:
        """
        التحميل يُعيد المصفوفة بنفس الشكل الأصلي.
        هذا هو الاختبار الأهم: دورة build → save → load.
        """
        original_shape = saved_indexer.tfidf_matrix.shape

        # ننشئ indexer جديد تماماً ونحمّل الفهرس
        new_indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        new_indexer.load_index("test_dataset")

        assert new_indexer.tfidf_matrix.shape == original_shape

    def test_load_restores_documents(
        self, saved_indexer: TFIDFIndexer, tmp_indexes_dir: Path
    ) -> None:
        """التحميل يُعيد قائمة الوثائق بنفس العدد."""
        new_indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        new_indexer.load_index("test_dataset")
        assert len(new_indexer.documents) == len(SAMPLE_DOCUMENTS)

    def test_load_restores_docid_map(
        self, saved_indexer: TFIDFIndexer, tmp_indexes_dir: Path
    ) -> None:
        """التحميل يُعيد خريطة doc_id → index."""
        new_indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        new_indexer.load_index("test_dataset")
        assert "d1" in new_indexer.doc_id_to_idx
        assert "d5" in new_indexer.doc_id_to_idx

    def test_load_nonexistent_raises_error(
        self, tmp_indexes_dir: Path
    ) -> None:
        """تحميل فهرس غير موجود يُطرح FileNotFoundError."""
        indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        with pytest.raises(FileNotFoundError):
            indexer.load_index("nonexistent_dataset")

    def test_metadata_preserved_after_save_load(
        self, saved_indexer: TFIDFIndexer, tmp_indexes_dir: Path
    ) -> None:
        """الـ metadata تُحفظ وتُسترجع بدقة."""
        original_meta = saved_indexer.metadata

        new_indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        loaded_meta = new_indexer.load_index("test_dataset")

        assert loaded_meta.dataset_name   == original_meta.dataset_name
        assert loaded_meta.num_documents  == original_meta.num_documents
        assert loaded_meta.apply_stemming == original_meta.apply_stemming
        assert loaded_meta.vocab_size     == original_meta.vocab_size


# =============================================================
# اختبارات transform_query
# =============================================================

class TestTransformQuery:
    """اختبارات تحويل الاستعلام لمتجه TF-IDF."""

    def test_transform_returns_sparse_matrix(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """
        transform_query يُرجع sparse matrix شكلها (1, vocab_size).
        هذا ما سيستخدمه Developer 2 لحساب Cosine Similarity.
        """
        query_vec = built_indexer.transform_query("cloud storage sync")
        assert query_vec is not None
        assert query_vec.shape[0] == 1
        assert query_vec.shape[1] == built_indexer.tfidf_matrix.shape[1]

    def test_transform_empty_query_returns_none(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """استعلام فارغ يُرجع None بدلاً من طرح خطأ."""
        result = built_indexer.transform_query("   ")
        assert result is None

    def test_transform_consistency(self, built_indexer: TFIDFIndexer) -> None:
        """
        نفس الاستعلام يجب أن يُنتج نفس المتجه دائماً.
        هذا يضمن نتائج بحث متسقة.
        """
        import numpy as np
        q1 = built_indexer.transform_query("information retrieval")
        q2 = built_indexer.transform_query("information retrieval")
        # المتجهان متطابقان
        diff = np.abs(q1 - q2).sum()
        assert diff == 0.0

    def test_transform_before_build_raises_error(
        self, indexer: TFIDFIndexer
    ) -> None:
        """transform_query قبل البناء يُطرح RuntimeError."""
        with pytest.raises(RuntimeError, match="غير مبني"):
            indexer.transform_query("test query")

    def test_transform_after_load_works(
        self, saved_indexer: TFIDFIndexer, tmp_indexes_dir: Path
    ) -> None:
        """
        transform_query يعمل بعد تحميل الفهرس من القرص.
        هذه الحالة هي الأكثر شيوعاً في الاستخدام الفعلي.
        """
        new_indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        new_indexer.load_index("test_dataset")
        result = new_indexer.transform_query("cloud storage")
        assert result is not None


# =============================================================
# اختبارات استرجاع الوثائق
# =============================================================

class TestDocumentRetrieval:
    """اختبارات الحصول على الوثائق من الفهرس."""

    def test_get_document_by_id_returns_correct_doc(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """get_document_by_id يُرجع الوثيقة الصحيحة."""
        doc = built_indexer.get_document_by_id("d1")
        assert doc is not None
        assert doc.doc_id == "d1"
        assert doc.title == "Cloud Storage"

    def test_get_document_by_id_returns_none_for_unknown(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """معرّف غير موجود يُرجع None بدلاً من خطأ."""
        doc = built_indexer.get_document_by_id("nonexistent_id")
        assert doc is None

    def test_get_document_by_index_returns_correct_doc(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """get_document_by_index يُرجع الوثيقة في الموضع الصحيح."""
        doc = built_indexer.get_document_by_index(0)
        assert doc is not None
        # الوثيقة الأولى هي d1
        assert doc.doc_id == "d1"

    def test_get_document_by_index_out_of_range(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """index خارج النطاق يُرجع None."""
        doc = built_indexer.get_document_by_index(9999)
        assert doc is None

    def test_all_documents_accessible_by_id(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """كل وثيقة في البيانات التجريبية قابلة للوصول بالـ ID."""
        for original_doc in SAMPLE_DOCUMENTS:
            retrieved = built_indexer.get_document_by_id(original_doc.doc_id)
            assert retrieved is not None, (
                f"الوثيقة {original_doc.doc_id} غير موجودة في الفهرس"
            )


# =============================================================
# اختبار التكامل: دورة كاملة تحاكي الاستخدام الفعلي
# =============================================================

class TestIntegration:
    """
    اختبار التكامل: يحاكي ما سيفعله Developer 2 بالضبط.

    السيناريو:
      1. Developer 1 يبني الفهرس ويحفظه
      2. Developer 2 يحمّل الفهرس في Retrieval Service
      3. Developer 2 يحوّل الاستعلام ويحسب التشابه
    """

    def test_full_cycle_build_save_load_query(
        self, tmp_indexes_dir: Path
    ) -> None:
        """
        دورة كاملة: بناء → حفظ → تحميل → تحويل استعلام → تشابه.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # === Developer 1: بناء وحفظ ===
        dev1_indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        dev1_indexer.build_index("test_dataset", apply_stemming=True)
        dev1_indexer.save_index("test_dataset")

        # === Developer 2: تحميل واستخدام ===
        dev2_indexer = TFIDFIndexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        dev2_indexer.load_index("test_dataset")

        # تحويل استعلام معالج
        query_vec = dev2_indexer.transform_query("cloud storag sync")

        # حساب التشابه مع كل الوثائق
        scores = cosine_similarity(query_vec, dev2_indexer.tfidf_matrix).flatten()

        # الترتيب تنازلياً
        ranked_indices = np.argsort(scores)[::-1]

        # الوثيقة الأولى يجب أن تكون d1 (تتحدث عن cloud storage)
        top_doc = dev2_indexer.get_document_by_index(ranked_indices[0])

        assert top_doc is not None
        assert scores[ranked_indices[0]] > 0, "أفضل وثيقة يجب أن يكون لها score > 0"
        assert top_doc.doc_id == "d1", (
            f"المتوقع d1 لكن حصلنا على {top_doc.doc_id}\n"
            f"Scores: {dict(zip([d.doc_id for d in dev2_indexer.documents], scores.round(3)))}"
        )

    def test_relevant_document_scores_higher_than_irrelevant(
        self, built_indexer: TFIDFIndexer
    ) -> None:
        """
        وثيقة ذات صلة يجب أن تحصل على score أعلى من وثيقة غير ذات صلة.
        هذا هو الاختبار الجوهري لجودة الفهرس.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # استعلام عن cloud storage
        query_vec = built_indexer.transform_query("cloud storag file sync")
        scores = cosine_similarity(
            query_vec, built_indexer.tfidf_matrix
        ).flatten()

        # d1 تتحدث عن cloud storage — يجب أن تكون score أعلى من d4 (Python)
        idx_d1 = built_indexer.doc_id_to_idx["d1"]
        idx_d4 = built_indexer.doc_id_to_idx["d4"]

        assert scores[idx_d1] > scores[idx_d4], (
            f"d1 (cloud) score={scores[idx_d1]:.3f} يجب أن يكون أعلى من "
            f"d4 (python) score={scores[idx_d4]:.3f}"
        )
