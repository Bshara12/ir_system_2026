"""
services/indexing/tests/test_vector_store.py
=============================================
اختبارات VectorStore — بدون تحميل نموذج حقيقي.

نستخدم Mock لأن:
  - SentenceTransformer يأخذ ~5 ثوانٍ للتحميل
  - FAISS يحتاج faiss-cpu مثبّتة
  - نريد اختبارات تعمل في ثوانٍ

تشغيل:
    cd ir_system_2026
    python -m pytest services/indexing/tests/test_vector_store.py -v
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

from services.indexing.tfidf_indexer import IndexedDocument


# =============================================================
# أدوات مساعدة للاختبار
# =============================================================

EMBEDDING_DIM = 384


def _make_normalized_vecs(n: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """ينشئ متجهات مُطبَّعة ثابتة للاختبار."""
    rng  = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32)


def _make_sample_docs(n: int = 5) -> List[IndexedDocument]:
    """ينشئ وثائق تجريبية."""
    texts = [
        ("d1", "Cloud storage is useful for syncing files.", "Cloud Storage"),
        ("d2", "AI assistants like Siri use voice recognition.", "AI Assistants"),
        ("d3", "BM25 ranks documents by term frequency.", "BM25 Ranking"),
        ("d4", "Python is used for machine learning.", None),
        ("d5", "Information retrieval finds relevant documents.", "IR Systems"),
    ]
    return [
        IndexedDocument(
            doc_id=t[0],
            original_text=t[1],
            processed_text=t[1].lower(),
            title=t[2],
        )
        for t in texts[:n]
    ]


def _make_built_indexer(n_docs: int = 5):
    """
    ينشئ EmbeddingIndexer mock مع FAISS index وهمي.
    يحاكي indexer مبني بالكامل.
    """
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss-cpu غير مثبتة")

    from services.indexing.embedding_indexer import (
        EmbeddingIndexer,
        EmbeddingIndexMetadata,
        DEFAULT_MODEL_NAME,
    )

    # نبني FAISS index حقيقي بمتجهات وهمية
    vecs  = _make_normalized_vecs(n_docs)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.ascontiguousarray(vecs))

    docs = _make_sample_docs(n_docs)

    import datetime
    meta = EmbeddingIndexMetadata(
        dataset_name="test-dataset",
        model_name=DEFAULT_MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        num_documents=n_docs,
        index_type="flat_ip",
        normalize_embeddings=True,
        build_time_seconds=1.0,
        build_timestamp=datetime.datetime.now().isoformat(),
        batch_size=64,
    )

    indexer              = MagicMock(spec=EmbeddingIndexer)
    indexer.faiss_index  = index
    indexer.embeddings   = vecs
    indexer.documents    = docs
    indexer.doc_id_to_idx = {d.doc_id: i for i, d in enumerate(docs)}
    indexer.metadata     = meta
    indexer.model_name   = DEFAULT_MODEL_NAME
    indexer.is_built.return_value  = True
    indexer.is_saved.return_value  = True

    # encode_query يُرجع متجه وهمي ثابت
    query_vec = _make_normalized_vecs(1)
    indexer.encode_query.return_value = query_vec

    # get_top_k يستدعي FAISS الحقيقي
    def real_get_top_k(query_embedding, k=10):
        if query_embedding is None:
            return []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        actual_k = min(k, index.ntotal)
        scores, indices = index.search(
            np.ascontiguousarray(query_embedding.astype(np.float32)),
            actual_k,
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if 0 <= idx < len(docs):
                results.append((docs[idx], float(score)))
        return results

    indexer.get_top_k.side_effect = real_get_top_k

    return indexer


# =============================================================
# Fixture
# =============================================================

@pytest.fixture
def ready_store(tmp_path):
    """VectorStore مع EmbeddingIndexer mock مبني."""
    from services.indexing.vector_store import VectorStore

    store = VectorStore(
        dataset_name="test-dataset",
        indexes_dir=str(tmp_path / "indexes"),
    )
    # نحقن الـ indexer المزيّف
    store._indexer = _make_built_indexer()
    return store


@pytest.fixture
def empty_store(tmp_path):
    """VectorStore فارغ (indexer غير مبني)."""
    from services.indexing.vector_store import VectorStore
    from services.indexing.embedding_indexer import EmbeddingIndexer

    store = VectorStore(
        dataset_name="test-dataset",
        indexes_dir=str(tmp_path / "indexes"),
    )
    mock_indexer = MagicMock(spec=EmbeddingIndexer)
    mock_indexer.is_built.return_value  = False
    mock_indexer.is_saved.return_value  = False
    store._indexer = mock_indexer
    return store


# =============================================================
# ① اختبارات is_ready
# =============================================================

class TestIsReady:

    def test_ready_when_indexer_built(self, ready_store) -> None:
        """
        ماذا يختبر: is_ready() يُرجع True عندما الفهرس مبني.
        لماذا مهم: Developer 2 يتحقق منها قبل كل search().
        """
        assert ready_store.is_ready() is True

    def test_not_ready_when_indexer_empty(self, empty_store) -> None:
        """
        ماذا يختبر: is_ready() يُرجع False عندما الفهرس غير مبني.
        """
        assert empty_store.is_ready() is False


# =============================================================
# ② اختبارات search
# =============================================================

class TestSearch:

    def test_search_returns_list(self, ready_store) -> None:
        """
        ماذا يختبر: search() يُرجع قائمة.
        لماذا مهم: Developer 2 يتوقع list يمكن iteration عليها.
        """
        results = ready_store.search("cloud storage", k=3)
        assert isinstance(results, list)

    def test_search_returns_tuples_with_correct_structure(
        self, ready_store
    ) -> None:
        """
        ماذا يختبر: كل نتيجة tuple من 4 عناصر (doc_id, score, text, title).
        لماذا مهم: هذا العقد الذي يعتمد عليه Developer 2.
        """
        results = ready_store.search("cloud storage", k=3)
        assert len(results) > 0
        for item in results:
            assert len(item) == 4, "يجب أن تكون tuple من 4 عناصر"
            doc_id, score, text, title = item
            assert isinstance(doc_id, str)
            assert isinstance(score,  float)
            assert isinstance(text,   str)
            # title يمكن أن يكون None

    def test_search_respects_k_limit(self, ready_store) -> None:
        """
        ماذا يختبر: النتائج لا تتجاوز k.
        """
        results = ready_store.search("test query", k=2)
        assert len(results) <= 2

    def test_search_empty_query_returns_empty(self, ready_store) -> None:
        """
        ماذا يختبر: استعلام فارغ → قائمة فارغة (لا استثناء).
        لماذا مهم: المستخدم قد يُرسل استعلاماً فارغاً.
        """
        results = ready_store.search("   ", k=5)
        assert results == []

    def test_search_not_ready_returns_empty(self, empty_store) -> None:
        """
        ماذا يختبر: البحث قبل التحميل → قائمة فارغة (لا استثناء).
        لماذا مهم: يمنع crash إذا نسي Developer 2 استدعاء load().
        """
        results = empty_store.search("cloud storage")
        assert results == []

    def test_search_scores_between_minus1_and_1(
        self, ready_store
    ) -> None:
        """
        ماذا يختبر: درجات cosine similarity بين -1 و 1.
        لماذا مهم: FAISS IndexFlatIP يُنتج هذا النطاق بعد L2 normalization.
        """
        results = ready_store.search("test", k=5)
        for _, score, _, _ in results:
            assert -1.1 <= score <= 1.1, f"score خارج النطاق: {score}"

    def test_search_calls_encode_query(self, ready_store) -> None:
        """
        ماذا يختبر: search() تستدعي encode_query() بالنص الصحيح.
        لماذا مهم: يضمن أن النص يصل للنموذج بدون تعديل.
        """
        ready_store.search("fever treatment", k=3)
        ready_store._indexer.encode_query.assert_called_once_with("fever treatment")


# =============================================================
# ③ اختبارات save و load
# =============================================================

class TestSaveLoad:

    def test_save_returns_true_when_ready(self, ready_store) -> None:
        """
        ماذا يختبر: save() يُرجع True عند النجاح.
        """
        ready_store._indexer.save_index = MagicMock(return_value=Path("."))
        result = ready_store.save()
        assert result is True

    def test_save_returns_false_when_not_ready(self, empty_store) -> None:
        """
        ماذا يختبر: save() يُرجع False إذا الفهرس غير مبني.
        لماذا مهم: لا يجب أن تُحفظ بيانات فارغة.
        """
        result = empty_store.save()
        assert result is False

    def test_save_calls_indexer_save(self, ready_store) -> None:
        """
        ماذا يختبر: save() تستدعي EmbeddingIndexer.save_index().
        لماذا مهم: يضمن أن الحفظ يصل للـ indexer الفعلي.
        """
        ready_store._indexer.save_index = MagicMock(return_value=Path("."))
        ready_store.save()
        ready_store._indexer.save_index.assert_called_once_with("test-dataset")

    def test_load_calls_indexer_load(self, empty_store) -> None:
        """
        ماذا يختبر: load() تستدعي EmbeddingIndexer.load_index().
        """
        empty_store._indexer.load_index = MagicMock()
        empty_store.load()
        empty_store._indexer.load_index.assert_called_once_with("test-dataset")

    def test_load_returns_false_on_file_not_found(
        self, empty_store
    ) -> None:
        """
        ماذا يختبر: load() يُرجع False إذا الملف غير موجود.
        لماذا مهم: يمنع crash إذا لم يُبنَ الفهرس بعد.
        """
        empty_store._indexer.load_index = MagicMock(
            side_effect=FileNotFoundError("not found")
        )
        result = empty_store.load()
        assert result is False


# =============================================================
# ④ اختبارات size و get_status
# =============================================================

class TestStatus:

    def test_size_when_ready(self, ready_store) -> None:
        """
        ماذا يختبر: size() يُرجع عدد الوثائق الصحيح.
        """
        assert ready_store.size() == 5

    def test_size_when_not_ready(self, empty_store) -> None:
        """
        ماذا يختبر: size() يُرجع 0 إذا الفهرس غير مبني.
        """
        assert empty_store.size() == 0

    def test_get_status_has_required_keys(self, ready_store) -> None:
        """
        ماذا يختبر: get_status() يُرجع dict بكل الحقول المطلوبة.
        لماذا مهم: Developer 3 يعرضها في الـ UI.
        """
        status = ready_store.get_status()
        required = [
            "dataset_name",
            "is_ready",
            "is_persisted",
            "num_documents",
        ]
        for key in required:
            assert key in status, f"الحقل مفقود: {key}"

    def test_get_status_shows_model_info_when_ready(
        self, ready_store
    ) -> None:
        """
        ماذا يختبر: get_status() يُظهر معلومات النموذج عند الجاهزية.
        """
        status = ready_store.get_status()
        assert "model_name"    in status
        assert "embedding_dim" in status


# =============================================================
# ⑤ اختبار indexer property
# =============================================================

class TestIndexerProperty:

    def test_indexer_returns_embedding_indexer(self, ready_store) -> None:
        """
        ماذا يختبر: store.indexer يُرجع EmbeddingIndexer.
        لماذا مهم: Developer 2 المتقدم يحتاجه للـ HybridRetriever.
        """
        from services.indexing.embedding_indexer import EmbeddingIndexer
        # المهم أن الكائن المُرجَع هو نفس self._indexer
        assert ready_store.indexer is ready_store._indexer


# =============================================================
# ⑥ اختبار التكامل — يحاكي Developer 2
# =============================================================

class TestIntegration:

    def test_full_search_workflow(self, ready_store) -> None:
        """
        يحاكي ما سيفعله Developer 2:
        1. يتحقق من is_ready()
        2. يبحث بنص طبيعي
        3. يُعالج النتائج

        هذا هو الاختبار الأهم — يضمن أن الواجهة تعمل كاملة.
        """
        # خطوة 1: التحقق
        assert ready_store.is_ready()

        # خطوة 2: البحث
        results = ready_store.search("cloud storage files", k=3)

        # خطوة 3: التحقق من النتائج
        assert len(results) > 0

        # كل نتيجة يجب أن تكون قابلة للاستخدام مباشرة
        for doc_id, score, text, title in results:
            assert doc_id  # ليس فارغاً
            assert text    # ليس فارغاً
            # score يمكن أن يكون أي رقم بين -1 و 1

    def test_get_status_reflects_state(self, ready_store) -> None:
        """
        ماذا يختبر: get_status() يعكس الحالة الفعلية.
        """
        status = ready_store.get_status()
        assert status["is_ready"]      == ready_store.is_ready()
        assert status["num_documents"] == ready_store.size()