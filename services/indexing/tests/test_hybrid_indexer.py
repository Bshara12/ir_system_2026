"""
tests/test_hybrid_indexer.py
==============================
اختبارات HybridIndexer بدون بناء فهارس حقيقية.

نستخدم Mock objects لأن:
  - بناء فهرس BM25 حقيقي يأخذ دقائق
  - بناء فهرس Embedding يأخذ وقتاً أطول (تحميل نموذج + encoding)
  - نريد اختبارات سريعة تعمل في ثوانٍ

ما الذي نختبره هنا؟
  1. أن HybridIndexer يستدعي BM25Indexer.build_index() صحيحاً
  2. أن HybridIndexer يستدعي EmbeddingIndexer.build_index() صحيحاً
  3. أن build_bm25=False يتخطى BM25
  4. أن build_embedding=False يتخطى Embedding
  5. أن is_built() يعمل صحيحاً
  6. أن is_saved() يعمل صحيحاً
  7. أن from_saved() يستدعي load_indexes صحيحاً
  8. أن get_status() يُرجع البنية الصحيحة
"""

import sys
import os
from unittest.mock import MagicMock, patch, call
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# نضيف مسار المشروع لكي تعمل الـ imports
# عدّل هذا المسار ليطابق مكان مشروعك
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # جذر المشروع
sys.path.insert(0, str(PROJECT_ROOT))


def make_mock_bm25(is_built=True, is_saved_result=True):
    """
    ينشئ BM25Indexer مزيّف للاختبار.

    لماذا MagicMock(spec=BM25Indexer)؟
    spec= يجعل الـ Mock يقبل فقط attributes و methods موجودة
    في BM25Indexer الحقيقي. هذا يُمسك الأخطاء مبكراً.
    """
    from services.indexing.bm25_indexer import BM25Indexer

    mock = MagicMock(spec=BM25Indexer)
    mock.is_built.return_value = is_built
    mock.is_saved.return_value = is_saved_result
    mock.metadata = MagicMock()
    mock.metadata.num_documents  = 100
    mock.metadata.k1             = 1.5
    mock.metadata.b              = 0.75
    mock.metadata.vocab_size     = 5000
    mock.metadata.avg_document_length = 45.2
    return mock


def make_mock_emb(is_built=True, is_saved_result=True):
    """ينشئ EmbeddingIndexer مزيّف للاختبار."""
    from services.indexing.embedding_indexer import EmbeddingIndexer

    mock = MagicMock(spec=EmbeddingIndexer)
    mock.is_built.return_value = is_built
    mock.is_saved.return_value = is_saved_result
    mock.metadata = MagicMock()
    mock.metadata.model_name    = "all-MiniLM-L6-v2"
    mock.metadata.num_documents = 100
    mock.metadata.embedding_dim = 384
    mock.metadata.index_type    = "flat_ip"
    return mock


# ══════════════════════════════════════════════════════════════
# الاختبار 1: build_indexes يستدعي كلا الـ indexers
# ══════════════════════════════════════════════════════════════

def test_build_indexes_calls_both():
    """
    يتحقق أن build_indexes تستدعي:
      - bm25.build_index() مرة واحدة
      - bm25.save_index() مرة واحدة
      - emb.build_index() مرة واحدة
      - emb.save_index() مرة واحدة
    """
    from services.indexing.hybrid_indexer import HybridIndexer

    mock_bm25 = make_mock_bm25()
    mock_emb  = make_mock_emb()

    hybrid = HybridIndexer(
        bm25_indexer=mock_bm25,
        embedding_indexer=mock_emb,
    )

    hybrid.build_indexes("test-dataset", max_docs=10)

    # التحقق من استدعاء BM25
    mock_bm25.build_index.assert_called_once()
    call_kwargs = mock_bm25.build_index.call_args
    assert call_kwargs.kwargs.get("dataset_name") == "test-dataset" or \
           call_kwargs.args[0] == "test-dataset", \
        "build_index لم يُستدعَ بـ dataset_name الصحيح"

    mock_bm25.save_index.assert_called_once_with("test-dataset")

    # التحقق من استدعاء Embedding
    mock_emb.build_index.assert_called_once()
    mock_emb.save_index.assert_called_once_with("test-dataset")

    print("✅ test_build_indexes_calls_both — نجح")


# ══════════════════════════════════════════════════════════════
# الاختبار 2: build_bm25=False يتخطى BM25
# ══════════════════════════════════════════════════════════════

def test_skip_bm25():
    """عند build_bm25=False يجب ألا يُستدعى BM25Indexer.build_index."""
    from services.indexing.hybrid_indexer import HybridIndexer

    mock_bm25 = make_mock_bm25()
    mock_emb  = make_mock_emb()

    hybrid = HybridIndexer(bm25_indexer=mock_bm25, embedding_indexer=mock_emb)
    hybrid.build_indexes("test-dataset", build_bm25=False, max_docs=5)

    # BM25 يجب ألا يُستدعى
    mock_bm25.build_index.assert_not_called()
    mock_bm25.save_index.assert_not_called()

    # Embedding يجب أن يُستدعى
    mock_emb.build_index.assert_called_once()
    mock_emb.save_index.assert_called_once()

    print("✅ test_skip_bm25 — نجح")


# ══════════════════════════════════════════════════════════════
# الاختبار 3: build_embedding=False يتخطى Embedding
# ══════════════════════════════════════════════════════════════

def test_skip_embedding():
    """عند build_embedding=False يجب ألا يُستدعى EmbeddingIndexer.build_index."""
    from services.indexing.hybrid_indexer import HybridIndexer

    mock_bm25 = make_mock_bm25()
    mock_emb  = make_mock_emb()

    hybrid = HybridIndexer(bm25_indexer=mock_bm25, embedding_indexer=mock_emb)
    hybrid.build_indexes("test-dataset", build_embedding=False, max_docs=5)

    mock_emb.build_index.assert_not_called()
    mock_emb.save_index.assert_not_called()
    mock_bm25.build_index.assert_called_once()

    print("✅ test_skip_embedding — نجح")


# ══════════════════════════════════════════════════════════════
# الاختبار 4: is_built صحيح
# ══════════════════════════════════════════════════════════════

def test_is_built():
    """
    is_built() يجب أن يُرجع True فقط عندما يكون
    كلا الفهرسين مبنيَّين.
    """
    from services.indexing.hybrid_indexer import HybridIndexer

    # الحالة 1: كلاهما مبني
    hybrid = HybridIndexer(
        bm25_indexer=make_mock_bm25(is_built=True),
        embedding_indexer=make_mock_emb(is_built=True),
    )
    assert hybrid.is_built() is True, "يجب أن يكون True عندما كلاهما مبني"

    # الحالة 2: BM25 غير مبني
    hybrid2 = HybridIndexer(
        bm25_indexer=make_mock_bm25(is_built=False),
        embedding_indexer=make_mock_emb(is_built=True),
    )
    assert hybrid2.is_built() is False, "يجب أن يكون False إذا BM25 غير مبني"

    # الحالة 3: Embedding غير مبني
    hybrid3 = HybridIndexer(
        bm25_indexer=make_mock_bm25(is_built=True),
        embedding_indexer=make_mock_emb(is_built=False),
    )
    assert hybrid3.is_built() is False, "يجب أن يكون False إذا Embedding غير مبني"

    print("✅ test_is_built — نجح")


# ══════════════════════════════════════════════════════════════
# الاختبار 5: is_saved صحيح
# ══════════════════════════════════════════════════════════════

def test_is_saved():
    """is_saved() يجب أن يتحقق من كلا الفهرسين."""
    from services.indexing.hybrid_indexer import HybridIndexer

    # كلاهما محفوظ
    hybrid = HybridIndexer(
        bm25_indexer=make_mock_bm25(is_saved_result=True),
        embedding_indexer=make_mock_emb(is_saved_result=True),
    )
    assert hybrid.is_saved("test") is True

    # BM25 غير محفوظ
    hybrid2 = HybridIndexer(
        bm25_indexer=make_mock_bm25(is_saved_result=False),
        embedding_indexer=make_mock_emb(is_saved_result=True),
    )
    assert hybrid2.is_saved("test") is False

    print("✅ test_is_saved — نجح")


# ══════════════════════════════════════════════════════════════
# الاختبار 6: load_indexes يستدعي كلا الـ indexers
# ══════════════════════════════════════════════════════════════

def test_load_indexes():
    """load_indexes يجب أن يستدعي load_index على كلا الـ indexers."""
    from services.indexing.hybrid_indexer import HybridIndexer

    mock_bm25 = make_mock_bm25()
    mock_emb  = make_mock_emb()

    hybrid = HybridIndexer(bm25_indexer=mock_bm25, embedding_indexer=mock_emb)
    hybrid.load_indexes("test-dataset")

    mock_bm25.load_index.assert_called_once_with("test-dataset")
    mock_emb.load_index.assert_called_once_with("test-dataset")

    print("✅ test_load_indexes — نجح")


# ══════════════════════════════════════════════════════════════
# الاختبار 7: get_status يُرجع البنية الصحيحة
# ══════════════════════════════════════════════════════════════

def test_get_status_structure():
    """get_status يجب أن يُرجع dict يحتوي hybrid_ready و bm25 و embedding."""
    from services.indexing.hybrid_indexer import HybridIndexer

    hybrid = HybridIndexer(
        bm25_indexer=make_mock_bm25(is_built=True),
        embedding_indexer=make_mock_emb(is_built=True),
    )

    status = hybrid.get_status()

    assert "hybrid_ready" in status,  "يجب أن يحتوي hybrid_ready"
    assert "bm25" in status,          "يجب أن يحتوي bm25"
    assert "embedding" in status,     "يجب أن يحتوي embedding"
    assert status["hybrid_ready"] is True

    assert "num_documents" in status["bm25"],     "bm25 يجب أن يحتوي num_documents"
    assert "model_name" in status["embedding"],   "embedding يجب أن يحتوي model_name"

    print("✅ test_get_status_structure — نجح")


# ══════════════════════════════════════════════════════════════
# الاختبار 8: from_saved يستدعي load_indexes
# ══════════════════════════════════════════════════════════════

def test_from_saved():
    """from_saved يجب أن ينشئ HybridIndexer ويستدعي load_indexes."""
    from services.indexing.hybrid_indexer import HybridIndexer

    with patch.object(HybridIndexer, "load_indexes") as mock_load:
        hybrid = HybridIndexer.from_saved("test-dataset")
        mock_load.assert_called_once_with("test-dataset")

    print("✅ test_from_saved — نجح")


# ══════════════════════════════════════════════════════════════
# تشغيل الاختبارات
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("تشغيل اختبارات HybridIndexer")
    print("="*55 + "\n")

    tests = [
        test_build_indexes_calls_both,
        test_skip_bm25,
        test_skip_embedding,
        test_is_built,
        test_is_saved,
        test_load_indexes,
        test_get_status_structure,
        test_from_saved,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"❌ {test_fn.__name__} — فشل: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"النتيجة: {passed} نجح، {failed} فشل")
    print(f"{'='*55}\n")

    if failed > 0:
        sys.exit(1)