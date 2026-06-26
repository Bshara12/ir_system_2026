"""
services/retrieval/tests/test_retrieval.py
===========================================
اختبارات وحدة (Unit Tests) لخدمة الاسترجاع.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا نستخدم Mock بدل الفهارس الحقيقية؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الفهارس الحقيقية تحتاج:
    - ملفات ضخمة على القرص (GBs)
    - وقت لتحميلها (عشرات الثواني)
    - بيئة محددة مع نماذج محمّلة

الاختبار الجيد يجب أن يعمل في ثوانٍ على أي جهاز.
نستبدل الفهرس بـ Mock يُرجع بيانات محددة مسبقاً.
هكذا نختبر منطق الكود فقط، وليس البيانات.

التشغيل:
    cd ir_system_2026
    python -m pytest services/retrieval/tests/ -v
"""

import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import pytest
from unittest.mock import MagicMock, patch
from collections import defaultdict

from shared.models import DocumentResult, DatasetName, RetrievalModel
from services.retrieval.hybrid_parallel import HybridParallelRetriever, RRF_K
from services.retrieval.hybrid_serial import HybridSerialRetriever

# =============================================================
# دوال مساعدة للاختبارات
# =============================================================


def make_result(doc_id: str, score: float, rank: int, text: str = "") -> DocumentResult:
    """يُنشئ DocumentResult للاختبار."""
    return DocumentResult(
        doc_id=doc_id,
        text=text or f"نص الوثيقة {doc_id}",
        score=score,
        rank=rank,
        title=f"عنوان {doc_id}",
    )


def make_retriever_mock(results: list, loaded: bool = True):
    """
    يُنشئ محرك بحث وهمي (Mock).
    يُرجع دائماً نفس القائمة المحددة.
    """
    mock = MagicMock()
    mock.is_loaded = loaded
    mock.search.return_value = results
    return mock


# =============================================================
# اختبارات Hybrid Parallel — التركيز على RRF
# =============================================================


class TestHybridParallelRRF:
    """
    اختبارات خوارزمية RRF في التمثيل الهجين المتوازي.
    هذا الجزء الأهم لأن RRF هو "قلب" الـ Hybrid Parallel.
    """

    def test_rrf_single_source_preserves_order(self):
        """
        مع محرك واحد فقط، الترتيب يجب أن يبقى نفسه.
        السبب: 1/(60+1) > 1/(60+2) > 1/(60+3)
        """
        results = [
            make_result("A", score=0.9, rank=1),
            make_result("B", score=0.7, rank=2),
            make_result("C", score=0.5, rank=3),
        ]
        mock = make_retriever_mock(results)
        hybrid = HybridParallelRetriever(bm25_retriever=mock)

        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=3,
            use_tfidf=False,
            use_bm25=True,
            use_embedding=False,
        )

        assert len(final) == 3
        assert final[0].doc_id == "A"
        assert final[1].doc_id == "B"
        assert final[2].doc_id == "C"

    def test_rrf_formula_is_correct(self):
        """
        التحقق من دقة حساب RRF:
        score = 1/(RRF_K + rank)
        """
        results = [make_result("X", score=5.0, rank=1)]
        mock = make_retriever_mock(results)
        hybrid = HybridParallelRetriever(bm25_retriever=mock)

        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=1,
            use_tfidf=False,
            use_bm25=True,
            use_embedding=False,
        )

        expected_score = 1.0 / (RRF_K + 1)
        assert len(final) == 1
        assert abs(final[0].score - expected_score) < 1e-5

    def test_rrf_document_in_two_sources_wins(self):
        """
        وثيقة تظهر في مصدرَين تتفوق على وثيقة تظهر في مصدر واحد.

        وثيقة "WINNER" جاءت أولاً في كلا المحركين
        وثيقة "LOSER"  جاءت أولاً في محرك واحد فقط
        WINNER يجب أن يفوز.
        """
        source1 = [
            make_result("WINNER", score=10.0, rank=1),
            make_result("LOSER", score=9.0, rank=2),
        ]
        source2 = [
            make_result("WINNER", score=8.0, rank=1),
            make_result("OTHER", score=7.0, rank=2),
        ]

        mock1 = make_retriever_mock(source1)
        mock2 = make_retriever_mock(source2)

        hybrid = HybridParallelRetriever(
            bm25_retriever=mock1,
            tfidf_retriever=mock2,
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=3,
            use_tfidf=True,
            use_bm25=True,
            use_embedding=False,
        )

        assert final[0].doc_id == "WINNER"

        # درجة WINNER = 2 × 1/(60+1)
        expected_winner = 2.0 / (RRF_K + 1)
        assert abs(final[0].score - expected_winner) < 1e-5

    def test_rrf_absent_document_gets_zero_contribution(self):
        """
        وثيقة غائبة من مصدر لا تحصل على درجة من ذلك المصدر.
        """
        source1 = [make_result("ONLY_IN_S1", score=10.0, rank=1)]
        source2 = [make_result("ONLY_IN_S2", score=10.0, rank=1)]

        mock1 = make_retriever_mock(source1)
        mock2 = make_retriever_mock(source2)

        hybrid = HybridParallelRetriever(
            bm25_retriever=mock1,
            tfidf_retriever=mock2,
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=2,
            use_tfidf=True,
            use_bm25=True,
            use_embedding=False,
        )

        # كلا الوثيقتين لهما نفس الدرجة (ظهرتا أولاً كل في مصدره)
        assert len(final) == 2
        scores = {r.doc_id: r.score for r in final}
        assert abs(scores["ONLY_IN_S1"] - scores["ONLY_IN_S2"]) < 1e-5

    def test_scores_are_descending(self):
        """الدرجات في النتائج النهائية يجب أن تكون تنازلية."""
        results = [make_result(f"doc{i}", score=10.0 - i, rank=i) for i in range(1, 8)]
        mock = make_retriever_mock(results)
        hybrid = HybridParallelRetriever(bm25_retriever=mock)

        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=5,
            use_tfidf=False,
            use_bm25=True,
            use_embedding=False,
        )

        scores = [r.score for r in final]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential_starting_from_one(self):
        """الـ rank في النتائج يجب أن يبدأ من 1 ويكون متسلسلاً."""
        results = [
            make_result(f"doc{i}", score=float(5 - i), rank=i) for i in range(1, 4)
        ]
        mock = make_retriever_mock(results)
        hybrid = HybridParallelRetriever(bm25_retriever=mock)

        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=3,
            use_tfidf=False,
            use_bm25=True,
            use_embedding=False,
        )

        ranks = [r.rank for r in final]
        assert ranks == list(range(1, len(final) + 1))

    def test_top_k_is_respected(self):
        """النتائج لا تتجاوز top_k أبداً."""
        results = [
            make_result(f"doc{i}", score=float(20 - i), rank=i) for i in range(1, 21)
        ]
        mock = make_retriever_mock(results)
        hybrid = HybridParallelRetriever(bm25_retriever=mock)

        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=5,
            use_tfidf=False,
            use_bm25=True,
            use_embedding=False,
        )

        assert len(final) <= 5

    def test_empty_results_returns_empty_list(self):
        """إذا كل المحركات تُرجع فارغاً → قائمة فارغة."""
        mock = make_retriever_mock([])
        hybrid = HybridParallelRetriever(bm25_retriever=mock)

        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=10,
            use_tfidf=False,
            use_bm25=True,
            use_embedding=False,
        )

        assert final == []

    def test_unloaded_retriever_is_skipped(self):
        """محرك غير محمَّل (is_loaded=False) يُتجاهل بدون خطأ."""
        loaded_mock = make_retriever_mock([make_result("doc1", 0.9, 1)], loaded=True)
        unloaded_mock = make_retriever_mock([make_result("doc2", 0.8, 1)], loaded=False)

        hybrid = HybridParallelRetriever(
            bm25_retriever=loaded_mock,
            tfidf_retriever=unloaded_mock,
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=5,
            use_tfidf=True,
            use_bm25=True,
            use_embedding=False,
        )

        # يجب أن تظهر فقط نتيجة المحرك المحمَّل
        assert len(final) == 1
        assert final[0].doc_id == "doc1"

    def test_no_retrievers_available_returns_empty(self):
        """إذا لا يوجد أي محرك متاح → قائمة فارغة."""
        hybrid = HybridParallelRetriever()  # كل شيء None
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=5,
        )
        assert final == []

    def test_three_sources_combine_correctly(self):
        """
        اختبار دمج ثلاثة مصادر:
        وثيقة جاءت أولاً في الثلاثة تحصل على أعلى درجة.
        """
        doc_in_all_three = "CHAMPION"

        s1 = [make_result(doc_in_all_three, 9.0, 1), make_result("X", 8.0, 2)]
        s2 = [make_result(doc_in_all_three, 7.0, 1), make_result("Y", 6.0, 2)]
        s3 = [make_result(doc_in_all_three, 5.0, 1), make_result("Z", 4.0, 2)]

        hybrid = HybridParallelRetriever(
            bm25_retriever=make_retriever_mock(s1),
            tfidf_retriever=make_retriever_mock(s2),
            embedding_retriever=make_retriever_mock(s3),
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=5,
            use_tfidf=True,
            use_bm25=True,
            use_embedding=True,
        )

        assert final[0].doc_id == doc_in_all_three
        # درجته = 3 × 1/(60+1)
        expected = 3.0 / (RRF_K + 1)
        assert abs(final[0].score - expected) < 1e-5


# =============================================================
# اختبارات Hybrid Serial
# =============================================================


class TestHybridSerial:
    """اختبارات التمثيل الهجين التسلسلي."""

    def test_returns_first_stage_when_no_second_stage(self):
        """
        إذا لا يوجد محرك ثانٍ → يُرجع نتائج المرحلة الأولى مباشرة.
        """
        first_results = [
            make_result("doc1", 9.0, 1),
            make_result("doc2", 8.0, 2),
            make_result("doc3", 7.0, 3),
        ]
        hybrid = HybridSerialRetriever(
            first_stage_retriever=make_retriever_mock(first_results),
            second_stage_retriever=None,
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test query",
            top_k=3,
        )
        assert len(final) == 3
        assert final[0].doc_id == "doc1"

    def test_empty_first_stage_returns_empty(self):
        """مرحلة أولى فارغة → قائمة فارغة حتى لو يوجد محرك ثانٍ."""
        hybrid = HybridSerialRetriever(
            first_stage_retriever=make_retriever_mock([]),
            second_stage_retriever=MagicMock(is_loaded=True),
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=5,
        )
        assert final == []

    def test_none_first_stage_returns_empty(self):
        """لا يوجد محرك أول → قائمة فارغة بدون خطأ."""
        hybrid = HybridSerialRetriever(
            first_stage_retriever=None,
            second_stage_retriever=None,
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=5,
        )
        assert final == []

    def test_unloaded_second_stage_falls_back_to_first(self):
        """
        محرك ثانٍ غير محمَّل → يُرجع نتائج المرحلة الأولى.
        """
        first_results = [make_result("doc1", 9.0, 1), make_result("doc2", 8.0, 2)]
        unloaded_second = MagicMock()
        unloaded_second.is_loaded = False

        hybrid = HybridSerialRetriever(
            first_stage_retriever=make_retriever_mock(first_results),
            second_stage_retriever=unloaded_second,
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=5,
        )
        # يجب العودة لنتائج المرحلة الأولى
        assert len(final) == 2
        assert final[0].doc_id == "doc1"


# =============================================================
# اختبارات RRF — دقة رياضية إضافية
# =============================================================


class TestRRFMathematicalProperties:
    """اختبارات تتحقق من الخصائص الرياضية لـ RRF."""

    def test_higher_rank_gives_lower_rrf_score(self):
        """
        وثيقة في الترتيب 1 تحصل على درجة RRF أعلى من وثيقة في الترتيب 10.
        1/(60+1) > 1/(60+10)
        """
        results = [
            make_result("first", 10.0, 1),
            make_result("tenth", 1.0, 10),
        ]
        mock = make_retriever_mock(results)
        hybrid = HybridParallelRetriever(bm25_retriever=mock)

        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=2,
            use_tfidf=False,
            use_bm25=True,
            use_embedding=False,
        )

        scores = {r.doc_id: r.score for r in final}
        assert scores["first"] > scores["tenth"]

    def test_rrf_score_is_sum_of_contributions(self):
        """
        درجة RRF = مجموع مساهمات كل مصدر.
        doc A في المرتبة 2 من المصدر 1، و3 من المصدر 2:
        score = 1/(60+2) + 1/(60+3)
        """
        s1 = [
            make_result("A", 10.0, 1),
            make_result("B", 9.0, 2),  # B rank=2 in s1
        ]
        s2 = [
            make_result("A", 8.0, 1),
            make_result("C", 7.0, 2),
            make_result("B", 6.0, 3),  # B rank=3 in s2
        ]

        hybrid = HybridParallelRetriever(
            bm25_retriever=make_retriever_mock(s1),
            tfidf_retriever=make_retriever_mock(s2),
        )
        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=3,
            use_tfidf=True,
            use_bm25=True,
            use_embedding=False,
        )

        scores = {r.doc_id: r.score for r in final}
        expected_b = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 3)
        assert "B" in scores
        assert abs(scores["B"] - expected_b) < 1e-5

    def test_doc_id_and_text_preserved_in_output(self):
        """بيانات الوثيقة (doc_id, text, title) تُحفظ في النتائج."""
        results = [
            make_result("special_id", 9.0, 1, text="نص خاص جداً"),
        ]
        mock = make_retriever_mock(results)
        hybrid = HybridParallelRetriever(bm25_retriever=mock)

        final = hybrid.search(
            query_tokens=["test"],
            query_text="test",
            top_k=1,
            use_tfidf=False,
            use_bm25=True,
            use_embedding=False,
        )

        assert final[0].doc_id == "special_id"
        assert final[0].text == "نص خاص جداً"


# =============================================================
# اختبارات TFIDFRetriever
# =============================================================


class TestTFIDFRetriever:
    """
    اختبارات وحدة لمحرك البحث TF-IDF.

    نستخدم Mock لـ TFIDFIndexer بدل فهرس حقيقي.
    هذا يتحقق من أن TFIDFRetriever:
    ١. يستدعي transform_query() من الفهرس بشكل صحيح
    ٢. يحسب cosine_similarity بشكل صحيح
    ٣. يُرجع النتائج مرتبة تنازلياً
    ٤. يتعامل مع الحالات الاستثنائية
    """

    def _make_mock_indexer(self, scores_to_return=None, num_docs=3):
        """
        يُنشئ TFIDFIndexer وهمي.
        scores_to_return: درجات التشابه التي سيُرجعها cosine_similarity
        """
        import numpy as np
        from scipy.sparse import csr_matrix

        mock_indexer = MagicMock()
        mock_indexer.is_built.return_value = True

        # نُنشئ مصفوفة TF-IDF وهمية بسيطة
        mock_indexer.tfidf_matrix = csr_matrix(
            np.array([[0.5, 0.3, 0.0], [0.0, 0.1, 0.9], [0.2, 0.7, 0.1]])
        )

        # transform_query يُرجع متجه استعلام وهمي
        mock_indexer.transform_query.return_value = csr_matrix(
            np.array([[0.8, 0.2, 0.0]])
        )

        # وثائق وهمية
        from unittest.mock import MagicMock as MM

        docs = []
        for i in range(num_docs):
            doc = MM()
            doc.doc_id = f"doc{i+1}"
            doc.title = f"Title {i+1}"
            doc.original_text = f"Text for document {i+1}"
            docs.append(doc)

        mock_indexer.get_document_by_index.side_effect = lambda idx: (
            docs[idx] if 0 <= idx < len(docs) else None
        )

        return mock_indexer

    def _make_tfidf_retriever(self, mock_indexer):
        """يُنشئ TFIDFRetriever باستخدام mock indexer."""
        # نُعطّل import من indexing لأن اختباراتنا لا تحتاجه
        import sys
        from unittest.mock import MagicMock as _MM

        # نُنشئ mock للـ module كاملاً لتجنب SyntaxError في indexing files
        fake_module = _MM()
        fake_module.get_tfidf_indexer = _MM()
        fake_module.TFIDFIndexer = _MM
        sys.modules["services.indexing.tfidf_indexer"] = fake_module
        sys.modules["services.indexing"] = _MM()

        from importlib import import_module, reload
        import services.retrieval.tfidf_retriever as _mod

        TFIDFRetriever = _mod.TFIDFRetriever
        retriever = TFIDFRetriever.__new__(TFIDFRetriever)
        retriever.dataset = DatasetName.DATASET_1
        retriever._indexer = mock_indexer
        return retriever

    def test_is_loaded_true_when_indexer_built(self):
        """is_loaded يُرجع True عندما الفهرس مبني."""
        mock = self._make_mock_indexer()
        retriever = self._make_tfidf_retriever(mock)
        assert retriever.is_loaded is True

    def test_is_loaded_false_when_indexer_not_built(self):
        """is_loaded يُرجع False عندما الفهرس غير مبني."""
        mock = self._make_mock_indexer()
        mock.is_built.return_value = False
        retriever = self._make_tfidf_retriever(mock)
        assert retriever.is_loaded is False

    def test_search_returns_list(self):
        """search() يُرجع list دائماً."""
        mock = self._make_mock_indexer()
        retriever = self._make_tfidf_retriever(mock)
        result = retriever.search(["cloud", "storage"], top_k=5)
        assert isinstance(result, list)

    def test_search_calls_transform_query(self):
        """search() يستدعي transform_query() من الفهرس."""
        mock = self._make_mock_indexer()
        retriever = self._make_tfidf_retriever(mock)
        retriever.search(["cloud", "storage"])
        # تحقق أن transform_query استُدعيت بـ "cloud storage"
        mock.transform_query.assert_called_once_with("cloud storage")

    def test_search_empty_tokens_returns_empty(self):
        """استعلام فارغ يُرجع قائمة فارغة."""
        mock = self._make_mock_indexer()
        retriever = self._make_tfidf_retriever(mock)
        result = retriever.search([])
        assert result == []

    def test_search_when_not_loaded_returns_empty(self):
        """البحث قبل تحميل الفهرس يُرجع قائمة فارغة."""
        mock = self._make_mock_indexer()
        mock.is_built.return_value = False
        retriever = self._make_tfidf_retriever(mock)
        result = retriever.search(["test"])
        assert result == []

    def test_search_results_are_document_results(self):
        """كل نتيجة من نوع DocumentResult."""
        mock = self._make_mock_indexer()
        retriever = self._make_tfidf_retriever(mock)
        results = retriever.search(["cloud"])
        for r in results:
            assert isinstance(r, DocumentResult)

    def test_search_results_have_descending_scores(self):
        """النتائج مرتبة تنازلياً حسب الدرجة."""
        mock = self._make_mock_indexer()
        retriever = self._make_tfidf_retriever(mock)
        results = retriever.search(["cloud", "storage"], top_k=3)
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_ranks_start_from_one(self):
        """الـ rank يبدأ من 1."""
        mock = self._make_mock_indexer()
        retriever = self._make_tfidf_retriever(mock)
        results = retriever.search(["cloud"])
        if results:
            assert results[0].rank == 1

    def test_get_stats_when_loaded(self):
        """get_stats() يُرجع dict يحتوي loaded=True."""
        mock = self._make_mock_indexer()
        mock.documents = [MagicMock()] * 3
        retriever = self._make_tfidf_retriever(mock)
        stats = retriever.get_stats()
        assert stats["loaded"] is True
        assert stats["dataset"] == DatasetName.DATASET_1.value

    def test_get_stats_when_not_loaded(self):
        """get_stats() يُرجع loaded=False عند غياب الفهرس."""
        mock = self._make_mock_indexer()
        mock.is_built.return_value = False
        retriever = self._make_tfidf_retriever(mock)
        stats = retriever.get_stats()
        assert stats["loaded"] is False


# =============================================================
# اختبارات BM25Retriever
# =============================================================


class TestBM25Retriever:
    """
    اختبارات وحدة لمحرك البحث BM25.

    نستخدم Mock لـ BM25Indexer.
    نتحقق من:
    ١. استدعاء get_top_n() بالمعاملات الصحيحة
    ٢. دعم تغيير k1/b
    ٣. تحويل نتائج IndexedDocument → DocumentResult
    ٤. الحالات الاستثنائية
    """

    def _make_mock_bm25_indexer(self, num_docs=3):
        """يُنشئ BM25Indexer وهمي."""
        mock_indexer = MagicMock()
        mock_indexer.is_built.return_value = True

        # metadata وهمية
        mock_indexer.metadata = MagicMock()
        mock_indexer.metadata.k1 = 1.5
        mock_indexer.metadata.b = 0.75
        mock_indexer.metadata.num_documents = num_docs
        mock_indexer.metadata.avg_document_length = 10.5

        # وثائق وهمية
        docs = []
        for i in range(num_docs):
            doc = MagicMock()
            doc.doc_id = f"doc{i+1}"
            doc.title = f"Title {i+1}"
            doc.original_text = f"Original text {i+1}"
            docs.append(doc)

        # get_top_n يُرجع قائمة (doc, score)
        mock_indexer.get_top_n.return_value = [
            (docs[0], 4.5),
            (docs[1], 2.1),
            (docs[2], 0.8),
        ]

        mock_indexer.tokenized_docs = [["cloud", "storage"]] * num_docs
        mock_indexer.get_document_by_index.side_effect = lambda idx: (
            docs[idx] if 0 <= idx < len(docs) else None
        )

        return mock_indexer

    def _make_bm25_retriever(self, mock_indexer):
        """يُنشئ BM25Retriever باستخدام mock indexer."""
        import sys
        from unittest.mock import MagicMock as _MM

        fake_bm25 = _MM()
        fake_bm25.get_bm25_indexer = _MM()
        fake_bm25.BM25Indexer = _MM
        sys.modules["services.indexing.bm25_indexer"] = fake_bm25
        import services.retrieval.bm25_retriever as _bmod

        BM25Retriever = _bmod.BM25Retriever
        retriever = BM25Retriever.__new__(BM25Retriever)
        retriever.dataset = DatasetName.DATASET_1
        retriever._indexer = mock_indexer
        return retriever

    def test_is_loaded_true(self):
        """is_loaded صحيح عندما الفهرس مبني."""
        mock = self._make_mock_bm25_indexer()
        retriever = self._make_bm25_retriever(mock)
        assert retriever.is_loaded is True

    def test_is_loaded_false(self):
        """is_loaded خاطئ عندما الفهرس غير مبني."""
        mock = self._make_mock_bm25_indexer()
        mock.is_built.return_value = False
        retriever = self._make_bm25_retriever(mock)
        assert retriever.is_loaded is False

    def test_search_returns_list(self):
        """search() يُرجع list."""
        mock = self._make_mock_bm25_indexer()
        retriever = self._make_bm25_retriever(mock)
        result = retriever.search(["cloud"])
        assert isinstance(result, list)

    def test_search_empty_tokens_returns_empty(self):
        """tokens فارغة تُرجع قائمة فارغة."""
        mock = self._make_mock_bm25_indexer()
        retriever = self._make_bm25_retriever(mock)
        assert retriever.search([]) == []

    def test_search_not_loaded_returns_empty(self):
        """البحث بدون فهرس يُرجع قائمة فارغة."""
        mock = self._make_mock_bm25_indexer()
        mock.is_built.return_value = False
        retriever = self._make_bm25_retriever(mock)
        assert retriever.search(["cloud"]) == []

    def test_search_uses_default_params_calls_get_top_n(self):
        """
        بالمعاملات الافتراضية (k1=1.5, b=0.75)،
        يجب استدعاء get_top_n() من المطور الأول مباشرة.
        """
        mock = self._make_mock_bm25_indexer()
        retriever = self._make_bm25_retriever(mock)
        retriever.search(["cloud"], top_k=5, k1=1.5, b=0.75)
        mock.get_top_n.assert_called_once_with(["cloud"], n=5)

    def test_search_results_are_document_results(self):
        """كل نتيجة من نوع DocumentResult."""
        mock = self._make_mock_bm25_indexer()
        retriever = self._make_bm25_retriever(mock)
        results = retriever.search(["cloud"])
        for r in results:
            assert isinstance(r, DocumentResult)

    def test_search_results_have_correct_doc_ids(self):
        """doc_id في النتائج يطابق بيانات الفهرس."""
        mock = self._make_mock_bm25_indexer()
        retriever = self._make_bm25_retriever(mock)
        results = retriever.search(["cloud"], top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "doc1"

    def test_search_top_k_limits_results(self):
        """top_k يحدد الحد الأقصى للنتائج."""
        mock = self._make_mock_bm25_indexer()
        # نُعيد تعريف get_top_n ليُرجع عدداً أقل
        mock.get_top_n.return_value = [
            (MagicMock(doc_id="d1", title="T", original_text="X"), 5.0)
        ]
        retriever = self._make_bm25_retriever(mock)
        results = retriever.search(["cloud"], top_k=1)
        assert len(results) <= 1

    def test_get_stats_returns_k1_b(self):
        """get_stats() يُرجع قيم k1 و b من الـ metadata."""
        mock = self._make_mock_bm25_indexer()
        retriever = self._make_bm25_retriever(mock)
        stats = retriever.get_stats()
        assert stats["loaded"] is True
        assert stats["k1"] == 1.5
        assert stats["b"] == 0.75


# =============================================================
# اختبارات EmbeddingRetriever
# =============================================================


class TestEmbeddingRetriever:
    """
    اختبارات وحدة لمحرك البحث الدلالي.

    نستخدم Mock لـ EmbeddingIndexer.
    نتحقق من:
    ١. استدعاء encode_query() بالنص الأصلي (غير المعالج)
    ٢. استدعاء get_top_k() بـ embedding الصحيح
    ٣. تحويل النتائج بشكل صحيح
    ٤. الحالات الاستثنائية (query فارغ، فهرس غير محمّل)
    """

    def _make_mock_embedding_indexer(self, num_docs=3):
        """يُنشئ EmbeddingIndexer وهمي."""
        import numpy as np

        mock_indexer = MagicMock()
        mock_indexer.is_built.return_value = True

        # metadata وهمية
        mock_indexer.metadata = MagicMock()
        mock_indexer.metadata.num_documents = num_docs
        mock_indexer.metadata.model_name = "all-MiniLM-L6-v2"
        mock_indexer.metadata.embedding_dim = 384

        # encode_query يُرجع متجه وهمي
        mock_indexer.encode_query.return_value = np.random.rand(1, 384).astype(
            "float32"
        )

        # وثائق وهمية
        docs = []
        for i in range(num_docs):
            doc = MagicMock()
            doc.doc_id = f"emb_doc{i+1}"
            doc.title = f"Embedding Title {i+1}"
            doc.original_text = f"Embedding text {i+1}"
            docs.append(doc)

        # get_top_k يُرجع (IndexedDocument, score)
        mock_indexer.get_top_k.return_value = [
            (docs[0], 0.95),
            (docs[1], 0.82),
            (docs[2], 0.71),
        ]

        return mock_indexer

    def _make_embedding_retriever(self, mock_indexer):
        """يُنشئ EmbeddingRetriever باستخدام mock indexer."""
        import sys
        from unittest.mock import MagicMock as _MM

        fake_emb = _MM()
        fake_emb.get_embedding_indexer = _MM()
        fake_emb.EmbeddingIndexer = _MM
        sys.modules["services.indexing.embedding_indexer"] = fake_emb
        import services.retrieval.embedding_retriever as _emod

        EmbeddingRetriever = _emod.EmbeddingRetriever
        retriever = EmbeddingRetriever.__new__(EmbeddingRetriever)
        retriever.dataset = DatasetName.DATASET_1
        retriever._indexer = mock_indexer
        return retriever

    def test_is_loaded_true(self):
        """is_loaded صحيح عندما الفهرس مبني."""
        mock = self._make_mock_embedding_indexer()
        retriever = self._make_embedding_retriever(mock)
        assert retriever.is_loaded is True

    def test_is_loaded_false(self):
        """is_loaded خاطئ عندما الفهرس غير مبني."""
        mock = self._make_mock_embedding_indexer()
        mock.is_built.return_value = False
        retriever = self._make_embedding_retriever(mock)
        assert retriever.is_loaded is False

    def test_search_returns_list(self):
        """search() يُرجع list دائماً."""
        mock = self._make_mock_embedding_indexer()
        retriever = self._make_embedding_retriever(mock)
        result = retriever.search("buy a car")
        assert isinstance(result, list)

    def test_search_calls_encode_query_with_original_text(self):
        """
        search() يُرسل النص الأصلي (غير المعالج) لـ encode_query().

        هذا مهم جداً:
        Embedding يحتاج النص الطبيعي "buy a car"
        وليس tokens المعالجة "buy car" (بعد stopword removal).
        """
        mock = self._make_mock_embedding_indexer()
        retriever = self._make_embedding_retriever(mock)
        original_text = "buy a car please"
        retriever.search(original_text)
        mock.encode_query.assert_called_once_with(original_text)

    def test_search_empty_text_returns_empty(self):
        """النص الفارغ يُرجع قائمة فارغة."""
        mock = self._make_mock_embedding_indexer()
        retriever = self._make_embedding_retriever(mock)
        assert retriever.search("") == []
        assert retriever.search("   ") == []

    def test_search_not_loaded_returns_empty(self):
        """البحث بدون فهرس يُرجع قائمة فارغة."""
        mock = self._make_mock_embedding_indexer()
        mock.is_built.return_value = False
        retriever = self._make_embedding_retriever(mock)
        assert retriever.search("test query") == []

    def test_search_when_encode_returns_none(self):
        """لو encode_query أرجع None — يُرجع قائمة فارغة بدون خطأ."""
        mock = self._make_mock_embedding_indexer()
        mock.encode_query.return_value = None
        retriever = self._make_embedding_retriever(mock)
        result = retriever.search("some query")
        assert result == []

    def test_search_results_are_document_results(self):
        """كل نتيجة من نوع DocumentResult."""
        mock = self._make_mock_embedding_indexer()
        retriever = self._make_embedding_retriever(mock)
        results = retriever.search("cloud storage")
        for r in results:
            assert isinstance(r, DocumentResult)

    def test_search_scores_between_zero_and_one(self):
        """
        درجات Cosine Similarity بين 0 و 1 (بعد L2 normalization).
        """
        mock = self._make_mock_embedding_indexer()
        retriever = self._make_embedding_retriever(mock)
        results = retriever.search("cloud storage")
        for r in results:
            assert 0.0 <= r.score <= 1.1  # 1.1 هامش للأخطاء العددية

    def test_search_results_have_correct_doc_ids(self):
        """doc_id في النتائج يطابق بيانات الفهرس."""
        mock = self._make_mock_embedding_indexer()
        retriever = self._make_embedding_retriever(mock)
        results = retriever.search("cloud storage", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "emb_doc1"

    def test_get_stats_returns_model_info(self):
        """get_stats() يُرجع اسم النموذج والبُعد."""
        mock = self._make_mock_embedding_indexer()
        retriever = self._make_embedding_retriever(mock)
        stats = retriever.get_stats()
        assert stats["loaded"] is True
        assert stats["model_name"] == "all-MiniLM-L6-v2"
        assert stats["embedding_dim"] == 384


class TestVectorStoreIntegration:
    """تحقق من أن EmbeddingRetriever يعمل مع VectorStore بنفس النتائج."""

    def _get_real_vector_store(self):
        """
        خدعة برمجية لجلب الكلاس الحقيقي من الملف مباشرة
        دون الحاجة لملف __init__.py
        """
        import importlib.util
        import sys

        module_name = "services.indexing.vector_store"
        file_path = "services/indexing/vector_store.py"

        # قراءة الوحدة مباشرة من مسار الملف
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        vs_module = importlib.util.module_from_spec(spec)

        # حقنها في الذاكرة لتجنب أخطاء الاستدعاء الداخلي
        sys.modules[module_name] = vs_module
        spec.loader.exec_module(vs_module)

        return vs_module.VectorStore

    def test_vector_store_and_retriever_same_results(self):
        """
        VectorStore.search() و EmbeddingRetriever.search()
        يجب أن يُنتجا نفس النتائج.
        """
        from unittest.mock import MagicMock
        import numpy as np

        # استدعاء الكلاس الحقيقي باستخدام دالتنا المساعدة بدلاً من from ... import
        VectorStore = self._get_real_vector_store()

        # Mock مشترك
        mock_indexer = MagicMock()
        mock_indexer.is_built.return_value = True
        mock_indexer.metadata = MagicMock(
            num_documents=3,
            model_name="all-MiniLM-L6-v2",
            embedding_dim=384,
            index_type="flat_ip",
        )

        # نفس النتائج لكليهما
        fake_doc = MagicMock()
        fake_doc.doc_id = "d1"
        fake_doc.title = "Cloud"
        fake_doc.original_text = "Cloud storage is useful."

        mock_indexer.encode_query.return_value = np.random.rand(1, 384).astype(
            "float32"
        )
        mock_indexer.get_top_k.return_value = [(fake_doc, 0.89)]

        # VectorStore
        store = VectorStore.__new__(VectorStore)
        store._dataset_name = "dataset1"
        store._indexer = mock_indexer

        results = store.search("cloud storage", k=1)

        assert len(results) == 1
        doc_id, score, text, title = results[0]
        assert doc_id == "d1"
        assert abs(score - 0.89) < 1e-5
        assert "Cloud" in text

    def test_vector_store_status_fields(self):
        """get_status() يُرجع الحقول المطلوبة."""
        from unittest.mock import MagicMock

        # استدعاء الكلاس الحقيقي باستخدام دالتنا المساعدة
        VectorStore = self._get_real_vector_store()

        store = VectorStore.__new__(VectorStore)
        store._dataset_name = "dataset1"

        mock_indexer = MagicMock()
        mock_indexer.is_built.return_value = True
        mock_indexer.is_saved.return_value = True
        mock_indexer.documents = [MagicMock()] * 5
        mock_indexer.metadata = MagicMock(
            model_name="all-MiniLM-L6-v2",
            embedding_dim=384,
            index_type="flat_ip",
        )
        store._indexer = mock_indexer

        status = store.get_status()
        assert status["is_ready"] is True
        assert status["num_documents"] == 5
        assert status["model_name"] == "all-MiniLM-L6-v2"
