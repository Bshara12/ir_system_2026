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
