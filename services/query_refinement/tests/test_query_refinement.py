"""
services/query_refinement/tests/test_query_refinement.py
=========================================================
اختبارات وحدة لخدمة تحسين الاستعلامات.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا نختبر Query Refinement؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
هذه الخدمة تؤثر على كل عملية بحث عند apply_refinement=true.
خطأ فيها = كل نتائج البحث تتأثر!

ما نختبره:
    - SpellCorrector: هل يُصحّح الأخطاء الإملائية بشكل صحيح؟
    - SynonymExpander: هل يُضيف مرادفات منطقية؟
    - QueryHistory: هل يحفظ ويسترجع السجل بشكل صحيح؟
    - SuggestionEngine: هل يُنتج اقتراحات منطقية؟
    - app.py endpoints: هل الـ API يعمل بشكل صحيح؟

التشغيل:
    cd ir_system_2026
    python -m pytest services/query_refinement/tests/ -v
"""

import sys
import os
import tempfile
import json

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from services.query_refinement.spell_corrector import SpellCorrector
from services.query_refinement.synonym_expander import SynonymExpander
from services.query_refinement.query_history import QueryHistory, QueryHistoryEntry
from services.query_refinement.suggestion_engine import SuggestionEngine

# =============================================================
# Fixtures
# =============================================================


@pytest.fixture
def spell_corrector():
    """نسخة SpellCorrector للاختبارات."""
    return SpellCorrector()


@pytest.fixture
def synonym_expander():
    """نسخة SynonymExpander للاختبارات."""
    return SynonymExpander(max_synonyms=3)


@pytest.fixture
def query_history(tmp_path):
    """
    نسخة QueryHistory تستخدم مجلداً مؤقتاً.
    tmp_path هو fixture من pytest يُنشئ مجلداً مؤقتاً يُحذف بعد الاختبار.
    هذا يمنع تلوّث بيانات الاختبارات الحقيقية.
    """
    return QueryHistory(history_dir=str(tmp_path / "history"))


@pytest.fixture
def suggestion_engine(query_history, synonym_expander):
    """نسخة SuggestionEngine تستخدم history و expander محددَين."""
    return SuggestionEngine(
        history=query_history,
        expander=synonym_expander,
        max_suggestions=5,
    )


# =============================================================
# اختبارات SpellCorrector
# =============================================================


class TestSpellCorrector:
    """اختبارات مُصحّح الإملاء."""

    def test_correct_returns_tuple(self, spell_corrector):
        """
        correct() يجب أن يُرجع tuple من (str, bool).
        str = الاستعلام المُصحَّح
        bool = هل تم تصحيح شيء؟
        """
        result = spell_corrector.correct("hello world")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], bool)

    def test_correct_word_unchanged(self, spell_corrector):
        """
        الكلمة الصحيحة لا تتغير.
        "information" موجودة في القاموس → تبقى كما هي.
        """
        corrected, was_fixed = spell_corrector.correct("information retrieval")
        assert "information" in corrected
        assert "retrieval" in corrected

    def test_empty_query_returns_unchanged(self, spell_corrector):
        """الاستعلام الفارغ يُرجع فارغاً بدون خطأ."""
        corrected, was_fixed = spell_corrector.correct("")
        assert corrected == ""
        assert was_fixed is False

    def test_whitespace_only_returns_unchanged(self, spell_corrector):
        """الاستعلام الذي يحتوي مسافات فقط يُرجع بدون تغيير."""
        corrected, was_fixed = spell_corrector.correct("   ")
        assert was_fixed is False

    def test_technical_terms_not_corrected(self, spell_corrector):
        """
        المصطلحات التقنية لا تُصحَّح.
        "bm25" ليست كلمة إنجليزية عادية لكن نُضيفها للقاموس.
        """
        corrected, _ = spell_corrector.correct("bm25 tfidf retrieval")
        # المصطلحات التقنية يجب أن تبقى
        assert "bm25" in corrected.lower()
        assert "tfidf" in corrected.lower()

    def test_short_words_not_corrected(self, spell_corrector):
        """
        الكلمات القصيرة (≤2 حرف) لا تُصحَّح.
        "IR", "AI", "ML" تبقى كما هي.
        """
        corrected, _ = spell_corrector.correct("IR AI ML systems")
        assert "ir" in corrected.lower() or "IR" in corrected

    def test_numbers_not_corrected(self, spell_corrector):
        """الأرقام لا تُصحَّح."""
        corrected, _ = spell_corrector.correct("top 10 results 2026")
        assert "10" in corrected
        assert "2026" in corrected

    def test_is_available_property(self, spell_corrector):
        """is_available يُرجع True أو False حسب وجود المكتبة."""
        # الخاصية يجب أن تعمل بدون خطأ
        result = spell_corrector.is_available
        assert isinstance(result, bool)

    def test_get_candidates_returns_list(self, spell_corrector):
        """get_candidates يُرجع قائمة (حتى لو فارغة)."""
        candidates = spell_corrector.get_candidates("retriveal")
        assert isinstance(candidates, list)

    def test_correct_preserves_original_if_no_fix(self, spell_corrector):
        """إذا لا يوجد تصحيح — يُرجع النص الأصلي بالضبط."""
        query = "machine learning"
        corrected, was_fixed = spell_corrector.correct(query)
        # machine و learning موجودتان في القاموس
        assert was_fixed is False or corrected == query


# =============================================================
# اختبارات SynonymExpander
# =============================================================


class TestSynonymExpander:
    """اختبارات موسّع المرادفات."""

    def test_expand_returns_string(self, synonym_expander):
        """expand() يُرجع string دائماً."""
        result = synonym_expander.expand("fast car")
        assert isinstance(result, str)

    def test_expand_contains_original_words(self, synonym_expander):
        """
        الاستعلام الموسّع يجب أن يحتوي الكلمات الأصلية.
        إذا كتبنا "car" — يجب أن "car" تبقى في النتيجة.
        """
        result = synonym_expander.expand("car")
        assert "car" in result.lower()

    def test_expand_empty_returns_empty(self, synonym_expander):
        """الاستعلام الفارغ يُرجع فارغاً."""
        result = synonym_expander.expand("")
        assert result == ""

    def test_expand_adds_words_or_keeps_same(self, synonym_expander):
        """
        الاستعلام الموسّع يكون أطول أو مساوياً للأصلي.
        (لا يمكن أن يكون أقصر — نحن نُضيف فقط، لا نحذف)
        """
        original = "fast car"
        expanded = synonym_expander.expand(original)
        # عدد الكلمات في النتيجة >= عدد كلمات الأصل
        assert len(expanded.split()) >= len(original.split())

    def test_no_duplicate_words(self, synonym_expander):
        """لا تكرار في الاستعلام الموسّع."""
        expanded = synonym_expander.expand("good information")
        words = expanded.lower().split()
        # عدد الكلمات الفريدة = عدد الكلمات الكلي (لا تكرار)
        assert len(words) == len(set(words))

    def test_technical_terms_not_expanded(self, synonym_expander):
        """المصطلحات التقنية لا تُوسَّع."""
        # "bm25" في قائمة _SKIP_WORDS
        result = synonym_expander.expand("bm25 ranking")
        # bm25 يجب أن يبقى كما هو (لا مرادفات له)
        assert "bm25" in result.lower()

    def test_get_synonyms_for_word_returns_list(self, synonym_expander):
        """get_synonyms_for_word يُرجع قائمة."""
        result = synonym_expander.get_synonyms_for_word("car")
        assert isinstance(result, list)

    def test_is_available_property(self, synonym_expander):
        """is_available يُرجع bool."""
        assert isinstance(synonym_expander.is_available, bool)

    def test_max_synonyms_respected(self, synonym_expander):
        """
        عدد المرادفات المُضافة لكل كلمة لا يتجاوز max_synonyms.
        عندنا max_synonyms=3، كلمة واحدة → أقصى 3 مرادفات مُضافة.
        """
        # كلمة واحدة
        original_count = 1  # كلمة واحدة "car"
        expanded = synonym_expander.expand("car")
        expanded_count = len(expanded.split())
        # أقصى = original + max_synonyms
        assert expanded_count <= original_count + synonym_expander.max_synonyms


# =============================================================
# اختبارات QueryHistory
# =============================================================


class TestQueryHistory:
    """اختبارات سجل الاستعلامات."""

    def test_add_and_get_recent(self, query_history):
        """
        بعد إضافة استعلام، يجب أن يظهر في get_recent().
        """
        query_history.add(
            session_id="test_session",
            query="machine learning",
        )
        recent = query_history.get_recent("test_session", limit=5)
        assert len(recent) == 1
        assert recent[0].query == "machine learning"

    def test_recent_is_newest_first(self, query_history):
        """
        get_recent() يُرجع الأحدث أولاً.
        """
        query_history.add("sess1", "first query")
        query_history.add("sess1", "second query")
        query_history.add("sess1", "third query")

        recent = query_history.get_recent("sess1", limit=3)
        # الأحدث أولاً
        assert recent[0].query == "third query"
        assert recent[1].query == "second query"
        assert recent[2].query == "first query"

    def test_empty_query_not_saved(self, query_history):
        """الاستعلام الفارغ لا يُحفظ."""
        query_history.add("sess1", "   ")
        recent = query_history.get_recent("sess1")
        assert len(recent) == 0

    def test_max_history_limit(self, query_history):
        """
        السجل لا يتجاوز MAX_HISTORY_PER_SESSION.
        """
        # نضيف أكثر من الحد
        limit = query_history.MAX_HISTORY_PER_SESSION
        for i in range(limit + 10):
            query_history.add("sess_limit", f"query {i}")

        recent = query_history.get_recent("sess_limit", limit=9999)
        assert len(recent) <= limit

    def test_different_sessions_isolated(self, query_history):
        """
        كل جلسة مستقلة — استعلامات جلسة A لا تظهر في جلسة B.
        """
        query_history.add("session_A", "query for A")
        query_history.add("session_B", "query for B")

        history_A = query_history.get_recent("session_A")
        history_B = query_history.get_recent("session_B")

        assert all(e.query == "query for A" for e in history_A)
        assert all(e.query == "query for B" for e in history_B)

    def test_clear_session(self, query_history):
        """بعد clear_session() السجل يصبح فارغاً."""
        query_history.add("sess_clear", "query 1")
        query_history.add("sess_clear", "query 2")

        query_history.clear_session("sess_clear")
        recent = query_history.get_recent("sess_clear")
        assert len(recent) == 0

    def test_get_similar_past_queries(self, query_history):
        """
        get_similar_past_queries() يجد استعلامات تشارك كلمات.
        """
        query_history.add("sess1", "machine learning algorithms")
        query_history.add("sess1", "deep learning models")
        query_history.add("sess1", "cloud storage systems")

        similar = query_history.get_similar_past_queries(
            session_id="sess1",
            current_query="machine learning",
            limit=5,
        )
        # "machine learning algorithms" و"deep learning models"
        # يشاركان كلمات مع "machine learning"
        assert len(similar) >= 1
        # كلها يجب أن تحتوي كلمة مشتركة مع "machine learning"
        for s in similar:
            words = set(s.lower().split())
            assert words & {"machine", "learning"}

    def test_get_all_queries_no_duplicates(self, query_history):
        """get_all_queries_for_session يُرجع بدون تكرار."""
        query_history.add("sess1", "machine learning")
        query_history.add("sess1", "machine learning")  # مكرر
        query_history.add("sess1", "cloud storage")

        all_q = query_history.get_all_queries_for_session("sess1")
        assert len(all_q) == len(set(all_q))  # لا تكرار

    def test_nonexistent_session_returns_empty(self, query_history):
        """جلسة غير موجودة تُرجع قائمة فارغة بدون خطأ."""
        recent = query_history.get_recent("nonexistent_session_xyz")
        assert recent == []

    def test_entry_has_timestamp(self, query_history):
        """كل إدخال يجب أن يحتوي timestamp."""
        query_history.add("sess1", "test query")
        recent = query_history.get_recent("sess1")
        assert recent[0].timestamp is not None
        assert len(recent[0].timestamp) > 0

    def test_entry_stores_model_and_dataset(self, query_history):
        """يجب حفظ نموذج البحث ومجموعة البيانات مع كل استعلام."""
        query_history.add(
            session_id="sess1",
            query="test",
            model="tfidf",
            dataset="dataset2",
            results_count=7,
        )
        recent = query_history.get_recent("sess1")
        assert recent[0].model == "tfidf"
        assert recent[0].dataset == "dataset2"
        assert recent[0].results_count == 7


# =============================================================
# اختبارات SuggestionEngine
# =============================================================


class TestSuggestionEngine:
    """اختبارات محرك الاقتراحات."""

    def test_suggest_returns_list(self, suggestion_engine):
        """suggest() يُرجع list دائماً."""
        result = suggestion_engine.suggest("machine", session_id="test")
        assert isinstance(result, list)

    def test_suggest_empty_returns_popular(self, suggestion_engine):
        """
        الاستعلام الفارغ يُرجع الاقتراحات الشائعة.
        """
        result = suggestion_engine.suggest("", session_id="test")
        assert len(result) > 0

    def test_suggest_respects_max_limit(self, suggestion_engine):
        """
        النتائج لا تتجاوز max_suggestions.
        """
        result = suggestion_engine.suggest("information", session_id="test")
        assert len(result) <= suggestion_engine.max_suggestions

    def test_suggest_no_duplicates(self, suggestion_engine):
        """الاقتراحات لا تحتوي تكراراً."""
        result = suggestion_engine.suggest("machine", session_id="test")
        # لكل اقتراح lowercase للمقارنة
        lower_results = [s.lower() for s in result]
        assert len(lower_results) == len(set(lower_results))

    def test_suggest_from_history(self, suggestion_engine, query_history):
        """
        الاقتراحات تشمل استعلامات من السجل الشخصي.
        """
        # نضيف استعلام للسجل
        query_history.add("hist_sess", "machine learning in practice")

        result = suggestion_engine.suggest(
            "machine",
            session_id="hist_sess",
        )
        # يجب أن يظهر الاستعلام من السجل ضمن الاقتراحات
        assert "machine learning in practice" in result

    def test_get_popular_queries_returns_list(self, suggestion_engine):
        """get_popular_queries يُرجع list غير فارغة."""
        popular = suggestion_engine.get_popular_queries(limit=5)
        assert isinstance(popular, list)
        assert len(popular) > 0
        assert len(popular) <= 5

    def test_suggest_original_not_in_suggestions(self, suggestion_engine):
        """
        الاستعلام الأصلي لا يظهر ضمن اقتراحاته هو.
        (لا معنى لاقتراح ما كتبه المستخدم بالفعل)
        """
        query = "machine learning"
        result = suggestion_engine.suggest(query, session_id="test")
        lower_results = [s.lower() for s in result]
        assert query.lower() not in lower_results


# =============================================================
# اختبارات API (Integration Tests)
# =============================================================


class TestQueryRefinementAPI:
    """اختبارات الـ FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """TestClient لاختبار الـ API."""
        from services.query_refinement.app import app

        return TestClient(app)

    def test_health_endpoint_returns_200(self, client):
        """GET /health يُرجع 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service_name"] == "query_refinement"

    def test_health_shows_components(self, client):
        """GET /health يُظهر حالة كل مكون."""
        response = client.get("/health")
        details = response.json()["details"]
        assert "spell_corrector" in details
        assert "synonym_expander" in details
        assert "query_history" in details
        assert "suggestion_engine" in details

    def test_refine_basic_request(self, client):
        """POST /refine يقبل طلب بسيط ويُرجع 200."""
        response = client.post(
            "/refine",
            json={
                "query": "machine learning",
                "session_id": "test_session",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "original_query" in data
        assert "refined_query" in data
        assert "spell_corrected" in data
        assert "synonyms_added" in data

    def test_refine_preserves_original_query(self, client):
        """POST /refine يحتفظ بالاستعلام الأصلي في original_query."""
        query = "information retrieval"
        response = client.post("/refine", json={"query": query})
        data = response.json()
        assert data["original_query"] == query

    def test_refine_empty_query_returns_422(self, client):
        """POST /refine باستعلام فارغ يُرجع 422 (validation error)."""
        response = client.post("/refine", json={"query": ""})
        assert response.status_code == 422

    def test_refine_with_spell_correction_disabled(self, client):
        """
        عند apply_spell_correction=false لا يتم التصحيح.
        """
        response = client.post(
            "/refine",
            json={
                "query": "infromation",
                "apply_spell_correction": False,
            },
        )
        data = response.json()
        assert data["spell_corrected"] is False

    def test_refine_corrections_field_is_list(self, client):
        """حقل corrections يجب أن يكون list دائماً."""
        response = client.post("/refine", json={"query": "machine learning"})
        data = response.json()
        assert isinstance(data["corrections"], list)

    def test_suggest_endpoint_returns_200(self, client):
        """GET /suggest يُرجع 200 مع اقتراحات."""
        response = client.get("/suggest?q=machine")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_suggest_without_q_returns_422(self, client):
        """GET /suggest بدون q يُرجع 422."""
        response = client.get("/suggest")
        assert response.status_code == 422

    def test_history_endpoint_returns_200(self, client):
        """GET /history/{session_id} يُرجع 200."""
        response = client.get("/history/test_session_123")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "queries" in data
        assert "total" in data

    def test_history_delete_returns_200(self, client):
        """DELETE /history/{session_id} يُرجع 200."""
        response = client.delete("/history/test_session_to_delete")
        assert response.status_code == 200

    def test_popular_endpoint_returns_200(self, client):
        """GET /popular يُرجع 200 مع قائمة."""
        response = client.get("/popular")
        assert response.status_code == 200
        data = response.json()
        assert "popular_queries" in data
        assert len(data["popular_queries"]) > 0

    def test_full_refinement_pipeline(self, client):
        """
        اختبار تكامل كامل:
        1. تحسين استعلام
        2. التحقق من السجل
        3. الحصول على اقتراحات
        """
        session = "integration_test_session"

        # الخطوة 1: تحسين استعلام
        refine_resp = client.post(
            "/refine",
            json={
                "query": "machine learning",
                "session_id": session,
                "save_to_history": True,
            },
        )
        assert refine_resp.status_code == 200

        # الخطوة 2: التحقق من السجل
        history_resp = client.get(f"/history/{session}")
        assert history_resp.status_code == 200
        history_data = history_resp.json()
        assert history_data["total"] >= 1

        # الخطوة 3: الحصول على اقتراحات
        suggest_resp = client.get(f"/suggest?q=machine&session_id={session}")
        assert suggest_resp.status_code == 200
        suggestions = suggest_resp.json()["suggestions"]
        # يجب أن يقترح "machine learning" من السجل
        assert "machine learning" in suggestions
