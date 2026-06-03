"""
services/query_refinement/app.py
=================================
FastAPI server لخدمة تحسين الاستعلامات — Query Refinement Service (port 8004).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما الذي تفعله هذه الخدمة؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
تستقبل استعلاماً من Retrieval Service وتُحسّنه قبل البحث.
التحسين يشمل:
    1. تصحيح الإملاء:  "infromation" → "information"
    2. توسيع بمرادفات: "car" → "car automobile vehicle"
    3. سجل الاستعلامات: حفظ كل بحث للاستخدام لاحقاً
    4. اقتراح استعلامات: "mach..." → "machine learning"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الـ Endpoints:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /refine          ← تحسين الاستعلام (الأهم)
GET  /suggest         ← اقتراح استعلامات
GET  /history/{id}    ← عرض السجل
DELETE /history/{id}  ← مسح السجل
GET  /health          ← حالة الخدمة

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
كيف تتواصل مع Retrieval Service؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retrieval Service يستدعي هذه الخدمة عندما:
    apply_refinement = true في طلب البحث

التدفق:
    User → Gateway → Retrieval (8003)
                         ↓ (إذا apply_refinement=true)
               Query Refinement (8004)
                         ↓
               استعلام مُحسَّن
                         ↓
               الفهرس (BM25/TFIDF/Embedding)
                         ↓
               نتائج أفضل للمستخدم

تشغيل:
    uvicorn services.query_refinement.app:app --port 8004 --reload

Swagger UI:
    http://localhost:8004/docs
"""

import sys
import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from shared.models import ServiceStatus, ErrorResponse
from shared.constants import QUERY_REFINEMENT_PORT

from services.query_refinement.spell_corrector import (
    get_spell_corrector,
    SpellCorrector,
)
from services.query_refinement.synonym_expander import (
    get_synonym_expander,
    SynonymExpander,
)
from services.query_refinement.query_history import get_query_history, QueryHistory
from services.query_refinement.suggestion_engine import (
    get_suggestion_engine,
    SuggestionEngine,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================
# نماذج البيانات الخاصة بهذه الخدمة
# =============================================================


class RefineRequest(BaseModel):
    """
    طلب تحسين استعلام.
    يأتي من: Retrieval Service أو مباشرة من الواجهة.
    """

    query: str = Field(
        ...,
        description="الاستعلام المراد تحسينه",
        min_length=1,
    )
    session_id: str = Field(
        default="default",
        description="معرّف الجلسة — لاستخدام السجل الشخصي",
    )
    apply_spell_correction: bool = Field(
        default=True,
        description="تطبيق تصحيح الإملاء",
    )
    apply_synonym_expansion: bool = Field(
        default=False,
        description="إضافة مرادفات للاستعلام (قد يُبطئ البحث قليلاً)",
    )
    save_to_history: bool = Field(
        default=True,
        description="حفظ الاستعلام في السجل",
    )


class RefineResponse(BaseModel):
    """
    نتيجة تحسين الاستعلام.
    """

    original_query: str = Field(description="الاستعلام الأصلي")
    refined_query: str = Field(description="الاستعلام بعد التحسين")
    spell_corrected: bool = Field(description="هل تم تصحيح الإملاء؟")
    synonyms_added: bool = Field(description="هل أُضيفت مرادفات؟")
    corrections: List[str] = Field(
        default=[],
        description="قائمة التصحيحات التي أُجريت",
    )


class SuggestResponse(BaseModel):
    """نتيجة طلب الاقتراحات."""

    query: str = Field(description="الاستعلام الجزئي")
    suggestions: List[str] = Field(description="قائمة الاقتراحات")


class HistoryResponse(BaseModel):
    """نتيجة طلب السجل."""

    session_id: str
    queries: List[dict]
    total: int


# =============================================================
# lifespan — تهيئة المكونات عند البدء
# =============================================================


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """يُهيّئ كل مكونات Query Refinement مرة واحدة عند البدء."""
    logger.info("[QueryRefinement] تهيئة المكونات...")
    get_spell_corrector()
    get_synonym_expander()
    get_query_history()
    get_suggestion_engine()
    logger.info("[QueryRefinement] ✅ جميع المكونات جاهزة")
    yield
    logger.info("[QueryRefinement] إيقاف الخدمة...")


# =============================================================
# إنشاء التطبيق
# =============================================================

app = FastAPI(
    title="Query Refinement Service",
    description=(
        "خدمة تحسين الاستعلامات في نظام IR 2026.\n\n"
        "**المكونات:**\n"
        "- **Spell Corrector**: تصحيح الأخطاء الإملائية (pyspellchecker)\n"
        "- **Synonym Expander**: توسيع بالمرادفات (NLTK WordNet)\n"
        "- **Query History**: سجل الاستعلامات السابقة\n"
        "- **Suggestion Engine**: اقتراح استعلامات ذكي\n\n"
        "**الاستخدام الرئيسي:**\n"
        "يُستدعى من Retrieval Service عند `apply_refinement=true`"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# Endpoints
# =============================================================


@app.get("/health", response_model=ServiceStatus, tags=["System"])
async def health_check() -> ServiceStatus:
    """
    يُرجع حالة الخدمة وتفاصيل المكونات المُحمَّلة.
    """
    corrector = get_spell_corrector()
    expander = get_synonym_expander()

    return ServiceStatus(
        service_name="query_refinement",
        status="healthy",
        details={
            "spell_corrector": {"available": corrector.is_available},
            "synonym_expander": {"available": expander.is_available},
            "query_history": {"available": True},
            "suggestion_engine": {"available": True},
        },
    )


@app.post(
    "/refine",
    response_model=RefineResponse,
    tags=["Refinement"],
    summary="تحسين استعلام البحث",
    responses={422: {"model": ErrorResponse}},
)
async def refine_query(request: RefineRequest) -> RefineResponse:
    """
    يُحسّن استعلام البحث بتصحيح الإملاء وإضافة المرادفات.

    **مثال:**
    ```json
    {
        "query": "infromation retreival",
        "session_id": "user_123",
        "apply_spell_correction": true,
        "apply_synonym_expansion": false
    }
    ```

    **النتيجة:**
    ```json
    {
        "original_query": "infromation retreival",
        "refined_query": "information retrieval",
        "spell_corrected": true,
        "synonyms_added": false,
        "corrections": ["infromation→information", "retreival→retrieval"]
    }
    ```
    """
    original = request.query
    current = request.query
    corrections: List[str] = []
    spell_corrected = False
    synonyms_added = False

    # ── الخطوة 1: تصحيح الإملاء ──────────────────────────────
    if request.apply_spell_correction:
        corrector = get_spell_corrector()
        corrected, was_fixed = corrector.correct(current)
        if was_fixed:
            # نُسجّل التصحيحات لإظهارها للمستخدم
            original_words = current.split()
            corrected_words = corrected.split()
            for orig_w, corr_w in zip(original_words, corrected_words):
                if orig_w != corr_w:
                    corrections.append(f"{orig_w}→{corr_w}")
            current = corrected
            spell_corrected = True
            logger.info(f"[Refine] إملاء: {original!r} → {current!r}")

    # ── الخطوة 2: توسيع بالمرادفات ──────────────────────────
    if request.apply_synonym_expansion:
        expander = get_synonym_expander()
        expanded = expander.expand(current)
        if expanded != current:
            current = expanded
            synonyms_added = True
            logger.info(f"[Refine] مرادفات: {expanded!r}")

    # ── الخطوة 3: حفظ في السجل ──────────────────────────────
    if request.save_to_history:
        history = get_query_history()
        history.add(
            session_id=request.session_id,
            query=original,
        )

    return RefineResponse(
        original_query=original,
        refined_query=current,
        spell_corrected=spell_corrected,
        synonyms_added=synonyms_added,
        corrections=corrections,
    )


@app.get(
    "/suggest",
    response_model=SuggestResponse,
    tags=["Suggestions"],
    summary="اقتراح استعلامات",
)
async def suggest_queries(
    q: str = QueryParam(..., description="الاستعلام الجزئي", min_length=1),
    session_id: str = QueryParam(default="default", description="معرّف الجلسة"),
    limit: int = QueryParam(default=5, ge=1, le=20, description="عدد الاقتراحات"),
) -> SuggestResponse:
    """
    يُرجع اقتراحات لاستعلام (أو استعلام جزئي).

    **مثال:**
    `GET /suggest?q=machine&session_id=user_123&limit=5`

    **النتيجة:**
    ```json
    {
        "query": "machine",
        "suggestions": [
            "machine learning",
            "machine learning algorithms",
            "machine vision",
            "machine translation"
        ]
    }
    ```
    """
    engine = get_suggestion_engine()
    suggestions = engine.suggest(
        partial_query=q,
        session_id=session_id,
    )
    return SuggestResponse(
        query=q,
        suggestions=suggestions[:limit],
    )


@app.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    tags=["History"],
    summary="عرض سجل الاستعلامات",
)
async def get_history(
    session_id: str,
    limit: int = QueryParam(default=20, ge=1, le=100),
) -> HistoryResponse:
    """
    يُرجع سجل استعلامات جلسة معينة.

    **مثال:**
    `GET /history/user_123?limit=10`
    """
    history = get_query_history()
    recent = history.get_recent(session_id=session_id, limit=limit)

    return HistoryResponse(
        session_id=session_id,
        queries=[entry.to_dict() for entry in recent],
        total=len(recent),
    )


@app.delete(
    "/history/{session_id}",
    tags=["History"],
    summary="مسح سجل الاستعلامات",
)
async def clear_history(session_id: str) -> dict:
    """
    يمسح سجل استعلامات جلسة معينة.
    مفيد لميزة "مسح سجل البحث" في الواجهة.
    """
    history = get_query_history()
    history.clear_session(session_id=session_id)
    return {"message": f"تم مسح سجل الجلسة: {session_id}"}


@app.get(
    "/popular",
    tags=["Suggestions"],
    summary="الاستعلامات الشائعة",
)
async def get_popular_queries(
    limit: int = QueryParam(default=10, ge=1, le=50),
) -> dict:
    """
    يُرجع قائمة الاستعلامات الشائعة في مجال IR.
    تُستخدم في الواجهة لعرض اقتراحات لمن لا يعرف ماذا يبحث.
    """
    engine = get_suggestion_engine()
    return {"popular_queries": engine.get_popular_queries(limit=limit)}


# =============================================================
# تشغيل مباشر
# =============================================================

if __name__ == "__main__":
    import uvicorn

    print(f"[Query Refinement Service] يبدأ على port {QUERY_REFINEMENT_PORT}")
    uvicorn.run(
        "services.query_refinement.app:app",
        host="0.0.0.0",
        port=QUERY_REFINEMENT_PORT,
        reload=True,
    )
