"""
services/retrieval/app.py
==========================
FastAPI server لخدمة الاسترجاع — Retrieval Service (port 8003).

هذه الخدمة هي قلب نظام البحث.
تستقبل طلبات البحث من Gateway وتُرجع الوثائق الأكثر صلة.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
تدفق طلب البحث:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Gateway يُرسل POST /search
2. نُرسل الاستعلام لـ Preprocessing Service (port 8001)
   ← يُرجع tokens معالجة
3. إذا apply_refinement=True → نُرسل لـ Query Refinement (port 8004)
4. نستدعي المحرك المطلوب (tfidf / bm25 / embedding / hybrid)
5. نُرجع RetrievalResponse

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
القاعدة الذهبية في IR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما طُبِّق على الوثائق عند الفهرسة
يجب أن يُطبَّق على الاستعلام عند البحث.

لذا: المطور الأول يحفظ في metadata الإعدادات التي استخدمها
      ونحن نُرسل الاستعلام لنفس Preprocessing Service بنفس الإعدادات.

تشغيل:
    cd ir_system_2026
    uvicorn services.retrieval.app:app --port 8003 --reload

Swagger UI:
    http://localhost:8003/docs
"""

import sys
import os
import time
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.indexing.vector_store import get_vector_store

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from shared.models import (
    RetrievalRequest,
    RetrievalResponse,
    DocumentResult,
    ServiceStatus,
    ErrorResponse,
    RetrievalModel,
    DatasetName,
    PreprocessRequest,
)
from shared.constants import (
    RETRIEVAL_PORT,
    PREPROCESSING_URL,
    QUERY_REFINEMENT_URL,
)

from services.retrieval.tfidf_retriever import get_tfidf_retriever, TFIDFRetriever
from services.retrieval.bm25_retriever import get_bm25_retriever, BM25Retriever
from services.retrieval.embedding_retriever import (
    get_embedding_retriever,
    EmbeddingRetriever,
)
from services.retrieval.hybrid_parallel import HybridParallelRetriever
from services.retrieval.hybrid_serial import HybridSerialRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================
# lifespan — تحميل الفهارس عند بدء التشغيل
# =============================================================


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """
    يُشغَّل مرة واحدة عند بدء الخدمة.
    نحمّل كل الفهارس مسبقاً حتى لا يتأخر أول طلب.
    """
    logger.info("[Retrieval] بدء تحميل الفهارس...")
    for dataset in DatasetName:
        logger.info(f"  ← {dataset.value}")
        get_tfidf_retriever(dataset)
        get_bm25_retriever(dataset)
        get_embedding_retriever(dataset)
    logger.info("[Retrieval] ✅ كل الفهارس جاهزة — الخدمة تعمل")
    yield
    logger.info("[Retrieval] إيقاف الخدمة...")


# =============================================================
# إنشاء التطبيق
# =============================================================

app = FastAPI(
    title="Retrieval Service",
    description=(
        "خدمة الاسترجاع الرئيسية في نظام IR 2026.\n\n"
        "**النماذج المدعومة:**\n"
        "- `tfidf`: TF-IDF + Cosine Similarity (VSM)\n"
        "- `bm25`: BM25 مع معاملات k1 و b قابلة للتعديل\n"
        "- `embedding`: Semantic Search (Sentence Transformers + FAISS)\n"
        "- `hybrid_parallel`: RRF Fusion (دمج متوازي)\n"
        "- `hybrid_serial`: Pipeline (BM25 تصفية + Embedding إعادة ترتيب)\n\n"
        "⚠️ يجب تشغيل Indexing Service أولاً لبناء الفهارس."
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
# دوال مساعدة
# =============================================================


def _basic_tokenize(text: str) -> list:
    """
    Tokenization بسيط يُستخدم كـ Fallback عند غياب Preprocessing Service.

    أفضل من split() العادي لأنه:
    - يُحوّل لأحرف صغيرة
    - يحذف علامات الترقيم
    - يحذف الكلمات القصيرة جداً
    """
    import re

    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = [t for t in cleaned.split() if len(t) >= 3]
    return tokens if tokens else text.lower().split()


async def _preprocess_query(query: str) -> list:
    """
    يُرسل الاستعلام لـ Preprocessing Service ويُرجع tokens.

    إذا الخدمة غير متاحة: نستخدم _basic_tokenize() كـ Fallback ذكي.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{PREPROCESSING_URL}/preprocess",
                json={"text": query},
            )
            resp.raise_for_status()
            tokens = resp.json().get("tokens", [])
            return tokens if tokens else _basic_tokenize(query)
    except httpx.ConnectError:
        logger.warning(
            "[Retrieval] Preprocessing Service غير متاح (port 8001). "
            "نستخدم basic tokenization كـ fallback."
        )
        return _basic_tokenize(query)
    except Exception as e:
        logger.error(f"[Retrieval] خطأ في Preprocessing: {e}")
        return _basic_tokenize(query)


async def _refine_query(query: str) -> str:
    """
    يُرسل الاستعلام لـ Query Refinement Service.
    إذا غير متاح، يُرجع الاستعلام الأصلي.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{QUERY_REFINEMENT_URL}/refine",
                json={"query": query},
            )
            if resp.status_code == 200:
                refined = resp.json().get("refined_query", query)
                if refined and refined.strip():
                    return refined
    except Exception:
        pass
    return query


# =============================================================
# Endpoints
# =============================================================


@app.get("/health", response_model=ServiceStatus, tags=["System"])
async def health_check() -> ServiceStatus:
    """
    يُرجع حالة الخدمة وتفاصيل الفهارس المحمّلة.
    يُستخدم من Gateway للتحقق قبل توجيه الطلبات.
    """
    details = {}
    for dataset in DatasetName:
        tfidf = get_tfidf_retriever(dataset)
        bm25 = get_bm25_retriever(dataset)
        emb = get_embedding_retriever(dataset)
        details[dataset.value] = {
            "tfidf": {"loaded": tfidf.is_loaded, **tfidf.get_stats()},
            "bm25": {"loaded": bm25.is_loaded, **bm25.get_stats()},
            "embedding": {"loaded": emb.is_loaded, **emb.get_stats()},
            "vector_store": get_vector_store(dataset.value).get_status(),
        }
    return ServiceStatus(
        service_name="retrieval",
        status="healthy",
        details=details,
    )


@app.post("/search/semantic-raw", tags=["Retrieval"])
async def search_semantic_raw(
    query: str,
    dataset: DatasetName,
    top_k: int = 10,
) -> dict:
    """
    بحث دلالي مباشر عبر VectorStore.
    يُرجع (doc_id, score, text, title) بدون تحويل لـ DocumentResult.
    """
    store = get_vector_store(dataset.value)

    if not store.is_ready():
        raise HTTPException(503, f"VectorStore لـ {dataset.value} غير جاهز")

    results = store.search(query, k=top_k)

    return {
        "query": query,
        "dataset": dataset.value,
        "results": [
            {
                "doc_id": doc_id,
                "score": score,
                "text": text[:200],
                "title": title,
                "rank": rank + 1,
            }
            for rank, (doc_id, score, text, title) in enumerate(results)
        ],
        "total": len(results),
    }


@app.post(
    "/search",
    response_model=RetrievalResponse,
    tags=["Retrieval"],
    summary="البحث في مجموعة البيانات",
    responses={
        422: {"model": ErrorResponse, "description": "بيانات الطلب غير صحيحة"},
        503: {"model": ErrorResponse, "description": "الفهرس غير متاح"},
    },
)
async def search(request: RetrievalRequest) -> RetrievalResponse:
    """
    يستقبل طلب بحث ويُرجع الوثائق الأكثر صلة.

    **مثال:**
    ```json
    {
        "query": "information retrieval systems",
        "dataset": "dataset1",
        "model": "bm25",
        "top_k": 10,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "apply_refinement": false
    }
    ```
    """
    start_time = time.time()

    # الخطوة 1: تحسين الاستعلام (اختياري)
    refined_query = None
    working_query = request.query

    if request.apply_refinement:
        refined = await _refine_query(request.query)
        if refined != request.query:
            refined_query = refined
            working_query = refined
            logger.info(f"[Retrieval] الاستعلام بعد التحسين: {refined_query}")

    # الخطوة 2: معالجة الاستعلام
    query_tokens = await _preprocess_query(working_query)
    logger.info(f"[Retrieval] tokens: {query_tokens}")

    # الخطوة 3: تشغيل المحرك المطلوب
    try:
        results = await _run_retrieval(
            model=request.model,
            dataset=request.dataset,
            query_tokens=query_tokens,
            query_text=working_query,
            top_k=request.top_k,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Retrieval] خطأ في البحث: {e}")
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")

    processing_time_ms = (time.time() - start_time) * 1000

    return RetrievalResponse(
        query=request.query,
        refined_query=refined_query,
        results=results,
        model_used=request.model,
        dataset=request.dataset,
        total_results=len(results),
        processing_time_ms=round(processing_time_ms, 2),
    )


async def _run_retrieval(
    model: RetrievalModel,
    dataset: DatasetName,
    query_tokens: list,
    query_text: str,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
) -> list:
    """
    يختار المحرك المناسب ويُشغّله.
    """
    tfidf = get_tfidf_retriever(dataset)
    bm25 = get_bm25_retriever(dataset)
    emb = get_embedding_retriever(dataset)

    # ─── TF-IDF ───
    if model == RetrievalModel.TFIDF:
        if not tfidf.is_loaded:
            raise HTTPException(
                503, f"فهرس TF-IDF لـ {dataset.value} غير متاح. شغّل Indexing Service."
            )
        return tfidf.search(query_tokens, top_k=top_k)

    # ─── BM25 ───
    elif model == RetrievalModel.BM25:
        if not bm25.is_loaded:
            raise HTTPException(
                503, f"فهرس BM25 لـ {dataset.value} غير متاح. شغّل Indexing Service."
            )
        return bm25.search(query_tokens, top_k=top_k, k1=bm25_k1, b=bm25_b)

    # ─── Embedding ───
    elif model == RetrievalModel.EMBEDDING:
        if not emb.is_loaded:
            raise HTTPException(
                503,
                f"فهرس Embedding لـ {dataset.value} غير متاح. شغّل Indexing Service.",
            )
        return emb.search(query_text, top_k=top_k)

    # ─── Hybrid Parallel (RRF) ───
    elif model == RetrievalModel.HYBRID_PARALLEL:
        hybrid = HybridParallelRetriever(
            tfidf_retriever=tfidf,
            bm25_retriever=bm25,
            embedding_retriever=emb,
        )
        return hybrid.search(
            query_tokens=query_tokens,
            query_text=query_text,
            top_k=top_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    # ─── Hybrid Serial (Pipeline) ───
    elif model == RetrievalModel.HYBRID_SERIAL:
        hybrid_serial = HybridSerialRetriever(
            first_stage_retriever=bm25,
            second_stage_retriever=emb,
        )
        return hybrid_serial.search(
            query_tokens=query_tokens,
            query_text=query_text,
            top_k=top_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    else:
        raise HTTPException(400, f"نموذج غير مدعوم: {model}")


# =============================================================
# تشغيل مباشر
# =============================================================

if __name__ == "__main__":
    import uvicorn

    print(f"[Retrieval Service] يبدأ على port {RETRIEVAL_PORT}")
    uvicorn.run(
        "services.retrieval.app:app",
        host="0.0.0.0",
        port=RETRIEVAL_PORT,
        reload=True,
    )
