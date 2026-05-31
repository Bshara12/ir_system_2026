"""
services/indexing/app.py
=========================
FastAPI server لخدمة الفهرسة.

يُغلّف الفهارس الأربعة في API موحّدة:
  - Inverted Index  (Boolean Retrieval)
  - TF-IDF Index    (VSM Retrieval)
  - BM25 Index      (Probabilistic Retrieval)
  - Embedding Index (Semantic Retrieval)

تشغيل:
    cd ir_system_2026
    uvicorn services.indexing.app:app --port 8002 --reload

Swagger UI:
    http://localhost:8002/docs
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.constants import INDEXING_PORT
from shared.models import ServiceStatus, ErrorResponse, DatasetName

from services.indexing.dataset_loader import get_dataset_loader
from services.indexing.inverted_index import get_inverted_index, InvertedIndex
from services.indexing.tfidf_indexer   import get_tfidf_indexer,  TFIDFIndexer
from services.indexing.bm25_indexer    import get_bm25_indexer,   BM25Indexer
from services.indexing.embedding_indexer import get_embedding_indexer, EmbeddingIndexer


# =============================================================
# نماذج طلبات هذه الخدمة
# =============================================================

class BuildIndexRequest(BaseModel):
    """طلب بناء فهرس."""
    dataset_name: str = Field(..., description="اسم مجموعة البيانات")
    index_type: str = Field(
        ...,
        description="نوع الفهرس: inverted | tfidf | bm25 | embedding | all"
    )
    max_docs: Optional[int] = Field(
        None, description="الحد الأقصى للوثائق (None = كل شيء)"
    )
    apply_stemming: bool = Field(default=True)
    remove_stopwords: bool = Field(default=True)
    # BM25 params
    bm25_k1: float = Field(default=1.5, ge=0.0, le=3.0)
    bm25_b: float  = Field(default=0.75, ge=0.0, le=1.0)
    # Embedding params
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_batch_size: int = Field(default=64, ge=1)


class BuildIndexResponse(BaseModel):
    """نتيجة بناء الفهرس."""
    dataset_name: str
    index_type: str
    status: str
    num_documents: int
    build_time_seconds: float
    details: Dict = Field(default_factory=dict)


class IndexStatusResponse(BaseModel):
    """حالة الفهارس لمجموعة بيانات."""
    dataset_name: str
    inverted: Dict
    tfidf: Dict
    bm25: Dict
    embedding: Dict


class BooleanSearchRequest(BaseModel):
    """طلب Boolean search باستخدام Inverted Index."""
    dataset_name: str
    operation: str = Field(..., description="and | or | not | and_not")
    terms: List[str] = Field(..., min_length=1)
    exclude_terms: List[str] = Field(default_factory=list)


# =============================================================
# تطبيق FastAPI
# =============================================================

app = FastAPI(
    title="Indexing Service",
    description=(
        "خدمة الفهرسة — تبني وتحفظ وتُدير الفهارس الأربعة:\n\n"
        "- **Inverted Index**: للـ Boolean Retrieval\n"
        "- **TF-IDF Index**: للـ VSM Retrieval\n"
        "- **BM25 Index**: للـ Probabilistic Retrieval\n"
        "- **Embedding Index**: للـ Semantic Retrieval\n\n"
        "تُستخدم من قِبَل: Retrieval Service و Gateway"
    ),
    version="1.0.0",
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
    return ServiceStatus(
        service_name="indexing",
        status="healthy",
        details={"port": INDEXING_PORT},
    )


@app.post(
    "/index/build",
    response_model=BuildIndexResponse,
    tags=["Indexing"],
    summary="بناء فهرس",
)
async def build_index(request: BuildIndexRequest) -> BuildIndexResponse:
    """
    يبني فهرساً واحداً أو كل الفهارس لمجموعة بيانات.

    index_type يمكن أن يكون:
    - `inverted`  : الفهرس المقلوب
    - `tfidf`     : فهرس TF-IDF
    - `bm25`      : فهرس BM25
    - `embedding` : فهرس Embeddings
    - `all`       : بناء كل الفهارس
    """
    valid_types = {"inverted", "tfidf", "bm25", "embedding", "all"}
    if request.index_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"index_type غير صالح. الخيارات: {valid_types}"
        )

    start = time.time()

    try:
        if request.index_type in ("inverted", "all"):
            idx = InvertedIndex()
            meta = idx.build_index(
                request.dataset_name,
                apply_stemming=request.apply_stemming,
                remove_stopwords=request.remove_stopwords,
                max_docs=request.max_docs,
            )
            idx.save_index(request.dataset_name)
            if request.index_type == "inverted":
                return BuildIndexResponse(
                    dataset_name=request.dataset_name,
                    index_type="inverted",
                    status="built",
                    num_documents=meta.num_documents,
                    build_time_seconds=meta.build_time_seconds,
                    details={"vocab_size": meta.vocab_size},
                )

        if request.index_type in ("tfidf", "all"):
            tidx = TFIDFIndexer()
            meta = tidx.build_index(
                request.dataset_name,
                apply_stemming=request.apply_stemming,
                remove_stopwords=request.remove_stopwords,
                max_docs=request.max_docs,
            )
            tidx.save_index(request.dataset_name)
            if request.index_type == "tfidf":
                return BuildIndexResponse(
                    dataset_name=request.dataset_name,
                    index_type="tfidf",
                    status="built",
                    num_documents=meta.num_documents,
                    build_time_seconds=meta.build_time_seconds,
                    details={"vocab_size": meta.vocab_size},
                )

        if request.index_type in ("bm25", "all"):
            bidx = BM25Indexer()
            meta = bidx.build_index(
                request.dataset_name,
                k1=request.bm25_k1,
                b=request.bm25_b,
                apply_stemming=request.apply_stemming,
                remove_stopwords=request.remove_stopwords,
                max_docs=request.max_docs,
            )
            bidx.save_index(request.dataset_name)
            if request.index_type == "bm25":
                return BuildIndexResponse(
                    dataset_name=request.dataset_name,
                    index_type="bm25",
                    status="built",
                    num_documents=meta.num_documents,
                    build_time_seconds=meta.build_time_seconds,
                    details={"k1": meta.k1, "b": meta.b},
                )

        if request.index_type in ("embedding", "all"):
            eidx = EmbeddingIndexer(model_name=request.embedding_model)
            meta = eidx.build_index(
                request.dataset_name,
                batch_size=request.embedding_batch_size,
                max_docs=request.max_docs,
            )
            eidx.save_index(request.dataset_name)
            if request.index_type == "embedding":
                return BuildIndexResponse(
                    dataset_name=request.dataset_name,
                    index_type="embedding",
                    status="built",
                    num_documents=meta.num_documents,
                    build_time_seconds=meta.build_time_seconds,
                    details={
                        "model": meta.model_name,
                        "dim": meta.embedding_dim,
                    },
                )

        # حالة "all"
        total_time = round(time.time() - start, 2)
        return BuildIndexResponse(
            dataset_name=request.dataset_name,
            index_type="all",
            status="all_built",
            num_documents=meta.num_documents,
            build_time_seconds=total_time,
            details={"message": "كل الفهارس الأربعة مبنية"},
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")


@app.get(
    "/index/status/{dataset_name}",
    response_model=IndexStatusResponse,
    tags=["Indexing"],
    summary="حالة الفهارس",
)
async def get_index_status(dataset_name: str) -> IndexStatusResponse:
    """
    يُرجع حالة كل الفهارس لمجموعة بيانات محددة.
    يُستخدم من Gateway للتحقق قبل توجيه الطلبات.
    """
    def _index_status(is_saved: bool, name: str) -> Dict:
        return {"saved": is_saved, "name": name}

    inv  = InvertedIndex()
    tidx = TFIDFIndexer()
    bidx = BM25Indexer()
    eidx = EmbeddingIndexer()

    return IndexStatusResponse(
        dataset_name=dataset_name,
        inverted=_index_status(inv.is_saved(dataset_name), "inverted"),
        tfidf=_index_status(tidx.is_saved(dataset_name), "tfidf"),
        bm25=_index_status(bidx.is_saved(dataset_name), "bm25"),
        embedding=_index_status(eidx.is_saved(dataset_name), "embedding"),
    )


@app.post(
    "/index/boolean-search",
    tags=["Retrieval-Basic"],
    summary="Boolean Search باستخدام Inverted Index",
)
async def boolean_search(request: BooleanSearchRequest) -> Dict:
    """
    Boolean Retrieval باستخدام الفهرس المقلوب.
    يُستخدم من Retrieval Service أو مباشرة للاختبار.
    """
    try:
        idx = get_inverted_index(request.dataset_name)
    except (FileNotFoundError, RuntimeError):
        raise HTTPException(
            status_code=404,
            detail=f"الفهرس المقلوب لـ '{request.dataset_name}' غير مبني."
        )
    if not idx.is_built():
        raise HTTPException(
            status_code=404,
            detail=f"الفهرس المقلوب لـ '{request.dataset_name}' غير مبني."
        )

    op = request.operation.lower()
    if op == "and":
        results = idx.search_and(request.terms)
    elif op == "or":
        results = idx.search_or(request.terms)
    elif op == "not":
        results = idx.search_not(request.terms[0])
    elif op == "and_not":
        results = idx.search_and_not(request.terms, request.exclude_terms)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"operation غير صالح: '{op}'. الخيارات: and | or | not | and_not"
        )

    return {
        "operation": op,
        "terms": request.terms,
        "exclude_terms": request.exclude_terms,
        "total_results": len(results),
        "doc_ids": results[:100],  # نُرجع أول 100 فقط
    }


@app.get(
    "/index/stats/{dataset_name}",
    tags=["Indexing"],
    summary="إحصائيات الفهارس",
)
async def get_stats(dataset_name: str) -> Dict:
    """إحصائيات مفصلة لكل الفهارس المبنية."""
    stats = {}

    inv = get_inverted_index()
    if inv.is_saved(dataset_name):
        inv.load_index(dataset_name)
        stats["inverted"] = inv.get_stats()
    else:
        stats["inverted"] = {"status": "not_built"}

    return {"dataset_name": dataset_name, "stats": stats}


if __name__ == "__main__":
    import uvicorn
    print(f"[Indexing Service] يبدأ على port {INDEXING_PORT}")
    uvicorn.run(
        "services.indexing.app:app",
        host="0.0.0.0",
        port=INDEXING_PORT,
        reload=True,
    )