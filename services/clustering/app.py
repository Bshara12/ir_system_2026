"""
services/clustering/app.py
============================
FastAPI server لخدمة تجميع الوثائق — Clustering Service (port 8006).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الـ Endpoints:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /cluster/dataset      ← تجميع كل وثائق dataset
POST /cluster/results      ← تجميع نتائج بحث محددة
GET  /cluster/optimal-k    ← إيجاد أفضل عدد clusters
GET  /health               ← حالة الخدمة

تشغيل:
    uvicorn services.clustering.app:app --port 8006 --reload

Swagger UI:
    http://localhost:8006/docs
"""

import sys
import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from shared.models import ServiceStatus, ErrorResponse
from shared.constants import DEFAULT_TOP_K

from services.clustering.clusterer import get_clusterer, ClusteringResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Port الخاص بهذه الخدمة
CLUSTERING_PORT = 8006


# =============================================================
# نماذج طلبات هذه الخدمة
# =============================================================


class ClusterDatasetRequest(BaseModel):
    """طلب تجميع كل وثائق dataset."""

    dataset_name: str = Field(
        ...,
        description="اسم مجموعة البيانات (dataset1 أو dataset2)",
        examples=["dataset1"],
    )
    n_clusters: int = Field(
        default=5,
        ge=2,
        le=20,
        description="عدد المجموعات (2-20)",
    )
    svd_components: int = Field(
        default=100,
        ge=10,
        le=300,
        description="أبعاد LSA لتقليل الأبعاد",
    )
    top_terms_per_cluster: int = Field(
        default=8,
        ge=3,
        le=20,
        description="عدد الكلمات المميِّزة لكل cluster",
    )


class ClusterResultsRequest(BaseModel):
    """طلب تجميع نتائج بحث محددة."""

    doc_ids: List[str] = Field(
        ...,
        description="معرّفات الوثائق المراد تجميعها",
        min_length=2,
    )
    dataset_name: str = Field(
        ...,
        description="اسم مجموعة البيانات المصدر",
    )
    n_clusters: int = Field(
        default=3,
        ge=2,
        le=10,
        description="عدد المجموعات",
    )


class OptimalKRequest(BaseModel):
    """طلب إيجاد أفضل عدد clusters."""

    dataset_name: str = Field(..., description="اسم مجموعة البيانات")
    k_min: int = Field(default=2, ge=2, le=5)
    k_max: int = Field(default=8, ge=3, le=15)


# =============================================================
# إنشاء التطبيق
# =============================================================

app = FastAPI(
    title="Clustering Service",
    description=(
        "خدمة تجميع الوثائق في نظام IR 2026.\n\n"
        "**الخوارزمية:** K-Means + LSA (TruncatedSVD)\n\n"
        "**الاستخدامات:**\n"
        "- تجميع كل وثائق مجموعة بيانات\n"
        "- تجميع نتائج بحث المستخدم لتنظيم العرض\n"
        "- إيجاد عدد المجموعات الأمثل تلقائياً\n\n"
        "**يعتمد على:** TF-IDF index من Indexing Service"
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
    """حالة خدمة التجميع."""
    return ServiceStatus(
        service_name="clustering",
        status="healthy",
        details={"port": CLUSTERING_PORT, "algorithm": "kmeans+lsa"},
    )


@app.post(
    "/cluster/dataset",
    tags=["Clustering"],
    summary="تجميع كل وثائق dataset",
    responses={
        200: {"description": "نجح التجميع"},
        503: {"model": ErrorResponse, "description": "فهرس TF-IDF غير متاح"},
        400: {"model": ErrorResponse, "description": "معاملات غير صحيحة"},
    },
)
async def cluster_dataset(request: ClusterDatasetRequest) -> dict:
    """
    يُجمّع كل وثائق مجموعة بيانات في clusters.

    **المتطلب:** يجب أن يكون TF-IDF index مبنياً لهذا الـ dataset.

    **مثال طلب:**
    ```json
    {
        "dataset_name": "dataset1",
        "n_clusters": 5,
        "top_terms_per_cluster": 8
    }
    ```

    **مثال نتيجة:**
    ```json
    {
        "n_clusters": 5,
        "silhouette_score": 0.312,
        "clusters": [
            {
                "cluster_id": 0,
                "label": "Cluster 1: cloud, storage, sync",
                "size": 42,
                "top_terms": ["cloud", "storage", "sync", "file", "backup"]
            }
        ]
    }
    ```
    """
    if request.k_min > request.n_clusters if hasattr(request, "k_min") else False:
        raise HTTPException(status_code=400, detail="n_clusters غير صحيح")

    clusterer = get_clusterer()

    try:
        result = clusterer.cluster(
            dataset_name=request.dataset_name,
            n_clusters=request.n_clusters,
            svd_components=request.svd_components,
            top_terms_per_cluster=request.top_terms_per_cluster,
        )
        return result.to_dict()

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Clustering] خطأ: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")


@app.post(
    "/cluster/results",
    tags=["Clustering"],
    summary="تجميع نتائج بحث محددة",
    responses={
        200: {"description": "نجح التجميع"},
        503: {"model": ErrorResponse, "description": "فهرس TF-IDF غير متاح"},
    },
)
async def cluster_search_results(request: ClusterResultsRequest) -> dict:
    """
    يُجمّع مجموعة محددة من نتائج البحث.

    يُستخدم لتنظيم نتائج البحث بدل عرضها كقائمة مسطحة.

    **مثال الاستخدام:**
    1. المستخدم يبحث عن "machine learning"
    2. Retrieval Service يُرجع 10 نتائج
    3. تُرسل doc_ids هنا للتجميع
    4. تُعرض النتائج مُنظَّمة في clusters

    **مثال طلب:**
    ```json
    {
        "doc_ids": ["d1", "d3", "d5", "d7", "d9"],
        "dataset_name": "dataset1",
        "n_clusters": 3
    }
    ```
    """
    clusterer = get_clusterer()

    try:
        result = clusterer.cluster_search_results(
            doc_ids=request.doc_ids,
            dataset_name=request.dataset_name,
            n_clusters=request.n_clusters,
        )
        return result.to_dict()

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[Clustering] خطأ في تجميع النتائج: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")


@app.post(
    "/cluster/optimal-k",
    tags=["Clustering"],
    summary="إيجاد أفضل عدد clusters",
)
async def find_optimal_k(request: OptimalKRequest) -> dict:
    """
    يجرّب قيم k مختلفة ويُرجع أفضلها بناءً على Silhouette Score.

    **تحذير:** هذا الـ endpoint أبطأ من /cluster/dataset
    لأنه يُشغّل K-Means عدة مرات.

    **مثال نتيجة:**
    ```json
    {
        "best_k": 4,
        "scores": {
            "2": 0.18,
            "3": 0.24,
            "4": 0.31,
            "5": 0.29
        }
    }
    ```
    """
    if request.k_min >= request.k_max:
        raise HTTPException(status_code=400, detail="k_min يجب أن يكون أصغر من k_max")

    clusterer = get_clusterer()

    try:
        result = clusterer.find_optimal_k(
            dataset_name=request.dataset_name,
            k_range=(request.k_min, request.k_max),
        )
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================
# تشغيل مباشر
# =============================================================

if __name__ == "__main__":
    import uvicorn

    print(f"[Clustering Service] يبدأ على port {CLUSTERING_PORT}")
    uvicorn.run(
        "services.clustering.app:app",
        host="0.0.0.0",
        port=CLUSTERING_PORT,
        reload=True,
    )
