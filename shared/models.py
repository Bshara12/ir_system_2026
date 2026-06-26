"""
shared/models.py
================
عقود البيانات (Data Contracts) المشتركة بين جميع الخدمات.

كل Request و Response في النظام يستخدم هذه النماذج.
Pydantic تتحقق تلقائياً من صحة البيانات عند كل طلب.

⚠️ لا تعرّف نماذج بيانات في ملفات الخدمات مباشرةً.
   كل شيء يجب أن يُعرَّف هنا ويُستورد من هنا.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

from shared.constants import (
    DEFAULT_LANGUAGE,
    DEFAULT_APPLY_STEMMING,
    DEFAULT_APPLY_LEMMATIZATION,
    DEFAULT_REMOVE_STOPWORDS,
    DEFAULT_REMOVE_PUNCTUATION,
    DEFAULT_LOWERCASE,
    BM25_DEFAULT_K1,
    BM25_DEFAULT_B,
    DEFAULT_TOP_K,
    MAX_TOP_K,
)


# =============================================================
# التعدادات (Enums) — القيم المسموح بها في النظام
# =============================================================

class RetrievalModel(str, Enum):
    """نماذج الاسترجاع المتاحة في النظام."""
    TFIDF           = "tfidf"
    BM25            = "bm25"
    EMBEDDING       = "embedding"
    HYBRID_PARALLEL = "hybrid_parallel"   # RRF fusion
    HYBRID_SERIAL   = "hybrid_serial"     # pipeline chaining


class DatasetName(str, Enum):
    """مجموعات البيانات المدعومة."""
    DATASET_1 = "dataset1"   # مجموعة البيانات الأولى
    DATASET_2 = "quora"   # مجموعة البيانات الثانية
    TREC_COVID = "trec-covid"   # مجموعة بيانات TREC-COVID
    QUORA = "quora"   # مجموعة بيانات Quora الكاملة


class Language(str, Enum):
    """اللغات المدعومة في المعالجة المسبقة."""
    ENGLISH = "english"
    ARABIC  = "arabic"


# =============================================================
# نماذج Preprocessing Service
# =============================================================

class PreprocessRequest(BaseModel):
    """
    طلب معالجة نص واحد.
    يُستخدم من: Indexing Service (لمعالجة الوثائق)
                Retrieval Service (لمعالجة الاستعلامات)
    """
    text: str = Field(
        ...,                          # مطلوب دائماً
        description="النص المراد معالجته",
        min_length=1,
    )
    language: Language = Field(
        default=DEFAULT_LANGUAGE,
        description="لغة النص",
    )
    lowercase: bool = Field(
        default=DEFAULT_LOWERCASE,
        description="تحويل إلى حروف صغيرة",
    )
    remove_punctuation: bool = Field(
        default=DEFAULT_REMOVE_PUNCTUATION,
        description="حذف علامات الترقيم",
    )
    remove_stopwords: bool = Field(
        default=DEFAULT_REMOVE_STOPWORDS,
        description="حذف الكلمات الوظيفية",
    )
    apply_stemming: bool = Field(
        default=DEFAULT_APPLY_STEMMING,
        description="تطبيق Stemming (تقليل للجذر)",
    )
    apply_lemmatization: bool = Field(
        default=DEFAULT_APPLY_LEMMATIZATION,
        description="تطبيق Lemmatization (تحويل للمعجمي) — لا يعمل مع Stemming",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        """يرفض النصوص الفارغة أو التي تحتوي على مسافات فقط."""
        if not v.strip():
            raise ValueError("النص لا يمكن أن يكون فارغاً أو مسافات فقط")
        return v

    @field_validator("apply_lemmatization")
    @classmethod
    def lemmatization_and_stemming_conflict(cls, v: bool, info: Any) -> bool:
        """
        Stemming و Lemmatization لا يعملان معاً.
        كلاهما يُحوّل الكلمة لشكل آخر — تطبيقهما معاً يُشوّه النتيجة.
        """
        if v and info.data.get("apply_stemming", False):
            raise ValueError(
                "لا يمكن تفعيل apply_stemming و apply_lemmatization معاً. "
                "اختر أحدهما فقط."
            )
        return v


class PreprocessResponse(BaseModel):
    """
    نتيجة معالجة نص واحد.
    """
    original_text: str = Field(description="النص الأصلي قبل المعالجة")
    processed_text: str = Field(description="النص بعد المعالجة (كنص واحد)")
    tokens: List[str] = Field(description="قائمة التوكنز النظيفة")
    token_count: int = Field(description="عدد التوكنز بعد المعالجة")
    steps_applied: List[str] = Field(description="قائمة الخطوات التي طُبِّقت")


class BatchPreprocessRequest(BaseModel):
    """
    طلب معالجة دفعة من النصوص دفعةً واحدة.
    يُستخدم من: Indexing Service لمعالجة آلاف الوثائق بكفاءة.

    لماذا Batch؟
    استدعاء الـ API مرة واحدة لكل 1000 وثيقة
    أسرع بكثير من 1000 استدعاء منفصل.
    """
    texts: List[str] = Field(
        ...,
        description="قائمة النصوص المراد معالجتها",
        min_length=1,
    )
    # نفس إعدادات PreprocessRequest تنطبق على كل النصوص
    language: Language = Field(default=Language.ENGLISH)
    lowercase: bool = Field(default=DEFAULT_LOWERCASE)
    remove_punctuation: bool = Field(default=DEFAULT_REMOVE_PUNCTUATION)
    remove_stopwords: bool = Field(default=DEFAULT_REMOVE_STOPWORDS)
    apply_stemming: bool = Field(default=DEFAULT_APPLY_STEMMING)
    apply_lemmatization: bool = Field(default=DEFAULT_APPLY_LEMMATIZATION)

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("القائمة لا يمكن أن تكون فارغة")
        return v


class BatchPreprocessResponse(BaseModel):
    """
    نتيجة معالجة دفعة من النصوص.
    """
    results: List[PreprocessResponse]
    total_processed: int = Field(description="العدد الإجمالي للنصوص التي عولجت")
    total_tokens: int = Field(description="العدد الإجمالي للتوكنز في كل النصوص")


# =============================================================
# نماذج Retrieval Service (تُعرَّف هنا مبكراً لأن Dev2 يحتاجها)
# =============================================================

class RetrievalRequest(BaseModel):
    """
    طلب بحث من المستخدم.
    يُستقبل من: Gateway → Retrieval Service
    """
    query: str = Field(
        ...,
        description="نص الاستعلام",
        min_length=1,
    )
    dataset: DatasetName = Field(description="مجموعة البيانات المراد البحث فيها")
    model: RetrievalModel = Field(
        default=RetrievalModel.BM25,
        description="نموذج الاسترجاع المستخدم",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description="عدد النتائج المطلوبة",
    )
    bm25_k1: float = Field(default=BM25_DEFAULT_K1, ge=0.0, le=3.0)
    bm25_b: float = Field(default=BM25_DEFAULT_B, ge=0.0, le=1.0)
    apply_refinement: bool = Field(
        default=False,
        description="تطبيق تحسين الاستعلام قبل البحث",
    )


class DocumentResult(BaseModel):
    """
    وثيقة واحدة في نتائج البحث.
    """
    doc_id: str = Field(description="معرّف الوثيقة")
    title: Optional[str] = Field(default=None, description="عنوان الوثيقة")
    text: str = Field(description="نص الوثيقة (أو مقتطف منها)")
    score: float = Field(description="درجة الصلة بالاستعلام")
    rank: int = Field(description="ترتيب الوثيقة في النتائج")


class RetrievalResponse(BaseModel):
    """
    نتيجة عملية البحث الكاملة.
    """
    query: str
    refined_query: Optional[str] = Field(
        default=None,
        description="الاستعلام بعد التحسين (إن طُلب)",
    )
    results: List[DocumentResult]
    model_used: RetrievalModel
    dataset: DatasetName
    total_results: int
    processing_time_ms: float = Field(description="وقت المعالجة بالميلي ثانية")


# =============================================================
# نماذج Evaluation Service
# =============================================================

class EvaluationRequest(BaseModel):
    """
    طلب تقييم نموذج على مجموعة بيانات.
    """
    dataset: DatasetName
    model: RetrievalModel
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    with_refinement: bool = Field(default=False)


class DatasetEvaluationRequest(BaseModel):
    """
    طلب تقييم مجموعة بيانات حقيقي باستخدام استرجاع الخدمة.
    """
    dataset_name: DatasetName = Field(..., alias="dataset_name")
    model: RetrievalModel
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    max_queries: int = Field(default=5, ge=1)
    bm25_k1: float = Field(default=BM25_DEFAULT_K1, ge=0.0, le=3.0)
    bm25_b: float = Field(default=BM25_DEFAULT_B, ge=0.0, le=1.0)
    apply_refinement: bool = Field(default=False)


class PerQueryEvaluation(BaseModel):
    query_id: str
    query: str
    retrieved_doc_ids: List[str]
    num_relevant: int
    precision_at_k: float
    recall_at_k: float
    average_precision_at_k: float
    ndcg_at_k: float


class DatasetEvaluationMetrics(BaseModel):
    MAP: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_ndcg_at_k: float


class DatasetEvaluationResponse(BaseModel):
    dataset_name: DatasetName
    model: RetrievalModel
    top_k: int
    max_queries: int
    evaluated_queries: int
    metrics: DatasetEvaluationMetrics
    per_query: List[PerQueryEvaluation]
    notes: str


class EvaluationMetrics(BaseModel):
    """
    نتائج التقييم لنموذج واحد.
    """
    model: RetrievalModel
    dataset: DatasetName
    top_k: int
    MAP: float = Field(description="Mean Average Precision")
    recall_at_k: float = Field(description="Recall@K")
    precision_at_k: float = Field(description="Precision@K")
    nDCG: float = Field(description="Normalized Discounted Cumulative Gain")
    with_refinement: bool = Field(description="هل استُخدم Query Refinement؟")
    evaluation_time_ms: float


# =============================================================
# نموذج الحالة العامة للخدمات (Health Check)
# =============================================================

class ServiceStatus(BaseModel):
    """
    حالة أي خدمة — تُستخدم في /health endpoint.
    يمكن للـ Gateway أن يتحقق من حالة كل خدمة قبل توجيه الطلبات.
    """
    service_name: str
    status: str = Field(description="'healthy' أو 'unhealthy'")
    version: str = Field(default="1.0.0")
    details: Optional[Dict[str, Any]] = Field(default=None)


# =============================================================
# نموذج الخطأ الموحّد
# =============================================================

class ErrorResponse(BaseModel):
    """
    شكل موحّد لرسائل الخطأ في كل الخدمات.
    بدلاً من أن يُرجع كل مطور خطأ بشكل مختلف.
    """
    error: str = Field(description="رسالة الخطأ")
    detail: Optional[str] = Field(default=None, description="تفاصيل إضافية")
    service: Optional[str] = Field(default=None, description="اسم الخدمة")
