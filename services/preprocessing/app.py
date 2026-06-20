"""
services/preprocessing/app.py
==============================
FastAPI server لخدمة المعالجة المسبقة.

هذا الملف مسؤول فقط عن:
  1. استقبال HTTP requests
  2. التحقق من صحة البيانات (يتولاها Pydantic تلقائياً)
  3. استدعاء preprocessor.py
  4. إرجاع HTTP response

لا يحتوي على أي منطق معالجة — كل ذلك في preprocessor.py.

تشغيل الخدمة:
    cd ir_system_2026
    uvicorn services.preprocessing.app:app --port 8001 --reload

اختبار Swagger UI:
    http://localhost:8001/docs
"""

import time
import sys
import os

# نضيف جذر المشروع لـ Python path حتى تعمل imports من shared/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.models import (
    PreprocessRequest,
    PreprocessResponse,
    BatchPreprocessRequest,
    BatchPreprocessResponse,
    ServiceStatus,
    ErrorResponse,
)
from shared.constants import PREPROCESSING_PORT
from services.preprocessing.preprocessor import TextPreprocessor, get_preprocessor


# =============================================================
# إنشاء تطبيق FastAPI
# =============================================================

app = FastAPI(
    title="Preprocessing Service",
    description=(
        "خدمة المعالجة المسبقة للنصوص في نظام IR.\n\n"
        "تُستخدم من قِبَل:\n"
        "- Indexing Service (لمعالجة الوثائق عند البناء)\n"
        "- Retrieval Service (لمعالجة استعلامات المستخدم)\n"
        "- Query Refinement Service (لتنظيف الاستعلام قبل التوسيع)\n\n"
        "⚠️ يجب أن تُعالَج الوثائق والاستعلامات بنفس الإعدادات دائماً."
    ),
    version="1.0.0",
)

# نسمح لأي خدمة محلية بالتواصل مع هذه الخدمة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # في الإنتاج: حدّد العناوين المسموح بها
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# Endpoints
# =============================================================

@app.get("/health", response_model=ServiceStatus, tags=["System"])
async def health_check() -> ServiceStatus:
    """
    التحقق من أن الخدمة تعمل بشكل صحيح.
    يستخدمها Gateway قبل توجيه الطلبات.
    """
    return ServiceStatus(
        service_name="preprocessing",
        status="healthy",
        details={"port": PREPROCESSING_PORT},
    )


@app.post(
    "/preprocess",
    response_model=PreprocessResponse,
    tags=["Preprocessing"],
    summary="معالجة نص واحد",
    responses={422: {"model": ErrorResponse}},
)
async def preprocess_text(
    request: PreprocessRequest,
    preprocessor: TextPreprocessor = Depends(get_preprocessor),
) -> PreprocessResponse:
    """
    يعالج نصاً واحداً ويُرجع التوكنز النظيفة.

    **يُستخدم من:**
    - Retrieval Service لمعالجة استعلام المستخدم
    - Query Refinement لتنظيف الاستعلام

    **مثال:**
    ```json
    {
        "text": "Running dogs jumped over fences!",
        "apply_stemming": true,
        "remove_stopwords": true
    }
    ```
    """
    # Pydantic تحققت من صحة البيانات قبل وصولنا هنا
    tokens, steps = preprocessor.process(
        text=request.text,
        language=request.language,
        # language=request.language.value,
        lowercase=request.lowercase,
        remove_punctuation=request.remove_punctuation,
        remove_stopwords=request.remove_stopwords,
        apply_stemming=request.apply_stemming,
        apply_lemmatization=request.apply_lemmatization,
    )

    return PreprocessResponse(
        original_text=request.text,
        processed_text=" ".join(tokens),
        tokens=tokens,
        token_count=len(tokens),
        steps_applied=steps,
    )


@app.post(
    "/preprocess/batch",
    response_model=BatchPreprocessResponse,
    tags=["Preprocessing"],
    summary="معالجة دفعة من النصوص",
)
async def preprocess_batch(
    request: BatchPreprocessRequest,
    preprocessor: TextPreprocessor = Depends(get_preprocessor),
) -> BatchPreprocessResponse:
    """
    يعالج قائمة من النصوص دفعةً واحدة.

    **يُستخدم من:**
    - Indexing Service لمعالجة آلاف الوثائق بكفاءة عالية

    **لماذا Batch؟**
    أسرع بكثير من استدعاء /preprocess لكل وثيقة على حدة.
    عند فهرسة 200,000 وثيقة، الفرق يكون في دقائق مقابل ساعات.
    """
    batch_results = preprocessor.process_batch(
        texts=request.texts,
        language=request.language,
        # language=request.language.value,
        lowercase=request.lowercase,
        remove_punctuation=request.remove_punctuation,
        remove_stopwords=request.remove_stopwords,
        apply_stemming=request.apply_stemming,
        apply_lemmatization=request.apply_lemmatization,
    )

    # نبني قائمة PreprocessResponse لكل نص
    results = [
        PreprocessResponse(
            original_text=original,
            processed_text=" ".join(tokens),
            tokens=tokens,
            token_count=len(tokens),
            steps_applied=steps,
        )
        for original, (tokens, steps) in zip(request.texts, batch_results)
    ]

    return BatchPreprocessResponse(
        results=results,
        total_processed=len(results),
        total_tokens=sum(r.token_count for r in results),
    )


# =============================================================
# نقطة الدخول عند تشغيل الملف مباشرةً
# =============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"[Preprocessing Service] يبدأ على port {PREPROCESSING_PORT}")
    uvicorn.run(
        "services.preprocessing.app:app",
        host="0.0.0.0",
        port=PREPROCESSING_PORT,
        reload=True,   # يُعيد التشغيل تلقائياً عند تغيير الكود
    )
