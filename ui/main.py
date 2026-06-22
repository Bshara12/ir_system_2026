"""
ui/main.py
==========
واجهة Streamlit الرئيسية لـ IR Search Engine 2026

وظائف:
- عرض واجهة بحث سهلة الاستخدام
- تجميع معاملات البحث من المستخدم
- استدعاء Gateway فقط (لا تستدعي الخدمات الأخرى مباشرة)
- عرض النتائج بشكل منسّق
- التحقق من صحة الخدمات
"""

import sys
import os

# إضافة المشروع للمسار ليمكن استيراد shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import httpx
from datetime import datetime
from typing import Optional, List

from shared.models import (
    RetrievalModel,
    DatasetName,
    ServiceStatus,
    DocumentResult,
    RetrievalResponse,
)
from ui.pages.clustering import main as clustering_page

# =============================================================
# إعدادات Streamlit
# =============================================================

st.set_page_config(
    page_title="IR Search Engine 2026",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS مخصص للواجهة
st.markdown(
    """
    <style>
    .result-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #1f77b4;
    }
    .result-title {
        font-weight: bold;
        font-size: 16px;
        color: #1f77b4;
        margin-bottom: 8px;
    }
    .result-meta {
        font-size: 12px;
        color: #666;
        margin-bottom: 8px;
    }
    .result-score {
        background-color: #ddd;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }
    .health-status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .health-status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
    .error-message {
        background-color: #f8d7da;
        padding: 12px;
        border-radius: 4px;
        border-left: 4px solid #dc3545;
        color: #721c24;
        margin-bottom: 12px;
    }
    .success-message {
        background-color: #d4edda;
        padding: 12px;
        border-radius: 4px;
        border-left: 4px solid #28a745;
        color: #155724;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# الدوال المساعدة
# =============================================================


def get_gateway_url() -> str:
    """الحصول على عنوان Gateway من Sidebar."""
    return st.session_state.get("gateway_url", "http://127.0.0.1:8000")


def check_services_health(gateway_url: str) -> Optional[List[ServiceStatus]]:
    """
    التحقق من صحة الخدمات عن طريق استدعاء Gateway.

    Args:
        gateway_url: عنوان الـ Gateway

    Returns:
        قائمة بحالات الخدمات أو None في حالة الفشل
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{gateway_url}/services/health")

        if response.status_code == 200:
            data = response.json()
            # تحويل البيانات إلى نماذج ServiceStatus
            services = [ServiceStatus(**item) for item in data]
            return services
        else:
            st.error(f"❌ خطأ من Gateway: {response.status_code}")
            return None

    except httpx.ConnectError:
        st.error(f"❌ لا يمكن الوصول إلى Gateway على {gateway_url}")
        return None
    except httpx.TimeoutException:
        st.error(f"⏱️ انتهت مهلة الاتصال بـ Gateway")
        return None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None


def perform_search(
    query: str,
    gateway_url: str,
    dataset: DatasetName,
    model: RetrievalModel,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
    apply_refinement: bool,
) -> Optional[RetrievalResponse]:
    """
    إجراء عملية البحث عن طريق استدعاء Gateway.

    Args:
        query: نص الاستعلام
        gateway_url: عنوان Gateway
        dataset: مجموعة البيانات
        model: نموذج الاسترجاع
        top_k: عدد النتائج المطلوبة
        bm25_k1: معامل BM25 k1
        bm25_b: معامل BM25 b
        apply_refinement: هل يتم تطبيق تحسين الاستعلام

    Returns:
        نتائج البحث أو None في حالة الفشل
    """
    try:
        payload = {
            "query": query,
            "dataset": dataset.value,
            "model": model.value,
            "top_k": top_k,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "apply_refinement": apply_refinement,
        }

        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                f"{gateway_url}/search",
                json=payload,
            )

        if response.status_code == 200:
            data = response.json()
            return RetrievalResponse(**data)
        else:
            error_msg = response.json().get("detail", "Unknown error")
            st.error(f"❌ خطأ في البحث: {error_msg}")
            return None

    except httpx.ConnectError:
        st.error(f"❌ لا يمكن الوصول إلى Gateway على {gateway_url}")
        return None
    except httpx.TimeoutException:
        st.error(f"⏱️ انتهت مهلة الاتصال (الطلب بطيء جداً)")
        return None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None


def display_result_card(result: DocumentResult, index: int):
    """عرض بطاقة نتيجة واحدة."""
    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            # العنوان والـ doc_id
            title_text = result.title if result.title else f"Document {result.doc_id}"
            st.markdown(
                f"<div class='result-title'>#{result.rank} - {title_text}</div>",
                unsafe_allow_html=True,
            )

            # البيانات الوصفية
            st.markdown(
                f"""
                <div class='result-meta'>
                <strong>Doc ID:</strong> {result.doc_id} | 
                <strong>Score:</strong> <span class='result-score'>{result.score:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # النص (مقتطف أول 300 حرف)
            text_snippet = result.text[:300]
            if len(result.text) > 300:
                text_snippet += "..."
            st.text(text_snippet)

        with col2:
            st.metric("Score", f"{result.score:.4f}")


# =============================================================
# الشريط الجانبي (Sidebar)
# =============================================================

st.sidebar.title("⚙️ إعدادات البحث")
st.sidebar.markdown("---")

# Gateway URL
gateway_url = st.sidebar.text_input(
    "🌐 عنوان Gateway",
    value="http://127.0.0.1:8000",
    help="عنوان الـ Gateway (نقطة الدخول الوحيدة للنظام)",
    key="gateway_url",
)

st.sidebar.markdown("---")

# اختيار مجموعة البيانات
dataset_options = [d.value for d in DatasetName]
selected_dataset = st.sidebar.selectbox(
    "📚 مجموعة البيانات",
    options=dataset_options,
    help="اختر مجموعة البيانات المراد البحث فيها",
)

# اختيار نموذج الاسترجاع
model_options = [m.value for m in RetrievalModel]
selected_model = st.sidebar.selectbox(
    "🤖 نموذج الاسترجاع",
    options=model_options,
    help="اختر نموذج البحث المستخدم",
)

st.sidebar.markdown("---")

# Top-K slider
top_k = st.sidebar.slider(
    "🔝 عدد النتائج المطلوبة (Top-K)",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="عدد الوثائق الأفضل المراد استرجاعها",
)

# BM25 parameters
st.sidebar.markdown("**معاملات BM25:**")

bm25_k1 = st.sidebar.slider(
    "k1 (معامل التكرار)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="يتحكم في تأثير تكرار المصطلح في الوثيقة",
)

bm25_b = st.sidebar.slider(
    "b (معامل الطول)",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="يتحكم في تأثير طول الوثيقة",
)

st.sidebar.markdown("---")

# Query Refinement checkbox
apply_refinement = st.sidebar.checkbox(
    "✨ تطبيق تحسين الاستعلام",
    value=False,
    help="تحسين نص الاستعلام قبل البحث (تصحيح إملائي، توسع المرادفات، إلخ)",
)

st.sidebar.markdown("---")

# زر فحص صحة الخدمات
if st.sidebar.button("🏥 فحص صحة الخدمات", use_container_width=True):
    st.sidebar.markdown("---")
    with st.spinner("🔄 جاري الفحص..."):
        services = check_services_health(gateway_url)

    if services:
        st.sidebar.markdown("**حالة الخدمات:**")
        for service in services:
            status_icon = "✅" if service.status == "healthy" else "❌"
            status_class = (
                "health-status-healthy"
                if service.status == "healthy"
                else "health-status-unhealthy"
            )
            status_text = "سليمة" if service.status == "healthy" else "معطلة"

            st.sidebar.markdown(
                f"{status_icon} **{service.service_name}**: <span class='{status_class}'>{status_text}</span>",
                unsafe_allow_html=True,
            )
        st.sidebar.markdown("---")


# =============================================================
# المنطقة الرئيسية (Main Area)
# =============================================================

# العنوان الرئيسي
st.title("🔍 IR Search Engine 2026")
st.markdown("محرك بحث استرجاع المعلومات - نظام بحث متقدم مع نماذج متعددة الاسترجاع")

st.markdown("---")

# منطقة الاستعلام
st.subheader("📝 ابدأ بحثك")

col1, col2 = st.columns([4, 1])

with col1:
    query_input = st.text_input(
        "اكتب استعلامك هنا",
        placeholder="مثال: cloud storage security",
        label_visibility="collapsed",
    )

with col2:
    search_button = st.button(
        "🔍 بحث",
        use_container_width=True,
        type="primary",
    )

st.markdown("---")

# تنفيذ البحث
if search_button:
    if not query_input.strip():
        st.error("❌ يرجى إدخال نص الاستعلام أولاً")
    else:
        with st.spinner("🔄 جاري البحث... يرجى الانتظار"):
            response = perform_search(
                query=query_input,
                gateway_url=gateway_url,
                dataset=DatasetName(selected_dataset),
                model=RetrievalModel(selected_model),
                top_k=top_k,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                apply_refinement=apply_refinement,
            )

        if response:
            # حفظ النتائج في Session State لعرضها حتى لو كان هناك تحديث آخر
            st.session_state.last_response = response
            st.session_state.show_results = True

# عرض النتائج إذا كانت موجودة
if st.session_state.get("show_results", False) and "last_response" in st.session_state:
    response = st.session_state.last_response

    st.markdown("---")

    # معلومات البحث
    st.subheader("📊 نتائج البحث")

    # إحصائيات البحث
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("نموذج البحث", response.model_used.value)

    with col2:
        st.metric("مجموعة البيانات", response.dataset.value)

    with col3:
        st.metric("عدد النتائج", response.total_results)

    with col4:
        st.metric("وقت المعالجة", f"{response.processing_time_ms:.2f} ms")

    with col5:
        st.metric("عدد النتائج المعروضة", len(response.results))

    st.markdown("---")

    # الاستعلام المحسّن (إذا تم تطبيقه)
    if response.refined_query and response.refined_query != response.query:
        st.info(f"✨ **الاستعلام بعد التحسين:** {response.refined_query}")

    st.markdown("---")

    # عرض النتائج
    if response.results:
        st.subheader(f"📄 النتائج ({len(response.results)} نتيجة)")

        for idx, result in enumerate(response.results, 1):
            display_result_card(result, idx)
            if idx < len(response.results):
                st.divider()
    else:
        st.info("📭 لم يتم العثور على نتائج تطابق الاستعلام")
else:
    # رسالة ترحيب عند عدم وجود بحث
    st.info(
        "👋 **مرحباً بك!**\n\n"
        "استخدم الشريط الجانبي لضبط إعدادات البحث، ثم ادخل استعلامك وانقر على 'بحث' "
        "للبدء في البحث عن المعلومات."
    )

st.markdown("---")

# =========================================================
# قسم التقييم التجريبي (Demo Evaluation Metrics)
# =========================================================

st.subheader("📊 Demo Evaluation Metrics")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(
        "<p style='color: #666; font-size: 13px;'>"
        "تقييم الأداء على بيانات تجريبية داخلية (بدون ملفات qrels على القرص)"
        "</p>",
        unsafe_allow_html=True,
    )

with col2:
    eval_button = st.button(
        "▶️ Run Demo Evaluation",
        use_container_width=True,
        key="eval_button",
    )

if eval_button:
    with st.spinner("🔄 جاري حساب المقاييس..."):
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(f"{gateway_url}/evaluate/demo")

            if response.status_code == 200:
                eval_data = response.json()
                st.session_state.eval_results = eval_data
                st.session_state.show_eval_results = True
            else:
                st.error(f"❌ خطأ من Gateway: {response.status_code}")
        except httpx.ConnectError:
            st.error(f"❌ لا يمكن الوصول إلى Gateway على {gateway_url}")
        except Exception as e:
            st.error(f"❌ خطأ: {str(e)}")

# عرض نتائج التقييم إذا كانت موجودة
if (
    st.session_state.get("show_eval_results", False)
    and "eval_results" in st.session_state
):
    eval_data = st.session_state.eval_results

    st.markdown("---")

    # عرض المقاييس الرئيسية
    st.subheader("🎯 المقاييس الرئيسية")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Precision@K", f"{eval_data.get('precision_at_k', 0):.4f}")

    with metric_col2:
        st.metric("Recall@K", f"{eval_data.get('recall_at_k', 0):.4f}")

    with metric_col3:
        st.metric("AP@K", f"{eval_data.get('average_precision_at_k', 0):.4f}")

    with metric_col4:
        st.metric("nDCG@K", f"{eval_data.get('ndcg_at_k', 0):.4f}")

    st.markdown("---")

    # عرض البيانات التجريبية
    with st.expander("📋 بيانات التقييم التجريبية", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**المستردة (Retrieved):**")
            st.code(str(eval_data.get("retrieved", [])), language="python")

            st.write("**ذات الصلة (Relevant):**")
            st.code(str(eval_data.get("relevant", [])), language="python")

        with col2:
            st.write("**درجات الصلة (Qrels):**")
            st.json(eval_data.get("qrels", {}))

            st.write(f"**Top-K:** {eval_data.get('k', 5)}")

st.markdown("---")

# =========================================================
# قسم التقييم الحقيقي باستخدام qrels
st.subheader("📌 Real Dataset Evaluation")

st.info(
    "⚠️ Real evaluation uses qrels. If the current index was built with max_docs=10000, "
    "scores may be low because many relevant documents may be outside the indexed subset."
)

real_dataset = st.selectbox(
    "📚 Dataset for evaluation",
    options=dataset_options,
    index=(
        dataset_options.index(selected_dataset)
        if selected_dataset in dataset_options
        else 0
    ),
)

real_model = st.selectbox(
    "🤖 Evaluation model",
    options=model_options,
    index=model_options.index(selected_model) if selected_model in model_options else 0,
)

real_top_k = st.slider(
    "🔝 Top-K for evaluation",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
)

real_max_queries = st.slider(
    "🧮 Max queries to evaluate",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
)

real_bm25_k1 = st.slider(
    "k1 (BM25)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
)

real_bm25_b = st.slider(
    "b (BM25)",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
)

real_apply_refinement = st.checkbox(
    "✨ Apply Query Refinement for evaluation",
    value=False,
)

real_eval_button = st.button(
    "▶️ Run Real Evaluation",
    use_container_width=True,
    key="real_eval_button",
)

if real_eval_button:
    with st.spinner("🔄 Running real dataset evaluation... Please wait"):
        try:
            payload = {
                "dataset_name": real_dataset,
                "model": real_model,
                "top_k": real_top_k,
                "max_queries": real_max_queries,
                "bm25_k1": real_bm25_k1,
                "bm25_b": real_bm25_b,
                "apply_refinement": real_apply_refinement,
            }
            with httpx.Client(timeout=180.0) as client:
                response = client.post(f"{gateway_url}/evaluate/dataset", json=payload)

            if response.status_code == 200:
                st.session_state.real_eval_results = response.json()
                st.session_state.show_real_eval_results = True
            else:
                error_msg = response.json().get("detail", response.text)
                st.error(f"❌ خطأ من Gateway: {response.status_code} - {error_msg}")
        except httpx.ConnectError:
            st.error(f"❌ لا يمكن الوصول إلى Gateway على {gateway_url}")
        except Exception as e:
            st.error(f"❌ خطأ: {str(e)}")

if (
    st.session_state.get("show_real_eval_results", False)
    and "real_eval_results" in st.session_state
):
    real_eval_data = st.session_state.real_eval_results
    st.markdown("---")
    st.subheader("📈 Real Evaluation Results")

    st.markdown(
        f"<strong>Dataset:</strong> {real_eval_data.get('dataset_name', '')}<br>"
        f"<strong>Model:</strong> {real_eval_data.get('model', '')}<br>"
        f"<strong>Evaluated Queries:</strong> {real_eval_data.get('evaluated_queries', 0)}",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAP", f"{real_eval_data.get('metrics', {}).get('MAP', 0.0):.4f}")
    with col2:
        st.metric(
            "Precision@K",
            f"{real_eval_data.get('metrics', {}).get('mean_precision_at_k', 0.0):.4f}",
        )
    with col3:
        st.metric(
            "Recall@K",
            f"{real_eval_data.get('metrics', {}).get('mean_recall_at_k', 0.0):.4f}",
        )
    with col4:
        st.metric(
            "nDCG@K",
            f"{real_eval_data.get('metrics', {}).get('mean_ndcg_at_k', 0.0):.4f}",
        )

    st.markdown("---")
    st.write(f"**Notes:** {real_eval_data.get('notes', '')}")
    st.markdown("---")

    if real_eval_data.get("per_query"):
        st.subheader("📑 Per-query results")
        st.table(real_eval_data["per_query"])

st.markdown("---")

# Footer
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 12px;'>"
    f"IR Search Engine 2026 | آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True,
)
