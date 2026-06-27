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

AUTO_AGENT_OPTION = "Auto (Agent)"
DATASET_OPTIONS = [
    DatasetName.DATASET_1.value,
    DatasetName.TREC_COVID.value,
    DatasetName.QUORA.value,
]
SEARCH_MODEL_OPTIONS = [
    AUTO_AGENT_OPTION,
    RetrievalModel.TFIDF.value,
    RetrievalModel.BM25.value,
    RetrievalModel.EMBEDDING.value,
    RetrievalModel.HYBRID_PARALLEL.value,
    RetrievalModel.HYBRID_SERIAL.value,
]
EVALUATION_MODEL_OPTIONS = [
    RetrievalModel.TFIDF.value,
    RetrievalModel.BM25.value,
    RetrievalModel.EMBEDDING.value,
    RetrievalModel.HYBRID_PARALLEL.value,
    RetrievalModel.HYBRID_SERIAL.value,
]


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
    dataset: str,
    model: str,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
    apply_refinement: bool,
) -> tuple[Optional[RetrievalResponse], Optional[dict]]:
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
        use_agent = model == AUTO_AGENT_OPTION
        payload = {
            "query": query,
            "dataset": dataset,
            "top_k": top_k,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "apply_refinement": apply_refinement,
        }
        if not use_agent:
            payload["model"] = model
        
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                f"{gateway_url}/agent/search" if use_agent else f"{gateway_url}/search",
                json=payload,
            )

        if response.status_code == 200:
            data = response.json()
            agent_decision = data.pop("agent_decision", None)
            return RetrievalResponse(**data), agent_decision
        else:
            error_msg = response.json().get("detail", "Unknown error")
            st.error(f"❌ خطأ في البحث: {error_msg}")
            return None, None
            
    except httpx.ConnectError:
        st.error(f"❌ لا يمكن الوصول إلى Gateway على {gateway_url}")
        return None, None
    except httpx.TimeoutException:
        st.error(f"⏱️ انتهت مهلة الاتصال (الطلب بطيء جداً)")
        return None, None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None, None

# =============================================================
# واجهة المستخدم الفعّالة (UI Elements)
# =============================================================

st.title("🔍 محرك البحث الدلالي (IR Search Engine)")

# 1. القائمة الجانبية (Sidebar) للإعدادات
with st.sidebar:
    st.header("⚙️ إعدادات النظام")
    
    # تحديد مسار الـ Gateway
    st.session_state["gateway_url"] = st.text_input(
        "مسار الـ Gateway:", 
        value="http://127.0.0.1:8000"
    )
    
    # زر فحص حالة الخدمات
    if st.button("🔄 فحص حالة الخدمات", use_container_width=True):
        with st.spinner("جاري فحص الخدمات..."):
            statuses = check_services_health(st.session_state["gateway_url"])
            if statuses:
                st.markdown("### حالة الخدمات:")
                for s in statuses:
                    icon = "✅" if s.status == "healthy" else "❌"
                    color_class = "health-status-healthy" if s.status == "healthy" else "health-status-unhealthy"
                    st.markdown(
                        f"<div style='margin-bottom: 5px;'><span class='{color_class}'>{icon} {s.service_name}</span></div>", 
                        unsafe_allow_html=True
                    )
    
    st.divider()
    
    # إعدادات البحث
    st.header("🎯 إعدادات البحث")
    selected_dataset = st.selectbox("مجموعة البيانات (Dataset):", DATASET_OPTIONS)
    selected_model = st.selectbox("نموذج البحث (Model):", SEARCH_MODEL_OPTIONS)
    top_k = st.slider("عدد النتائج (Top K):", min_value=1, max_value=50, value=10)
    apply_refinement = st.checkbox("تفعيل تحسين الاستعلام (Query Refinement)", value=False)
    
    with st.expander("معاملات BM25 (متقدم)"):
        bm25_k1 = st.number_input("k1", value=1.5, step=0.1)
        bm25_b = st.number_input("b", value=0.75, step=0.1)

# 2. منطقة البحث الرئيسية
st.markdown("### أدخل استعلامك هنا:")
query = st.text_input("شريط البحث", label_visibility="collapsed", placeholder="اكتب ما تبحث عنه... (مثال: cloud computing architecture)")

# زر البحث
search_button = st.button("🔍 بحث", type="primary", use_container_width=True)

if search_button:
    if not query.strip():
        st.warning("⚠️ يرجى إدخال نص للبحث عنه أولاً.")
    else:
        with st.spinner("جاري البحث في الوثائق..."):
            # استدعاء دالة البحث
            response, agent_decision = perform_search(
                query=query,
                gateway_url=get_gateway_url(),
                dataset=selected_dataset,
                model=selected_model,
                top_k=top_k,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                apply_refinement=apply_refinement,
            )
            
            if response:
                st.markdown(
                    f"<div class='success-message'>✅ تم العثور على <strong>{response.total_results}</strong> نتيجة في <strong>{response.processing_time:.3f}</strong> ثانية.</div>",
                    unsafe_allow_html=True
                )
                
                # إذا تم استخدام الوكيل الذكي (AUTO_AGENT)، نعرض قراره
                if agent_decision:
                    with st.expander("🤖 قرار الوكيل الذكي (Agent Reasoning)"):
                        st.json(agent_decision)
                
                st.markdown("---")
                
                # عرض النتائج باستخدام الـ CSS المخصص
                if not response.results:
                    st.info("لم يتم العثور على وثائق مطابقة.")
                else:
                    for doc in response.results:
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-title">📄 وثيقة رقم: {doc.doc_id}</div>
                                <div class="result-meta">
                                    <span class="result-score">⭐ التقييم (Score): {doc.score:.4f}</span>
                                </div>
                                <div style="color: #333; line-height: 1.6;">
                                    {doc.text[:400]}{'...' if len(doc.text) > 400 else ''}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

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
dataset_options = DATASET_OPTIONS
selected_dataset = st.sidebar.selectbox(
    "📚 مجموعة البيانات",
    options=dataset_options,
    help="اختر مجموعة البيانات المراد البحث فيها",
)

# اختيار نموذج الاسترجاع
search_model_options = SEARCH_MODEL_OPTIONS
evaluation_model_options = EVALUATION_MODEL_OPTIONS
selected_model = st.sidebar.selectbox(
    "🤖 نموذج الاسترجاع",
    options=search_model_options,
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
            response, agent_decision = perform_search(
                query=query_input,
                gateway_url=gateway_url,
                dataset=selected_dataset,
                model=selected_model,
                top_k=top_k,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                apply_refinement=apply_refinement,
            )

        if response:
            # حفظ النتائج في Session State لعرضها حتى لو كان هناك تحديث آخر
            st.session_state.last_response = response
            st.session_state.last_agent_decision = agent_decision
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
        st.info(
            f"✨ **الاستعلام بعد التحسين:** {response.refined_query}"
        )
    
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

st.warning(
    "The old trec-covid max_docs=10000 setup was only for local testing. "
    "Final evaluation should use the full `quora` dataset or another complete manageable dataset with qrels."
)
st.caption(
    "Auto Agent is used for single-query search only. Real evaluation requires a fixed retrieval model."
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
    options=evaluation_model_options,
    index=evaluation_model_options.index(selected_model) if selected_model in evaluation_model_options else 0,
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

    metrics = real_eval_data.get("metrics", {})
    summary_table = [{
        "dataset": real_eval_data.get("dataset_name", ""),
        "model": real_eval_data.get("model", ""),
        "top_k": real_eval_data.get("top_k", 0),
        "max_queries": real_eval_data.get("max_queries", 0),
        "MAP": metrics.get("MAP", 0.0),
        "Precision@K": metrics.get("mean_precision_at_k", 0.0),
        "Recall@K": metrics.get("mean_recall_at_k", 0.0),
        "nDCG@K": metrics.get("mean_ndcg_at_k", 0.0),
        "notes": real_eval_data.get("notes", ""),
    }]
    st.subheader("📋 Report summary")
    st.table(summary_table)
    st.markdown("---")

    if real_eval_data.get("per_query"):
        st.subheader("📑 Per-query results")
        st.table(real_eval_data["per_query"])

st.markdown("---")


# =========================================================
# STANDARD_EVALUATION_REPORT_BLOCK
# تقرير تقييم ثابت حسب متطلب المشروع
# =========================================================

if "selected_page" not in globals() or selected_page == "🔍 محرك البحث والتقييم":
    st.markdown("---")
    st.subheader("📊 Standard Evaluation Report")

    st.caption(
        "This benchmark uses the official dataset evaluation queries and qrels. "
        "Manual search-box queries are not used for evaluation."
    )

    st.info(
        "Baseline = بدون Query Refinement. "
        "With Query Refinement = بعد تطبيق ميزة تحسين الاستعلام. "
        "Top-K is fixed to 10, so Precision@K is reported as Precision@10."
    )

    try:
        import pandas as pd
        import altair as alt

        standard_dataset_options = globals().get(
            "DATASET_OPTIONS",
            globals().get("dataset_options", ["dataset1", "trec-covid", "quora"])
        )

        if not isinstance(standard_dataset_options, list):
            standard_dataset_options = list(standard_dataset_options)

        default_standard_dataset_index = (
            standard_dataset_options.index("quora")
            if "quora" in standard_dataset_options
            else 0
        )

        standard_dataset = st.selectbox(
            "📚 Dataset for standard benchmark",
            options=standard_dataset_options,
            index=default_standard_dataset_index,
            key="standard_eval_dataset",
        )

        st.caption(
            f"Evaluation source: data/datasets/{standard_dataset}/queries.jsonl "
            f"+ data/datasets/{standard_dataset}/qrels.jsonl"
        )

        standard_max_queries = st.slider(
            "🧮 Max official test queries",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="standard_eval_max_queries",
            help="Use the same fixed number of official queries for all models to ensure fair comparison.",
        )

        col_std_1, col_std_2 = st.columns(2)

        with col_std_1:
            standard_bm25_k1 = st.slider(
                "k1 for benchmark BM25",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                key="standard_eval_bm25_k1",
            )

        with col_std_2:
            standard_bm25_b = st.slider(
                "b for benchmark BM25",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                key="standard_eval_bm25_b",
            )

        STANDARD_EVAL_MODELS = [
            "tfidf",
            "bm25",
            "embedding",
            "hybrid_parallel",
            "hybrid_serial",
        ]

        st.write("**Models included in this benchmark:**")
        st.code(", ".join(STANDARD_EVAL_MODELS))

        run_standard_eval = st.button(
            "▶️ Run Standard Evaluation Benchmark",
            use_container_width=True,
            key="run_standard_evaluation_benchmark",
        )

        if run_standard_eval:
            rows = []
            progress_total = len(STANDARD_EVAL_MODELS) * 2
            progress_done = 0
            progress_bar = st.progress(0)

            with st.spinner("Running fixed benchmark on official test queries and qrels..."):
                for model_name in STANDARD_EVAL_MODELS:
                    for feature_label, refinement_value in [
                        ("Baseline", False),
                        ("With Query Refinement", True),
                    ]:
                        payload = {
                            "dataset_name": standard_dataset,
                            "model": model_name,
                            "top_k": 10,
                            "max_queries": standard_max_queries,
                            "bm25_k1": standard_bm25_k1,
                            "bm25_b": standard_bm25_b,
                            "apply_refinement": refinement_value,
                        }

                        row = {
                            "Dataset": standard_dataset,
                            "Model": model_name,
                            "Feature Setting": feature_label,
                            "Top-K": 10,
                            "Max Queries": standard_max_queries,
                            "MAP": None,
                            "Precision@10": None,
                            "Recall@10": None,
                            "nDCG@10": None,
                            "Error": "",
                        }

                        try:
                            with httpx.Client(timeout=240.0) as client:
                                resp = client.post(
                                    f"{gateway_url}/evaluate/dataset",
                                    json=payload,
                                )

                            if resp.status_code == 200:
                                data = resp.json()
                                metrics = data.get("metrics", {})

                                row["MAP"] = metrics.get("MAP", 0.0)
                                row["Precision@10"] = metrics.get("mean_precision_at_k", 0.0)
                                row["Recall@10"] = metrics.get("mean_recall_at_k", 0.0)
                                row["nDCG@10"] = metrics.get("mean_ndcg_at_k", 0.0)
                            else:
                                try:
                                    err = resp.json().get("detail", resp.text)
                                except Exception:
                                    err = resp.text
                                row["Error"] = f"HTTP {resp.status_code}: {err}"

                        except Exception as e:
                            row["Error"] = str(e)

                        rows.append(row)
                        progress_done += 1
                        progress_bar.progress(progress_done / progress_total)

            report_df = pd.DataFrame(rows)
            st.session_state.standard_eval_report_df = report_df
            st.success("✅ Standard evaluation benchmark finished.")

        if "standard_eval_report_df" in st.session_state:
            report_df = st.session_state.standard_eval_report_df

            st.markdown("---")
            st.subheader("📋 Fixed Evaluation Table")
            st.dataframe(report_df, use_container_width=True)

            csv_bytes = report_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download evaluation table as CSV",
                data=csv_bytes,
                file_name=f"standard_evaluation_{standard_dataset}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_standard_eval_csv",
            )

            numeric_df = report_df[report_df["Error"].fillna("") == ""].copy()

            if not numeric_df.empty:
                st.markdown("---")
                st.subheader("📈 Evaluation Charts")

                for metric_name in ["MAP", "Precision@10", "Recall@10", "nDCG@10"]:
                    st.write(f"**{metric_name} comparison**")

                    chart_data = numeric_df[
                        ["Model", "Feature Setting", metric_name]
                    ].copy()

                    chart_data[metric_name] = pd.to_numeric(
                        chart_data[metric_name],
                        errors="coerce",
                    )

                    chart = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Model:N",
                                title="Model",
                                sort=[
                                    "tfidf",
                                    "bm25",
                                    "embedding",
                                    "hybrid_parallel",
                                    "hybrid_serial",
                                ],
                            ),
                            xOffset=alt.XOffset(
                                "Feature Setting:N",
                                title="Feature Setting",
                            ),
                            y=alt.Y(
                                f"{metric_name}:Q",
                                title=metric_name,
                                scale=alt.Scale(domain=[0, 1]),
                            ),
                            color=alt.Color(
                                "Feature Setting:N",
                                title="Feature Setting",
                            ),
                            tooltip=[
                                alt.Tooltip("Dataset:N"),
                                alt.Tooltip("Model:N"),
                                alt.Tooltip("Feature Setting:N"),
                                alt.Tooltip(f"{metric_name}:Q", format=".4f"),
                            ],
                        )
                        .properties(height=360)
                    )

                    st.altair_chart(chart, use_container_width=True)

                st.markdown("---")
                st.subheader("🧠 Interpretation Notes")
                st.markdown(
                    """
                    - **BM25** is expected to perform well when relevant documents share exact terms with the query.
                    - **TF-IDF** is a strong lexical baseline but may be weaker than BM25 because it does not handle document length and term saturation in the same way.
                    - **Embedding** can improve semantic matching when the query and document use different words with similar meaning.
                    - **Hybrid models** should be compared against single models using the same Top-K and the same official queries.
                    - **Query Refinement** may improve or reduce scores depending on whether the refined query still matches the official qrels.
                    """
                )
            else:
                st.warning("No successful benchmark rows to chart yet.")

    except Exception as e:
        st.error(f"❌ Standard Evaluation Report failed: {str(e)}")


# Footer
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 12px;'>"
    f"IR Search Engine 2026 | آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True,
)
