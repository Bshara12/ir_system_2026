"""
ui/pages/clustering.py
========================
صفحة Streamlit لعرض تجميع الوثائق.

وظائف:
- اختيار dataset وعدد الـ clusters
- عرض نتائج التجميع بشكل مرئي (بطاقات)
- دعم تجميع نتائج البحث مباشرة
- عرض Silhouette Score كمقياس للجودة

الاتصال:
    هذه الصفحة تتواصل فقط مع Clustering Service (port 8006)
    ولا تستدعي الخدمات الأخرى مباشرة.
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import streamlit as st
import httpx
from typing import Optional, List, Dict, Any

# عنوان Clustering Service
CLUSTERING_URL = "http://127.0.0.1:8006"

# ألوان الـ clusters (حتى 10 clusters)
CLUSTER_COLORS = [
    "#1f77b4",  # أزرق
    "#ff7f0e",  # برتقالي
    "#2ca02c",  # أخضر
    "#d62728",  # أحمر
    "#9467bd",  # بنفسجي
    "#8c564b",  # بني
    "#e377c2",  # وردي
    "#7f7f7f",  # رمادي
    "#bcbd22",  # زيتوني
    "#17becf",  # سماوي
]


# =============================================================
# دوال الاتصال بالخدمة
# =============================================================


def cluster_dataset(
    dataset_name: str,
    n_clusters: int,
    svd_components: int = 100,
    top_terms: int = 8,
) -> Optional[Dict[str, Any]]:
    """يستدعي Clustering Service لتجميع كل وثائق dataset."""
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{CLUSTERING_URL}/cluster/dataset",
                json={
                    "dataset_name": dataset_name,
                    "n_clusters": n_clusters,
                    "svd_components": svd_components,
                    "top_terms_per_cluster": top_terms,
                },
            )
        if response.status_code == 200:
            return response.json()
        else:
            detail = response.json().get("detail", "خطأ غير معروف")
            st.error(f"❌ خطأ من Clustering Service: {detail}")
            return None
    except httpx.ConnectError:
        st.error(f"❌ لا يمكن الوصول إلى Clustering Service على {CLUSTERING_URL}")
        return None
    except httpx.TimeoutException:
        st.error("⏱️ انتهت مهلة الاتصال — التجميع يأخذ وقتاً على datasets كبيرة")
        return None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None


def cluster_results(
    doc_ids: List[str],
    dataset_name: str,
    n_clusters: int,
) -> Optional[Dict[str, Any]]:
    """يُجمّع قائمة محددة من نتائج البحث."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{CLUSTERING_URL}/cluster/results",
                json={
                    "doc_ids": doc_ids,
                    "dataset_name": dataset_name,
                    "n_clusters": n_clusters,
                },
            )
        if response.status_code == 200:
            return response.json()
        else:
            detail = response.json().get("detail", "خطأ")
            st.error(f"❌ {detail}")
            return None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None


def find_optimal_k(
    dataset_name: str,
    k_min: int = 2,
    k_max: int = 8,
) -> Optional[Dict[str, Any]]:
    """يجد أفضل عدد clusters تلقائياً."""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{CLUSTERING_URL}/cluster/optimal-k",
                json={
                    "dataset_name": dataset_name,
                    "k_min": k_min,
                    "k_max": k_max,
                },
            )
        if response.status_code == 200:
            return response.json()
        else:
            detail = response.json().get("detail", "خطأ")
            st.error(f"❌ {detail}")
            return None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None


# =============================================================
# دوال العرض
# =============================================================


def render_cluster_card(cluster: Dict[str, Any], color: str) -> None:
    """يعرض بطاقة cluster واحد."""
    cluster_id = cluster["cluster_id"]
    size = cluster["size"]
    label = cluster["label"]
    top_terms = cluster.get("top_terms", [])
    doc_ids = cluster.get("doc_ids", [])
    centroid_score = cluster.get("centroid_score", 0.0)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}30);
            border: 2px solid {color};
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        ">
            <div style="
                background-color: {color};
                color: white;
                padding: 6px 14px;
                border-radius: 20px;
                display: inline-block;
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 10px;
            ">
                📁 {label}
            </div>
            <div style="
                display: flex;
                gap: 20px;
                margin-bottom: 10px;
                font-size: 13px;
                color: #444;
            ">
                <span>📄 <strong>{size}</strong> وثيقة</span>
                <span>🎯 تماسك: <strong>{centroid_score:.2f}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # الكلمات المفتاحية
    if top_terms:
        terms_html = " ".join(
            f'<span style="'
            f"background-color: {color}25; "
            f"border: 1px solid {color}; "
            f"color: {color}; "
            f"padding: 3px 10px; "
            f"border-radius: 12px; "
            f"font-size: 12px; "
            f"font-weight: bold; "
            f'margin: 2px;">'
            f"{term}</span>"
            for term in top_terms[:8]
        )
        st.markdown(
            f"<div style='margin-bottom:8px;'>🔑 <strong>كلمات مفتاحية:</strong><br>"
            f"<div style='margin-top:6px;'>{terms_html}</div></div>",
            unsafe_allow_html=True,
        )

    # عرض بعض doc_ids
    if doc_ids:
        preview_ids = doc_ids[:5]
        more = len(doc_ids) - 5
        ids_text = ", ".join(preview_ids)
        if more > 0:
            ids_text += f" ... (+{more} أخرى)"
        st.caption(f"🗂️ أمثلة: {ids_text}")


def render_quality_gauge(silhouette: float) -> None:
    """يعرض مقياس جودة التجميع بصري."""
    # تحديد الجودة
    if silhouette >= 0.5:
        quality = "ممتازة 🌟"
        color = "#28a745"
    elif silhouette >= 0.25:
        quality = "جيدة ✅"
        color = "#17a2b8"
    elif silhouette >= 0.0:
        quality = "مقبولة ⚠️"
        color = "#ffc107"
    else:
        quality = "ضعيفة ❌"
        color = "#dc3545"

    # شريط تقدم
    normalized = max(0.0, min(1.0, (silhouette + 1) / 2))  # تحويل [-1,1] → [0,1]
    percentage = int(normalized * 100)

    st.markdown(
        f"""
        <div style="
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px 16px;
            border: 1px solid #dee2e6;
            margin-bottom: 16px;
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <strong>جودة التجميع (Silhouette Score)</strong>
                <span style="color: {color}; font-weight: bold;">
                    {silhouette:.3f} — {quality}
                </span>
            </div>
            <div style="
                background: #e9ecef;
                border-radius: 4px;
                height: 12px;
                overflow: hidden;
            ">
                <div style="
                    background: {color};
                    width: {percentage}%;
                    height: 100%;
                    border-radius: 4px;
                    transition: width 0.5s;
                "></div>
            </div>
            <div style="
                font-size: 11px;
                color: #6c757d;
                margin-top: 6px;
            ">
                -1 (سيء) ←————————————→ +1 (مثالي)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_clusters_summary(result: Dict[str, Any]) -> None:
    """يعرض ملخص إحصائي للتجميع."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📦 عدد المجموعات", result["n_clusters"])
    with col2:
        st.metric("📄 إجمالي الوثائق", result["n_documents"])
    with col3:
        st.metric("⏱️ وقت التجميع", f"{result['build_time_seconds']:.2f}s")
    with col4:
        st.metric("🔬 أبعاد LSA", result["svd_components"])


# =============================================================
# الصفحة الرئيسية
# =============================================================


def main():
    st.title("📊 تجميع الوثائق")
    st.markdown("تجميع الوثائق المتشابهة في مجموعات باستخدام **K-Means + LSA**")
    st.markdown("---")

    # ── Sidebar: إعدادات التجميع ──────────────────────────────
    with st.sidebar:
        st.header("⚙️ إعدادات التجميع")

        clustering_url_input = st.text_input(
            "عنوان Clustering Service",
            value=CLUSTERING_URL,
        )

        st.markdown("---")

        dataset_name = st.selectbox(
            "📚 مجموعة البيانات",
            options=["dataset1", "dataset2", "trec-covid"],
            help="يجب أن يكون TF-IDF index مبنياً لهذا الـ dataset",
        )

        n_clusters = st.slider(
            "📁 عدد المجموعات (K)",
            min_value=2,
            max_value=15,
            value=5,
            step=1,
            help="كلما زاد العدد كانت المجموعات أدق لكن أصغر",
        )

        st.markdown("---")

        with st.expander("⚙️ إعدادات متقدمة"):
            svd_components = st.slider(
                "أبعاد LSA",
                min_value=20,
                max_value=200,
                value=100,
                step=10,
                help="أبعاد أكبر = دقة أعلى لكن أبطأ",
            )
            top_terms = st.slider(
                "كلمات مفتاحية لكل cluster",
                min_value=3,
                max_value=15,
                value=8,
            )

    # ── Tabs: أنواع التجميع ──────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        [
            "🗂️ تجميع كل الوثائق",
            "🔍 تجميع نتائج بحث",
            "📈 إيجاد أفضل K",
        ]
    )

    # ════════════════════════════════
    # Tab 1: تجميع كل الوثائق
    # ════════════════════════════════
    with tab1:
        st.subheader("🗂️ تجميع كل وثائق مجموعة البيانات")
        st.info(
            "يُجمّع **كل** وثائق الـ dataset في مجموعات متشابهة. "
            "مفيد لفهم هيكل مجموعة البيانات."
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            cluster_btn = st.button(
                "🚀 ابدأ التجميع",
                use_container_width=True,
                type="primary",
                key="cluster_dataset_btn",
            )

        if cluster_btn:
            with st.spinner(f"⏳ جاري التجميع في {n_clusters} مجموعة..."):
                result = cluster_dataset(
                    dataset_name=dataset_name,
                    n_clusters=n_clusters,
                    svd_components=svd_components,
                    top_terms=top_terms,
                )

            if result:
                st.session_state["cluster_result"] = result
                st.session_state["show_cluster_result"] = True

        # عرض النتائج
        if (
            st.session_state.get("show_cluster_result")
            and "cluster_result" in st.session_state
        ):
            result = st.session_state["cluster_result"]

            st.markdown("---")
            st.subheader("📊 نتائج التجميع")

            # ملخص إحصائي
            render_clusters_summary(result)
            st.markdown("---")

            # جودة التجميع
            render_quality_gauge(result["silhouette_score"])

            # بطاقات الـ clusters
            st.subheader(f"📁 المجموعات ({result['n_clusters']} مجموعة)")

            clusters = result.get("clusters", [])
            for i, cluster in enumerate(clusters):
                color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                render_cluster_card(cluster, color)

            # عرض البيانات الخام (اختياري)
            with st.expander("🔍 عرض البيانات الكاملة (JSON)", expanded=False):
                st.json(result)

    # ════════════════════════════════
    # Tab 2: تجميع نتائج بحث
    # ════════════════════════════════
    with tab2:
        st.subheader("🔍 تجميع نتائج البحث")
        st.info(
            "أدخل معرّفات الوثائق الناتجة عن بحث لتُجمَّع في مجموعات. "
            "هذا يُنظّم نتائج البحث بدل عرضها كقائمة."
        )

        doc_ids_input = st.text_area(
            "معرّفات الوثائق (واحد في كل سطر أو مفصولة بفاصلة)",
            height=120,
            placeholder="d1\nd2\nd3\nd5\nd8\nd10",
            help="أدخل doc_ids من نتائج بحث سابق",
        )

        n_clusters_results = st.slider(
            "عدد المجموعات للنتائج",
            min_value=2,
            max_value=8,
            value=3,
            key="n_clusters_results",
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            cluster_results_btn = st.button(
                "🚀 جمّع النتائج",
                use_container_width=True,
                type="primary",
                key="cluster_results_btn",
            )

        if cluster_results_btn:
            if not doc_ids_input.strip():
                st.error("❌ أدخل معرّفات الوثائق أولاً")
            else:
                # تحليل المدخلات
                raw = doc_ids_input.replace(",", "\n")
                doc_ids = [d.strip() for d in raw.split("\n") if d.strip()]

                if len(doc_ids) < 2:
                    st.error("❌ يجب إدخال وثيقتين على الأقل")
                else:
                    with st.spinner(f"⏳ جاري تجميع {len(doc_ids)} وثيقة..."):
                        result = cluster_results(
                            doc_ids=doc_ids,
                            dataset_name=dataset_name,
                            n_clusters=n_clusters_results,
                        )

                    if result:
                        st.markdown("---")
                        st.subheader("📊 نتائج التجميع")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📦 المجموعات", result["n_clusters"])
                        with col2:
                            st.metric("📄 الوثائق", result["n_documents"])
                        with col3:
                            st.metric(
                                "🎯 Silhouette", f"{result['silhouette_score']:.3f}"
                            )

                        st.markdown("---")
                        render_quality_gauge(result["silhouette_score"])

                        for i, cluster in enumerate(result.get("clusters", [])):
                            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                            render_cluster_card(cluster, color)

    # ════════════════════════════════
    # Tab 3: إيجاد أفضل K
    # ════════════════════════════════
    with tab3:
        st.subheader("📈 إيجاد أفضل عدد مجموعات (K)")
        st.info(
            "يجرّب قيم K مختلفة ويختار الأفضل بناءً على **Silhouette Score**. "
            "⚠️ قد يستغرق دقيقة على الـ datasets الكبيرة."
        )

        col1, col2 = st.columns(2)
        with col1:
            k_min = st.number_input("K الأدنى", min_value=2, max_value=5, value=2)
        with col2:
            k_max = st.number_input("K الأقصى", min_value=3, max_value=12, value=7)

        col1, col2 = st.columns([3, 1])
        with col2:
            optimal_btn = st.button(
                "🔍 ابحث عن أفضل K",
                use_container_width=True,
                type="primary",
                key="optimal_k_btn",
            )

        if optimal_btn:
            if k_min >= k_max:
                st.error("❌ K الأدنى يجب أن يكون أصغر من K الأقصى")
            else:
                with st.spinner(f"⏳ جاري اختبار K من {k_min} إلى {k_max}..."):
                    result = find_optimal_k(
                        dataset_name=dataset_name,
                        k_min=int(k_min),
                        k_max=int(k_max),
                    )

                if result:
                    st.markdown("---")

                    # أفضل K
                    best_k = result["best_k"]
                    st.success(f"✅ **أفضل عدد مجموعات: K = {best_k}**")

                    # رسم بياني للـ scores
                    scores = result["scores"]
                    k_values = list(scores.keys())
                    score_values = list(scores.values())

                    st.subheader("📊 Silhouette Score لكل قيمة K")

                    # عرض كجدول
                    table_data = []
                    for k, score in sorted(scores.items(), key=lambda x: int(x[0])):
                        is_best = "⭐ أفضل" if int(k) == best_k else ""
                        table_data.append(
                            {
                                "K": int(k),
                                "Silhouette Score": f"{score:.4f}",
                                "الجودة": (
                                    "ممتازة 🌟"
                                    if score >= 0.5
                                    else (
                                        "جيدة ✅"
                                        if score >= 0.25
                                        else "مقبولة ⚠️" if score >= 0 else "ضعيفة ❌"
                                    )
                                ),
                                "": is_best,
                            }
                        )
                    st.table(table_data)

                    # نصيحة
                    st.info(
                        f"💡 **التوصية:** استخدم K = {best_k} للحصول على "
                        f"أفضل تجميع لهذا الـ dataset."
                    )


# =============================================================
# تشغيل مباشر
# =============================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Document Clustering",
        page_icon="📊",
        layout="wide",
    )
    main()
else:
    # عند الاستيراد من main.py
    main()
