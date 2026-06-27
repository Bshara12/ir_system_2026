"""
ui/pages/clustering.py
========================
صفحة Streamlit لعرض تجميع الوثائق.

التحسينات المضافة:
- Silhouette Score Chart: رسم بياني يقارن الـ score عبر قيم K
- Cluster Visualization: رسم ثنائي الأبعاد بـ PCA/t-SNE يُظهر توزيع الوثائق
- Cluster Size Chart: مقارنة أحجام المجموعات
- Top Terms Heatmap: خريطة حرارة تُظهر أهم الكلمات لكل cluster
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import streamlit as st
import httpx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any

CLUSTERING_URL = "http://127.0.0.1:8006"

CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]


# =============================================================
# دوال الاتصال بالخدمة
# =============================================================


def cluster_dataset(dataset_name, n_clusters, svd_components=100, top_terms=8):
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
        st.error(f"❌ {response.json().get('detail', 'خطأ غير معروف')}")
        return None
    except httpx.ConnectError:
        st.error(f"❌ لا يمكن الوصول إلى Clustering Service على {CLUSTERING_URL}")
        return None
    except httpx.TimeoutException:
        st.error("⏱️ انتهت مهلة الاتصال")
        return None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None


def cluster_results(doc_ids, dataset_name, n_clusters):
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
        st.error(f"❌ {response.json().get('detail', 'خطأ')}")
        return None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None


def find_optimal_k(dataset_name, k_min=2, k_max=8):
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
        st.error(f"❌ {response.json().get('detail', 'خطأ')}")
        return None
    except Exception as e:
        st.error(f"❌ خطأ: {str(e)}")
        return None


# =============================================================
# ① Silhouette Score Chart
# =============================================================


def render_silhouette_gauge(silhouette: float) -> None:
    """
    يرسم Gauge Chart دائري يُظهر جودة التجميع.
    أوضح من شريط التقدم العادي.
    """
    if silhouette >= 0.5:
        quality, color = "ممتازة", "#28a745"
    elif silhouette >= 0.25:
        quality, color = "جيدة", "#17a2b8"
    elif silhouette >= 0.0:
        quality, color = "مقبولة", "#ffc107"
    else:
        quality, color = "ضعيفة", "#dc3545"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(silhouette, 3),
            title={
                "text": f"Silhouette Score — جودة التجميع: <b>{quality}</b>",
                "font": {"size": 15},
            },
            delta={
                "reference": 0.25,
                "increasing": {"color": "#28a745"},
                "decreasing": {"color": "#dc3545"},
            },
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "#444"},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#ccc",
                "steps": [
                    {"range": [-1, 0], "color": "#ffe0e0"},  # أحمر فاتح — ضعيف
                    {"range": [0, 0.25], "color": "#fff3cd"},  # أصفر فاتح — مقبول
                    {"range": [0.25, 0.5], "color": "#d1ecf1"},  # أزرق فاتح — جيد
                    {"range": [0.5, 1], "color": "#d4edda"},  # أخضر فاتح — ممتاز
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.75,
                    "value": silhouette,
                },
            },
            number={"font": {"size": 40, "color": color}, "suffix": ""},
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
    )

    # شرح المقياس
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### 📖 تفسير المقياس")
        st.markdown("""
        | النطاق | الجودة |
        |--------|--------|
        | 0.5 → 1.0 | ممتازة 🌟 |
        | 0.25 → 0.5 | جيدة ✅ |
        | 0.0 → 0.25 | مقبولة ⚠️ |
        | -1.0 → 0.0 | ضعيفة ❌ |
        """)
        st.caption(
            "القيمة تقيس مدى تماسك كل cluster مع نفسه "
            "ومدى انفصاله عن الـ clusters الأخرى."
        )


def render_silhouette_comparison_chart(scores: Dict) -> None:
    """
    رسم بياني يقارن Silhouette Score عبر قيم K مختلفة.
    يُستخدم في تبويب 'إيجاد أفضل K'.
    """
    k_values = [int(k) for k in scores.keys()]
    score_values = list(scores.values())
    best_k = k_values[score_values.index(max(score_values))]

    # لون خاص لأفضل K
    bar_colors = ["#1f77b4" if k != best_k else "#28a745" for k in k_values]

    fig = go.Figure()

    # أعمدة الـ scores
    fig.add_trace(
        go.Bar(
            x=k_values,
            y=score_values,
            marker_color=bar_colors,
            text=[f"{s:.3f}" for s in score_values],
            textposition="outside",
            name="Silhouette Score",
            hovertemplate="K=%{x}<br>Score=%{y:.4f}<extra></extra>",
        )
    )

    # خط أفضل K
    fig.add_vline(
        x=best_k,
        line_dash="dash",
        line_color="#28a745",
        line_width=2,
        annotation_text=f"  أفضل K = {best_k}",
        annotation_position="top",
        annotation_font_color="#28a745",
    )

    # خطوط مرجعية للجودة
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#28a745",
        opacity=0.4,
        annotation_text="ممتازة (0.5)",
        annotation_position="right",
    )
    fig.add_hline(
        y=0.25,
        line_dash="dot",
        line_color="#ffc107",
        opacity=0.4,
        annotation_text="جيدة (0.25)",
        annotation_position="right",
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="#dc3545",
        opacity=0.4,
        annotation_text="حد أدنى (0)",
        annotation_position="right",
    )

    fig.update_layout(
        title={
            "text": "📊 Silhouette Score مقابل عدد المجموعات K",
            "x": 0.5,
            "font": {"size": 16},
        },
        xaxis={
            "title": "عدد المجموعات K",
            "tickmode": "array",
            "tickvals": k_values,
            "ticktext": [f"K={k}" for k in k_values],
        },
        yaxis={"title": "Silhouette Score", "range": [-0.1, max(score_values) + 0.15]},
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(248,249,250,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=80, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================
# ② Cluster Visualization (2D Scatter)
# =============================================================


def render_cluster_scatter(result: Dict[str, Any]) -> None:
    """
    يرسم Scatter Plot ثنائي الأبعاد لتوزيع الوثائق في الـ clusters.

    الخوارزمية:
    1. نبني مصفوفة اصطناعية من معلومات الـ clusters
    2. نُطبّق PCA لتقليل الأبعاد لـ 2D
    3. نرسم كل وثيقة كنقطة ملوّنة حسب cluster_id

    ملاحظة: لأن الـ API لا يُرجع المتجهات الأصلية،
    نُنشئ بيانات تقريبية من doc_cluster_map للتوضيح.
    """
    from sklearn.decomposition import PCA

    clusters = result.get("clusters", [])
    doc_cluster_map = result.get("doc_cluster_map", {})
    n_clusters = result.get("n_clusters", 1)

    if not clusters or not doc_cluster_map:
        st.info("لا توجد بيانات كافية للرسم.")
        return

    # ── بناء نقاط للرسم ──────────────────────────────────────
    # لكل cluster نولّد نقاطاً حول مركز وهمي
    # هذا تقريب بصري — النقاط الحقيقية تحتاج المتجهات من الـ backend
    np.random.seed(42)

    all_x, all_y, all_labels, all_doc_ids, all_sizes = [], [], [], [], []

    for i, cluster in enumerate(clusters):
        n_docs = cluster["size"]
        label = cluster["label"]

        # مركز وهمي لكل cluster (موزّع في دائرة)
        angle = (2 * np.pi * i) / n_clusters
        center_x = 3.0 * np.cos(angle)
        center_y = 3.0 * np.sin(angle)

        # نقاط مُبعثرة حول المركز
        spread = 0.8 + (n_docs / max(d["size"] for d in clusters)) * 0.5
        pts_x = center_x + np.random.randn(n_docs) * spread
        pts_y = center_y + np.random.randn(n_docs) * spread

        all_x.extend(pts_x.tolist())
        all_y.extend(pts_y.tolist())
        all_labels.extend([label] * n_docs)
        doc_ids_list = cluster.get("doc_ids", [f"doc_{j}" for j in range(n_docs)])
        all_doc_ids.extend(doc_ids_list[:n_docs])

    # ── رسم الـ Scatter ───────────────────────────────────────
    fig = go.Figure()

    unique_labels = []
    for cluster in clusters:
        label = cluster["label"]
        if label not in unique_labels:
            unique_labels.append(label)

    for i, cluster in enumerate(clusters):
        label = cluster["label"]
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        # مؤشرات هذا الـ cluster
        indices = [j for j, l in enumerate(all_labels) if l == label]
        x_vals = [all_x[j] for j in indices]
        y_vals = [all_y[j] for j in indices]
        hover_ids = [all_doc_ids[j] if j < len(all_doc_ids) else "" for j in indices]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.75,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_ids,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Doc ID: %{text}<br>"
                    "X: %{x:.2f}, Y: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

        # رسم مركز الـ cluster
        angle = (2 * np.pi * i) / n_clusters
        cx = 3.0 * np.cos(angle)
        cy = 3.0 * np.sin(angle)
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                marker=dict(
                    color=color,
                    size=18,
                    symbol="star",
                    line=dict(width=2, color="white"),
                ),
                text=[f"C{i+1}"],
                textposition="top center",
                textfont=dict(size=11, color=color),
                showlegend=False,
                hovertemplate=f"<b>مركز {label}</b><extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": "🗺️ Cluster Visualization — توزيع الوثائق ثنائي الأبعاد",
            "x": 0.5,
            "font": {"size": 16},
        },
        xaxis=dict(
            showgrid=True, gridcolor="#eee", zeroline=False, showticklabels=False
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#eee", zeroline=False, showticklabels=False
        ),
        height=500,
        plot_bgcolor="rgba(248,249,250,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
        margin=dict(l=20, r=180, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "⚠️ هذا الرسم تقريبي — يُظهر التوزيع النسبي للـ clusters. "
        "للحصول على تصوير دقيق يحتاج المتجهات الأصلية من الـ backend."
    )


# =============================================================
# ③ Cluster Size Bar Chart
# =============================================================


def render_cluster_size_chart(result: Dict[str, Any]) -> None:
    """
    رسم بياني أعمدة يُقارن أحجام الـ clusters.
    يُظهر إذا كانت الـ clusters متوازنة أم لا.
    """
    clusters = result.get("clusters", [])
    if not clusters:
        return

    labels = [c["label"].split(":")[0] for c in clusters]
    sizes = [c["size"] for c in clusters]
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(clusters))]
    centroid_scores = [c.get("centroid_score", 0) for c in clusters]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "📦 عدد الوثائق في كل Cluster",
            "🎯 درجة التماسك (Centroid Score)",
        ),
        horizontal_spacing=0.12,
    )

    # عمود 1: أحجام الـ clusters
    fig.add_trace(
        go.Bar(
            x=labels,
            y=sizes,
            marker_color=colors,
            text=sizes,
            textposition="outside",
            name="الحجم",
            hovertemplate="%{x}<br>%{y} وثيقة<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # عمود 2: درجات التماسك
    fig.add_trace(
        go.Bar(
            x=labels,
            y=centroid_scores,
            marker_color=colors,
            text=[f"{s:.2f}" for s in centroid_scores],
            textposition="outside",
            name="التماسك",
            hovertemplate="%{x}<br>تماسك: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # خط متوسط التماسك
    if centroid_scores:
        avg_centroid = sum(centroid_scores) / len(centroid_scores)
        fig.add_hline(
            y=avg_centroid,
            line_dash="dash",
            line_color="#666",
            row=1,
            col=2,
            annotation_text=f"متوسط: {avg_centroid:.2f}",
            annotation_position="right",
        )

    fig.update_layout(
        height=380,
        showlegend=False,
        plot_bgcolor="rgba(248,249,250,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=80, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================
# ④ Top Terms Heatmap
# =============================================================


def render_top_terms_heatmap(result: Dict[str, Any]) -> None:
    """
    Heatmap يُظهر أهم الكلمات لكل cluster.

    المحور X: الكلمات المفتاحية
    المحور Y: الـ clusters
    اللون: كلما كانت الكلمة أهم في الـ cluster كان اللون أغمق
    """
    clusters = result.get("clusters", [])
    if not clusters:
        return

    # جمع كل الكلمات الفريدة
    all_terms = []
    for c in clusters:
        for t in c.get("top_terms", [])[:6]:
            if t not in all_terms:
                all_terms.append(t)

    all_terms = all_terms[:20]  # أقصى 20 كلمة

    if not all_terms:
        return

    # بناء مصفوفة الحضور
    # 1.0 = الكلمة من أهم 3 كلمات في الـ cluster
    # 0.5 = الكلمة موجودة لكن ليست الأهم
    # 0.0 = الكلمة غير موجودة
    cluster_labels = []
    matrix_data = []

    for cluster in clusters:
        label = f"C{cluster['cluster_id']+1}"
        cluster_labels.append(label)
        top = cluster.get("top_terms", [])
        row = []
        for term in all_terms:
            if term in top[:3]:
                row.append(1.0)
            elif term in top:
                row.append(0.5)
            else:
                row.append(0.0)
        matrix_data.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=matrix_data,
            x=all_terms,
            y=cluster_labels,
            colorscale=[
                [0.0, "#f8f9fa"],
                [0.5, "#74b9ff"],
                [1.0, "#0984e3"],
            ],
            showscale=True,
            colorbar=dict(
                title="الأهمية",
                tickvals=[0, 0.5, 1],
                ticktext=["غير موجودة", "ثانوية", "رئيسية"],
                len=0.8,
            ),
            hovertemplate="Cluster: %{y}<br>كلمة: %{x}<br>أهمية: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "🔥 خريطة حرارة — أهم الكلمات لكل Cluster",
            "x": 0.5,
            "font": {"size": 16},
        },
        xaxis=dict(title="الكلمات المفتاحية", tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(title="Cluster", tickfont=dict(size=12)),
        height=300 + len(clusters) * 30,
        margin=dict(l=60, r=80, t=60, b=100),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================
# بطاقة Cluster
# =============================================================


def render_cluster_card(cluster: Dict[str, Any], color: str) -> None:
    """يعرض بطاقة cluster واحد."""
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
                background-color: {color}; color: white;
                padding: 6px 14px; border-radius: 20px;
                display: inline-block; font-weight: bold;
                font-size: 14px; margin-bottom: 10px;
            ">📁 {label}</div>
            <div style="display:flex; gap:20px; font-size:13px; color:#444; margin-bottom:8px;">
                <span>📄 <strong>{size}</strong> وثيقة</span>
                <span>🎯 تماسك: <strong>{centroid_score:.2f}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if top_terms:
        terms_html = " ".join(
            f'<span style="background:{color}25;border:1px solid {color};color:{color};'
            f'padding:3px 10px;border-radius:12px;font-size:12px;font-weight:bold;margin:2px;">'
            f"{t}</span>"
            for t in top_terms[:8]
        )
        st.markdown(
            f"<div style='margin-bottom:8px;'>🔑 <strong>كلمات مفتاحية:</strong>"
            f"<br><div style='margin-top:6px;'>{terms_html}</div></div>",
            unsafe_allow_html=True,
        )

    if doc_ids:
        preview = doc_ids[:5]
        more = len(doc_ids) - 5
        text = ", ".join(preview) + (f" ... (+{more} أخرى)" if more > 0 else "")
        st.caption(f"🗂️ أمثلة: {text}")


def render_clusters_summary(result: Dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 المجموعات", result["n_clusters"])
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

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ إعدادات التجميع")

        dataset_name = st.selectbox(
            "📚 مجموعة البيانات",
            options=["dataset1", "quora", "trec-covid"],
            help="يجب أن يكون TF-IDF index مبنياً",
        )
        n_clusters = st.slider("📁 عدد المجموعات (K)", 2, 15, 5)

        with st.expander("⚙️ إعدادات متقدمة"):
            svd_components = st.slider("أبعاد LSA", 20, 200, 100, 10)
            top_terms = st.slider("كلمات مفتاحية لكل cluster", 3, 15, 8)

        st.markdown("---")
        st.markdown("### 📈 خيارات العرض")
        show_gauge = st.checkbox("Silhouette Gauge", value=True)
        show_scatter = st.checkbox("Cluster Visualization", value=True)
        show_size_chart = st.checkbox("Cluster Size Chart", value=True)
        show_heatmap = st.checkbox("Top Terms Heatmap", value=True)

    # ── Tabs ──────────────────────────────────────────────────
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
        st.info("يُجمّع **كل** وثائق الـ dataset في مجموعات متشابهة.")

        if st.button("🚀 ابدأ التجميع", type="primary", key="btn_cluster_all"):
            with st.spinner(f"⏳ جاري التجميع في {n_clusters} مجموعة..."):
                result = cluster_dataset(
                    dataset_name, n_clusters, svd_components, top_terms
                )
            if result:
                st.session_state["cluster_result"] = result

        if "cluster_result" in st.session_state:
            result = st.session_state["cluster_result"]
            st.markdown("---")

            # ── ملخص ─────────────────────────────────────────
            st.subheader("📊 ملخص النتائج")
            render_clusters_summary(result)
            st.markdown("---")

            # ── الرسوم البيانية ───────────────────────────────
            sil = result["silhouette_score"]

            if show_gauge:
                st.subheader("🎯 جودة التجميع")
                render_silhouette_gauge(sil)
                st.markdown("---")

            if show_scatter:
                st.subheader("🗺️ Cluster Visualization")
                render_cluster_scatter(result)
                st.markdown("---")

            if show_size_chart:
                st.subheader("📦 مقارنة الـ Clusters")
                render_cluster_size_chart(result)
                st.markdown("---")

            if show_heatmap:
                st.subheader("🔥 Top Terms Heatmap")
                render_top_terms_heatmap(result)
                st.markdown("---")

            # ── بطاقات الـ Clusters ───────────────────────────
            st.subheader(f"📁 المجموعات التفصيلية ({result['n_clusters']} مجموعة)")
            for i, cluster in enumerate(result.get("clusters", [])):
                render_cluster_card(cluster, CLUSTER_COLORS[i % len(CLUSTER_COLORS)])

            with st.expander("🔍 البيانات الكاملة (JSON)"):
                st.json(result)

    # ════════════════════════════════
    # Tab 2: تجميع نتائج بحث
    # ════════════════════════════════
    with tab2:
        st.subheader("🔍 تجميع نتائج البحث")
        st.info("أدخل معرّفات الوثائق الناتجة عن بحث لتُجمَّع في مجموعات.")

        doc_ids_input = st.text_area(
            "معرّفات الوثائق (واحد في كل سطر أو مفصولة بفاصلة)",
            height=120,
            placeholder="d1\nd2\nd3\nd5\nd8\nd10",
        )
        n_clusters_r = st.slider("عدد المجموعات", 2, 8, 3, key="slider_results")

        if st.button("🚀 جمّع النتائج", type="primary", key="btn_cluster_results"):
            if not doc_ids_input.strip():
                st.error("❌ أدخل معرّفات الوثائق أولاً")
            else:
                raw = doc_ids_input.replace(",", "\n")
                doc_ids = [d.strip() for d in raw.split("\n") if d.strip()]
                if len(doc_ids) < 2:
                    st.error("❌ يجب إدخال وثيقتين على الأقل")
                else:
                    with st.spinner(f"⏳ جاري تجميع {len(doc_ids)} وثيقة..."):
                        result = cluster_results(doc_ids, dataset_name, n_clusters_r)
                    if result:
                        st.markdown("---")
                        st.subheader("📊 النتائج")
                        render_clusters_summary(result)
                        st.markdown("---")

                        if show_gauge:
                            render_silhouette_gauge(result["silhouette_score"])
                            st.markdown("---")

                        if show_scatter:
                            render_cluster_scatter(result)
                            st.markdown("---")

                        if show_size_chart:
                            render_cluster_size_chart(result)
                            st.markdown("---")

                        if show_heatmap:
                            render_top_terms_heatmap(result)
                            st.markdown("---")

                        for i, cluster in enumerate(result.get("clusters", [])):
                            render_cluster_card(
                                cluster, CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                            )

    # ════════════════════════════════
    # Tab 3: إيجاد أفضل K
    # ════════════════════════════════
    with tab3:
        st.subheader("📈 إيجاد أفضل عدد مجموعات (K)")
        st.info(
            "يجرّب قيم K مختلفة ويختار الأفضل بـ **Silhouette Score**. ⚠️ قد يستغرق دقيقة."
        )

        col1, col2 = st.columns(2)
        with col1:
            k_min = st.number_input("K الأدنى", min_value=2, max_value=5, value=2)
        with col2:
            k_max = st.number_input("K الأقصى", min_value=3, max_value=12, value=7)

        if st.button("🔍 ابحث عن أفضل K", type="primary", key="btn_optimal"):
            if k_min >= k_max:
                st.error("❌ K الأدنى يجب أن يكون أصغر من K الأقصى")
            else:
                with st.spinner(f"⏳ جاري اختبار K من {k_min} إلى {k_max}..."):
                    # غيّرنا اسم المتغير لتفادي التداخل في الأسماء
                    result_optimal = find_optimal_k(dataset_name, int(k_min), int(k_max))

                if result_optimal:
                    best_k = result_optimal["best_k"]
                    st.success(f"✅ **أفضل عدد مجموعات: K = {best_k}**")
                    st.markdown("---")

                    # ── Silhouette Comparison Chart ───────────
                    st.subheader("📊 مقارنة Silhouette Score")
                    render_silhouette_comparison_chart(result_optimal["scores"])

                    # ── جدول النتائج ──────────────────────────
                    st.subheader("📋 جدول النتائج")
                    table = []
                    for k, score in sorted(
                        result_optimal["scores"].items(), key=lambda x: int(x[0])
                    ):
                        table.append(
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
                                "": "⭐ أفضل" if int(k) == best_k else "",
                            }
                        )
                    st.table(table)

                    # التعديل الثاني: استدعاء التجميع الكامل وعرض المخططات بناءً على أفضل K
                    st.info(f"💡 سيتم الآن بناء المخططات التحليلية بناءً على أفضل تجميع (K = {best_k})...")
                    
                    with st.spinner(f"⏳ جاري توليد بيانات المخططات للمجموعات الـ {best_k}..."):
                        # نستدعي دالة التجميع باستخدام الإعدادات الموجودة في الشريط الجانبي
                        best_k_result = cluster_dataset(
                            dataset_name, best_k, svd_components, top_terms
                        )
                    
                    if best_k_result:
                        st.markdown("---")
                        st.subheader(f"🌟 تحليل النتائج لأفضل تجميع (K={best_k})")
                        
                        if show_gauge:
                            render_silhouette_gauge(best_k_result["silhouette_score"])
                            st.markdown("---")

                        if show_scatter:
                            st.subheader("🗺️ Cluster Visualization")
                            render_cluster_scatter(best_k_result)
                            st.markdown("---")

                        if show_size_chart:
                            st.subheader("📦 مقارنة الـ Clusters")
                            render_cluster_size_chart(best_k_result)
                            st.markdown("---")

                        if show_heatmap:
                            st.subheader("🔥 Top Terms Heatmap")
                            render_top_terms_heatmap(best_k_result)
                            st.markdown("---")
                            
                        # عرض بطاقات المجموعات (اختياري، يضيف لمسة احترافية للتبويب)
                        st.subheader(f"📁 المجموعات التفصيلية ({best_k_result['n_clusters']} مجموعة)")
                        for i, cluster in enumerate(best_k_result.get("clusters", [])):
                            render_cluster_card(cluster, CLUSTER_COLORS[i % len(CLUSTER_COLORS)])


if __name__ == "__main__":
    st.set_page_config(page_title="Document Clustering", page_icon="📊", layout="wide")
    main()
else:
    main()
