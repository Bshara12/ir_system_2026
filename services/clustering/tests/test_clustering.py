"""
services/clustering/tests/test_clustering.py
=============================================
اختبارات وحدة شاملة لخدمة تجميع الوثائق.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
استراتيجية الاختبار
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
نستخدم Mock للـ TF-IDF index بدلاً من الفهرس الحقيقي.
هذا يجعل الاختبارات:
    - سريعة (أجزاء من الثانية)
    - مستقلة (لا تحتاج ملفات على القرص)
    - موثوقة (نتائج ثابتة دائماً)

ما نختبره:
    - DocumentClusterer: منطق التجميع
    - ClusterInfo / ClusteringResult: نماذج البيانات
    - FastAPI endpoints: الـ API

التشغيل:
    cd ir_system_2026
    python -m pytest services/clustering/tests/test_clustering.py -v
"""

import sys
import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.clustering.clusterer import (
    DocumentClusterer,
    ClusterInfo,
    ClusteringResult,
    get_clusterer,
)

# =============================================================
# بيانات تجريبية مشتركة
# =============================================================


def make_mock_tfidf_data(n_docs: int = 20, n_features: int = 50):
    """
    ينشئ TF-IDF matrix وهمية ووثائق للاختبار.
    يدعم أي n_docs بشكل ديناميكي مع 3 clusters واضحة.
    """
    rng = np.random.default_rng(seed=42)
    data = np.zeros((n_docs, n_features))

    # نقسّم الوثائق لـ 3 مجموعات ديناميكياً
    third = n_docs // 3
    rest = n_docs - 2 * third

    # حدود الكلمات لكل cluster (تتكيف مع n_features)
    f1 = n_features // 3
    f2 = 2 * (n_features // 3)

    # Cluster 1
    if third > 0:
        data[:third, :f1] = rng.random((third, f1)) * 0.8 + 0.2

    # Cluster 2
    if third > 0:
        data[third : 2 * third, f1:f2] = rng.random((third, f2 - f1)) * 0.8 + 0.2

    # Cluster 3
    if rest > 0:
        data[2 * third :, f2:] = rng.random((rest, n_features - f2)) * 0.8 + 0.2

    matrix = csr_matrix(data.astype(np.float32))

    class FakeDoc:
        def __init__(self, doc_id, title, text):
            self.doc_id = doc_id
            self.title = title
            self.original_text = text

    documents = [
        FakeDoc(f"d{i}", f"Title {i}", f"Document text number {i}")
        for i in range(n_docs)
    ]

    feature_names = [f"word_{i}" for i in range(n_features)]
    return matrix, documents, feature_names


def make_clusterer_with_mock(n_docs=20, n_features=50) -> DocumentClusterer:
    """ينشئ DocumentClusterer بـ mock للفهرس."""
    clusterer = DocumentClusterer()
    matrix, documents, feature_names = make_mock_tfidf_data(n_docs, n_features)

    # نستبدل _load_tfidf_index بدالة تُرجع بياناتنا الوهمية
    clusterer._load_tfidf_index = lambda dataset_name: (
        matrix,
        documents,
        feature_names,
    )
    return clusterer


# =============================================================
# اختبارات ClusterInfo
# =============================================================


class TestClusterInfo:

    def test_to_dict_contains_required_fields(self):
        """
        ماذا يختبر: to_dict() يُرجع كل الحقول المطلوبة.
        لماذا مهم: API يُرجع هذا الـ dict للواجهة.
        """
        cluster = ClusterInfo(
            cluster_id=0,
            size=10,
            top_terms=["cloud", "storage"],
            label="Cluster 1: cloud, storage",
            doc_ids=["d1", "d2"],
            centroid_score=0.75,
        )
        d = cluster.to_dict()
        assert d["cluster_id"] == 0
        assert d["size"] == 10
        assert d["top_terms"] == ["cloud", "storage"]
        assert d["label"] == "Cluster 1: cloud, storage"
        assert d["doc_ids"] == ["d1", "d2"]
        assert d["centroid_score"] == 0.75

    def test_cluster_id_is_integer(self):
        """cluster_id يجب أن يكون int."""
        cluster = ClusterInfo(0, 5, [], "C1", [], 0.0)
        assert isinstance(cluster.cluster_id, int)

    def test_doc_ids_is_list(self):
        """doc_ids يجب أن تكون list."""
        cluster = ClusterInfo(0, 3, [], "C1", ["d1", "d2", "d3"], 0.5)
        assert isinstance(cluster.doc_ids, list)
        assert len(cluster.doc_ids) == 3


# =============================================================
# اختبارات ClusteringResult
# =============================================================


class TestClusteringResult:

    def make_result(self, n_clusters=3) -> ClusteringResult:
        clusters = [
            ClusterInfo(i, 5, [f"term{i}"], f"Cluster {i+1}", [f"d{i}"], 0.7)
            for i in range(n_clusters)
        ]
        return ClusteringResult(
            dataset_name="test",
            n_clusters=n_clusters,
            n_documents=15,
            algorithm="kmeans+lsa",
            svd_components=50,
            silhouette_score=0.35,
            build_time_seconds=1.5,
            clusters=clusters,
            doc_cluster_map={f"d{i}": i % n_clusters for i in range(15)},
        )

    def test_to_dict_has_all_fields(self):
        """to_dict() يحتوي كل الحقول الأساسية."""
        result = self.make_result()
        d = result.to_dict()
        required = [
            "dataset_name",
            "n_clusters",
            "n_documents",
            "algorithm",
            "silhouette_score",
            "clusters",
            "doc_cluster_map",
            "build_time_seconds",
        ]
        for field in required:
            assert field in d, f"الحقل المفقود: {field}"

    def test_silhouette_score_rounded(self):
        """silhouette_score مُقرَّب لـ 4 أرقام عشرية."""
        result = self.make_result()
        d = result.to_dict()
        # نتأكد أن الرقم لا يحتوي أكثر من 4 أرقام عشرية
        score_str = str(d["silhouette_score"])
        if "." in score_str:
            decimal_part = score_str.split(".")[1]
            assert len(decimal_part) <= 4

    def test_clusters_count_matches_n_clusters(self):
        """عدد الـ clusters في القائمة = n_clusters."""
        result = self.make_result(n_clusters=4)
        assert len(result.clusters) == 4

    def test_doc_cluster_map_values_are_valid(self):
        """قيم doc_cluster_map يجب أن تكون cluster_ids صحيحة."""
        result = self.make_result(n_clusters=3)
        valid_ids = {c.cluster_id for c in result.clusters}
        for doc_id, cluster_id in result.doc_cluster_map.items():
            assert (
                cluster_id in valid_ids
            ), f"doc_id={doc_id} assigned to invalid cluster_id={cluster_id}"


# =============================================================
# اختبارات DocumentClusterer.cluster()
# =============================================================


class TestDocumentClustererCluster:

    def test_returns_clustering_result(self):
        """
        ماذا يختبر: cluster() يُرجع ClusteringResult.
        لماذا مهم: API يعتمد على هذا النوع.
        """
        clusterer = make_clusterer_with_mock()
        result = clusterer.cluster("test_dataset", n_clusters=3)
        assert isinstance(result, ClusteringResult)

    def test_correct_number_of_clusters(self):
        """
        ماذا يختبر: عدد الـ clusters = n_clusters المطلوب.
        لماذا مهم: المستخدم طلب k=3 يجب أن يحصل على 3 clusters.
        """
        clusterer = make_clusterer_with_mock()
        result = clusterer.cluster("test_dataset", n_clusters=3)
        assert result.n_clusters == 3

    def test_all_documents_assigned(self):
        """
        ماذا يختبر: كل وثيقة معيّنة لـ cluster.
        لماذا مهم: وثيقة بدون cluster = خطأ في العرض.
        """
        clusterer = make_clusterer_with_mock(n_docs=20)
        result = clusterer.cluster("test_dataset", n_clusters=3)
        # doc_cluster_map يجب أن يحتوي كل الوثائق
        assert len(result.doc_cluster_map) == 20

    def test_cluster_sizes_sum_to_total(self):
        """
        ماذا يختبر: مجموع أحجام الـ clusters = عدد الوثائق الكلي.
        لماذا مهم: لا يجوز فقدان وثائق.
        """
        clusterer = make_clusterer_with_mock(n_docs=20)
        result = clusterer.cluster("test_dataset", n_clusters=3)
        total_in_clusters = sum(c.size for c in result.clusters)
        assert total_in_clusters == result.n_documents

    def test_top_terms_not_empty(self):
        """
        ماذا يختبر: كل cluster له كلمات مميِّزة.
        لماذا مهم: بدون top_terms لا يمكن تسمية الـ cluster.
        """
        clusterer = make_clusterer_with_mock()
        result = clusterer.cluster("test_dataset", n_clusters=3)
        for cluster in result.clusters:
            assert (
                len(cluster.top_terms) > 0
            ), f"Cluster {cluster.cluster_id} بلا top_terms"

    def test_cluster_labels_contain_terms(self):
        """
        ماذا يختبر: تسمية الـ cluster تحتوي كلمات مفتاحية.
        لماذا مهم: التسمية تُعرض للمستخدم.
        """
        clusterer = make_clusterer_with_mock()
        result = clusterer.cluster("test_dataset", n_clusters=3)
        for cluster in result.clusters:
            assert len(cluster.label) > 0
            assert "Cluster" in cluster.label

    def test_silhouette_score_in_valid_range(self):
        """
        ماذا يختبر: Silhouette Score بين -1 و 1.
        لماذا مهم: قيمة خارج النطاق تعني خطأ في الحساب.
        """
        clusterer = make_clusterer_with_mock()
        result = clusterer.cluster("test_dataset", n_clusters=3)
        assert -1.0 <= result.silhouette_score <= 1.0

    def test_n_documents_matches_matrix(self):
        """عدد الوثائق في النتيجة = عدد صفوف المصفوفة."""
        clusterer = make_clusterer_with_mock(n_docs=20)
        result = clusterer.cluster("test_dataset", n_clusters=3)
        assert result.n_documents == 20

    def test_doc_ids_in_clusters_are_valid(self):
        """
        ماذا يختبر: doc_ids في كل cluster موجودة فعلاً.
        لماذا مهم: doc_id خاطئ يُفشل استرجاع الوثيقة لاحقاً.
        """
        clusterer = make_clusterer_with_mock(n_docs=20)
        result = clusterer.cluster("test_dataset", n_clusters=3)

        all_doc_ids = set(result.doc_cluster_map.keys())
        for cluster in result.clusters:
            for doc_id in cluster.doc_ids:
                assert (
                    doc_id in all_doc_ids
                ), f"doc_id={doc_id} في cluster لكن غير موجود في doc_cluster_map"

    def test_cluster_ids_are_unique(self):
        """كل cluster له cluster_id فريد."""
        clusterer = make_clusterer_with_mock()
        result = clusterer.cluster("test_dataset", n_clusters=4)
        ids = [c.cluster_id for c in result.clusters]
        assert len(ids) == len(set(ids))

    def test_clusters_sorted_by_size_descending(self):
        """
        ماذا يختبر: الـ clusters مرتبة تنازلياً حسب الحجم.
        لماذا مهم: الـ cluster الأكبر يظهر أولاً في الواجهة.
        """
        clusterer = make_clusterer_with_mock(n_docs=20)
        result = clusterer.cluster("test_dataset", n_clusters=3)
        sizes = [c.size for c in result.clusters]
        assert sizes == sorted(sizes, reverse=True)

    def test_large_n_clusters_auto_reduced(self):
        """
        ماذا يختبر: n_clusters > n_docs يُقلَّل تلقائياً.
        لماذا مهم: K-Means يفشل إذا k >= n_docs.
        """
        clusterer = make_clusterer_with_mock(n_docs=10)
        # n_clusters=50 أكبر بكثير من n_docs=10
        result = clusterer.cluster("test_dataset", n_clusters=50)
        # يجب أن ينجح بدون خطأ ويُقلّل k تلقائياً
        assert result.n_clusters < 50
        assert result.n_clusters >= 2

    def test_algorithm_field_is_correct(self):
        """الخوارزمية المُستخدمة محفوظة في النتيجة."""
        clusterer = make_clusterer_with_mock()
        result = clusterer.cluster("test_dataset", n_clusters=3)
        assert "kmeans" in result.algorithm.lower()

    def test_build_time_is_positive(self):
        """وقت البناء يجب أن يكون موجباً."""
        clusterer = make_clusterer_with_mock()
        result = clusterer.cluster("test_dataset", n_clusters=3)
        assert result.build_time_seconds >= 0


# =============================================================
# اختبارات cluster_search_results()
# =============================================================


class TestClusterSearchResults:

    def test_clusters_only_given_doc_ids(self):
        """
        ماذا يختبر: تجميع نتائج بحث محددة.
        لماذا مهم: نريد تجميع نتائج البحث فقط، لا كل الـ dataset.
        """
        clusterer = make_clusterer_with_mock(n_docs=20)
        # نُرسل فقط أول 10 وثائق
        doc_ids = [f"d{i}" for i in range(10)]
        result = clusterer.cluster_search_results(
            doc_ids=doc_ids,
            dataset_name="test_dataset",
            n_clusters=3,
        )
        # النتيجة يجب أن تحتوي فقط الوثائق المطلوبة
        assert result.n_documents == 10

    def test_doc_cluster_map_only_contains_requested_docs(self):
        """
        ماذا يختبر: doc_cluster_map لا يحتوي وثائق خارج القائمة المطلوبة.
        """
        clusterer = make_clusterer_with_mock(n_docs=20)
        doc_ids = ["d0", "d1", "d5", "d10", "d15"]
        result = clusterer.cluster_search_results(
            doc_ids=doc_ids,
            dataset_name="test_dataset",
            n_clusters=2,
        )
        # كل doc_id في النتيجة يجب أن يكون من الـ 5 المطلوبة
        for doc_id in result.doc_cluster_map.keys():
            assert doc_id in doc_ids

    def test_single_cluster_for_very_few_docs(self):
        """
        ماذا يختبر: عند وثيقة واحدة → cluster واحد بدون خطأ.
        لماذا مهم: K-Means يفشل مع doc واحدة.
        """
        clusterer = make_clusterer_with_mock(n_docs=20)
        result = clusterer.cluster_search_results(
            doc_ids=["d0"],  # وثيقة واحدة فقط
            dataset_name="test_dataset",
            n_clusters=3,
        )
        # يجب أن يتعامل معها بدون خطأ
        assert result.n_documents <= 1
        assert len(result.clusters) >= 1

    def test_returns_clustering_result_type(self):
        """النتيجة من النوع الصحيح."""
        clusterer = make_clusterer_with_mock(n_docs=20)
        doc_ids = [f"d{i}" for i in range(8)]
        result = clusterer.cluster_search_results(
            doc_ids=doc_ids,
            dataset_name="test_dataset",
            n_clusters=2,
        )
        assert isinstance(result, ClusteringResult)


# =============================================================
# اختبارات find_optimal_k()
# =============================================================


class TestFindOptimalK:

    def test_returns_best_k(self):
        """
        ماذا يختبر: find_optimal_k يُرجع best_k.
        لماذا مهم: الواجهة تعرض best_k للمستخدم.
        """
        clusterer = make_clusterer_with_mock(n_docs=30)
        result = clusterer.find_optimal_k("test_dataset", k_range=(2, 5))
        assert "best_k" in result
        assert isinstance(result["best_k"], int)

    def test_best_k_within_range(self):
        """best_k يجب أن يكون ضمن النطاق المطلوب."""
        clusterer = make_clusterer_with_mock(n_docs=30)
        result = clusterer.find_optimal_k("test_dataset", k_range=(2, 6))
        assert 2 <= result["best_k"] <= 6

    def test_scores_dict_has_all_k_values(self):
        """scores يحتوي نتيجة لكل قيمة k."""
        clusterer = make_clusterer_with_mock(n_docs=30)
        result = clusterer.find_optimal_k("test_dataset", k_range=(2, 4))
        for k in [2, 3, 4]:
            assert k in result["scores"], f"k={k} غير موجود في scores"

    def test_scores_in_valid_range(self):
        """كل silhouette score بين -1 و 1."""
        clusterer = make_clusterer_with_mock(n_docs=30)
        result = clusterer.find_optimal_k("test_dataset", k_range=(2, 4))
        for k, score in result["scores"].items():
            assert -1.0 <= score <= 1.0, f"score={score} لـ k={k} خارج النطاق المسموح"

    def test_best_k_has_highest_score(self):
        """best_k هو الـ k ذو أعلى silhouette score."""
        clusterer = make_clusterer_with_mock(n_docs=30)
        result = clusterer.find_optimal_k("test_dataset", k_range=(2, 5))
        best_score = result["scores"][result["best_k"]]
        for k, score in result["scores"].items():
            assert (
                score <= best_score + 1e-6
            ), f"k={k} (score={score}) أفضل من best_k={result['best_k']} (score={best_score})"


# =============================================================
# اختبارات منطقية إضافية
# =============================================================


class TestClustererLogic:

    def test_cosine_sim_identical_vectors(self):
        """Cosine similarity بين متجهين متطابقين = 1."""
        clusterer = DocumentClusterer()
        v = np.array([0.5, 0.3, 0.8, 0.1])
        sim = clusterer._cosine_sim(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_sim_orthogonal_vectors(self):
        """Cosine similarity بين متجهين متعامدين = 0."""
        clusterer = DocumentClusterer()
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = clusterer._cosine_sim(v1, v2)
        assert abs(sim) < 1e-5

    def test_cosine_sim_zero_vector(self):
        """متجه صفري → similarity = 0 بدون خطأ."""
        clusterer = DocumentClusterer()
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        sim = clusterer._cosine_sim(v1, v2)
        assert sim == 0.0

    def test_singleton_returns_same_instance(self):
        """get_clusterer() يُرجع نفس النسخة دائماً."""
        c1 = get_clusterer()
        c2 = get_clusterer()
        assert c1 is c2

    def test_cluster_with_clear_structure(self):
        """
        اختبار مهم: مع بيانات ذات clusters واضحة،
        K-Means يجب أن يُنتج silhouette score > 0.
        """
        clusterer = make_clusterer_with_mock(n_docs=30, n_features=60)
        result = clusterer.cluster("test_dataset", n_clusters=3)
        # مع clusters واضحة → silhouette > 0
        assert result.silhouette_score > 0.0, (
            f"Silhouette score={result.silhouette_score} يجب أن يكون > 0 "
            f"مع بيانات ذات clusters واضحة"
        )


# =============================================================
# اختبارات FastAPI Endpoints
# =============================================================


class TestClusteringAPI:
    """اختبارات الـ HTTP endpoints."""

    @pytest.fixture
    def client(self):
        """TestClient للاختبار."""
        # نُضيف PROJECT_ROOT للـ path حتى تعمل imports من shared/
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))

        # نُنشئ shared modules مؤقتة إذا لم تكن موجودة
        import types

        if "shared" not in sys.modules:
            shared = types.ModuleType("shared")
            sys.modules["shared"] = shared

        if "shared.models" not in sys.modules:
            models = types.ModuleType("shared.models")

            from pydantic import BaseModel
            from typing import Optional, Dict, Any

            class ServiceStatus(BaseModel):
                service_name: str
                status: str
                version: str = "1.0.0"
                details: Optional[Dict[str, Any]] = None

            class ErrorResponse(BaseModel):
                error: str
                detail: Optional[str] = None

            models.ServiceStatus = ServiceStatus
            models.ErrorResponse = ErrorResponse
            sys.modules["shared.models"] = models

        if "shared.constants" not in sys.modules:
            constants = types.ModuleType("shared.constants")
            constants.DEFAULT_TOP_K = 10
            sys.modules["shared.constants"] = constants

        from fastapi.testclient import TestClient
        from services.clustering.app import app

        return TestClient(app)

    @pytest.fixture
    def mock_clusterer(self):
        """يُنشئ mock للـ clusterer لتجنب الحاجة لفهرس حقيقي."""
        mock = MagicMock()
        fake_cluster = ClusterInfo(
            cluster_id=0,
            size=5,
            top_terms=["cloud", "storage", "sync"],
            label="Cluster 1: cloud, storage, sync",
            doc_ids=["d1", "d2", "d3"],
            centroid_score=0.75,
        )
        fake_result = ClusteringResult(
            dataset_name="dataset1",
            n_clusters=2,
            n_documents=10,
            algorithm="kmeans+lsa",
            svd_components=50,
            silhouette_score=0.35,
            build_time_seconds=0.5,
            clusters=[fake_cluster],
            doc_cluster_map={"d1": 0, "d2": 0, "d3": 0},
        )
        mock.cluster.return_value = fake_result
        mock.cluster_search_results.return_value = fake_result
        mock.find_optimal_k.return_value = {
            "best_k": 3,
            "scores": {2: 0.2, 3: 0.35, 4: 0.28},
            "dataset_name": "dataset1",
            "n_documents": 10,
        }
        return mock

    def test_health_returns_200(self, client):
        """GET /health يُرجع 200."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
        assert r.json()["service_name"] == "clustering"

    def test_health_shows_algorithm(self, client):
        """GET /health يُظهر الخوارزمية المُستخدمة."""
        r = client.get("/health")
        assert "algorithm" in r.json()["details"]

    def test_cluster_dataset_with_mock(self, client, mock_clusterer):
        """POST /cluster/dataset يُرجع 200 مع نتيجة صحيحة."""
        with patch(
            "services.clustering.app.get_clusterer", return_value=mock_clusterer
        ):
            r = client.post(
                "/cluster/dataset",
                json={
                    "dataset_name": "dataset1",
                    "n_clusters": 3,
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert "clusters" in data
        assert "silhouette_score" in data
        assert "n_clusters" in data

    def test_cluster_dataset_invalid_n_clusters(self, client):
        """n_clusters < 2 → 422 validation error."""
        r = client.post(
            "/cluster/dataset",
            json={
                "dataset_name": "dataset1",
                "n_clusters": 1,  # أقل من الحد الأدنى 2
            },
        )
        assert r.status_code == 422

    def test_cluster_dataset_n_clusters_too_large(self, client):
        """n_clusters > 20 → 422 validation error."""
        r = client.post(
            "/cluster/dataset",
            json={
                "dataset_name": "dataset1",
                "n_clusters": 25,  # أكبر من الحد الأقصى 20
            },
        )
        assert r.status_code == 422

    def test_cluster_results_with_mock(self, client, mock_clusterer):
        """POST /cluster/results يُرجع 200."""
        with patch(
            "services.clustering.app.get_clusterer", return_value=mock_clusterer
        ):
            r = client.post(
                "/cluster/results",
                json={
                    "doc_ids": ["d1", "d2", "d3", "d4", "d5"],
                    "dataset_name": "dataset1",
                    "n_clusters": 2,
                },
            )
        assert r.status_code == 200

    def test_cluster_results_empty_doc_ids_returns_422(self, client):
        """قائمة doc_ids فارغة → 422."""
        r = client.post(
            "/cluster/results",
            json={
                "doc_ids": [],
                "dataset_name": "dataset1",
                "n_clusters": 2,
            },
        )
        assert r.status_code == 422

    def test_optimal_k_with_mock(self, client, mock_clusterer):
        """POST /cluster/optimal-k يُرجع 200 مع best_k."""
        with patch(
            "services.clustering.app.get_clusterer", return_value=mock_clusterer
        ):
            r = client.post(
                "/cluster/optimal-k",
                json={
                    "dataset_name": "dataset1",
                    "k_min": 2,
                    "k_max": 6,
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert "best_k" in data
        assert "scores" in data

    def test_optimal_k_invalid_range(self, client, mock_clusterer):
        """k_min >= k_max → 400."""
        with patch(
            "services.clustering.app.get_clusterer", return_value=mock_clusterer
        ):
            r = client.post(
                "/cluster/optimal-k",
                json={
                    "dataset_name": "dataset1",
                    "k_min": 5,
                    "k_max": 3,  # k_min > k_max
                },
            )
        assert r.status_code == 400

    def test_cluster_dataset_unavailable_index_returns_503(self, client):
        """
        فهرس TF-IDF غير متاح → 503.
        ماذا يختبر: رسالة خطأ واضحة بدل crash.
        """
        mock = MagicMock()
        mock.cluster.side_effect = RuntimeError("فهرس TF-IDF غير مبني")
        with patch("services.clustering.app.get_clusterer", return_value=mock):
            r = client.post(
                "/cluster/dataset",
                json={
                    "dataset_name": "nonexistent_dataset",
                    "n_clusters": 3,
                },
            )
        assert r.status_code == 503
