"""
services/clustering/clusterer.py
==================================
منطق تجميع الوثائق (Document Clustering).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما هو Document Clustering؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
بدل عرض نتائج البحث كقائمة عشوائية،
نُجمّع الوثائق المتشابهة في مجموعات (clusters).

مثال:
    بحثت عن "machine learning"
    بدون clustering: 10 نتائج مختلطة
    مع clustering:
        📁 Cluster 1 - Deep Learning (3 وثائق)
        📁 Cluster 2 - Classical ML (4 وثائق)
        📁 Cluster 3 - Applications (3 وثائق)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا هو سهل التطبيق في مشروعنا؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TF-IDF Matrix موجودة من المطور الأول ← نطبّق K-Means مباشرة
لا نحتاج أي بيانات جديدة أو مكتبات خاصة.

الخوارزمية المستخدمة: K-Means
    - سريعة وبسيطة
    - sklearn موجودة في المشروع أصلاً
    - تعمل مباشرة على TF-IDF matrix
    - نتائجها قابلة للتفسير

تحسين: TruncatedSVD (LSA)
    - يُقلّل أبعاد المصفوفة قبل التجميع
    - يُحسّن جودة الـ clusters
    - يُسرّع الحساب على datasets كبيرة
"""

import sys
import os
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


# =============================================================
# نماذج البيانات
# =============================================================


@dataclass
class ClusterInfo:
    """
    معلومات مجموعة واحدة (Cluster).
    """

    cluster_id: int
    size: int  # عدد الوثائق في المجموعة
    top_terms: List[str]  # أهم الكلمات المميِّزة للمجموعة
    label: str  # تسمية تلقائية مثل "Cluster 1: cloud storage"
    doc_ids: List[str]  # معرّفات الوثائق في المجموعة
    centroid_score: float  # متوسط قرب الوثائق من المركز

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ClusteringResult:
    """
    نتيجة عملية التجميع الكاملة.
    """

    dataset_name: str
    n_clusters: int
    n_documents: int
    algorithm: str  # "kmeans"
    svd_components: int  # أبعاد LSA المُستخدمة
    silhouette_score: float  # جودة التجميع (-1 إلى 1، كلما زاد كان أفضل)
    build_time_seconds: float
    clusters: List[ClusterInfo]
    doc_cluster_map: Dict[str, int]  # doc_id → cluster_id

    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "n_clusters": self.n_clusters,
            "n_documents": self.n_documents,
            "algorithm": self.algorithm,
            "svd_components": self.svd_components,
            "silhouette_score": round(self.silhouette_score, 4),
            "build_time_seconds": self.build_time_seconds,
            "clusters": [c.to_dict() for c in self.clusters],
            "doc_cluster_map": self.doc_cluster_map,
        }


# =============================================================
# DocumentClusterer — الكلاس الرئيسي
# =============================================================


class DocumentClusterer:
    """
    يُجمّع وثائق مجموعة بيانات باستخدام K-Means على TF-IDF.

    كيف يعمل؟
    ━━━━━━━━━━
    1. يحمّل TF-IDF matrix من فهرس المطور الأول
    2. يُقلّل الأبعاد بـ TruncatedSVD (LSA) → أسرع وأدق
    3. يُطبّق K-Means للتجميع
    4. يستخرج أهم الكلمات لكل cluster
    5. يُرجع نتيجة منظمة لعرضها في الواجهة

    الاستخدام:
        clusterer = DocumentClusterer()
        result = clusterer.cluster("dataset1", n_clusters=5)
        for cluster in result.clusters:
            print(f"Cluster {cluster.cluster_id}: {cluster.label}")
            print(f"  وثائق: {cluster.size}")
            print(f"  كلمات مفتاحية: {cluster.top_terms}")
    """

    def __init__(self) -> None:
        self._last_result: Optional[ClusteringResult] = None

    def cluster(
        self,
        dataset_name: str,
        n_clusters: int = 5,
        svd_components: int = 100,
        max_iter: int = 300,
        random_state: int = 42,
        top_terms_per_cluster: int = 8,
    ) -> ClusteringResult:
        """
        يُجمّع وثائق مجموعة البيانات.

        المعاملات:
            dataset_name:         اسم مجموعة البيانات
            n_clusters:           عدد المجموعات (2-20)
            svd_components:       أبعاد LSA (50-200، كلما زاد كان أدق لكن أبطأ)
            max_iter:             أقصى عدد تكرارات K-Means
            random_state:         للنتائج القابلة للتكرار
            top_terms_per_cluster: عدد الكلمات المميِّزة لكل cluster

        الإرجاع:
            ClusteringResult يحتوي كل المعلومات

        الخطوات:
            1. تحميل TF-IDF matrix من فهرس المطور الأول
            2. TruncatedSVD لتقليل الأبعاد
            3. L2 Normalization (لتحسين K-Means)
            4. K-Means clustering
            5. استخراج top terms لكل cluster
            6. حساب Silhouette Score
        """
        logger.info(f"[Clusterer] بدء التجميع: dataset={dataset_name}, k={n_clusters}")
        start_time = time.time()

        # ── الخطوة 1: تحميل الفهرس من المطور الأول ──────────
        tfidf_matrix, documents, feature_names = self._load_tfidf_index(dataset_name)

        n_docs = tfidf_matrix.shape[0]
        logger.info(f"[Clusterer] المصفوفة: {tfidf_matrix.shape}")

        # التحقق من صحة n_clusters
        if n_clusters >= n_docs:
            n_clusters = max(2, n_docs // 2)
            logger.warning(f"[Clusterer] n_clusters قُلِّل إلى {n_clusters}")

        # ── الخطوة 2: TruncatedSVD (LSA) ─────────────────────
        # نُقلّل الأبعاد من vocab_size (50,000+) إلى svd_components (100)
        # هذا يُحسّن جودة التجميع ويُسرّعه بشكل كبير
        actual_components = min(svd_components, tfidf_matrix.shape[1] - 1, n_docs - 1)
        logger.info(
            f"[Clusterer] SVD: {tfidf_matrix.shape[1]} → {actual_components} بُعد"
        )

        svd = TruncatedSVD(n_components=actual_components, random_state=random_state)
        reduced_matrix = svd.fit_transform(tfidf_matrix)

        # ── الخطوة 3: L2 Normalization ────────────────────────
        # K-Means يستخدم Euclidean distance
        # بعد Normalization: Euclidean ≈ Cosine → نتائج أفضل
        reduced_normalized = normalize(reduced_matrix, norm="l2")

        # ── الخطوة 4: K-Means ─────────────────────────────────
        # MiniBatchKMeans أسرع من KMeans على datasets كبيرة
        # كلاهما يُنتج نفس الجودة تقريباً
        if n_docs > 10_000:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter,
                n_init=3,
            )
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter,
                n_init=10,
            )

        labels = kmeans.fit_predict(reduced_normalized)
        logger.info(f"[Clusterer] K-Means اكتمل")

        # ── الخطوة 5: Silhouette Score ────────────────────────
        # يقيس جودة التجميع: 1 = مثالي، 0 = عشوائي، -1 = سيء
        # نحسبه على عينة إذا كانت البيانات كبيرة (أداء)
        try:
            sample_size = min(n_docs, 5000)
            if sample_size < n_docs:
                idx = np.random.choice(n_docs, sample_size, replace=False)
                sil_score = silhouette_score(
                    reduced_normalized[idx], labels[idx], metric="euclidean"
                )
            else:
                sil_score = silhouette_score(
                    reduced_normalized, labels, metric="euclidean"
                )
        except Exception:
            sil_score = 0.0

        # ── الخطوة 6: بناء معلومات كل Cluster ───────────────
        clusters = self._build_cluster_info(
            labels=labels,
            documents=documents,
            tfidf_matrix=tfidf_matrix,
            feature_names=feature_names,
            n_clusters=n_clusters,
            top_terms_per_cluster=top_terms_per_cluster,
        )

        # خريطة doc_id → cluster_id
        doc_cluster_map = {
            doc.doc_id: int(labels[i]) for i, doc in enumerate(documents)
        }

        build_time = round(time.time() - start_time, 2)
        logger.info(
            f"[Clusterer] ✅ اكتمل في {build_time}s | " f"silhouette={sil_score:.3f}"
        )

        result = ClusteringResult(
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            n_documents=n_docs,
            algorithm="kmeans+lsa",
            svd_components=actual_components,
            silhouette_score=float(sil_score),
            build_time_seconds=build_time,
            clusters=clusters,
            doc_cluster_map=doc_cluster_map,
        )

        self._last_result = result
        return result

    def cluster_search_results(
        self,
        doc_ids: List[str],
        dataset_name: str,
        n_clusters: int = 3,
    ) -> ClusteringResult:
        """
        يُجمّع مجموعة محددة من نتائج البحث فقط (وليس كل الـ dataset).

        يُستخدم لتجميع نتائج بحث المستخدم مباشرة.
        مثال: المستخدم بحث وحصل على 20 نتيجة → نُجمّعها في 3 clusters.

        المعاملات:
            doc_ids:      معرّفات الوثائق المراد تجميعها
            dataset_name: المصدر لاستخراج التمثيلات
            n_clusters:   عدد المجموعات
        """
        logger.info(f"[Clusterer] تجميع {len(doc_ids)} نتيجة بحث")

        tfidf_matrix_full, documents_full, feature_names = self._load_tfidf_index(
            dataset_name
        )

        # بناء خريطة doc_id → index
        doc_id_to_idx = {doc.doc_id: i for i, doc in enumerate(documents_full)}

        # تصفية الوثائق المطلوبة فقط
        valid_indices = []
        valid_doc_ids = []
        for doc_id in doc_ids:
            if doc_id in doc_id_to_idx:
                valid_indices.append(doc_id_to_idx[doc_id])
                valid_doc_ids.append(doc_id)

        if len(valid_indices) < 2:
            logger.warning("[Clusterer] عدد الوثائق قليل جداً للتجميع")
            return self._make_single_cluster(valid_doc_ids, dataset_name)

        # استخراج صفوف المصفوفة للوثائق المحددة
        subset_matrix = tfidf_matrix_full[valid_indices]
        subset_documents = [documents_full[i] for i in valid_indices]

        # تعديل n_clusters إذا لزم
        n_clusters = min(n_clusters, len(valid_indices) - 1)
        n_clusters = max(2, n_clusters)

        # LSA على المجموعة الفرعية
        actual_components = min(50, subset_matrix.shape[1] - 1, len(valid_indices) - 1)
        svd = TruncatedSVD(n_components=actual_components, random_state=42)
        reduced = normalize(svd.fit_transform(subset_matrix))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced)

        clusters = self._build_cluster_info(
            labels=labels,
            documents=subset_documents,
            tfidf_matrix=subset_matrix,
            feature_names=feature_names,
            n_clusters=n_clusters,
        )

        try:
            sil_score = float(silhouette_score(reduced, labels))
        except Exception:
            sil_score = 0.0

        doc_cluster_map = {
            subset_documents[i].doc_id: int(labels[i])
            for i in range(len(subset_documents))
        }

        return ClusteringResult(
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            n_documents=len(valid_indices),
            algorithm="kmeans+lsa",
            svd_components=actual_components,
            silhouette_score=sil_score,
            build_time_seconds=0.0,
            clusters=clusters,
            doc_cluster_map=doc_cluster_map,
        )

    def find_optimal_k(
        self,
        dataset_name: str,
        k_range: Tuple[int, int] = (2, 10),
    ) -> Dict[str, Any]:
        """
        يجد عدد المجموعات الأمثل تلقائياً باستخدام Elbow Method.

        الفكرة: نجرّب قيم k مختلفة ونختار تلك التي تُعطي
        أفضل Silhouette Score.

        مثال:
            result = clusterer.find_optimal_k("dataset1", k_range=(2, 8))
            print(f"أفضل k: {result['best_k']}")
        """
        tfidf_matrix, _, _ = self._load_tfidf_index(dataset_name)
        n_docs = tfidf_matrix.shape[0]

        # تقليل الأبعاد مرة واحدة
        components = min(100, tfidf_matrix.shape[1] - 1, n_docs - 1)
        svd = TruncatedSVD(n_components=components, random_state=42)
        reduced = normalize(svd.fit_transform(tfidf_matrix))

        k_min, k_max = k_range
        k_max = min(k_max, n_docs - 1)

        scores = {}
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            labels = km.fit_predict(reduced)
            try:
                score = float(
                    silhouette_score(reduced, labels, sample_size=min(2000, n_docs))
                )
            except Exception:
                score = 0.0
            scores[k] = round(score, 4)
            logger.info(f"[Clusterer] k={k}: silhouette={score:.4f}")

        best_k = max(scores, key=scores.get)

        return {
            "best_k": best_k,
            "scores": scores,
            "dataset_name": dataset_name,
            "n_documents": n_docs,
        }

    # ----------------------------------------------------------
    # دوال مساعدة خاصة
    # ----------------------------------------------------------

    def _load_tfidf_index(self, dataset_name: str):
        """
        يحمّل TF-IDF index من فهرس المطور الأول.

        يُرجع:
            tfidf_matrix: sparse matrix (n_docs × vocab)
            documents:    List[IndexedDocument]
            feature_names: List[str] أسماء الكلمات
        """
        try:
            from services.indexing.tfidf_indexer import get_tfidf_indexer

            indexer = get_tfidf_indexer(dataset_name)

            if not indexer.is_built():
                raise RuntimeError(
                    f"فهرس TF-IDF لـ '{dataset_name}' غير مبني.\n"
                    f"شغّل Indexing Service أولاً."
                )

            feature_names = indexer.vectorizer.get_feature_names_out().tolist()
            return indexer.tfidf_matrix, indexer.documents, feature_names

        except ImportError:
            raise RuntimeError("لا يمكن استيراد TFIDFIndexer من المطور الأول.")

    def _build_cluster_info(
        self,
        labels: np.ndarray,
        documents,
        tfidf_matrix,
        feature_names: List[str],
        n_clusters: int,
        top_terms_per_cluster: int = 8,
    ) -> List[ClusterInfo]:
        """
        يبني معلومات تفصيلية لكل cluster.

        لاستخراج أهم الكلمات:
        نحسب متوسط TF-IDF لكل كلمة داخل الـ cluster
        الكلمات ذات أعلى متوسط هي الأكثر تميُّزاً للـ cluster.
        """
        clusters = []

        for cluster_id in range(n_clusters):
            # مؤشرات الوثائق في هذا الـ cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # وثائق الـ cluster
            cluster_doc_ids = [documents[i].doc_id for i in cluster_indices]

            # استخراج أهم الكلمات
            # نأخذ صفوف المصفوفة للوثائق في هذا الـ cluster
            cluster_matrix = tfidf_matrix[cluster_indices]

            # متوسط TF-IDF لكل كلمة في الـ cluster
            # .toarray() لتحويل sparse → dense للحساب
            mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).flatten()

            # أعلى N كلمة
            top_indices = np.argsort(mean_tfidf)[::-1][:top_terms_per_cluster]
            top_terms = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]

            # تسمية تلقائية من أول 3 كلمات
            label_words = top_terms[:3]
            label = f"Cluster {cluster_id + 1}"
            if label_words:
                label = f"Cluster {cluster_id + 1}: {', '.join(label_words)}"

            # متوسط المسافة من المركز (قرب الوثائق من بعضها)
            if cluster_matrix.shape[0] > 0:
                centroid = np.asarray(cluster_matrix.mean(axis=0))
                centroid_score = float(
                    np.mean(
                        [
                            self._cosine_sim(
                                np.asarray(cluster_matrix[i].todense()).flatten(),
                                centroid.flatten(),
                            )
                            for i in range(min(cluster_matrix.shape[0], 100))
                        ]
                    )
                )
            else:
                centroid_score = 0.0

            clusters.append(
                ClusterInfo(
                    cluster_id=cluster_id,
                    size=len(cluster_indices),
                    top_terms=top_terms,
                    label=label,
                    doc_ids=cluster_doc_ids,
                    centroid_score=round(centroid_score, 4),
                )
            )

        # ترتيب الـ clusters تنازلياً حسب الحجم
        clusters.sort(key=lambda c: c.size, reverse=True)
        return clusters

    def _make_single_cluster(
        self, doc_ids: List[str], dataset_name: str
    ) -> ClusteringResult:
        """يُنشئ نتيجة تجميع من cluster واحد (عند عدد وثائق قليل جداً)."""
        cluster = ClusterInfo(
            cluster_id=0,
            size=len(doc_ids),
            top_terms=[],
            label="Cluster 1: All Documents",
            doc_ids=doc_ids,
            centroid_score=1.0,
        )
        return ClusteringResult(
            dataset_name=dataset_name,
            n_clusters=1,
            n_documents=len(doc_ids),
            algorithm="single_cluster",
            svd_components=0,
            silhouette_score=0.0,
            build_time_seconds=0.0,
            clusters=[cluster],
            doc_cluster_map={doc_id: 0 for doc_id in doc_ids},
        )

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """يحسب Cosine Similarity بين متجهين."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================
# Singleton
# =============================================================

_clusterer_instance: Optional[DocumentClusterer] = None


def get_clusterer() -> DocumentClusterer:
    """يُرجع النسخة الوحيدة من DocumentClusterer."""
    global _clusterer_instance
    if _clusterer_instance is None:
        _clusterer_instance = DocumentClusterer()
    return _clusterer_instance
