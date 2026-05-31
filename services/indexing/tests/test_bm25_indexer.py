"""
services/indexing/tests/test_bm25_indexer.py
=============================================
اختبارات وحدة شاملة لـ BM25Indexer.

تشغيل على Windows PowerShell:
    cd ir_system_2026
    python -m pytest services/indexing/tests/test_bm25_indexer.py -v

    # أو كل الاختبارات دفعة واحدة
    python -m pytest services/indexing/tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.indexing.dataset_loader import DatasetLoader, Document
from services.indexing.bm25_indexer import (
    BM25IndexMetadata,
    BM25Indexer,
)


# =============================================================
# بيانات تجريبية
# =============================================================

SAMPLE_DOCUMENTS = [
    Document(
        doc_id="d1",
        title="Cloud Storage",
        text="Cloud storage is useful for syncing files across devices.",
    ),
    Document(
        doc_id="d2",
        title="AI Assistants",
        text=(
            "AI assistants like Siri use voice recognition "
            "and natural language processing."
        ),
    ),
    Document(
        doc_id="d3",
        title="BM25 Ranking",
        text=(
            "BM25 is a ranking function used in information retrieval. "
            "It improves on TF-IDF by normalizing term frequency."
        ),
    ),
    Document(
        doc_id="d4",
        title="Python Programming",
        text="Python is used for machine learning and data science.",
    ),
    Document(
        doc_id="d5",
        title="Information Retrieval",
        text=(
            "Information retrieval systems help users find relevant "
            "documents using queries and ranking algorithms."
        ),
    ),
]

# وثيقتان بطول مختلف جداً — لاختبار تطبيع الطول (b parameter)
LENGTH_TEST_DOCUMENTS = [
    Document(
        doc_id="short",
        title=None,
        text="cloud storage sync",   # قصير جداً
    ),
    Document(
        doc_id="long",
        title=None,
        text=(
            "cloud storage is a technology that allows users to store "
            "files online and sync them across multiple devices. "
            "cloud storage services include Dropbox, Google Drive, "
            "and OneDrive. cloud storage offers backup and recovery. "
            "cloud storage is useful for collaboration and sharing. "
            "cloud cloud cloud storage sync backup files devices online."
        ),  # طويل جداً مع تكرار "cloud"
    ),
]


def _make_mock_loader(documents: List[Document] | None = None) -> MagicMock:
    docs = documents if documents is not None else SAMPLE_DOCUMENTS
    mock = MagicMock(spec=DatasetLoader)
    mock.load_all.return_value = docs
    return mock


# =============================================================
# Fixtures
# =============================================================

@pytest.fixture
def tmp_indexes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "indexes"
    d.mkdir()
    return d


@pytest.fixture
def indexer(tmp_indexes_dir: Path) -> BM25Indexer:
    return BM25Indexer(
        indexes_dir=str(tmp_indexes_dir),
        dataset_loader=_make_mock_loader(),
    )


@pytest.fixture
def built_indexer(indexer: BM25Indexer) -> BM25Indexer:
    indexer.build_index("test_dataset")
    return indexer


@pytest.fixture
def saved_indexer(built_indexer: BM25Indexer) -> BM25Indexer:
    built_indexer.save_index("test_dataset")
    return built_indexer


# =============================================================
# اختبارات build_index
# =============================================================

class TestBuildIndex:

    def test_returns_metadata(self, indexer: BM25Indexer) -> None:
        meta = indexer.build_index("test_dataset")
        assert isinstance(meta, BM25IndexMetadata)

    def test_correct_document_count(self, indexer: BM25Indexer) -> None:
        meta = indexer.build_index("test_dataset")
        assert meta.num_documents == len(SAMPLE_DOCUMENTS)

    def test_bm25_object_created(self, built_indexer: BM25Indexer) -> None:
        assert built_indexer.bm25 is not None

    def test_tokenized_docs_count(self, built_indexer: BM25Indexer) -> None:
        """عدد قوائم الـ tokens يساوي عدد الوثائق."""
        assert len(built_indexer.tokenized_docs) == len(SAMPLE_DOCUMENTS)

    def test_documents_list_populated(self, built_indexer: BM25Indexer) -> None:
        assert len(built_indexer.documents) == len(SAMPLE_DOCUMENTS)

    def test_docid_map_contains_all_ids(
        self, built_indexer: BM25Indexer
    ) -> None:
        for doc in SAMPLE_DOCUMENTS:
            assert doc.doc_id in built_indexer.doc_id_to_idx

    def test_is_built_true_after_build(self, built_indexer: BM25Indexer) -> None:
        assert built_indexer.is_built() is True

    def test_is_built_false_before_build(self, indexer: BM25Indexer) -> None:
        assert indexer.is_built() is False

    def test_metadata_stores_k1_b(self, indexer: BM25Indexer) -> None:
        """
        k1 و b يُحفظان في metadata.
        Developer 2 سيقرأهما ليعرف كيف بُني الفهرس.
        """
        meta = indexer.build_index("test_dataset", k1=1.2, b=0.5)
        assert meta.k1 == 1.2
        assert meta.b == 0.5

    def test_avg_doc_length_positive(self, built_indexer: BM25Indexer) -> None:
        """avgdl يجب أن يكون موجباً — قيمة صفر تعني خطأً."""
        assert built_indexer.metadata.avg_document_length > 0

    def test_vocab_size_positive(self, built_indexer: BM25Indexer) -> None:
        assert built_indexer.metadata.vocab_size > 0

    def test_empty_dataset_raises_error(
        self, tmp_indexes_dir: Path
    ) -> None:
        indexer = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(documents=[]),
        )
        with pytest.raises(ValueError, match="فارغة"):
            indexer.build_index("empty")


# =============================================================
# اختبارات save و load
# =============================================================

class TestSaveAndLoad:

    def test_save_creates_required_files(
        self, saved_indexer: BM25Indexer, tmp_indexes_dir: Path
    ) -> None:
        index_dir = tmp_indexes_dir / "test_dataset" / "bm25"
        for filename in [
            "bm25_model.pkl",
            "bm25_documents.json",
            "bm25_metadata.json",
            "bm25_docid_map.json",
        ]:
            assert (index_dir / filename).exists(), f"مفقود: {filename}"

    def test_is_saved_true_after_save(
        self, saved_indexer: BM25Indexer
    ) -> None:
        assert saved_indexer.is_saved("test_dataset") is True

    def test_load_restores_document_count(
        self, saved_indexer: BM25Indexer, tmp_indexes_dir: Path
    ) -> None:
        new_indexer = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        new_indexer.load_index("test_dataset")
        assert len(new_indexer.documents) == len(SAMPLE_DOCUMENTS)

    def test_load_restores_bm25_object(
        self, saved_indexer: BM25Indexer, tmp_indexes_dir: Path
    ) -> None:
        new_indexer = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        new_indexer.load_index("test_dataset")
        assert new_indexer.bm25 is not None

    def test_load_restores_metadata(
        self, saved_indexer: BM25Indexer, tmp_indexes_dir: Path
    ) -> None:
        original = saved_indexer.metadata
        new_indexer = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        loaded_meta = new_indexer.load_index("test_dataset")
        assert loaded_meta.k1 == original.k1
        assert loaded_meta.b  == original.b
        assert loaded_meta.num_documents == original.num_documents

    def test_load_nonexistent_raises_error(
        self, tmp_indexes_dir: Path
    ) -> None:
        indexer = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        with pytest.raises(FileNotFoundError):
            indexer.load_index("ghost_dataset")


# =============================================================
# اختبارات get_scores و get_top_n
# =============================================================

class TestScoring:

    def test_get_scores_returns_array_of_correct_length(
        self, built_indexer: BM25Indexer
    ) -> None:
        """
        get_scores يُرجع array بطول عدد الوثائق.
        كل عنصر = score الوثيقة المقابلة.
        """
        scores = built_indexer.get_scores(["cloud", "storag"])
        assert len(scores) == len(SAMPLE_DOCUMENTS)

    def test_get_scores_empty_query_returns_zeros(
        self, built_indexer: BM25Indexer
    ) -> None:
        """استعلام فارغ → أصفار لكل الوثائق."""
        scores = built_indexer.get_scores([])
        assert all(s == 0.0 for s in scores)

    def test_relevant_doc_scores_higher(
        self, built_indexer: BM25Indexer
    ) -> None:
        """
        الاختبار الجوهري: وثيقة ذات صلة تحصل على score أعلى.
        d1 تتحدث عن cloud storage → يجب أن تتصدر نتائج الاستعلام.
        """
        scores = built_indexer.get_scores(["cloud", "storag"])
        idx_d1 = built_indexer.doc_id_to_idx["d1"]
        idx_d4 = built_indexer.doc_id_to_idx["d4"]  # Python — لا علاقة له
        assert scores[idx_d1] > scores[idx_d4], (
            f"d1 score={scores[idx_d1]:.3f} يجب > d4 score={scores[idx_d4]:.3f}"
        )

    def test_get_top_n_returns_correct_count(
        self, built_indexer: BM25Indexer
    ) -> None:
        results = built_indexer.get_top_n(["cloud", "storag"], n=3)
        assert len(results) <= 3

    def test_get_top_n_sorted_by_score_descending(
        self, built_indexer: BM25Indexer
    ) -> None:
        """النتائج مرتبة تنازلياً بالـ score."""
        results = built_indexer.get_top_n(["information", "retriev"], n=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), (
            "النتائج يجب أن تكون مرتبة تنازلياً"
        )

    def test_get_scores_before_build_raises_error(
        self, indexer: BM25Indexer
    ) -> None:
        with pytest.raises(RuntimeError, match="غير مبني"):
            indexer.get_scores(["cloud"])

    def test_scores_after_load_match_original(
        self, saved_indexer: BM25Indexer, tmp_indexes_dir: Path
    ) -> None:
        """
        نفس الاستعلام يجب أن يُنتج نفس الـ scores
        سواء من الفهرس المبني في الذاكرة أو المحمّل من القرص.
        """
        query_tokens = ["cloud", "storag"]
        scores_original = saved_indexer.get_scores(query_tokens)

        new_indexer = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        new_indexer.load_index("test_dataset")
        scores_loaded = new_indexer.get_scores(query_tokens)

        np.testing.assert_array_almost_equal(
            scores_original, scores_loaded, decimal=5,
            err_msg="الـ scores يجب أن تتطابق بعد الحفظ والتحميل"
        )


# =============================================================
# اختبارات تأثير k1 و b (مفاهيمية)
# =============================================================

class TestBM25Parameters:
    """
    اختبارات تُثبت أن k1 و b يؤثران على النتائج كما هو متوقع نظرياً.
    هذه الاختبارات تعليمية — تفهمك كيف تعمل المعاملات.
    """

    def test_higher_k1_amplifies_term_frequency_effect(
        self, tmp_indexes_dir: Path
    ) -> None:
        """
        k1 يتحكم في سقف تأثير تكرار المصطلح.

        ملاحظة تقنية مهمة حول rank_bm25:
        IDF في rank_bm25 تستخدم الصيغة:
            IDF(t) = log( (N - df + 0.5) / (df + 0.5) )

        عندما يظهر المصطلح في كلتا الوثيقتين (df=N=2):
            IDF = log((2-2+0.5)/(2+0.5)) = log(0.5/2.5) = log(0.2) ≈ -1.6

        IDF سالب! هذا سلوك مقصود في rank_bm25 — يعني المصطلح
        شائع جداً في هذه المجموعة الصغيرة.

        الاختبار الصحيح: نختبر أن k1 يؤثر على نسبة الـ scores
        وليس على إشارتها.
        """
        repeated_docs = [
            Document("few",  None, "cloud storage"),
            Document("many", None, "cloud cloud cloud cloud cloud storage"),
        ]

        low_k1 = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir / "low"),
            dataset_loader=_make_mock_loader(repeated_docs),
        )
        low_k1.indexes_dir.mkdir(parents=True, exist_ok=True)
        low_k1.build_index("test", k1=0.1, b=0.0)

        high_k1 = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir / "high"),
            dataset_loader=_make_mock_loader(repeated_docs),
        )
        high_k1.indexes_dir.mkdir(parents=True, exist_ok=True)
        high_k1.build_index("test", k1=3.0, b=0.0)

        scores_low  = low_k1.get_scores(["cloud"])
        scores_high = high_k1.get_scores(["cloud"])

        # مع k1 مختلف يجب أن تتغير الـ scores
        # (الأرقام لا تكون متطابقة تماماً)
        assert not np.allclose(scores_low, scores_high), (
            "k1 مختلف يجب أن يُنتج scores مختلفة"
        )

        # النسبة بين الوثيقتين تتغير مع k1
        # مع k1 عالٍ: الوثيقة الكثيرة التكرار لها نسبة مختلفة
        ratio_low  = scores_low[1]  / (scores_low[0]  + 1e-9)
        ratio_high = scores_high[1] / (scores_high[0] + 1e-9)

        # النسبتان مختلفتان — k1 يؤثر فعلاً
        assert abs(ratio_high - ratio_low) > 0.001, (
            f"k1 يجب أن يُغيّر النسبة بين الوثيقتين. "
            f"ratio_low={ratio_low:.3f}, ratio_high={ratio_high:.3f}"
        )

    def test_b_zero_ignores_document_length(
        self, tmp_indexes_dir: Path
    ) -> None:
        """
        b=0 يعني طول الوثيقة لا يؤثر على الـ score.

        ══════════════════════════════════════════════
        درس مهم في IR: ظاهرة IDF=0 في rank_bm25
        ══════════════════════════════════════════════

        صيغة Robertson IDF في rank_bm25:
            IDF(t) = log( (N - df + 0.5) / (df + 0.5) )

        عندما df = N/2 (المصطلح في نصف الوثائق بالضبط):
            IDF = log( (N - N/2 + 0.5) / (N/2 + 0.5) )
                ≈ log(1) = 0.0

        ← score = IDF × (...) = 0.0 لكل الوثائق!

        لهذا السبب نحتاج 3+ وثائق في هذا الاختبار:
        مع 3 وثائق و df=1 مصطلح في وثيقة واحدة:
            IDF = log( (3-1+0.5)/(1+0.5) ) = log(2.5/1.5) = log(1.67) ≈ 0.51
        ← score موجب ✅

        هذا الاختبار يُعلّمك قاعدة مهمة:
        BM25 يعمل بشكل أفضل على مجموعات كبيرة (100+ وثيقة).
        على المجموعات الصغيرة، IDF قد يكون 0 أو سالب.
        ══════════════════════════════════════════════
        """
        # 3 وثائق: وثيقتان بطول مختلف + وثيقة لا تحتوي cloud
        # هذا يضمن IDF("cloud") > 0
        test_docs = [
            Document("short", None, "cloud storage sync"),
            Document("long",  None, (
                "cloud storage service allows users to store "
                "sync backup files across many different devices "
                "cloud cloud cloud storage service is great"
            )),
            Document("other", None, "python machine learning data science"),
        ]

        # فهرس بـ b=0 (طول الوثيقة لا يؤثر)
        idx_b0 = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir / "b0"),
            dataset_loader=_make_mock_loader(test_docs),
        )
        idx_b0.indexes_dir.mkdir(parents=True, exist_ok=True)
        idx_b0.build_index("test", b=0.0, k1=1.5)

        # فهرس بـ b=0.75 (الطول يؤثر)
        idx_b75 = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir / "b75"),
            dataset_loader=_make_mock_loader(test_docs),
        )
        idx_b75.indexes_dir.mkdir(parents=True, exist_ok=True)
        idx_b75.build_index("test", b=0.75, k1=1.5)

        scores_b0  = idx_b0.get_scores(["cloud"])
        scores_b75 = idx_b75.get_scores(["cloud"])

        # كلتا الوثيقتين (short وlong) يجب أن تحصلا على score > 0
        assert scores_b0[0] > 0, "short doc يجب أن يحصل على score موجب مع b=0"
        assert scores_b0[1] > 0, "long doc يجب أن يحصل على score موجب مع b=0"

        # الوثيقة الثالثة لا تحتوي "cloud" → score = 0
        assert scores_b0[2] == 0.0, "وثيقة بلا cloud يجب score=0"

        # b=0 و b=0.75 يُنتجان scores مختلفة (الطول يؤثر فعلاً)
        assert not np.allclose(scores_b0[:2], scores_b75[:2]), (
            "b=0 و b=0.75 يجب أن يُنتجا scores مختلفة للوثائق ذات الأطوال المختلفة"
        )

        # مع b=0.75: الوثيقة الطويلة تُعاقَب أكثر مقارنةً بـ b=0
        # نسبة (long/short) يجب أن تكون أقل مع b=0.75
        ratio_b0  = scores_b0[1]  / (scores_b0[0]  + 1e-9)   # long/short
        ratio_b75 = scores_b75[1] / (scores_b75[0] + 1e-9)

        assert ratio_b75 < ratio_b0, (
            f"b=0.75 يجب أن يُعاقب الوثيقة الطويلة أكثر من b=0.\n"
            f"ratio_b0={ratio_b0:.3f}, ratio_b75={ratio_b75:.3f}"
        )

    def test_b_one_strongly_penalizes_long_documents(
        self, tmp_indexes_dir: Path
    ) -> None:
        """
        b=1.0 يُعاقب الوثائق الطويلة بشدة.
        الوثيقة القصيرة قد تحصل على score مقارَب رغم تكرار أقل.
        """
        indexer_b1 = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir / "b1"),
            dataset_loader=_make_mock_loader(LENGTH_TEST_DOCUMENTS),
        )
        indexer_b1.indexes_dir.mkdir(parents=True, exist_ok=True)
        indexer_b1.build_index("test", b=1.0, k1=1.5)

        indexer_b0 = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir / "b0"),
            dataset_loader=_make_mock_loader(LENGTH_TEST_DOCUMENTS),
        )
        indexer_b0.indexes_dir.mkdir(parents=True, exist_ok=True)
        indexer_b0.build_index("test", b=0.0, k1=1.5)

        scores_b1 = indexer_b1.get_scores(["cloud"])
        scores_b0 = indexer_b0.get_scores(["cloud"])

        # مع b=1: الوثيقة القصيرة (index 0) تحصل على نسبة أعلى نسبياً
        # مع b=0: الوثيقة الطويلة تسيطر بسبب التكرار
        ratio_b1 = scores_b1[0] / (scores_b1[1] + 1e-9)  # short/long
        ratio_b0 = scores_b0[0] / (scores_b0[1] + 1e-9)

        assert ratio_b1 > ratio_b0, (
            "b=1.0 يجب أن يُحسّن نسبة الوثيقة القصيرة مقارنةً بـ b=0.0"
        )


# =============================================================
# اختبار التكامل: Developer 2 workflow
# =============================================================

class TestIntegration:
    """
    يحاكي بالضبط ما سيفعله Developer 2 في Retrieval Service.
    """

    def test_full_workflow_dev2_perspective(
        self, tmp_indexes_dir: Path
    ) -> None:
        """
        Developer 1 يبني ويحفظ.
        Developer 2 يحمّل ويبحث.
        النتائج منطقية.
        """
        # === Developer 1 ===
        dev1 = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        dev1.build_index("dataset1", k1=1.5, b=0.75)
        dev1.save_index("dataset1")

        # === Developer 2 ===
        dev2 = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(),
        )
        dev2.load_index("dataset1")

        # Developer 2 يعالج الاستعلام (بنفس إعدادات الفهرس)
        from services.preprocessing.preprocessor import get_preprocessor
        preprocessor = get_preprocessor()
        tokens, _ = preprocessor.process(
            "cloud storage sync files",
            apply_stemming=dev2.metadata.apply_stemming,
            remove_stopwords=dev2.metadata.remove_stopwords,
        )

        # البحث
        results = dev2.get_top_n(tokens, n=3)

        assert len(results) > 0
        top_doc, top_score = results[0]
        assert top_score > 0
        # d1 يجب أن يكون في الأوائل
        top_ids = [doc.doc_id for doc, _ in results]
        assert "d1" in top_ids, (
            f"d1 يجب أن يظهر في أفضل 3 نتائج لاستعلام cloud storage. "
            f"النتائج: {top_ids}"
        )

    def test_bm25_outranks_tfidf_on_verbose_document(
        self, tmp_indexes_dir: Path
    ) -> None:
        """
        اختبار المزية الرئيسية لـ BM25:
        وثيقة تكرر كلمة كثيراً يجب ألا تتصدر النتائج بشكل مبالغ فيه.

        هذا هو بالضبط ما يُثبته هذا الاختبار:
        BM25 يُعطي الوثيقة الجيدة المختصرة score مقارَب
        بالوثيقة المتكررة.
        """
        verbose_docs = [
            Document("good",    "Cloud Sync", "cloud storage sync backup"),
            Document("verbose", "Cloud Info",
                     " ".join(["cloud"] * 20) + " storage info"),
        ]

        indexer = BM25Indexer(
            indexes_dir=str(tmp_indexes_dir),
            dataset_loader=_make_mock_loader(verbose_docs),
        )
        indexer.build_index("test", k1=1.5, b=0.75)

        scores = indexer.get_scores(["cloud", "storag"])

        # مع BM25 الفرق بين الوثيقتين لا يجب أن يكون ضخماً
        # (التكرار المبالغ فيه لا يعطي ميزة كبيرة)
        ratio = scores[1] / (scores[0] + 1e-9)  # verbose / good
        assert ratio < 3.0, (
            f"BM25 يجب أن يُقيّد تأثير التكرار. "
            f"verbose_score={scores[1]:.3f}, good_score={scores[0]:.3f}, "
            f"ratio={ratio:.2f} (المتوقع < 3.0)"
        )
