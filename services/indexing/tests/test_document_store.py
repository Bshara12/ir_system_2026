"""
services/indexing/tests/test_document_store.py
===============================================
اختبارات DocumentStore — لا تحتاج أي مكتبات خارجية.

SQLite مبنية في Python — كل الاختبارات تعمل بدون تثبيت إضافي.

تشغيل:
    cd ir_system_2026
    python -m pytest services/indexing/tests/test_document_store.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.indexing.document_store import DocumentStore, get_document_store


# =============================================================
# بيانات تجريبية — تُحاكي IndexedDocument الحقيقي
# =============================================================

@dataclass
class FakeIndexedDocument:
    """
    نسخة مبسّطة من IndexedDocument لأغراض الاختبار.
    نفس الـ attributes التي يقرأها add_batch().
    """
    doc_id:         str
    original_text:  str
    processed_text: str
    title:          Optional[str] = None


SAMPLE_DOCS = [
    FakeIndexedDocument(
        doc_id="d1",
        original_text="Cloud storage is useful for syncing files across devices.",
        processed_text="cloud storag use sync file devic",
        title="Cloud Storage",
    ),
    FakeIndexedDocument(
        doc_id="d2",
        original_text="AI assistants like Siri use voice recognition.",
        processed_text="ai assist siri use voic recognit",
        title="AI Assistants",
    ),
    FakeIndexedDocument(
        doc_id="d3",
        original_text="BM25 is a ranking function used in information retrieval.",
        processed_text="bm25 rank function use inform retriev",
        title="BM25 Ranking",
    ),
    FakeIndexedDocument(
        doc_id="d4",
        original_text="Python is used for machine learning and data science.",
        processed_text="python use machin learn data scienc",
        title=None,  # بدون عنوان — لنتحقق أن None يعمل
    ),
    FakeIndexedDocument(
        doc_id="d5",
        original_text="Information retrieval systems help users find relevant documents.",
        processed_text="inform retriev system help user find relev document",
        title="Information Retrieval",
    ),
]


# =============================================================
# Fixture
# =============================================================

@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    """DocumentStore فارغ في مجلد مؤقت."""
    return DocumentStore(
        indexes_dir=str(tmp_path / "indexes"),
        dataset_name="test-dataset",
    )


@pytest.fixture
def populated_store(tmp_path: Path) -> DocumentStore:
    """DocumentStore مملوء بـ SAMPLE_DOCS."""
    s = DocumentStore(
        indexes_dir=str(tmp_path / "indexes"),
        dataset_name="test-dataset",
    )
    s.add_batch(SAMPLE_DOCS)
    return s


# =============================================================
# ① اختبارات الإنشاء
# =============================================================

class TestInit:

    def test_creates_db_file(self, tmp_path: Path) -> None:
        """
        ماذا يختبر: __init__ ينشئ ملف DB على القرص.
        لماذا مهم: إذا لم يُنشأ الملف = كل شيء يفشل.
        """
        indexes_dir = str(tmp_path / "indexes")
        DocumentStore(indexes_dir=indexes_dir, dataset_name="test")
        db_path = tmp_path / "indexes" / "test" / "documents.db"
        assert db_path.exists(), "ملف documents.db يجب أن يُنشأ"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """
        ماذا يختبر: المجلدات تُنشأ تلقائياً حتى لو غير موجودة.
        لماذا مهم: في بيئة جديدة لا يوجد data/indexes مسبقاً.
        """
        deep_path = str(tmp_path / "a" / "b" / "c" / "indexes")
        DocumentStore(indexes_dir=deep_path, dataset_name="test")
        assert (Path(deep_path) / "test" / "documents.db").exists()

    def test_init_twice_is_safe(self, tmp_path: Path) -> None:
        """
        ماذا يختبر: استدعاء __init__ مرتين لا يحذف البيانات.
        لماذا مهم: إذا أعدنا تشغيل الخادم = البيانات تبقى.
        """
        indexes_dir = str(tmp_path / "indexes")
        s = DocumentStore(indexes_dir=indexes_dir, dataset_name="test")
        s.add("doc1", "hello world")

        # إنشاء ثانٍ — يجب ألا يمسح البيانات
        s2 = DocumentStore(indexes_dir=indexes_dir, dataset_name="test")
        assert s2.count() == 1, "IF NOT EXISTS يجب أن يحفظ البيانات"


# =============================================================
# ② اختبارات الإضافة
# =============================================================

class TestAdd:

    def test_add_single_document(self, store: DocumentStore) -> None:
        """
        ماذا يختبر: إضافة وثيقة واحدة بـ add().
        """
        store.add("doc1", "hello world", title="Test Doc")
        assert store.count() == 1

    def test_add_batch_returns_correct_count(
        self, store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: add_batch يُرجع عدد الوثائق المُضافة فعلاً.
        لماذا مهم: للتحقق أن كل الوثائق وُلِدت بالكامل.
        """
        count = store.add_batch(SAMPLE_DOCS)
        assert count == len(SAMPLE_DOCS)

    def test_add_batch_stores_all_docs(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: كل الوثائق محفوظة فعلاً في DB.
        """
        assert populated_store.count() == len(SAMPLE_DOCS)

    def test_add_batch_with_indexed_document_objects(
        self, store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: add_batch يقرأ IndexedDocument بشكل صحيح.
        لماذا مهم: هذا هو النوع الفعلي الذي يُمرَّر من BM25Indexer.
        """
        store.add_batch(SAMPLE_DOCS)
        doc = store.get("d1")
        assert doc is not None
        # original_text يجب أن يُحفظ في raw_text
        assert "Cloud storage" in doc["raw_text"]

    def test_add_batch_with_dict_objects(
        self, store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: add_batch يقبل dicts أيضاً.
        لماذا مهم: مرونة — بعض الأكواد قد تُمرّر dicts.
        """
        dicts = [
            {"doc_id": "x1", "raw_text": "hello", "title": "T1"},
            {"doc_id": "x2", "original_text": "world", "title": None},
        ]
        store.add_batch(dicts)
        assert store.count() == 2
        assert store.get("x1")["raw_text"] == "hello"

    def test_add_replaces_existing(self, store: DocumentStore) -> None:
        """
        ماذا يختبر: إضافة doc_id موجود → يُحدَّث (لا يُضاعَف).
        لماذا مهم: عند إعادة بناء الفهرس = تحديث وليس تكرار.
        """
        store.add("doc1", "النص الأول")
        store.add("doc1", "النص المحدَّث")
        assert store.count() == 1
        assert store.get("doc1")["raw_text"] == "النص المحدَّث"

    def test_add_batch_with_small_batch_size(
        self, store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: batch_size صغير يعمل صحيحاً.
        لماذا مهم: يضمن أن الدفعة الأخيرة (أصغر من batch_size) تُحفظ.
        """
        count = store.add_batch(SAMPLE_DOCS, batch_size=2)
        assert count == len(SAMPLE_DOCS)
        assert store.count() == len(SAMPLE_DOCS)

    def test_add_empty_batch(self, store: DocumentStore) -> None:
        """
        ماذا يختبر: add_batch([]) لا يُطرح استثناء ويُرجع 0.
        """
        count = store.add_batch([])
        assert count == 0
        assert store.count() == 0

    def test_title_none_stored_correctly(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: وثيقة بدون عنوان (title=None) تُخزَّن صحيحاً.
        لماذا مهم: بعض الوثائق في MSMARCO ليس لها عنوان.
        """
        doc = populated_store.get("d4")
        assert doc is not None
        assert doc["title"] is None


# =============================================================
# ③ اختبارات الجلب
# =============================================================

class TestGet:

    def test_get_existing_doc(self, populated_store: DocumentStore) -> None:
        """
        ماذا يختبر: get() يُرجع الوثيقة الصحيحة.
        """
        doc = populated_store.get("d1")
        assert doc is not None
        assert doc["doc_id"]   == "d1"
        assert doc["title"]    == "Cloud Storage"
        assert "Cloud storage" in doc["raw_text"]

    def test_get_returns_correct_text(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: raw_text يطابق original_text عند الإدخال.
        لماذا مهم: هذا هو الهدف الجوهري من DocumentStore.
        """
        for sample in SAMPLE_DOCS:
            doc = populated_store.get(sample.doc_id)
            assert doc is not None
            assert doc["raw_text"] == sample.original_text, (
                f"doc_id={sample.doc_id}: النص لا يتطابق"
            )

    def test_get_nonexistent_returns_none(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: doc_id غير موجود → None (لا استثناء).
        لماذا مهم: Retriever يجب أن يتعامل مع None بأمان.
        """
        result = populated_store.get("nonexistent_id_xyz")
        assert result is None

    def test_get_returns_dict_with_correct_keys(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: شكل الـ dict المُرجَع يحتوي كل الحقول.
        لماذا مهم: Developer 2 يعتمد على doc["raw_text"].
        """
        doc = populated_store.get("d1")
        assert "doc_id"   in doc
        assert "raw_text" in doc
        assert "title"    in doc
        assert "metadata" in doc

    def test_get_many_returns_all_found(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: get_many يُرجع كل الوثائق الموجودة.
        """
        ids    = ["d1", "d2", "d3"]
        result = populated_store.get_many(ids)
        assert len(result) == 3
        assert "d1" in result
        assert "d2" in result
        assert "d3" in result

    def test_get_many_ignores_nonexistent(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: get_many يتجاهل IDs غير موجودة بصمت.
        """
        ids    = ["d1", "FAKE_ID", "d3"]
        result = populated_store.get_many(ids)
        assert len(result) == 2
        assert "FAKE_ID" not in result

    def test_get_many_empty_list(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: get_many([]) يُرجع {} بدون استثناء.
        """
        result = populated_store.get_many([])
        assert result == {}

    def test_exists_true(self, populated_store: DocumentStore) -> None:
        assert populated_store.exists("d1") is True

    def test_exists_false(self, populated_store: DocumentStore) -> None:
        assert populated_store.exists("nonexistent") is False


# =============================================================
# ④ اختبارات الإحصائيات
# =============================================================

class TestStats:

    def test_count_empty(self, store: DocumentStore) -> None:
        assert store.count() == 0

    def test_count_after_add(self, populated_store: DocumentStore) -> None:
        assert populated_store.count() == len(SAMPLE_DOCS)

    def test_is_populated_false(self, store: DocumentStore) -> None:
        assert store.is_populated() is False

    def test_is_populated_true(self, populated_store: DocumentStore) -> None:
        assert populated_store.is_populated() is True

    def test_get_status_structure(
        self, populated_store: DocumentStore
    ) -> None:
        """
        ماذا يختبر: get_status يُرجع dict بكل الحقول المطلوبة.
        لماذا مهم: Developer 3 يعرضها في الـ UI.
        """
        status = populated_store.get_status()
        assert "dataset_name"  in status
        assert "db_path"       in status
        assert "num_documents" in status
        assert "db_size_mb"    in status
        assert "is_populated"  in status
        assert status["num_documents"] == len(SAMPLE_DOCS)
        assert status["is_populated"]  is True


# =============================================================
# ⑤ اختبارات Class Methods
# =============================================================

class TestClassMethods:

    def test_open_returns_store(self, tmp_path: Path) -> None:
        """
        ماذا يختبر: DocumentStore.open() يُرجع كائن صحيح.
        """
        indexes_dir  = str(tmp_path / "indexes")
        original     = DocumentStore(indexes_dir=indexes_dir, dataset_name="ds")
        original.add("x1", "hello")

        opened = DocumentStore.open(indexes_dir=indexes_dir, dataset_name="ds")
        assert opened.count() == 1

    def test_db_exists_true(self, tmp_path: Path) -> None:
        """
        ماذا يختبر: db_exists يُرجع True إذا DB موجودة ومملوءة.
        """
        indexes_dir = str(tmp_path / "indexes")
        s = DocumentStore(indexes_dir=indexes_dir, dataset_name="ds")
        s.add("x1", "hello")
        assert DocumentStore.db_exists(indexes_dir, "ds") is True

    def test_db_exists_false_no_file(self, tmp_path: Path) -> None:
        """
        ماذا يختبر: db_exists يُرجع False إذا لم يكن الملف موجوداً.
        """
        assert DocumentStore.db_exists(str(tmp_path), "nonexistent") is False


# =============================================================
# ⑥ اختبار التكامل — يحاكي سيناريو Developer 1 + Developer 2
# =============================================================

class TestIntegration:

    def test_build_then_retrieve_workflow(self, tmp_path: Path) -> None:
        """
        يحاكي بالضبط ما يحدث في النظام الحقيقي:

        Developer 1 → يبني الفهارس ويحفظ الوثائق في SQLite
        Developer 2 → يفتح DB ويجلب النص عند كل بحث

        هذا هو الاختبار الأهم.
        """
        indexes_dir = str(tmp_path / "indexes")

        # === Developer 1: وقت البناء ===
        build_store = DocumentStore(
            indexes_dir=indexes_dir,
            dataset_name="msmarco-passage",
        )
        added = build_store.add_batch(SAMPLE_DOCS)
        assert added == len(SAMPLE_DOCS)

        # === Developer 2: وقت الاستعلام (خادم جديد) ===
        # ننشئ كائن جديد تماماً — يحاكي إعادة تشغيل الخادم
        retrieval_store = DocumentStore.open(
            indexes_dir=indexes_dir,
            dataset_name="msmarco-passage",
        )

        # يجلب نص الوثيقة بعد البحث
        doc_ids_from_bm25 = ["d1", "d3", "d5"]
        docs = retrieval_store.get_many(doc_ids_from_bm25)

        assert len(docs) == 3
        for doc_id in doc_ids_from_bm25:
            assert doc_id in docs
            assert docs[doc_id]["raw_text"]  # النص ليس فارغاً
            assert docs[doc_id]["doc_id"] == doc_id

        # التحقق من محتوى محدد
        assert "Cloud storage" in docs["d1"]["raw_text"]

    def test_data_survives_store_recreation(self, tmp_path: Path) -> None:
        """
        ماذا يختبر: البيانات تبقى بعد إغلاق وإعادة فتح الـ store.
        لماذا مهم: SQLite يحفظ على القرص — هذا الهدف الأساسي.
        """
        indexes_dir = str(tmp_path / "indexes")

        # إنشاء وملء
        s1 = DocumentStore(indexes_dir=indexes_dir, dataset_name="test")
        s1.add_batch(SAMPLE_DOCS)

        # إعادة الفتح (يحاكي restart الخادم)
        s2 = DocumentStore(indexes_dir=indexes_dir, dataset_name="test")
        assert s2.count() == len(SAMPLE_DOCS)

        # كل وثيقة لا تزال موجودة بنصها الصحيح
        for sample in SAMPLE_DOCS:
            doc = s2.get(sample.doc_id)
            assert doc is not None
            assert doc["raw_text"] == sample.original_text

    def test_singleton_returns_same_instance(self, tmp_path: Path) -> None:
        """
        ماذا يختبر: get_document_store يُرجع نفس الكائن.
        لماذا مهم: Singleton يمنع فتح اتصالات DB متعددة.
        """
        indexes_dir = str(tmp_path / "indexes")
        s1 = get_document_store("test-ds", indexes_dir=indexes_dir)
        s2 = get_document_store("test-ds", indexes_dir=indexes_dir)
        assert s1 is s2, "يجب أن يكون نفس الكائن (Singleton)"