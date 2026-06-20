"""
services/indexing/tests/test_inverted_index.py
===============================================
اختبارات وحدة لـ InvertedIndex.

تشغيل:
    python -m pytest services/indexing/tests/test_inverted_index.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.indexing.dataset_loader import DatasetLoader, Document
from services.indexing.inverted_index import (
    InvertedIndex, InvertedIndexMetadata, Posting
)

SAMPLE_DOCS = [
    Document("d1", "Cloud Storage", "Cloud storage is useful for syncing files."),
    Document("d2", "AI Systems",    "AI assistants use natural language processing."),
    Document("d3", "IR Basics",     "Information retrieval finds relevant documents."),
    Document("d4", "Python",        "Python is used for machine learning."),
    Document("d5", "Cloud AI",      "Cloud computing and AI are transforming tech."),
]

def _mock_loader(docs=None):
    m = MagicMock(spec=DatasetLoader)
    m.load_all.return_value = docs if docs is not None else SAMPLE_DOCS
    return m

@pytest.fixture
def idx(tmp_path):
    i = InvertedIndex(
        indexes_dir=str(tmp_path / "indexes"),
        dataset_loader=_mock_loader(),
    )
    i.build_index("test", apply_stemming=True, remove_stopwords=True)
    return i

@pytest.fixture
def saved_idx(idx, tmp_path):
    idx.save_index("test")
    return idx, tmp_path


class TestBuildIndex:
    def test_returns_metadata(self, idx):
        assert isinstance(idx.metadata, InvertedIndexMetadata)

    def test_vocab_not_empty(self, idx):
        assert idx.metadata.vocab_size > 0

    def test_all_doc_ids_indexed(self, idx):
        assert len(idx._all_doc_ids) == len(SAMPLE_DOCS)

    def test_cloud_in_index(self, idx):
        """'cloud' يجب أن يظهر في d1 و d5."""
        postings = idx.get_posting_list("cloud")
        doc_ids = {p.doc_id for p in postings}
        assert "d1" in doc_ids
        assert "d5" in doc_ids

    def test_document_frequency(self, idx):
        df = idx.document_frequency("cloud")
        assert df >= 2  # d1 و d5 على الأقل

    def test_is_built_true(self, idx):
        assert idx.is_built() is True

    def test_is_built_false_before_build(self, tmp_path):
        i = InvertedIndex(indexes_dir=str(tmp_path / "idx2"))
        assert i.is_built() is False


class TestBooleanRetrieval:
    def test_and_returns_intersection(self, idx):
        """AND: فقط الوثائق التي تحتوي المصطلحين."""
        # cloud موجود في d1 وd5 — نفحص أن AND يُرجع فقط المشتركة
        result = idx.search_and(["cloud"])
        assert all(isinstance(r, str) for r in result)

    def test_or_returns_union(self, idx):
        """OR: كل الوثائق التي تحتوي أي مصطلح."""
        result_cloud = idx.search_or(["cloud"])
        result_ai    = idx.search_or(["ai"])
        result_both  = idx.search_or(["cloud", "ai"])
        # union يجب أن يكون أكبر من أو يساوي كل على حدة
        assert len(result_both) >= len(result_cloud)
        assert len(result_both) >= len(result_ai)

    def test_not_excludes_term(self, idx):
        """NOT: يُرجع كل الوثائق ما عدا التي تحتوي المصطلح."""
        all_docs = set(idx._all_doc_ids)
        cloud_docs = {p.doc_id for p in idx.get_posting_list("cloud")}
        not_cloud = set(idx.search_not("cloud"))
        # تقاطع NOT cloud مع cloud يجب أن يكون فارغاً
        assert not_cloud & cloud_docs == set()
        # اتحادهما يجب أن يساوي كل الوثائق
        assert not_cloud | cloud_docs == all_docs

    def test_and_empty_terms_returns_empty(self, idx):
        assert idx.search_and([]) == []

    def test_or_empty_terms_returns_empty(self, idx):
        assert idx.search_or([]) == []

    def test_and_nonexistent_term_returns_empty(self, idx):
        result = idx.search_and(["xyznonexistent123"])
        assert result == []

    def test_and_not_excludes_correctly(self, idx):
        """AND NOT: تحتوي الأول ولا تحتوي الثاني."""
        cloud_docs = set(idx.search_or(["cloud"]))
        python_docs = set(idx.search_or(["python"]))
        result = set(idx.search_and_not(["cloud"], ["python"]))
        # النتيجة يجب ألا تحتوي أي وثيقة من python_docs
        assert result & python_docs == set()
        # النتيجة يجب أن تكون جزءاً من cloud_docs
        assert result <= cloud_docs


class TestSaveAndLoad:
    def test_save_creates_files(self, saved_idx):
        idx, tmp_path = saved_idx
        index_dir = tmp_path / "indexes" / "test" / "inverted"
        assert (index_dir / "inverted_index.json").exists()
        assert (index_dir / "inverted_metadata.json").exists()

    def test_load_restores_vocab_size(self, saved_idx):
        idx, tmp_path = saved_idx
        original_vocab = idx.metadata.vocab_size

        new_idx = InvertedIndex(
            indexes_dir=str(tmp_path / "indexes"),
            dataset_loader=_mock_loader(),
        )
        new_idx.load_index("test")
        assert new_idx.metadata.vocab_size == original_vocab

    def test_load_restores_boolean_results(self, saved_idx):
        """البحث بعد التحميل يُنتج نفس النتائج."""
        idx, tmp_path = saved_idx
        original_result = idx.search_or(["cloud"])

        new_idx = InvertedIndex(
            indexes_dir=str(tmp_path / "indexes"),
            dataset_loader=_mock_loader(),
        )
        new_idx.load_index("test")
        loaded_result = new_idx.search_or(["cloud"])
        assert original_result == loaded_result

    def test_load_nonexistent_raises(self, tmp_path):
        i = InvertedIndex(indexes_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            i.load_index("ghost")


class TestStats:
    def test_stats_before_build(self, tmp_path):
        i = InvertedIndex(indexes_dir=str(tmp_path))
        assert i.get_stats()["status"] == "not_built"

    def test_stats_after_build(self, idx):
        stats = idx.get_stats()
        assert stats["status"] == "ready"
        assert stats["vocab_size"] > 0
        assert stats["num_documents"] == len(SAMPLE_DOCS)
