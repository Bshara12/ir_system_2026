"""
services/indexing/tests/test_dataset_loader.py
===============================================
اختبارات شاملة لـ DatasetLoader و IrDatasetsAdapter.

ماذا نختبر؟
  1. تحميل الوثائق من ملفات محلية (JSONL, TSV, CSV, TXT)
  2. تحميل الاستعلامات (Queries)
  3. تحميل الـ Qrels
  4. الكشف التلقائي عن المصدر (ir_datasets vs local)
  5. معالجة الأخطاء
  6. نماذج البيانات (Document, Query, Qrel)
  7. IrDatasetsAdapter مع Mock (بدون إنترنت)

تشغيل:
    cd ir_system_2026
    python -m pytest services/indexing/tests/test_dataset_loader.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.indexing.ir_datasets_adapter import (
    Document, Query, Qrel,
    IrDatasetsAdapter,
    SUPPORTED_DATASETS,
)
from services.indexing.dataset_loader import DatasetLoader


# =============================================================
# Fixtures — بيانات تجريبية
# =============================================================

@pytest.fixture
def tmp_datasets_dir(tmp_path: Path) -> Path:
    """مجلد مؤقت يحاكي data/datasets/"""
    d = tmp_path / "datasets"
    d.mkdir()
    return d


@pytest.fixture
def sample_jsonl_dataset(tmp_datasets_dir: Path) -> Path:
    """
    ينشئ dataset بصيغة JSONL للاختبار.
    يحاكي بنية BEIR (الأكثر شيوعاً في ir_datasets).
    """
    ds_dir = tmp_datasets_dir / "test_dataset"
    ds_dir.mkdir()
    corpus_path = ds_dir / "corpus.jsonl"
    docs = [
        {"_id": "d1", "title": "Cloud Storage",  "text": "Cloud storage syncs files."},
        {"_id": "d2", "title": "AI Assistants",  "text": "AI helps with tasks."},
        {"_id": "d3", "title": "Blockchain",      "text": "Blockchain is decentralized."},
        {"_id": "d4", "title": "",                "text": "No title document."},
        {"_id": "d5", "title": None,              "text": "Another doc without title."},
    ]
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    return tmp_datasets_dir


@pytest.fixture
def sample_tsv_dataset(tmp_datasets_dir: Path) -> Path:
    """ينشئ dataset بصيغة TSV: doc_id TAB title TAB text"""
    ds_dir = tmp_datasets_dir / "tsv_dataset"
    ds_dir.mkdir()
    tsv_path = ds_dir / "corpus.tsv"
    rows = [
        ("d1", "Cloud", "Cloud storage is useful."),
        ("d2", "AI", "AI helps with many tasks."),
        ("d3", "Python", "Python is a programming language."),
    ]
    with open(tsv_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")
    return tmp_datasets_dir


@pytest.fixture
def sample_csv_dataset(tmp_datasets_dir: Path) -> Path:
    """ينشئ dataset بصيغة CSV مع header"""
    ds_dir = tmp_datasets_dir / "csv_dataset"
    ds_dir.mkdir()
    csv_path = ds_dir / "corpus.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,title,text\n")
        f.write('d1,Cloud,"Cloud storage syncs files across devices."\n')
        f.write('d2,AI,"AI helps with tasks using machine learning."\n')
    return tmp_datasets_dir


@pytest.fixture
def sample_txt_dataset(tmp_datasets_dir: Path) -> Path:
    """ينشئ dataset بصيغة TXT: كل سطر = وثيقة"""
    ds_dir = tmp_datasets_dir / "txt_dataset"
    ds_dir.mkdir()
    txt_path = ds_dir / "corpus.txt"
    lines = [
        "Cloud storage syncs files across devices.",
        "AI helps with tasks using machine learning.",
        "Blockchain decentralizes finance.",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return tmp_datasets_dir


@pytest.fixture
def dataset_with_queries_and_qrels(tmp_datasets_dir: Path) -> Path:
    """
    ينشئ dataset كامل مع corpus + queries + qrels.
    يحاكي بنية trec-covid المحلية بعد save_to_jsonl().
    """
    ds_dir = tmp_datasets_dir / "full_dataset"
    ds_dir.mkdir()

    # corpus
    corpus = [
        {"_id": "d1", "text": "COVID-19 treatment options."},
        {"_id": "d2", "text": "Vaccine development for coronavirus."},
        {"_id": "d3", "text": "PCR testing accuracy."},
    ]
    with open(ds_dir / "corpus.jsonl", "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    # queries
    queries = [
        {"query_id": "q1", "text": "What are COVID treatments?"},
        {"query_id": "q2", "text": "How effective are vaccines?"},
    ]
    with open(ds_dir / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    # qrels
    qrels = [
        {"query_id": "q1", "doc_id": "d1", "relevance": 2},
        {"query_id": "q1", "doc_id": "d3", "relevance": 1},
        {"query_id": "q2", "doc_id": "d2", "relevance": 2},
    ]
    with open(ds_dir / "qrels.jsonl", "w") as f:
        for qr in qrels:
            f.write(json.dumps(qr) + "\n")

    return tmp_datasets_dir


def make_loader(datasets_dir: Path) -> DatasetLoader:
    """يُنشئ DatasetLoader بـ datasets_dir مخصص بدون ir_datasets."""
    mock_adapter = MagicMock(spec=IrDatasetsAdapter)
    mock_adapter.is_supported.return_value = False
    mock_adapter.list_supported_datasets.return_value = list(SUPPORTED_DATASETS.keys())
    return DatasetLoader(
        datasets_dir=str(datasets_dir),
        ir_adapter=mock_adapter,
    )


# =============================================================
# 1. اختبارات نموذج Document
# =============================================================

class TestDocument:
    def test_get_full_text_with_title(self):
        """
        ماذا يختبر: get_full_text() يدمج العنوان مع النص.
        لماذا مهم: الفهارس تستخدم get_full_text() وليس text مباشرة.
        """
        doc = Document(doc_id="d1", text="Cloud syncs files.", title="Cloud Storage")
        result = doc.get_full_text()
        assert "Cloud Storage" in result
        assert "Cloud syncs files." in result

    def test_get_full_text_without_title(self):
        """
        ماذا يختبر: بدون عنوان → يُرجع النص فقط.
        """
        doc = Document(doc_id="d1", text="Cloud syncs files.", title=None)
        assert doc.get_full_text() == "Cloud syncs files."

    def test_to_dict(self):
        """
        ماذا يختبر: to_dict() يُرجع dict كامل.
        لماذا مهم: يُستخدم عند الحفظ كـ JSONL.
        """
        doc = Document(doc_id="d1", text="Test", title="T")
        d = doc.to_dict()
        assert d["doc_id"] == "d1"
        assert d["text"] == "Test"
        assert d["title"] == "T"


# =============================================================
# 2. اختبارات نموذج Query
# =============================================================

class TestQuery:
    def test_to_dict(self):
        q = Query(query_id="q1", text="cloud storage")
        d = q.to_dict()
        assert d["query_id"] == "q1"
        assert d["text"] == "cloud storage"


# =============================================================
# 3. اختبارات نموذج Qrel
# =============================================================

class TestQrel:
    def test_to_dict(self):
        qr = Qrel(query_id="q1", doc_id="d1", relevance=2)
        d = qr.to_dict()
        assert d["query_id"] == "q1"
        assert d["doc_id"] == "d1"
        assert d["relevance"] == 2


# =============================================================
# 4. اختبارات IrDatasetsAdapter
# =============================================================

class TestIrDatasetsAdapter:
    def test_is_supported_known_dataset(self):
        """
        ماذا يختبر: msmarco و trec-covid مدعومان.
        لماذا مهم: DatasetLoader يعتمد على is_supported() للتوجيه.
        """
        adapter = IrDatasetsAdapter()
        assert adapter.is_supported("msmarco") is True
        assert adapter.is_supported("trec-covid") is True

    def test_is_supported_unknown_dataset(self):
        """
        ماذا يختبر: اسم غير موجود → False.
        ما الخطأ الذي يمنعه: DatasetLoader يحاول ir_datasets على dataset محلي.
        """
        adapter = IrDatasetsAdapter()
        assert adapter.is_supported("nonexistent_xyz") is False

    def test_list_supported_datasets_not_empty(self):
        """
        ماذا يختبر: قائمة الـ datasets المدعومة غير فارغة.
        """
        adapter = IrDatasetsAdapter()
        datasets = adapter.list_supported_datasets()
        assert len(datasets) > 0
        assert "msmarco" in datasets
        assert "trec-covid" in datasets

    def test_get_dataset_info_msmarco(self):
        """
        ماذا يختبر: معلومات msmarco صحيحة.
        """
        adapter = IrDatasetsAdapter()
        info = adapter.get_dataset_info("msmarco")
        assert info is not None
        assert info["doc_count"] > 200_000   # شرط المشروع
        assert info["has_qrels"] is True
        assert "ir_datasets_id" in info

    def test_get_dataset_info_trec_covid(self):
        """
        ماذا يختبر: معلومات trec-covid صحيحة.
        """
        adapter = IrDatasetsAdapter()
        info = adapter.get_dataset_info("trec-covid")
        assert info is not None
        assert info["doc_count"] > 100_000
        assert info["has_qrels"] is True

    def test_get_dataset_info_unknown_returns_none(self):
        """
        ماذا يختبر: dataset غير موجود → None.
        """
        adapter = IrDatasetsAdapter()
        assert adapter.get_dataset_info("nonexistent") is None

    def test_load_unknown_dataset_raises_value_error(self):
        """
        ماذا يختبر: محاولة تحميل dataset غير مدعوم → ValueError.
        ما الخطأ الذي يمنعه: خطأ غامض بدون رسالة واضحة.
        """
        adapter = IrDatasetsAdapter()
        with pytest.raises(ValueError, match="غير مدعوم"):
            list(adapter.stream_documents("nonexistent_xyz"))

    def test_convert_document_with_mock(self):
        """
        ماذا يختبر: _convert_document() يعمل مع بنية msmarco.
        كيف نختبر بدون إنترنت: نصنع كائناً يحاكي raw_doc.
        """
        adapter = IrDatasetsAdapter()

        # نصنع mock يحاكي وثيقة ir_datasets
        raw_doc = MagicMock()
        raw_doc.doc_id = "msmarco_d1"
        raw_doc.body   = "Passage text about cloud storage."
        raw_doc.title  = "Cloud Storage"
        raw_doc.text   = None

        doc = adapter._convert_document(raw_doc)
        assert doc is not None
        assert doc.doc_id == "msmarco_d1"
        assert "cloud storage" in doc.text.lower()
        assert doc.title == "Cloud Storage"

    def test_convert_document_beir_style(self):
        """
        ماذا يختبر: _convert_document() مع بنية BEIR (trec-covid).
        BEIR يستخدم .text بدلاً من .body
        """
        adapter = IrDatasetsAdapter()

        raw_doc = MagicMock()
        raw_doc.doc_id = "covid_d1"
        raw_doc.text   = "COVID-19 treatment options include..."
        raw_doc.title  = "COVID Treatment"
        raw_doc.body   = None

        doc = adapter._convert_document(raw_doc)
        assert doc is not None
        assert doc.doc_id == "covid_d1"
        assert "COVID-19" in doc.text

    def test_convert_document_empty_text_returns_none(self):
        """
        ماذا يختبر: وثيقة بنص فارغ → None (نتجاوزها).
        ما الخطأ الذي يمنعه: وثائق فارغة تدخل الفهرس وتُفسد النتائج.
        """
        adapter = IrDatasetsAdapter()

        raw_doc = MagicMock()
        raw_doc.doc_id = "empty_d"
        raw_doc.text   = "   "
        raw_doc.body   = None
        raw_doc.title  = None

        doc = adapter._convert_document(raw_doc)
        assert doc is None

    def test_convert_query_with_mock(self):
        """ماذا يختبر: _convert_query() يعمل مع كائن mock."""
        adapter = IrDatasetsAdapter()
        raw_q = MagicMock()
        raw_q.query_id = "q42"
        raw_q.text     = "what is cloud computing"

        q = adapter._convert_query(raw_q)
        assert q is not None
        assert q.query_id == "q42"
        assert q.text == "what is cloud computing"

    def test_convert_qrel_with_mock(self):
        """ماذا يختبر: _convert_qrel() يعمل مع كائن mock."""
        adapter = IrDatasetsAdapter()
        raw_qr = MagicMock()
        raw_qr.query_id  = "q1"
        raw_qr.doc_id    = "d1"
        raw_qr.relevance = 2

        qr = adapter._convert_qrel(raw_qr)
        assert qr is not None
        assert qr.query_id == "q1"
        assert qr.doc_id == "d1"
        assert qr.relevance == 2

    def test_stream_documents_mocked(self):
        """
        ماذا يختبر: stream_documents() يمر على كل وثائق ir_datasets.
        كيف: نصنع mock dataset يُرجع 3 وثائق.
        لماذا مهم: نتأكد أن المنطق الداخلي صحيح بدون إنترنت.
        """
        adapter = IrDatasetsAdapter()

        # صنع وثائق mock
        raw_docs = []
        for i in range(3):
            rd = MagicMock()
            rd.doc_id = f"d{i}"
            rd.text   = f"Document {i} text content."
            rd.body   = None
            rd.title  = f"Doc {i}"
            raw_docs.append(rd)

        mock_ds = MagicMock()
        mock_ds.docs_iter.return_value = iter(raw_docs)
        adapter._loaded["msmarco"] = mock_ds

        docs = list(adapter.stream_documents("msmarco"))
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].doc_id == "d0"
        assert docs[2].doc_id == "d2"

    def test_stream_documents_respects_max_docs(self):
        """
        ماذا يختبر: max_docs يوقف التدفق عند الحد المحدد.
        ما الخطأ الذي يمنعه: تحميل 8.8M وثيقة بدلاً من 1000.
        """
        adapter = IrDatasetsAdapter()

        raw_docs = []
        for i in range(100):
            rd = MagicMock()
            rd.doc_id = f"d{i}"
            rd.text   = f"Document text {i}."
            rd.body   = None
            rd.title  = None
            raw_docs.append(rd)

        mock_ds = MagicMock()
        mock_ds.docs_iter.return_value = iter(raw_docs)
        adapter._loaded["msmarco"] = mock_ds

        docs = list(adapter.stream_documents("msmarco", max_docs=10))
        assert len(docs) == 10


# =============================================================
# 5. اختبارات DatasetLoader — ملفات محلية
# =============================================================

class TestDatasetLoaderLocalFiles:
    def test_stream_jsonl_docs(self, sample_jsonl_dataset):
        """
        ماذا يختبر: تحميل وثائق من JSONL يعمل.
        لماذا مهم: JSONL هو الصيغة الرئيسية لـ BEIR datasets.
        """
        loader = make_loader(sample_jsonl_dataset)
        docs = list(loader.stream_documents("test_dataset"))
        assert len(docs) == 5
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].doc_id == "d1"
        assert "Cloud" in docs[0].title

    def test_stream_tsv_docs(self, sample_tsv_dataset):
        """
        ماذا يختبر: تحميل من TSV يعمل.
        لماذا مهم: بعض datasets تأتي بصيغة TSV.
        """
        loader = make_loader(sample_tsv_dataset)
        docs = list(loader.stream_documents("tsv_dataset"))
        assert len(docs) == 3
        assert docs[0].doc_id == "d1"
        assert "Cloud" in docs[0].title

    def test_stream_csv_docs(self, sample_csv_dataset):
        """ماذا يختبر: تحميل من CSV مع header يعمل."""
        loader = make_loader(sample_csv_dataset)
        docs = list(loader.stream_documents("csv_dataset"))
        assert len(docs) == 2
        assert docs[0].doc_id == "d1"

    def test_stream_txt_docs(self, sample_txt_dataset):
        """ماذا يختبر: تحميل من TXT — كل سطر وثيقة."""
        loader = make_loader(sample_txt_dataset)
        docs = list(loader.stream_documents("txt_dataset"))
        assert len(docs) == 3
        # doc_id = رقم السطر
        assert docs[0].doc_id == "1"
        assert "Cloud" in docs[0].text

    def test_load_all_returns_list(self, sample_jsonl_dataset):
        """ماذا يختبر: load_all() يُرجع List[Document]."""
        loader = make_loader(sample_jsonl_dataset)
        docs = loader.load_all("test_dataset")
        assert isinstance(docs, list)
        assert len(docs) == 5

    def test_load_all_respects_max_docs(self, sample_jsonl_dataset):
        """
        ماذا يختبر: max_docs يحدّ عدد الوثائق المحمّلة.
        ما الخطأ الذي يمنعه: تحميل كل 200K وثيقة عند الاختبار.
        """
        loader = make_loader(sample_jsonl_dataset)
        docs = loader.load_all("test_dataset", max_docs=3)
        assert len(docs) == 3

    def test_nonexistent_dataset_raises_error(self, tmp_datasets_dir):
        """
        ماذا يختبر: dataset غير موجود → FileNotFoundError.
        ما الخطأ الذي يمنعه: خطأ صامت أو crash بدون رسالة.
        """
        loader = make_loader(tmp_datasets_dir)
        with pytest.raises(FileNotFoundError, match="nonexistent_xyz"):
            list(loader.stream_documents("nonexistent_xyz"))

    def test_document_ids_are_strings(self, sample_jsonl_dataset):
        """
        ماذا يختبر: doc_id دائماً string حتى لو JSON رقم.
        ما الخطأ الذي يمنعه: type mismatch عند المقارنة مع qrels.
        """
        loader = make_loader(sample_jsonl_dataset)
        docs = list(loader.stream_documents("test_dataset"))
        assert all(isinstance(d.doc_id, str) for d in docs)

    def test_empty_text_documents_are_skipped(self, tmp_datasets_dir):
        """
        ماذا يختبر: وثائق بنص فارغ تُتخطّى.
        ما الخطأ الذي يمنعه: وثائق فارغة تُفسد الفهارس.
        """
        ds_dir = tmp_datasets_dir / "empty_docs"
        ds_dir.mkdir()
        with open(ds_dir / "corpus.jsonl", "w") as f:
            f.write('{"_id": "d1", "text": "Valid doc."}\n')
            f.write('{"_id": "d2", "text": "   "}\n')       # فارغ → يُتخطى
            f.write('{"_id": "d3", "text": ""}\n')          # فارغ → يُتخطى
            f.write('{"_id": "d4", "text": "Another valid."}\n')

        loader = make_loader(tmp_datasets_dir)
        docs = list(loader.stream_documents("empty_docs"))
        assert len(docs) == 2
        assert docs[0].doc_id == "d1"
        assert docs[1].doc_id == "d4"

    def test_invalid_json_lines_are_skipped(self, tmp_datasets_dir):
        """
        ماذا يختبر: سطر JSON فاسد لا يوقف التحميل.
        ما الخطأ الذي يمنعه: crash أثناء فهرسة 200K وثيقة بسبب سطر واحد فاسد.
        """
        ds_dir = tmp_datasets_dir / "broken_jsonl"
        ds_dir.mkdir()
        with open(ds_dir / "corpus.jsonl", "w") as f:
            f.write('{"_id": "d1", "text": "Valid line."}\n')
            f.write('NOT VALID JSON {\n')                   # فاسد → يُتخطى
            f.write('{"_id": "d3", "text": "Also valid."}\n')

        loader = make_loader(tmp_datasets_dir)
        docs = list(loader.stream_documents("broken_jsonl"))
        assert len(docs) == 2


# =============================================================
# 6. اختبارات Queries و Qrels المحلية
# =============================================================

class TestQueriesAndQrels:
    def test_load_queries(self, dataset_with_queries_and_qrels):
        """
        ماذا يختبر: load_queries() يُرجع List[Query].
        لماذا مهم: Dev2 يحتاج قائمة الاستعلامات للبحث.
        """
        loader = make_loader(dataset_with_queries_and_qrels)
        queries = loader.load_queries("full_dataset")
        assert len(queries) == 2
        assert all(isinstance(q, Query) for q in queries)
        assert queries[0].query_id == "q1"
        assert "COVID" in queries[0].text

    def test_stream_queries_is_lazy(self, dataset_with_queries_and_qrels):
        """
        ماذا يختبر: stream_queries() يُرجع generator.
        لماذا مهم: لا نريد تحميل كل الاستعلامات مرة واحدة.
        """
        import types
        loader = make_loader(dataset_with_queries_and_qrels)
        gen = loader.stream_queries("full_dataset")
        assert isinstance(gen, types.GeneratorType)

    def test_load_qrels(self, dataset_with_queries_and_qrels):
        """
        ماذا يختبر: load_qrels() يُرجع List[Qrel].
        لماذا مهم: Dev3 يحتاج الـ qrels لحساب MAP/nDCG.
        """
        loader = make_loader(dataset_with_queries_and_qrels)
        qrels = loader.load_qrels("full_dataset")
        assert len(qrels) == 3
        assert all(isinstance(qr, Qrel) for qr in qrels)

    def test_qrels_relevance_values(self, dataset_with_queries_and_qrels):
        """
        ماذا يختبر: قيم relevance صحيحة (0, 1, 2).
        ما الخطأ الذي يمنعه: قيم خاطئة تُعطي نتائج تقييم مزورة.
        """
        loader = make_loader(dataset_with_queries_and_qrels)
        qrels = loader.load_qrels("full_dataset")
        relevances = {qr.relevance for qr in qrels}
        assert relevances.issubset({0, 1, 2})
        assert 2 in relevances  # يوجد qrel بدرجة عالية

    def test_missing_queries_file_raises_error(self, sample_jsonl_dataset):
        """
        ماذا يختبر: dataset بدون queries.jsonl → FileNotFoundError.
        ما الخطأ الذي يمنعه: crash صامت في Retrieval Service.
        """
        loader = make_loader(sample_jsonl_dataset)
        with pytest.raises(FileNotFoundError):
            list(loader.stream_queries("test_dataset"))

    def test_missing_qrels_file_raises_error(self, sample_jsonl_dataset):
        """
        ماذا يختبر: dataset بدون qrels.jsonl → FileNotFoundError.
        ما الخطأ الذي يمنعه: Evaluation Service يُعطي نتائج فارغة بصمت.
        """
        loader = make_loader(sample_jsonl_dataset)
        with pytest.raises(FileNotFoundError):
            list(loader.stream_qrels("test_dataset"))


# =============================================================
# 7. اختبارات get_dataset_info
# =============================================================

class TestGetDatasetInfo:
    def test_info_for_existing_dataset(self, sample_jsonl_dataset):
        """ماذا يختبر: معلومات dataset موجود تُرجع exists=True."""
        loader = make_loader(sample_jsonl_dataset)
        info = loader.get_dataset_info("test_dataset")
        assert info["exists"] is True
        assert info["name"] == "test_dataset"

    def test_info_for_nonexistent_dataset(self, tmp_datasets_dir):
        """ماذا يختبر: dataset غير موجود → exists=False."""
        loader = make_loader(tmp_datasets_dir)
        info = loader.get_dataset_info("nonexistent")
        assert info["exists"] is False

    def test_info_includes_source(self, sample_jsonl_dataset):
        """
        ماذا يختبر: info يُضمّن مصدر البيانات.
        لماذا مهم: الـ UI يعرض "ملف محلي" أو "ir_datasets".
        """
        loader = make_loader(sample_jsonl_dataset)
        info = loader.get_dataset_info("test_dataset")
        assert "source" in info
        assert info["source"] == "local_file"


# =============================================================
# 8. اختبار save_to_jsonl
# =============================================================

class TestSaveToJsonl:
    def test_save_to_jsonl_creates_files(self, tmp_path):
        """
        ماذا يختبر: save_to_jsonl() يُنشئ corpus.jsonl + queries.jsonl + qrels.jsonl
        كيف: نصنع mock للـ ir_datasets.
        لماذا مهم: هذه الدالة هي التي تُنزّل وتحفظ الـ datasets للعمل offline.
        """
        adapter = IrDatasetsAdapter()

        # Mock docs
        raw_docs = [MagicMock() for _ in range(5)]
        for i, rd in enumerate(raw_docs):
            rd.doc_id = f"d{i}"
            rd.text   = f"Document text {i}."
            rd.body   = None
            rd.title  = f"Doc {i}"

        # Mock queries
        raw_queries = [MagicMock() for _ in range(2)]
        for i, rq in enumerate(raw_queries):
            rq.query_id = f"q{i}"
            rq.text     = f"Query text {i}"

        # Mock qrels
        raw_qrels = [MagicMock() for _ in range(3)]
        for i, rqr in enumerate(raw_qrels):
            rqr.query_id  = "q0"
            rqr.doc_id    = f"d{i}"
            rqr.relevance = 1

        mock_ds = MagicMock()
        mock_ds.docs_iter.return_value    = iter(raw_docs)
        mock_ds.queries_iter.return_value = iter(raw_queries)
        mock_ds.qrels_iter.return_value   = iter(raw_qrels)
        mock_ds.has_attr = lambda x: True

        # نحقن الـ mock
        adapter._loaded["msmarco"] = mock_ds

        out_dir = str(tmp_path / "msmarco_output")
        paths = adapter.save_to_jsonl(
            "msmarco",
            output_dir=out_dir,
            save_queries=True,
            save_qrels=True,
        )

        assert "corpus" in paths
        assert Path(paths["corpus"]).exists()

        # تحقق من محتوى corpus.jsonl
        lines = Path(paths["corpus"]).read_text().strip().split("\n")
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert first["doc_id"] == "d0"
        assert "Document text 0" in first["text"]


# =============================================================
# 9. اختبار DatasetLoader مع ir_datasets Mocked
# =============================================================

class TestDatasetLoaderWithIrDatasets:
    def test_loader_routes_to_ir_adapter(self, tmp_path):
        """
        ماذا يختبر: DatasetLoader يُوجّه msmarco إلى IrDatasetsAdapter.
        ما الخطأ الذي يمنعه: loader يبحث عن ملف محلي باسم msmarco.
        """
        mock_adapter = MagicMock(spec=IrDatasetsAdapter)
        mock_adapter.is_supported.return_value = True
        mock_adapter.stream_documents.return_value = iter([
            Document(doc_id="d1", text="MS MARCO doc 1."),
            Document(doc_id="d2", text="MS MARCO doc 2."),
        ])

        loader = DatasetLoader(
            datasets_dir=str(tmp_path),
            ir_adapter=mock_adapter,
        )

        docs = list(loader.stream_documents("msmarco"))
        assert len(docs) == 2
        mock_adapter.stream_documents.assert_called_once_with("msmarco", max_docs=None)

    def test_loader_info_for_ir_dataset(self):
        """
        ماذا يختبر: get_dataset_info() يُرجع doc_count صحيح لـ ir_datasets.
        لماذا مهم: الـ UI يعرض عدد الوثائق للمستخدم.
        """
        mock_adapter = MagicMock(spec=IrDatasetsAdapter)
        mock_adapter.is_supported.return_value = True
        mock_adapter.get_dataset_info.return_value = {
            "ir_datasets_id": "msmarco-passage/dev/small",
            "friendly_name":  "MS MARCO",
            "domain":         "General Web",
            "doc_count":      8_841_823,
            "query_count":    6_980,
            "qrel_count":     7_437,
            "has_qrels":      True,
            "size_gb":        3.8,
            "notes":          "",
        }

        loader = DatasetLoader(ir_adapter=mock_adapter)
        info = loader.get_dataset_info("msmarco")

        assert info["exists"] is True
        assert info["source"] == "ir_datasets"
        assert info["doc_count"] > 200_000   # شرط المشروع
        assert info["has_qrels"] is True

    def test_dataset_exists_checks_both_sources(self, tmp_path):
        """
        ماذا يختبر: dataset_exists() يتحقق من ir_datasets والملفات المحلية.
        ما الخطأ الذي يمنعه: loader يُرجع False لـ dataset موجود.
        """
        mock_adapter = MagicMock(spec=IrDatasetsAdapter)
        mock_adapter.is_supported.side_effect = lambda name: name == "msmarco"

        loader = DatasetLoader(datasets_dir=str(tmp_path), ir_adapter=mock_adapter)

        assert loader.dataset_exists("msmarco") is True       # موجود في ir_datasets
        assert loader.dataset_exists("unknown_xyz") is False  # غير موجود في أي مكان