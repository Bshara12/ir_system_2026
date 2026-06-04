"""
services/indexing/dataset_loader.py
====================================
مسؤول عن تحميل مجموعات البيانات من مصدرين:
  1. ملفات محلية  (JSONL / TSV / CSV / TXT)
  2. ir_datasets  (msmarco, trec-covid, ...)

مبدأ التصميم — Strategy Pattern:
═══════════════════════════════
DatasetLoader يحدد المصدر تلقائياً:
  - إذا الاسم موجود في SUPPORTED_DATASETS → يستخدم IrDatasetsAdapter
  - إذا الاسم موجود كملف محلي           → يقرأه مباشرة
  - إذا لم يجد شيئاً                    → FileNotFoundError واضح

التدفق الكامل:
══════════════
  loader = DatasetLoader()

  # تحميل من ir_datasets (يُنزّل تلقائياً إذا لم يكن موجوداً)
  for doc in loader.stream_documents("msmarco"):
      tfidf_indexer.add(doc)

  # تحميل الاستعلامات (لـ Dev2 - Retrieval Service)
  queries = loader.load_queries("msmarco")

  # تحميل الـ Qrels (لـ Dev3 - Evaluation Service)
  qrels = loader.load_qrels("trec-covid")

  # تحميل من ملف محلي (كما كان سابقاً)
  for doc in loader.stream_documents("my_custom_dataset"):
      bm25_indexer.add(doc)
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Any

from shared.constants import DATASETS_DIR

# نستورد من ir_datasets_adapter لإعادة استخدام نماذج البيانات
from services.indexing.ir_datasets_adapter import (
    Document,
    Query,
    Qrel,
    IrDatasetsAdapter,
    SUPPORTED_DATASETS,
    get_ir_datasets_adapter,
)

logger = logging.getLogger(__name__)


# =============================================================
# DatasetLoader الرئيسي (المُحسَّن)
# =============================================================

class DatasetLoader:
    """
    يحمّل مجموعات البيانات من مصدرين: ir_datasets أو ملفات محلية.

    لماذا دمجنا المصدرين في loader واحد؟
    ════════════════════════════════════
    أي خدمة أخرى (Retrieval, Evaluation) تحتاج فقط أن تقول:
        loader.stream_documents("msmarco")
    بدون أن تعرف هل البيانات في ir_datasets أو ملف محلي.
    هذا هو مبدأ Loose Coupling.

    الاستخدام:
        loader = DatasetLoader()

        # وثائق (للفهرسة)
        for doc in loader.stream_documents("msmarco", max_docs=10_000):
            tfidf_indexer.add_document(doc)

        # استعلامات (لـ Retrieval Service)
        for query in loader.stream_queries("trec-covid"):
            results = retrieval_service.search(query.text)

        # qrels (لـ Evaluation Service)
        for qrel in loader.stream_qrels("trec-covid"):
            evaluator.add_qrel(qrel)
    """

    # الامتدادات المدعومة لملفات الـ corpus المحلية
    _SUPPORTED_FORMATS = {
        ".jsonl": "_load_jsonl",
        ".json":  "_load_jsonl",
        ".tsv":   "_load_tsv",
        ".csv":   "_load_csv",
        ".txt":   "_load_txt",
    }

    def __init__(
        self,
        datasets_dir: str = DATASETS_DIR,
        ir_adapter: Optional[IrDatasetsAdapter] = None,
    ) -> None:
        """
        datasets_dir: مجلد ملفات الـ datasets المحلية
        ir_adapter:   للاختبار يمكن حقن adapter مختلف
        """
        self.datasets_dir = Path(datasets_dir)
        # استخدام الـ Singleton إذا لم يُحقن adapter خاص
        self._ir_adapter = ir_adapter or get_ir_datasets_adapter()

    # ----------------------------------------------------------
    # تحديد مصدر الـ Dataset
    # ----------------------------------------------------------

    def _is_ir_dataset(self, dataset_name: str) -> bool:
        """هل هذا الاسم مدعوم في ir_datasets registry؟"""
        return self._ir_adapter.is_supported(dataset_name)

    def _is_local_dataset(self, dataset_name: str) -> bool:
        """هل يوجد ملف محلي بهذا الاسم؟"""
        return self._find_corpus_path(dataset_name) is not None

    def dataset_exists(self, dataset_name: str) -> bool:
        """هل هذا الـ dataset موجود في أي مصدر؟"""
        return self._is_ir_dataset(dataset_name) or self._is_local_dataset(dataset_name)

    # ----------------------------------------------------------
    # واجهة الوثائق (Documents)
    # ----------------------------------------------------------

    def stream_documents(
        self,
        dataset_name: str,
        max_docs: Optional[int] = None,
    ) -> Generator[Document, None, None]:
        """
        يقرأ وثائق الـ dataset تدفقياً (سطر بسطر).

        يحدد المصدر تلقائياً:
          - ir_datasets (msmarco, trec-covid, ...) ← يُنزّل إذا لزم
          - ملف محلي (corpus.jsonl, data.tsv, ...) ← يقرأ مباشرة

        لماذا Lazy Loading ضروري؟
        ══════════════════════════
        msmarco: 8.8M وثيقة × ~200 حرف = ~1.8 GB نصوص
        لو حمّلنا كل شيء في RAM = الذاكرة تمتلئ وتتوقف العملية.
        بالتدفق: نقرأ وثيقة → نفهرسها → نتجاوزها، الذاكرة تبقى ~50MB.

        مثال:
            for doc in loader.stream_documents("trec-covid"):
                bm25_indexer.add_document(doc.get_full_text(), doc.doc_id)
        """
        if self._is_ir_dataset(dataset_name):
            # ── المسار 1: ir_datasets ──────────────────────
            logger.info(f"[DatasetLoader] تحميل '{dataset_name}' من ir_datasets")
            yield from self._ir_adapter.stream_documents(dataset_name, max_docs=max_docs)

        elif self._is_local_dataset(dataset_name):
            # ── المسار 2: ملف محلي ─────────────────────────
            logger.info(f"[DatasetLoader] تحميل '{dataset_name}' من ملف محلي")
            corpus_path = self._find_corpus_path(dataset_name)
            count = 0
            for doc in self._load_file(corpus_path):
                yield doc
                count += 1
                if max_docs and count >= max_docs:
                    break

        else:
            raise FileNotFoundError(
                f"مجموعة البيانات '{dataset_name}' غير موجودة.\n"
                f"الـ datasets المدعومة في ir_datasets: "
                f"{self._ir_adapter.list_supported_datasets()}\n"
                f"أو ضع ملفاً في: {self.datasets_dir / dataset_name}/"
            )

    def load_all(
        self,
        dataset_name: str,
        max_docs: Optional[int] = None,
    ) -> List[Document]:
        """
        يحمّل كل وثائق مجموعة البيانات في قائمة.

        ⚠️ تحذير: للـ datasets الكبيرة (200K+) استخدم stream_documents()
           هذه الدالة مناسبة فقط للاختبار أو الـ datasets الصغيرة.

        مثال:
            docs = loader.load_all("trec-covid", max_docs=1000)
            print(f"حُمِّل {len(docs)} وثيقة")
        """
        docs = list(self.stream_documents(dataset_name, max_docs=max_docs))
        logger.info(f"[DatasetLoader] حُمِّل {len(docs):,} وثيقة من '{dataset_name}'")
        return docs

    # ----------------------------------------------------------
    # واجهة الاستعلامات (Queries) — جديد
    # ----------------------------------------------------------

    def stream_queries(
        self,
        dataset_name: str,
    ) -> Generator[Query, None, None]:
        """
        يقرأ الاستعلامات التدفقياً.

        من يستخدم هذه الدالة؟
        ═════════════════════
        Dev2 - Retrieval Service:
            for query in loader.stream_queries("msmarco"):
                results = retrieval_service.search(query.text, top_k=10)
                save_results(query.query_id, results)

        Dev3 - Evaluation Service:
            for query in loader.stream_queries("trec-covid"):
                eval_results = evaluator.evaluate_query(query.query_id)
        """
        if self._is_ir_dataset(dataset_name):
            yield from self._ir_adapter.stream_queries(dataset_name)

        elif self._is_local_dataset(dataset_name):
            # ابحث عن ملف queries.jsonl محلي
            queries_path = self._find_queries_path(dataset_name)
            if queries_path is None:
                raise FileNotFoundError(
                    f"لا توجد استعلامات لـ '{dataset_name}'.\n"
                    f"ضع ملف queries.jsonl في: {self.datasets_dir / dataset_name}/"
                )
            yield from self._load_queries_jsonl(queries_path)

        else:
            raise FileNotFoundError(
                f"مجموعة البيانات '{dataset_name}' غير موجودة."
            )

    def load_queries(self, dataset_name: str) -> List[Query]:
        """
        يحمّل كل الاستعلامات في قائمة.
        الاستعلامات أعداد صغيرة (50 - 7000) → تحميل كامل آمن.
        """
        return list(self.stream_queries(dataset_name))

    # ----------------------------------------------------------
    # واجهة الـ Qrels — جديد
    # ----------------------------------------------------------

    def stream_qrels(
        self,
        dataset_name: str,
    ) -> Generator[Qrel, None, None]:
        """
        يقرأ الـ Qrels التدفقياً.

        ما هو الـ Qrel؟
        ═══════════════
        Qrel = Query Relevance Judgment
        يخبرنا: "الوثيقة X ذات صلة بالاستعلام Y بدرجة Z"

        مثال من trec-covid:
            query_id="1", doc_id="d1234", relevance=2
            → الوثيقة d1234 ذات صلة عالية بالاستعلام رقم 1

        Dev3 يستخدمها لحساب MAP, nDCG, Recall, P@10
        """
        if self._is_ir_dataset(dataset_name):
            yield from self._ir_adapter.stream_qrels(dataset_name)

        elif self._is_local_dataset(dataset_name):
            qrels_path = self._find_qrels_path(dataset_name)
            if qrels_path is None:
                raise FileNotFoundError(
                    f"لا توجد qrels لـ '{dataset_name}'.\n"
                    f"ضع ملف qrels.jsonl في: {self.datasets_dir / dataset_name}/"
                )
            yield from self._load_qrels_jsonl(qrels_path)

        else:
            raise FileNotFoundError(
                f"مجموعة البيانات '{dataset_name}' غير موجودة."
            )

    def load_qrels(self, dataset_name: str) -> List[Qrel]:
        """
        يحمّل كل الـ Qrels في قائمة.

        trec-covid: 69K qrels → حجم معقول في الذاكرة
        msmarco:    7.4K qrels → صغير جداً
        """
        return list(self.stream_qrels(dataset_name))

    # ----------------------------------------------------------
    # معلومات الـ Dataset
    # ----------------------------------------------------------

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        يُرجع معلومات كاملة عن مجموعة البيانات.
        يُستخدم في API /index/status ولوحة التحكم في الواجهة.
        """
        # ── معلومات من ir_datasets registry ─────────────────
        if self._is_ir_dataset(dataset_name):
            info = self._ir_adapter.get_dataset_info(dataset_name)
            local_path = self._find_corpus_path(dataset_name)
            return {
                "exists":        True,
                "source":        "ir_datasets",
                "name":          dataset_name,
                "friendly_name": info.get("friendly_name", dataset_name),
                "domain":        info.get("domain", "Unknown"),
                "doc_count":     info.get("doc_count", "Unknown"),
                "query_count":   info.get("query_count", "Unknown"),
                "qrel_count":    info.get("qrel_count", "Unknown"),
                "has_qrels":     info.get("has_qrels", False),
                "size_gb":       info.get("size_gb", "Unknown"),
                "local_cache":   str(local_path) if local_path else None,
                "notes":         info.get("notes", ""),
            }

        # ── معلومات من ملف محلي ──────────────────────────────
        local_path = self._find_corpus_path(dataset_name)
        if local_path:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            return {
                "exists":      True,
                "source":      "local_file",
                "name":        dataset_name,
                "path":        str(local_path),
                "format":      local_path.suffix,
                "size_mb":     round(size_mb, 2),
                "has_qrels":   self._find_qrels_path(dataset_name) is not None,
                "has_queries": self._find_queries_path(dataset_name) is not None,
            }

        return {"exists": False, "name": dataset_name}

    def list_available_datasets(self) -> Dict[str, List[str]]:
        """
        يُرجع كل الـ datasets المتاحة مقسّمة حسب المصدر.

        مثال:
            {
              "ir_datasets": ["msmarco", "trec-covid"],
              "local":       ["my_custom_dataset"]
            }
        """
        ir_datasets_list = self._ir_adapter.list_supported_datasets()

        local_datasets = []
        if self.datasets_dir.exists():
            for entry in self.datasets_dir.iterdir():
                if entry.is_dir() and entry.name not in ir_datasets_list:
                    if self._find_corpus_path(entry.name):
                        local_datasets.append(entry.name)
                elif entry.is_file() and entry.suffix in self._SUPPORTED_FORMATS:
                    name = entry.stem
                    if name not in ir_datasets_list:
                        local_datasets.append(name)

        return {
            "ir_datasets": ir_datasets_list,
            "local":       local_datasets,
        }

    # ----------------------------------------------------------
    # البحث عن الملفات المحلية (Private)
    # ----------------------------------------------------------

    def _find_corpus_path(self, dataset_name: str) -> Optional[Path]:
        """
        يبحث عن ملف corpus المحلي.
        يبحث في أسماء قياسية: corpus, documents, data, {dataset_name}
        """
        # بحث مباشر في datasets_dir
        for ext in self._SUPPORTED_FORMATS:
            direct = self.datasets_dir / f"{dataset_name}{ext}"
            if direct.exists():
                return direct

        # بحث داخل مجلد الـ dataset
        dataset_dir = self.datasets_dir / dataset_name
        if dataset_dir.is_dir():
            for filename in ["corpus", "documents", "data", dataset_name]:
                for ext in self._SUPPORTED_FORMATS:
                    candidate = dataset_dir / f"{filename}{ext}"
                    if candidate.exists():
                        return candidate
            # أي ملف بامتداد مدعوم
            for ext in self._SUPPORTED_FORMATS:
                matches = list(dataset_dir.glob(f"*{ext}"))
                if matches:
                    return matches[0]

        return None

    def _find_queries_path(self, dataset_name: str) -> Optional[Path]:
        """يبحث عن ملف queries.jsonl المحلي."""
        dataset_dir = self.datasets_dir / dataset_name
        if dataset_dir.is_dir():
            for filename in ["queries", "test_queries", "dev_queries"]:
                for ext in [".jsonl", ".json", ".tsv", ".csv"]:
                    candidate = dataset_dir / f"{filename}{ext}"
                    if candidate.exists():
                        return candidate
        return None

    def _find_qrels_path(self, dataset_name: str) -> Optional[Path]:
        """يبحث عن ملف qrels.jsonl المحلي."""
        dataset_dir = self.datasets_dir / dataset_name
        if dataset_dir.is_dir():
            for filename in ["qrels", "test_qrels", "relevance_judgments"]:
                for ext in [".jsonl", ".json", ".tsv", ".txt"]:
                    candidate = dataset_dir / f"{filename}{ext}"
                    if candidate.exists():
                        return candidate
        return None

    # ----------------------------------------------------------
    # قراءة الملفات المحلية (Private)
    # ----------------------------------------------------------

    def _load_file(self, path: Path) -> Iterator[Document]:
        """يختار طريقة القراءة المناسبة بناءً على امتداد الملف."""
        suffix = path.suffix.lower()
        method_name = self._SUPPORTED_FORMATS.get(suffix)
        if method_name is None:
            raise ValueError(f"امتداد غير مدعوم: {suffix}")
        method = getattr(self, method_name)
        yield from method(path)

    def _load_jsonl(self, path: Path) -> Iterator[Document]:
        """
        يقرأ ملف JSONL — كل سطر = JSON مستقل.

        الصيغ المقبولة:
          {"id": "1", "text": "...", "title": "..."}
          {"_id": "1", "text": "..."}    ← صيغة BEIR
          {"doc_id": "1", "body": "..."}
        """
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"[DatasetLoader] سطر {line_num} غير صالح: {e}")
                    continue

                doc_id = (
                    obj.get("id") or obj.get("_id") or
                    obj.get("doc_id") or obj.get("docid") or
                    str(line_num)
                )
                text = (
                    obj.get("text") or obj.get("passage") or
                    obj.get("contents") or obj.get("body") or ""
                )
                title = obj.get("title") or obj.get("heading")

                if not str(text).strip():
                    continue

                yield Document(
                    doc_id=str(doc_id),
                    text=str(text),
                    title=str(title) if title else None,
                )

    def _load_tsv(self, path: Path) -> Iterator[Document]:
        """
        يقرأ TSV: doc_id TAB text
                 أو: doc_id TAB title TAB text
        """
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num, row in enumerate(reader, start=1):
                if not row or not any(r.strip() for r in row):
                    continue
                if len(row) == 2:
                    doc_id, text, title = row[0], row[1], None
                elif len(row) >= 3:
                    doc_id, title, text = row[0], row[1], row[2]
                else:
                    continue
                if not str(text).strip():
                    continue
                yield Document(
                    doc_id=str(doc_id),
                    text=str(text),
                    title=str(title) if title else None,
                )

    def _load_csv(self, path: Path) -> Iterator[Document]:
        """يقرأ CSV مع header: id/doc_id, text, title (اختياري)"""
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line_num, row in enumerate(reader, start=1):
                doc_id = (
                    row.get("id") or row.get("_id") or
                    row.get("doc_id") or str(line_num)
                )
                text = (
                    row.get("text") or row.get("passage") or
                    row.get("contents") or ""
                )
                title = row.get("title") or row.get("heading")
                if not str(text).strip():
                    continue
                yield Document(
                    doc_id=str(doc_id),
                    text=str(text),
                    title=str(title) if title else None,
                )

    def _load_txt(self, path: Path) -> Iterator[Document]:
        """يقرأ TXT: كل سطر = وثيقة (للاختبار السريع)"""
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                yield Document(doc_id=str(line_num), text=text)

    def _load_queries_jsonl(self, path: Path) -> Iterator[Query]:
        """يقرأ استعلامات من ملف JSONL محلي."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query_id = str(
                    obj.get("query_id") or obj.get("id") or obj.get("_id") or ""
                )
                text = str(obj.get("text") or obj.get("query") or "")
                if query_id and text.strip():
                    yield Query(query_id=query_id, text=text)

    def _load_qrels_jsonl(self, path: Path) -> Iterator[Qrel]:
        """يقرأ qrels من ملف JSONL محلي."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query_id  = str(obj.get("query_id") or "")
                doc_id    = str(obj.get("doc_id") or "")
                relevance = int(obj.get("relevance") or 0)
                if query_id and doc_id:
                    yield Qrel(
                        query_id=query_id,
                        doc_id=doc_id,
                        relevance=relevance,
                    )


# =============================================================
# Singleton
# =============================================================

_loader_instance: Optional[DatasetLoader] = None


def get_dataset_loader() -> DatasetLoader:
    """
    يُرجع النسخة الوحيدة من DatasetLoader (Singleton).
    يُستخدم كـ Dependency في FastAPI وفي الـ Indexers.
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DatasetLoader()
    return _loader_instance