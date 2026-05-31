"""
services/indexing/dataset_loader.py
====================================
مسؤول عن تحميل مجموعات البيانات (datasets) من القرص.

مبدأ التصميم: كل فهرس (TF-IDF, BM25, Embedding) يستخدم
هذا الملف لتحميل البيانات. لا أحد يكتب كود تحميل بيانات
في مكان آخر.

الصيغ المدعومة:
  - JSONL : كل سطر = {"id":"..","title":"..","text":".."}
  - TSV   : doc_id  title  text  (بدون header, tab-separated)
  - CSV   : doc_id, title, text  (مع header)

⚠️ خطر هندسي مهم:
  إذا حمّلت 200,000 وثيقة دفعةً واحدة في RAM ستحتاج ~2GB.
  لهذا السبب نوفر stream_documents() التي تقرأ سطراً بسطر.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional

from shared.constants import DATASETS_DIR


# =============================================================
# نموذج الوثيقة الأساسي
# =============================================================

@dataclass
class Document:
    """
    وثيقة واحدة في مجموعة البيانات.

    dataclass بدلاً من dict لأنها:
    - أوضح (تعرف حقولها مسبقاً)
    - أسرع في الوصول
    - تدعم type hints
    """
    doc_id: str
    text: str
    title: Optional[str] = None

    def get_full_text(self) -> str:
        """
        يُرجع النص الكامل للفهرسة.
        ندمج العنوان مع النص لأن العنوان غالباً يحمل كلمات مفتاحية مهمة.

        مثال:
            title = "Cloud Storage Guide"
            text  = "Cloud storage allows syncing..."
            → "Cloud Storage Guide Cloud storage allows syncing..."

        لماذا ندمج العنوان مرتين؟ في IR يُسمّى هذا "title boosting"
        — نعطي العنوان وزناً أكبر بتكراره في النص.
        """
        if self.title:
            return f"{self.title} {self.text}"
        return self.text


# =============================================================
# DatasetLoader الرئيسي
# =============================================================

class DatasetLoader:
    """
    يحمّل مجموعات البيانات بصيغ مختلفة.

    الاستخدام:
        loader = DatasetLoader()

        # تحميل كل شيء في الذاكرة (للـ datasets الصغيرة)
        docs = loader.load_all(dataset_name="dataset1")

        # قراءة تدفقية (للـ datasets الكبيرة 200k+)
        for doc in loader.stream_documents(dataset_name="dataset1"):
            process(doc)
    """

    # الامتدادات المدعومة وطريقة قراءتها
    _SUPPORTED_FORMATS = {
        ".jsonl": "_load_jsonl",
        ".json":  "_load_jsonl",   # بعض الملفات JSONL لكن بامتداد .json
        ".tsv":   "_load_tsv",
        ".csv":   "_load_csv",
        ".txt":   "_load_txt",
    }

    def __init__(self, datasets_dir: str = DATASETS_DIR) -> None:
        self.datasets_dir = Path(datasets_dir)

    # ----------------------------------------------------------
    # الدوال العامة (Public Interface)
    # ----------------------------------------------------------

    def load_all(
        self,
        dataset_name: str,
        max_docs: Optional[int] = None,
    ) -> List[Document]:
        """
        يحمّل كل وثائق مجموعة البيانات في قائمة.

        ⚠️ استخدم هذه الدالة فقط إذا كان حجم البيانات معقولاً (<50k وثيقة)
           أو عند الاختبار. للبيانات الكبيرة استخدم stream_documents().

        المعاملات:
            dataset_name : اسم المجلد داخل data/datasets/
            max_docs     : الحد الأقصى للتحميل (None = كل شيء)

        الإرجاع:
            List[Document]
        """
        docs = []
        for doc in self.stream_documents(dataset_name):
            docs.append(doc)
            if max_docs and len(docs) >= max_docs:
                break
        print(f"[DatasetLoader] حُمِّل {len(docs):,} وثيقة من '{dataset_name}'")
        return docs

    def stream_documents(
        self,
        dataset_name: str,
    ) -> Generator[Document, None, None]:
        """
        يقرأ الوثائق سطراً بسطر بدون تحميلها كلها في الذاكرة.

        هذا هو الأسلوب الصحيح لـ 200,000+ وثيقة.
        كل مرة تستدعيه يُرجع وثيقة واحدة ثم تتوقف حتى تطلب التالية.

        مثال:
            for doc in loader.stream_documents("dataset1"):
                index.add(doc)    # يُضاف مباشرة دون تخزين الكل
        """
        dataset_path = self._find_dataset_path(dataset_name)

        if dataset_path is None:
            raise FileNotFoundError(
                f"مجموعة البيانات '{dataset_name}' غير موجودة في: "
                f"{self.datasets_dir}\n"
                f"تأكد من وضع الملف في: data/datasets/{dataset_name}/"
            )

        suffix = dataset_path.suffix.lower()
        loader_method_name = self._SUPPORTED_FORMATS.get(suffix)

        if loader_method_name is None:
            raise ValueError(
                f"صيغة الملف '{suffix}' غير مدعومة. "
                f"الصيغ المدعومة: {list(self._SUPPORTED_FORMATS.keys())}"
            )

        loader_method = getattr(self, loader_method_name)
        yield from loader_method(dataset_path)

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        يُرجع معلومات عن مجموعة البيانات (بدون تحميلها كلها).
        مفيد للـ UI ولعرض حالة النظام.
        """
        dataset_path = self._find_dataset_path(dataset_name)
        if dataset_path is None:
            return {"exists": False, "name": dataset_name}

        file_size_mb = dataset_path.stat().st_size / (1024 * 1024)

        return {
            "exists": True,
            "name": dataset_name,
            "path": str(dataset_path),
            "format": dataset_path.suffix,
            "size_mb": round(file_size_mb, 2),
        }

    # ----------------------------------------------------------
    # دوال القراءة الخاصة
    # ----------------------------------------------------------

    def _find_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """
        يبحث عن ملف مجموعة البيانات بأي امتداد مدعوم.

        يبحث في:
        1. data/datasets/{dataset_name}.jsonl
        2. data/datasets/{dataset_name}/{dataset_name}.jsonl
        3. data/datasets/{dataset_name}/corpus.jsonl
        4. وهكذا لكل الامتدادات المدعومة
        """
        # البحث في المجلد مباشرة
        for ext in self._SUPPORTED_FORMATS:
            direct = self.datasets_dir / f"{dataset_name}{ext}"
            if direct.exists():
                return direct

        # البحث داخل مجلد باسم مجموعة البيانات
        dataset_dir = self.datasets_dir / dataset_name
        if dataset_dir.is_dir():
            # الأسماء القياسية للملفات
            for filename in ["corpus", dataset_name, "documents", "data"]:
                for ext in self._SUPPORTED_FORMATS:
                    candidate = dataset_dir / f"{filename}{ext}"
                    if candidate.exists():
                        return candidate

            # أي ملف بامتداد مدعوم داخل المجلد
            for ext in self._SUPPORTED_FORMATS:
                matches = list(dataset_dir.glob(f"*{ext}"))
                if matches:
                    return matches[0]

        return None

    def _load_jsonl(self, path: Path) -> Iterator[Document]:
        """
        يقرأ ملف JSONL — كل سطر JSON مستقل.

        الصيغ المقبولة:
            {"id": "1", "text": "...", "title": "..."}
            {"_id": "1", "text": "...", "title": "..."}   ← صيغة BEIR
            {"docid": "1", "passage": "...", "query": "..."}
        """
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[DatasetLoader] تحذير: سطر {line_num} غير صالح: {e}")
                    continue

                doc_id = (
                    obj.get("id")
                    or obj.get("_id")
                    or obj.get("docid")
                    or str(line_num)  # fallback: رقم السطر كـ ID
                )
                text = (
                    obj.get("text")
                    or obj.get("passage")
                    or obj.get("contents")
                    or obj.get("body")
                    or ""
                )
                title = obj.get("title") or obj.get("heading")

                if not text.strip():
                    continue  # نتجاوز الوثائق الفارغة

                yield Document(
                    doc_id=str(doc_id),
                    text=text,
                    title=title,
                )

    def _load_tsv(self, path: Path) -> Iterator[Document]:
        """
        يقرأ ملف TSV بالصيغة: doc_id \\t text
        أو: doc_id \\t title \\t text
        """
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num, row in enumerate(reader, start=1):
                if not row or not any(row):
                    continue
                if len(row) == 2:
                    doc_id, text = row[0], row[1]
                    title = None
                elif len(row) >= 3:
                    doc_id, title, text = row[0], row[1], row[2]
                else:
                    continue

                if not text.strip():
                    continue

                yield Document(
                    doc_id=str(doc_id),
                    text=text,
                    title=title or None,
                )

    def _load_csv(self, path: Path) -> Iterator[Document]:
        """
        يقرأ ملف CSV مع header.
        يتوقع أعمدة: id/doc_id, text/passage, title (اختياري)
        """
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line_num, row in enumerate(reader, start=1):
                doc_id = (
                    row.get("id") or row.get("doc_id")
                    or row.get("docid") or str(line_num)
                )
                text = (
                    row.get("text") or row.get("passage")
                    or row.get("contents") or ""
                )
                title = row.get("title") or row.get("heading")

                if not text.strip():
                    continue

                yield Document(
                    doc_id=str(doc_id),
                    text=text,
                    title=title or None,
                )

    def _load_txt(self, path: Path) -> Iterator[Document]:
        """
        يقرأ ملف نصي بسيط — كل سطر = وثيقة.
        يُستخدم عادةً للاختبار السريع.
        """
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                yield Document(
                    doc_id=str(line_num),
                    text=text,
                    title=None,
                )


# =============================================================
# Singleton
# =============================================================

_loader_instance: Optional[DatasetLoader] = None


def get_dataset_loader() -> DatasetLoader:
    """
    يُرجع النسخة الوحيدة من DatasetLoader.
    يُستخدم كـ Dependency في FastAPI.
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DatasetLoader()
    return _loader_instance
