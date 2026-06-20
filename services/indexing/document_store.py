"""
services/indexing/document_store.py
=====================================
SQLite-based Document Store لتخزين الوثائق الأصلية.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا هذا الملف موجود؟ (طلب المعلم)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

المعلم طلب:
  - تخزين Raw Documents داخل Database
  - جلب الوثيقة الأصلية بـ doc_id عند الاستعلام
  - عدم الاعتماد على JSON وقت الاستعلام

الحل: SQLite جدول واحد بسيط.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
بنية قاعدة البيانات
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  الملف: data/indexes/{dataset_name}/documents.db

  جدول واحد:
    doc_id   TEXT PRIMARY KEY  ← فريد، بحث O(log N)
    raw_text TEXT NOT NULL      ← النص الأصلي
    title    TEXT               ← العنوان (قد يكون فارغاً)
    metadata TEXT               ← JSON إضافي

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما الذي لا يلمسه هذا الملف؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✗ لا يلمس منطق TF-IDF
  ✗ لا يلمس منطق BM25
  ✗ لا يلمس منطق Embeddings
  ✗ لا يغيّر Ranking أو Scoring
  ✗ لا يغيّر واجهات الـ Retrievers
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional

# ─────────────────────────────────────────────────────────────
# ثوابت
# ─────────────────────────────────────────────────────────────

_DB_FILENAME = "documents.db"

# حجم الدفعة للإدخال الجماعي
# 500 = توازن بين سرعة الكتابة وحجم الذاكرة المستخدمة
_DEFAULT_BATCH_SIZE = 500


class DocumentStore:
    """
    SQLite-based store للوثائق الأصلية.

    الاستخدام في BUILD (Developer 1):
    ─────────────────────────────────
        store = DocumentStore("data/indexes", "msmarco-passage")
        store.add_batch(indexed_documents)

    الاستخدام في RETRIEVAL (Developer 2):
    ──────────────────────────────────────
        store = DocumentStore.open("data/indexes", "msmarco-passage")
        doc   = store.get("doc_id_123")
        text  = doc["raw_text"]
    """

    def __init__(self, indexes_dir: str, dataset_name: str) -> None:
        """
        يُنشئ أو يفتح DocumentStore.

        المعاملات:
            indexes_dir  : مجلد الفهارس (نفس مجلد BM25/Embedding)
            dataset_name : اسم مجموعة البيانات

        الملف الناتج: {indexes_dir}/{dataset_name}/documents.db
        """
        db_dir = Path(indexes_dir) / dataset_name
        db_dir.mkdir(parents=True, exist_ok=True)

        self._db_path      = db_dir / _DB_FILENAME
        self._dataset_name = dataset_name

        self._initialize_db()

    # ──────────────────────────────────────────────────────────
    # إنشاء الجدول
    # ──────────────────────────────────────────────────────────

    def _initialize_db(self) -> None:
        """
        ينشئ الجدول إذا لم يكن موجوداً.

        IF NOT EXISTS: آمن للاستدعاء أكثر من مرة.

        WAL (Write-Ahead Logging):
          يسمح بقراءات متزامنة أثناء الكتابة.
          مهم عند تعدد الـ workers في FastAPI.
        """
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id   TEXT PRIMARY KEY,
                    raw_text TEXT NOT NULL,
                    title    TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

    # ──────────────────────────────────────────────────────────
    # الاتصال
    # ──────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager للاتصال بـ SQLite.

        row_factory = sqlite3.Row:
          يجعل النتائج قابلة للوصول بالاسم:
          row["doc_id"] بدلاً من row[0]

        check_same_thread=False:
          ضروري مع FastAPI الذي يعمل بـ async threads.
        """
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ──────────────────────────────────────────────────────────
    # الكتابة
    # ──────────────────────────────────────────────────────────

    def add_batch(
        self,
        documents: list,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        يضيف وثائق بالجملة — الطريقة الأساسية للاستخدام.

        لماذا batch وليس واحدة واحدة؟
        ─────────────────────────────────
        200,000 وثيقة × insert واحد = 200,000 transaction
        كل transaction تكتب للقرص = بطيء جداً

        مع batch_size=500:
        400 transaction فقط = أسرع بـ ~100×

        يقبل:
          List[IndexedDocument]  (من tfidf_indexer.py)
          أو List[dict]

        الإرجاع:
          عدد الوثائق المُضافة
        """
        if not documents:
            return 0

        total_added   = 0
        current_batch: List[tuple] = []

        with self._connect() as conn:
            for doc in documents:
                # نستخرج الحقول بطريقة تعمل مع IndexedDocument أو dict
                if hasattr(doc, "doc_id"):
                    # IndexedDocument dataclass
                    doc_id   = doc.doc_id
                    raw_text = doc.original_text
                    title    = getattr(doc, "title", None)
                    meta     = None
                else:
                    # dict
                    doc_id   = doc["doc_id"]
                    raw_text = doc.get("raw_text") or doc.get("original_text", "")
                    title    = doc.get("title")
                    meta     = doc.get("metadata")

                # metadata → JSON string
                if isinstance(meta, dict):
                    meta = json.dumps(meta, ensure_ascii=False)

                current_batch.append((doc_id, raw_text, title, meta))

                if len(current_batch) >= batch_size:
                    self._insert_batch(conn, current_batch)
                    total_added   += len(current_batch)
                    current_batch  = []

            # الدفعة الأخيرة
            if current_batch:
                self._insert_batch(conn, current_batch)
                total_added += len(current_batch)

        return total_added

    @staticmethod
    def _insert_batch(conn: sqlite3.Connection, batch: List[tuple]) -> None:
        """
        يُدرج دفعة واحدة داخل transaction.

        INSERT OR REPLACE:
          إذا كان doc_id موجوداً → يُحدَّث بدل رفع خطأ.
          مفيد عند إعادة بناء الفهرس.
        """
        conn.executemany(
            """
            INSERT OR REPLACE INTO documents
            (doc_id, raw_text, title, metadata)
            VALUES (?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()

    def add(
        self,
        doc_id:   str,
        raw_text: str,
        title:    Optional[str]  = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        يضيف وثيقة واحدة.
        للاختبارات أو الحالات الفردية فقط.
        في البناء الفعلي: استخدم add_batch.
        """
        meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, raw_text, title, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, raw_text, title, meta_str),
            )
            conn.commit()

    # ──────────────────────────────────────────────────────────
    # القراءة
    # ──────────────────────────────────────────────────────────

    def get(self, doc_id: str) -> Optional[Dict]:
        """
        يجلب وثيقة بالـ ID.

        السرعة: O(log N) بفضل PRIMARY KEY.
        لـ 200,000 وثيقة: أقل من 1ms.

        الإرجاع:
          {"doc_id": ..., "raw_text": ..., "title": ..., "metadata": ...}
          أو None إذا لم توجد الوثيقة.

        مثال:
          doc = store.get("d1234")
          if doc:
              print(doc["raw_text"])
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT doc_id, raw_text, title, metadata "
                "FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            row = cursor.fetchone()

        return self._row_to_dict(row) if row else None

    def get_many(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """
        يجلب عدة وثائق دفعة واحدة — أسرع من get() في حلقة.

        متى تستخدمها؟
        بعد البحث: لديك 10 doc_ids من BM25/Embedding
        وتريد نصوصها كلها في طلب واحد لقاعدة البيانات.

        الإرجاع:
          {doc_id → document_dict}
          الوثائق غير الموجودة لا تظهر في النتيجة.
        """
        if not doc_ids:
            return {}

        placeholders = ",".join("?" * len(doc_ids))
        query = (
            f"SELECT doc_id, raw_text, title, metadata "
            f"FROM documents WHERE doc_id IN ({placeholders})"
        )

        with self._connect() as conn:
            cursor = conn.execute(query, doc_ids)
            rows   = cursor.fetchall()

        return {row["doc_id"]: self._row_to_dict(row) for row in rows}

    def exists(self, doc_id: str) -> bool:
        """هل الوثيقة موجودة؟ (بدون جلب محتواها — أسرع)"""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1",
                (doc_id,),
            )
            return cursor.fetchone() is not None

    # ──────────────────────────────────────────────────────────
    # الإحصائيات والحالة
    # ──────────────────────────────────────────────────────────

    def count(self) -> int:
        """عدد الوثائق المخزّنة."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]

    def is_populated(self) -> bool:
        """هل قاعدة البيانات تحتوي وثائق؟"""
        return self.count() > 0

    def get_status(self) -> Dict:
        """
        حالة قاعدة البيانات — مفيد لـ /health endpoints.

        مثال الإخراج:
        {
            "dataset_name": "msmarco-passage",
            "db_path": "data/indexes/msmarco-passage/documents.db",
            "num_documents": 6980,
            "db_size_mb": 45.3,
            "is_populated": True
        }
        """
        num_docs = self.count()
        db_size  = (
            self._db_path.stat().st_size / (1024 * 1024)
            if self._db_path.exists() else 0.0
        )
        return {
            "dataset_name":  self._dataset_name,
            "db_path":       str(self._db_path),
            "num_documents": num_docs,
            "db_size_mb":    round(db_size, 2),
            "is_populated":  num_docs > 0,
        }

    # ──────────────────────────────────────────────────────────
    # Class Methods
    # ──────────────────────────────────────────────────────────

    @classmethod
    def open(cls, indexes_dir: str, dataset_name: str) -> "DocumentStore":
        """
        يفتح DocumentStore موجود.

        نفس __init__ لكن باسم أوضح للقارئ:
        "أنا أفتح store موجود، لا أُنشئ جديداً"

        مثال (في Retrieval Service):
            store = DocumentStore.open("data/indexes", "msmarco-passage")
            doc   = store.get(doc_id)
        """
        return cls(indexes_dir=indexes_dir, dataset_name=dataset_name)

    @staticmethod
    def db_exists(indexes_dir: str, dataset_name: str) -> bool:
        """
        هل ملف DB موجود ومملوء؟

        استخدمها قبل open() للتحقق:
            if DocumentStore.db_exists("data/indexes", "msmarco"):
                store = DocumentStore.open(...)
            else:
                raise RuntimeError("شغّل build_index أولاً")
        """
        db_path = Path(indexes_dir) / dataset_name / _DB_FILENAME
        return db_path.exists() and db_path.stat().st_size > 0

    # ──────────────────────────────────────────────────────────
    # دوال مساعدة
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        """يحوّل sqlite3.Row → dict Python."""
        meta = row["metadata"]
        if meta and isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                pass  # نبقيه string إذا فشل التحليل
        return {
            "doc_id":   row["doc_id"],
            "raw_text": row["raw_text"],
            "title":    row["title"],
            "metadata": meta,
        }


# =============================================================
# Singleton Factory
# =============================================================

_store_instances: Dict[str, DocumentStore] = {}


def get_document_store(
    dataset_name: str,
    indexes_dir:  str = "data/indexes",
) -> DocumentStore:
    """
    يُرجع DocumentStore — نسخة واحدة لكل dataset (Singleton).

    مثال (Developer 2 — Retrieval Service):
        store = get_document_store("msmarco-passage")
        doc   = store.get(result.doc_id)
        text  = doc["raw_text"] if doc else ""
    """
    global _store_instances
    key = f"{dataset_name}::{indexes_dir}"

    if key not in _store_instances:
        _store_instances[key] = DocumentStore(
            indexes_dir=indexes_dir,
            dataset_name=dataset_name,
        )

    return _store_instances[key]