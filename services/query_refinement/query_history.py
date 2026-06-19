"""
services/query_refinement/query_history.py
===========================================
سجل الاستعلامات — الجزء الثالث من Query Refinement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا نحتاج سجل الاستعلامات؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
من متطلبات المشروع:
"تتقيل استعلام المستخدم من سجله السابق"

يعني: نحتفظ بتاريخ بحث المستخدم حتى:
    1. نقترح عليه استعلامات سبق أن بحث عنها
    2. نُحسّن استعلامه الحالي بناءً على تاريخه
    3. نعرض "استعلامات حديثة" في الواجهة

مثال حقيقي (مثل Google):
    بحثت أمس عن: "machine learning algorithms"
    اليوم تكتب: "machine..."
    Google يقترح: "machine learning algorithms" ← من سجلك!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
التخزين:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
نستخدم ملف JSON بسيط لتخزين السجل.
لكل جلسة (session_id) نحفظ قائمة استعلاماتها.

البنية:
{
    "session_abc123": [
        {
            "query": "machine learning",
            "timestamp": "2026-06-01T10:30:00",
            "model": "bm25",
            "results_count": 5
        },
        ...
    ]
}

لماذا JSON وليس قاعدة بيانات؟
لأن هذا مشروع أكاديمي — JSON كافٍ وبسيط.
في الإنتاج الحقيقي نستخدم Redis أو PostgreSQL.
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── مسار ملف السجل ──────────────────────────────────────────
# نحفظه في data/ مثل باقي ملفات النظام
_HISTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "query_history",
)


class QueryHistoryEntry:
    """
    سجل استعلام واحد.
    يحتوي على الاستعلام + وقته + معلومات إضافية.
    """

    def __init__(
        self,
        query: str,
        model: str = "bm25",
        dataset: str = "dataset1",
        results_count: int = 0,
        timestamp: Optional[str] = None,
    ) -> None:
        self.query = query
        self.model = model
        self.dataset = dataset
        self.results_count = results_count
        # نحفظ الوقت بصيغة ISO 8601: "2026-06-01T10:30:00"
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """تحويل للتخزين في JSON."""
        return {
            "query": self.query,
            "model": self.model,
            "dataset": self.dataset,
            "results_count": self.results_count,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QueryHistoryEntry":
        """استعادة من JSON."""
        return cls(
            query=data.get("query", ""),
            model=data.get("model", "bm25"),
            dataset=data.get("dataset", "dataset1"),
            results_count=data.get("results_count", 0),
            timestamp=data.get("timestamp"),
        )


class QueryHistory:
    """
    يُدير سجل استعلامات المستخدم.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    طريقة العمل:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - كل مستخدم له session_id فريد
    - عند كل بحث: نُسجّل الاستعلام
    - عند طلب الاقتراحات: نجلب الاستعلامات السابقة

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    الحد الأقصى للسجل:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    نحتفظ بآخر 100 استعلام فقط لكل جلسة.
    القديمة تُحذف تلقائياً (FIFO: First In First Out).
    """

    MAX_HISTORY_PER_SESSION = 100  # أقصى عدد استعلامات محفوظة لكل جلسة

    def __init__(self, history_dir: str = _HISTORY_DIR) -> None:
        self.history_dir = Path(history_dir)
        # نُنشئ المجلد إذا لم يكن موجوداً
        self.history_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[QueryHistory] مجلد السجل: {self.history_dir}")

    # ─────────────────────────────────────────────────────────
    # الدوال الرئيسية
    # ─────────────────────────────────────────────────────────

    def add(
        self,
        session_id: str,
        query: str,
        model: str = "bm25",
        dataset: str = "dataset1",
        results_count: int = 0,
    ) -> None:
        """
        يُضيف استعلاماً جديداً لسجل الجلسة.

        المعاملات:
            session_id:    معرّف الجلسة (مثل "user_abc123")
            query:         الاستعلام المُنفَّذ
            model:         النموذج المُستخدم (bm25, tfidf...)
            dataset:       مجموعة البيانات
            results_count: عدد النتائج التي أُرجعت

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        مثال:
        add("user_123", "machine learning", "bm25", "dataset1", 5)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        if not query.strip():
            return

        # تحميل السجل الحالي
        history = self._load_session(session_id)

        # إنشاء إدخال جديد
        entry = QueryHistoryEntry(
            query=query.strip(),
            model=model,
            dataset=dataset,
            results_count=results_count,
        )

        # إضافة في البداية (الأحدث أولاً)
        history.insert(0, entry.to_dict())

        # حذف القديم إذا تجاوز الحد
        if len(history) > self.MAX_HISTORY_PER_SESSION:
            history = history[: self.MAX_HISTORY_PER_SESSION]

        # حفظ
        self._save_session(session_id, history)
        logger.debug(f"[QueryHistory] سُجِّل: {query!r} للجلسة {session_id}")

    def get_recent(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[QueryHistoryEntry]:
        """
        يُرجع آخر N استعلام للجلسة.

        المعاملات:
            session_id: معرّف الجلسة
            limit:      أقصى عدد يُرجَع (افتراضي 10)

        الإرجاع:
            List[QueryHistoryEntry] — الأحدث أولاً

        مثال:
            get_recent("user_123", 3)
            → [
                QueryHistoryEntry("information retrieval"),
                QueryHistoryEntry("machine learning"),
                QueryHistoryEntry("cloud storage"),
              ]
        """
        history_dicts = self._load_session(session_id)[:limit]
        return [QueryHistoryEntry.from_dict(d) for d in history_dicts]

    def get_similar_past_queries(
        self,
        session_id: str,
        current_query: str,
        limit: int = 5,
    ) -> List[str]:
        """
        يجد استعلامات سابقة مشابهة للاستعلام الحالي.

        كيف يعمل؟
        ━━━━━━━━━━
        يبحث عن استعلامات سابقة تشترك في كلمات مع الاستعلام الحالي.

        مثال:
            الاستعلام الحالي: "machine learning"
            السجل السابق:
                - "deep learning algorithms"   ← تشترك: "learning" ✓
                - "machine vision"             ← تشترك: "machine" ✓
                - "cloud storage"              ← لا تشارك ✗

            يُرجع: ["deep learning algorithms", "machine vision"]

        يُستخدم في:
            - اقتراح استعلامات للمستخدم
            - تحسين الاستعلام الحالي بمعلومات من التاريخ
        """
        if not current_query.strip():
            return []

        history = self._load_session(session_id)
        current_words = set(current_query.lower().split())

        # نحسب درجة تشابه بسيطة (عدد الكلمات المشتركة)
        scored_queries: List[tuple] = []
        for entry_dict in history:
            past_query = entry_dict.get("query", "")
            if not past_query or past_query.lower() == current_query.lower():
                continue

            past_words = set(past_query.lower().split())
            shared = len(current_words & past_words)

            if shared > 0:
                scored_queries.append((past_query, shared))

        # ترتيب حسب التشابه (الأكثر تشابهاً أولاً)
        scored_queries.sort(key=lambda x: x[1], reverse=True)

        return [q for q, _ in scored_queries[:limit]]

    def clear_session(self, session_id: str) -> None:
        """
        يحذف سجل جلسة معينة.
        مفيد لميزة "مسح السجل" في الواجهة.
        """
        file_path = self._get_session_file(session_id)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"[QueryHistory] تم مسح سجل الجلسة: {session_id}")

    def get_all_queries_for_session(
        self,
        session_id: str,
    ) -> List[str]:
        """
        يُرجع كل الاستعلامات الفريدة لجلسة (بدون تكرار).
        مفيد لعرض "تاريخ البحث" في الواجهة.
        """
        history = self._load_session(session_id)
        seen: set = set()
        unique_queries: List[str] = []

        for entry_dict in history:
            q = entry_dict.get("query", "")
            if q and q not in seen:
                unique_queries.append(q)
                seen.add(q)

        return unique_queries

    # ─────────────────────────────────────────────────────────
    # دوال مساعدة خاصة للتخزين
    # ─────────────────────────────────────────────────────────

    def cleanup_old_sessions(self, max_files: int = 500) -> int:
        """
        يحذف ملفات الجلسات القديمة إذا تجاوز عددها الحد الأقصى.

        لماذا نحتاج هذا؟
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        مع مرور الوقت واستخدام النظام، يتراكم آلاف ملفات JSON
        في مجلد data/query_history/ (ملف لكل جلسة).
        هذا يملأ القرص الصلب تدريجياً!

        الحل: نحتفظ بأحدث max_files ملف فقط.
        الأقدم يُحذف تلقائياً (FIFO: الأقدم يخرج أولاً).

        المعاملات:
            max_files: أقصى عدد ملفات مسموح به (افتراضي 500)

        الإرجاع:
            عدد الملفات التي حُذفت

        مثال:
            لو عندنا 600 ملف و max_files=500 → نحذف 100 ملف قديم
        """
        try:
            # نجمع كل ملفات JSON مع تاريخ آخر تعديل
            json_files = sorted(
                self.history_dir.glob("*.json"),
                key=lambda f: f.stat().st_mtime,  # نرتب حسب تاريخ التعديل
                reverse=False,  # الأقدم أولاً
            )

            deleted = 0
            if len(json_files) > max_files:
                files_to_delete = json_files[: len(json_files) - max_files]
                for f in files_to_delete:
                    f.unlink()
                    deleted += 1
                logger.info(
                    f"[QueryHistory] Cleanup: حذف {deleted} ملف قديم "
                    f"(تبقّى {len(json_files) - deleted})"
                )
            return deleted

        except Exception as e:
            logger.error(f"[QueryHistory] خطأ في Cleanup: {e}")
            return 0

    def get_session_count(self) -> int:
        """يُرجع عدد الجلسات المحفوظة حالياً. مفيد للـ /health endpoint."""
        try:
            return len(list(self.history_dir.glob("*.json")))
        except Exception:
            return 0

    def _get_session_file(self, session_id: str) -> Path:
        """يُرجع مسار ملف JSON للجلسة."""
        # نُنظّف session_id من أي أحرف غير آمنة
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.history_dir / f"{safe_id}.json"

    def _load_session(self, session_id: str) -> List[Dict]:
        """يُحمّل سجل الجلسة من ملف JSON."""
        file_path = self._get_session_file(session_id)
        if not file_path.exists():
            return []  # جلسة جديدة — سجل فارغ

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"[QueryHistory] خطأ في تحميل {file_path}: {e}")
            return []

    def _save_session(self, session_id: str, history: List[Dict]) -> None:
        """يحفظ سجل الجلسة في ملف JSON."""
        file_path = self._get_session_file(session_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"[QueryHistory] خطأ في الحفظ {file_path}: {e}")


# =============================================================
# Singleton Pattern
# =============================================================

_query_history_instance: Optional[QueryHistory] = None


def get_query_history() -> QueryHistory:
    """يُرجع النسخة الوحيدة من QueryHistory (Singleton)."""
    global _query_history_instance
    if _query_history_instance is None:
        _query_history_instance = QueryHistory()
    return _query_history_instance
