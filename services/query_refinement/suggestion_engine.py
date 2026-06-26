"""
services/query_refinement/suggestion_engine.py
===============================================
محرك الاقتراحات — الجزء الرابع من Query Refinement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما هو Suggestion Engine؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
من متطلبات المشروع:
"query formulation assistance, query suggestion"

هو النظام الذي يقترح عليك استعلامات بينما تكتب.
مثل ما تراه في Google عندما تكتب "machine..."
ويقترح: "machine learning", "machine vision"...

المصادر التي نقترح منها:
    1. السجل الشخصي للمستخدم (query_history)
    2. الاستعلامات الشائعة في النظام
    3. اقتراحات مبنية على Prefix (ما بدأت تكتبه)
    4. اقتراحات مبنية على المرادفات

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الفرق بين Suggestion و Expansion:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expansion:  يُعدّل الاستعلام الحالي (يضيف كلمات)
            "car" → "car automobile vehicle"

Suggestion: يقترح استعلامات بديلة كاملة
            "car" → اقتراح 1: "car purchase guide"
                    اقتراح 2: "car maintenance tips"
                    اقتراح 3: "best car brands 2026"
"""

import logging
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# استيراد المكونات الأخرى من Query Refinement
from services.query_refinement.query_history import QueryHistory, get_query_history
from services.query_refinement.synonym_expander import (
    SynonymExpander,
    get_synonym_expander,
)


class SuggestionEngine:
    """
    يُنتج اقتراحات لاستعلامات بحث.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    المصادر (بالأولوية):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. PREFIX MATCHING (أعلى أولوية):
       استعلامات من السجل تبدأ بنفس ما كتبه المستخدم
       مثال: كتب "mach..." → يقترح "machine learning" من سجله

    2. SIMILAR PAST QUERIES:
       استعلامات سابقة تشارك كلمات مع الاستعلام الحالي
       مثال: يكتب "learning" → يقترح "deep learning" من سجله

    3. SYNONYM-BASED SUGGESTIONS:
       اقتراحات مبنية على مرادفات الكلمات
       مثال: يكتب "fast" → يقترح "rapid search", "quick retrieval"

    4. POPULAR QUERIES (أدنى أولوية):
       قائمة ثابتة من الاستعلامات الشائعة في IR
    """

    # اقتراحات افتراضية شائعة في مجال IR
    # تُستخدم عندما لا يوجد سجل للمستخدم
    _DEFAULT_SUGGESTIONS = [
        "information retrieval systems",
        "machine learning algorithms",
        "natural language processing",
        "text classification methods",
        "document ranking algorithms",
        "inverted index construction",
        "cosine similarity calculation",
        "BM25 ranking function",
        "neural network embeddings",
        "vector space model",
        "query expansion techniques",
        "relevance feedback methods",
        "precision and recall evaluation",
        "tf-idf weighting scheme",
        "semantic search approaches",
    ]

    def __init__(
        self,
        history: Optional[QueryHistory] = None,
        expander: Optional[SynonymExpander] = None,
        max_suggestions: int = 5,
    ) -> None:
        """
        المعاملات:
            history:         سجل الاستعلامات
            expander:        موسّع المرادفات
            max_suggestions: أقصى عدد اقتراحات تُرجَع
        """
        self._history = history or get_query_history()
        self._expander = expander or get_synonym_expander()
        self.max_suggestions = max_suggestions

    # ─────────────────────────────────────────────────────────
    # الدالة الرئيسية
    # ─────────────────────────────────────────────────────────

    def suggest(
        self,
        partial_query: str,
        session_id: str = "default",
        include_history: bool = True,
        include_synonyms: bool = True,
    ) -> List[str]:
        """
        يُنتج اقتراحات لاستعلام (أو استعلام جزئي).

        المعاملات:
            partial_query:    ما كتبه المستخدم حتى الآن
                              مثال: "mach" أو "machine learning"
            session_id:       معرّف الجلسة للبحث في السجل الشخصي
            include_history:  هل نشمل نتائج السجل الشخصي؟
            include_synonyms: هل نشمل اقتراحات المرادفات؟

        الإرجاع:
            List[str] — قائمة اقتراحات مرتبة حسب الصلة
                        أقصى max_suggestions اقتراح

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        مثال:
            suggest("machine", "user_123")
            → [
                "machine learning",          ← من السجل
                "machine learning algorithms", ← من السجل
                "machine vision",             ← من المرادفات
                "machine learning systems",   ← من الافتراضي
              ]
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        if not partial_query.strip():
            # إذا فارغ → نُرجع الشائعة مباشرة
            return self._DEFAULT_SUGGESTIONS[: self.max_suggestions]

        partial_lower = partial_query.lower().strip()
        suggestions: List[str] = []
        # الاستعانة بالـ Set هنا ضرورة هندسية لتفادي مشكلة التكرار، ولأن التحقق من وجود عنصر داخل الـ Set يستغرق زمناً ثابتاً قدره $O(1)$ مقارنة بالمصفوفات التي تستغرق $O(N)$
        seen: Set[str] = set()

        def add_suggestion(s: str) -> bool:
            """يُضيف اقتراحاً إذا لم يكن مكرراً."""
            s_clean = s.strip()
            if (
                s_clean
                and s_clean.lower() not in seen
                and s_clean.lower() != partial_lower
            ):
                suggestions.append(s_clean)
                seen.add(s_clean.lower())
                return True
            return False

        # ────────────────────────────────────────────
        # المصدر 1: Prefix Matching من السجل الشخصي
        # ────────────────────────────────────────────
        if include_history:
            prefix_matches = self._get_prefix_matches(
                session_id=session_id,
                prefix=partial_lower,
            )
            for match in prefix_matches:
                add_suggestion(match)
                if len(suggestions) >= self.max_suggestions:
                    return suggestions

        # ────────────────────────────────────────────
        # المصدر 2: استعلامات مشابهة من السجل
        # ────────────────────────────────────────────
        if include_history and len(suggestions) < self.max_suggestions:
            similar = self._history.get_similar_past_queries(
                session_id=session_id,
                current_query=partial_lower,
                limit=self.max_suggestions,
            )
            for s in similar:
                add_suggestion(s)
                if len(suggestions) >= self.max_suggestions:
                    return suggestions

        # ────────────────────────────────────────────
        # المصدر 3: اقتراحات مبنية على المرادفات
        # ────────────────────────────────────────────
        if include_synonyms and len(suggestions) < self.max_suggestions:
            synonym_suggestions = self._get_synonym_suggestions(partial_lower)
            for s in synonym_suggestions:
                add_suggestion(s)
                if len(suggestions) >= self.max_suggestions:
                    return suggestions

        # ────────────────────────────────────────────
        # المصدر 4: Prefix Matching من الاقتراحات الافتراضية
        # ────────────────────────────────────────────
        if len(suggestions) < self.max_suggestions:
            for default_q in self._DEFAULT_SUGGESTIONS:
                if default_q.lower().startswith(partial_lower):
                    add_suggestion(default_q)
                if len(suggestions) >= self.max_suggestions:
                    return suggestions

        # ────────────────────────────────────────────
        # المصدر 5: كلمات مشتركة من الاقتراحات الافتراضية
        # ────────────────────────────────────────────
        if len(suggestions) < self.max_suggestions:
            partial_words = set(partial_lower.split())
            for default_q in self._DEFAULT_SUGGESTIONS:
                default_words = set(default_q.lower().split())
                if partial_words & default_words:  # تقاطع غير فارغ
                    add_suggestion(default_q)
                if len(suggestions) >= self.max_suggestions:
                    return suggestions

        return suggestions

    def get_popular_queries(self, limit: int = 10) -> List[str]:
        """
        يُرجع قائمة الاستعلامات الشائعة.
        تُستخدم في الواجهة لعرض "اقتراحات بحث شائعة".
        """
        return self._DEFAULT_SUGGESTIONS[:limit]

    # ─────────────────────────────────────────────────────────
    # دوال مساعدة خاصة
    # ─────────────────────────────────────────────────────────

    def _get_prefix_matches(
        self,
        session_id: str,
        prefix: str,
    ) -> List[str]:
        """
        يجد استعلامات من السجل تبدأ بـ prefix.

        مثال:
            prefix = "machine"
            السجل: ["machine learning", "deep learning", "machine vision"]
            النتيجة: ["machine learning", "machine vision"]
        """
        recent = self._history.get_recent(session_id=session_id, limit=50)
        matches: List[str] = []

        for entry in recent:
            if entry.query.lower().startswith(prefix):
                matches.append(entry.query)

        return matches

    def _get_synonym_suggestions(self, query: str) -> List[str]:
        """
        يُنتج اقتراحات بدمج المرادفات مع الاستعلام.

        مثال:
            query = "fast retrieval"
            مرادفات "fast": quick, rapid, swift
            اقتراحات: "quick retrieval", "rapid retrieval"
        """
        if not self._expander.is_available:
            return []

        words = query.split()
        suggestions: List[str] = []

        for i, word in enumerate(words):
            # نجد مرادفات الكلمة
            synonyms = self._expander.get_synonyms_for_word(word)[:2]

            for syn in synonyms:
                # ننشئ استعلاماً جديداً بتبديل الكلمة بمرادفها
                new_words = words[:i] + [syn] + words[i + 1 :]
                new_query = " ".join(new_words)
                if new_query != query:
                    suggestions.append(new_query)

        return suggestions[:3]  # أقصى 3 اقتراحات من المرادفات


# =============================================================
# Singleton Pattern
# =============================================================

_suggestion_engine_instance: Optional[SuggestionEngine] = None


def get_suggestion_engine() -> SuggestionEngine:
    """يُرجع النسخة الوحيدة من SuggestionEngine (Singleton)."""
    global _suggestion_engine_instance
    if _suggestion_engine_instance is None:
        _suggestion_engine_instance = SuggestionEngine()
    return _suggestion_engine_instance
