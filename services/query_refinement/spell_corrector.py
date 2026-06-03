"""
services/query_refinement/spell_corrector.py
=============================================
مُصحّح الإملاء — الجزء الأول من Query Refinement Service.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما هو Query Refinement؟ (من متطلبات المشروع)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
المشروع يطلب:
"تطبيق تحسينات على الاستعلامات لزيادة دقة النتائج مثل تتقيل
استعلام المستخدم من سجله السابق، أو إضافة مرادفات على استعلامه،
أو تصحيح الاستعلام لغوياً."

يعني: قبل أن نبحث، نُحسّن الاستعلام أولاً.
هذا الملف مسؤول عن: تصحيح الأخطاء الإملائية.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا تصحيح الإملاء مهم في IR؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
مثال حقيقي:
    المستخدم يكتب: "infromation retreival"
    الفهرس يحتوي: "information retrieval"

    بدون تصحيح: نتائج = صفر (لا تطابق!)
    بعد التصحيح: "information retrieval" → نتائج ممتازة

المكتبة المُستخدمة: pyspellchecker
    مذكورة في المحاضرات كأداة مقترحة.
    تستخدم "مسافة Edit Distance" لإيجاد أقرب كلمة صحيحة.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ما هي Edit Distance؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
هي عدد العمليات اللازمة لتحويل كلمة إلى أخرى:
    "infromation" → "information"
    العمليات: نقل حرفين (r و m)  = مسافة 2
    هذا قريب جداً → التصحيح صحيح!

    "cat" → "dog"
    العمليات: تغيير 3 أحرف = مسافة 3
    هذا بعيد → لا تصحيح
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ─── نحاول استيراد المكتبة ───────────────────────────────────
try:
    from spellchecker import SpellChecker

    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logger.warning(
        "[SpellCorrector] pyspellchecker غير مثبّت.\n"
        "  شغّل: pip install pyspellchecker"
    )


class SpellCorrector:
    """
    يُصحّح الأخطاء الإملائية في استعلامات المستخدم.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    كيف تعمل pyspellchecker؟
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. تحمّل قاموساً ضخماً من الكلمات الإنجليزية وترددات استخدامها
    2. لكل كلمة مدخلة، تحسب Edit Distance مع كلمات القاموس
    3. تختار الكلمة الأقرب التي لها أعلى تردد في اللغة

    مثال:
        "retreival" → تجد "retrieval" (مسافة 1) → تُصحَّح
        "information" → موجودة في القاموس → تُترك كما هي
        "BERT" → نُضيفها لقاموس مخصص → لا تُصحَّح

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    المصطلحات التقنية (لا تُصحَّح):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    مصطلحات IR مثل "bm25" و "tfidf" ليست في القاموس العادي.
    نُضيفها يدوياً حتى لا يُصحَّح "bm25" إلى شيء آخر!
    """

    # ─── مصطلحات تقنية لا يجب تصحيحها أبداً ───────────────
    TECHNICAL_TERMS = {
        # مصطلحات IR
        "tfidf",
        "bm25",
        "idf",
        "tf",
        "inverted",
        "retrieval",
        "indexing",
        "preprocessing",
        "tokenization",
        "lemmatization",
        "stemming",
        "stopwords",
        "cosine",
        "similarity",
        "embedding",
        "embeddings",
        "dataset",
        "corpus",
        "query",
        "queries",
        "ranking",
        "precision",
        "recall",
        "ndcg",
        "mrr",
        "map",
        # نماذج AI
        "bert",
        "gpt",
        "lstm",
        "rnn",
        "cnn",
        "nlp",
        "faiss",
        "sklearn",
        "numpy",
        "pandas",
        # أسماء مجموعات بيانات
        "trec",
        "msmarco",
        "beir",
        "covid",
        # مصطلحات أخرى
        "api",
        "url",
        "json",
        "http",
    }

    def __init__(self, language: str = "en") -> None:
        """
        ينشئ مُصحّح الإملاء.

        المعاملات:
            language: لغة القاموس
                "en" = الإنجليزية
                "ar" = العربية (إذا أردنا دعمها لاحقاً)
        """
        self.language = language
        self._checker: Optional[SpellChecker] = None

        if SPELLCHECKER_AVAILABLE:
            try:
                self._checker = SpellChecker(language=language)
                # نُضيف المصطلحات التقنية للقاموس
                # حتى لا تُصحَّح بشكل خاطئ
                self._checker.word_frequency.load_words(self.TECHNICAL_TERMS)
                logger.info(f"[SpellCorrector] جاهز. اللغة: {language}")
            except Exception as e:
                logger.error(f"[SpellCorrector] فشل الإنشاء: {e}")
                self._checker = None

    # ─────────────────────────────────────────────────────────
    # الدالة الرئيسية
    # ─────────────────────────────────────────────────────────

    def correct(self, query: str) -> Tuple[str, bool]:
        """
        يُصحّح الأخطاء الإملائية في الاستعلام.

        المعاملات:
            query: الاستعلام الأصلي مثل "infromation retreival"

        الإرجاع:
            Tuple[str, bool]:
                [0] الاستعلام المُصحَّح مثل "information retrieval"
                [1] True إذا تم تصحيح شيء، False إذا لا

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        الخوارزمية خطوة بخطوة:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. نُقسّم الاستعلام إلى كلمات: "infromation retreival" → ["infromation", "retreival"]
        2. لكل كلمة:
           a. هل هي قصيرة جداً (≤2 حرف)؟ → تجاوز (NLP, AI...)
           b. هل هي رقم؟ → تجاوز
           c. هل هي مصطلح تقني؟ → تجاوز
           d. هل هي في القاموس؟ → تجاوز
           e. إذا لا: ابحث عن أقرب كلمة صحيحة
        3. ادمج الكلمات من جديد

        مثال تفصيلي:
            "infromation" → ليست في القاموس
                          → أقرب كلمة = "information" (مسافة 2)
                          → نُصحَّح!

            "retreival"   → ليست في القاموس
                          → أقرب كلمة = "retrieval" (مسافة 1)
                          → نُصحَّح!

            "bm25"        → مصطلح تقني (في قائمتنا)
                          → نتجاوز، لا تصحيح
        """
        # إذا المكتبة غير متاحة، نُرجع الأصلي بدون تغيير
        if self._checker is None:
            return query, False

        if not query or not query.strip():
            return query, False

        # تقسيم الاستعلام إلى كلمات
        words = query.split()
        corrected_words = []
        was_corrected = False

        for word in words:
            corrected_word = self._correct_word(word)
            if corrected_word != word:
                was_corrected = True
                logger.debug(f"[SpellCorrector] {word!r} → {corrected_word!r}")
            corrected_words.append(corrected_word)

        corrected_query = " ".join(corrected_words)
        return corrected_query, was_corrected

    def _correct_word(self, word: str) -> str:
        """
        يُصحّح كلمة واحدة.
        دالة مساعدة خاصة — تُستدعى من correct() فقط.
        """
        # نتجاوز الكلمات القصيرة جداً (مثل "AI", "ML", "IR")
        if len(word) <= 2:
            return word

        # نتجاوز الأرقام والرموز
        if not word.isalpha():
            return word

        word_lower = word.lower()

        # نتجاوز المصطلحات التقنية
        if word_lower in self.TECHNICAL_TERMS:
            return word

        # نتجاوز الكلمات الموجودة في القاموس
        if word_lower in self._checker:
            return word

        # نجد التصحيح الأنسب
        correction = self._checker.correction(word_lower)

        # إذا لا يوجد تصحيح أو التصحيح نفس الكلمة
        if correction is None or correction == word_lower:
            return word

        # نحافظ على حالة الأحرف الأصلية إذا كانت الكلمة بأحرف كبيرة
        if word[0].isupper():
            return correction.capitalize()

        return correction

    # ─────────────────────────────────────────────────────────
    # دوال مساعدة إضافية
    # ─────────────────────────────────────────────────────────

    def get_candidates(self, word: str) -> List[str]:
        """
        يُرجع قائمة كلمات مقترحة لكلمة مشبوهة.
        مفيد لميزة "هل تقصد؟" في الواجهة.

        مثال:
            get_candidates("retreival")
            → ["retrieval", "retrieval"]  ← مرتبة حسب الاحتمالية
        """
        if self._checker is None or not word:
            return []
        candidates = self._checker.candidates(word.lower())
        return sorted(candidates) if candidates else []

    def is_word_correct(self, word: str) -> bool:
        """
        يتحقق هل الكلمة صحيحة إملائياً.

        مثال:
            is_word_correct("information") → True
            is_word_correct("infromation") → False
        """
        if self._checker is None:
            return True
        return word.lower() in self._checker

    @property
    def is_available(self) -> bool:
        """هل المكتبة متاحة وجاهزة؟"""
        return self._checker is not None


# =============================================================
# Singleton Pattern
# =============================================================
# نسخة واحدة مشتركة — تحميل القاموس مرة واحدة فقط عند بدء التشغيل

_spell_corrector_instance: Optional[SpellCorrector] = None


def get_spell_corrector() -> SpellCorrector:
    """
    يُرجع النسخة الوحيدة من SpellCorrector (Singleton).
    تحميل القاموس يستغرق بضع ثوانٍ — نريده مرة واحدة فقط.
    """
    global _spell_corrector_instance
    if _spell_corrector_instance is None:
        _spell_corrector_instance = SpellCorrector()
    return _spell_corrector_instance
