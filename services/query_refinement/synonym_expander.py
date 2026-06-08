"""
services/query_refinement/synonym_expander.py
==============================================
موسّع المرادفات — الجزء الثاني من Query Refinement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لماذا نحتاج المرادفات في IR؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
مشكلة TF-IDF و BM25 الأساسية:
يبحثان عن الكلمة بالضبط — لا يفهمان المعنى.

مثال:
    استعلام: "car purchase"
    وثيقة:   "automobile buying guide"

    BM25 لا يجد تطابق! (كلمات مختلفة)
    لكن المعنى واحد!

الحل: نُوسّع الاستعلام بالمرادفات:
    "car"      → أضف: "automobile", "vehicle"
    "purchase" → أضف: "buy", "acquire"

الاستعلام الموسّع: "car automobile vehicle purchase buy acquire"
الآن BM25 يجد الوثيقة! ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
المكتبة المُستخدمة: NLTK WordNet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WordNet هي قاعدة بيانات لغوية ضخمة من جامعة برينستون.
تحتوي على:
    - 155,000+ كلمة إنجليزية
    - مجموعات كلمات بنفس المعنى (Synsets)
    - علاقات بين الكلمات (مرادف، متضاد، أعم، أخص...)

مذكورة في المحاضرات كأداة مقترحة لتوسيع الاستعلام.

مثال من WordNet:
    كلمة "car":
    Synset: {car, auto, automobile, machine, motorcar}
    كلها تعني نفس الشيء!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
تحذير مهم — Query Drift:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
لو أضفنا كل مرادفات كل كلمة، قد يصبح الاستعلام
بعيداً عن المعنى الأصلي!

مثال سيء:
    "bank" → {bank, depository, financial_institution, river_bank}
    هل يقصد بنك مالي أم ضفة نهر؟!

الحل: نُحدّد max_synonyms=3 فقط لكل كلمة.
      ونستبعد الكلمات الغامضة (التي لها معانٍ كثيرة جداً).
"""

import logging
from typing import List, Set, Optional

logger = logging.getLogger(__name__)

# ─── تحميل NLTK WordNet ──────────────────────────────────────
try:
    import nltk
    from nltk.corpus import wordnet

    # تأكد من تحميل البيانات المطلوبة
    def _ensure_wordnet():
        for resource in ["wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info(f"[SynonymExpander] تحميل {resource}...")
                nltk.download(resource, quiet=True)

    _ensure_wordnet()
    WORDNET_AVAILABLE = True

except ImportError:
    WORDNET_AVAILABLE = False
    logger.warning("[SynonymExpander] NLTK غير مثبّت. pip install nltk")


class SynonymExpander:
    """
    يُوسّع الاستعلام بإضافة مرادفات الكلمات الرئيسية.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    كيف يعمل؟
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. نقسّم الاستعلام إلى كلمات
    2. لكل كلمة: نبحث في WordNet عن مرادفاتها
    3. نأخذ أفضل 3 مرادفات فقط (لتجنب Query Drift)
    4. ندمج الكلمات الأصلية + المرادفات

    مثال:
        الاستعلام: "fast car"
        توسيع "fast"   → quick, rapid, swift
        توسيع "car"    → automobile, vehicle, auto
        النتيجة: "fast car quick rapid swift automobile vehicle auto"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    الكلمات التي لا نُوسّعها:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Stopwords (the, is, a...) → لا فائدة من مرادفاتها
    - كلمات قصيرة (≤2 حرف)
    - أرقام
    - مصطلحات تقنية محددة (bm25, tfidf...)
    """

    # كلمات لا نُوسّعها لأن مرادفاتها قد تُشوّش النتيجة
    _SKIP_WORDS: Set[str] = {
        # stopwords أساسية
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "and",
        "or",
        "but",
        "not",
        "if",
        "as",
        # مصطلحات تقنية IR (لها معنى محدد لا نريد توسيعه)
        "bm25",
        "tfidf",
        "tf",
        "idf",
        "bert",
        "gpt",
        "retrieval",
        "indexing",
        "embedding",
    }

    def __init__(self, max_synonyms: int = 3) -> None:
        """
        المعاملات:
            max_synonyms: أقصى عدد مرادفات لكل كلمة (افتراضي 3)
                          3 هو التوازن المثالي بين الغنى وتجنب Query Drift
        """
        self.max_synonyms = max_synonyms

        if WORDNET_AVAILABLE:
            logger.info(f"[SynonymExpander] جاهز. max_synonyms={max_synonyms}")

    # ─────────────────────────────────────────────────────────
    # الدالة الرئيسية
    # ─────────────────────────────────────────────────────────

    def expand(self, query: str) -> str:
        """
        يُوسّع الاستعلام بإضافة مرادفات.

        المعاملات:
            query: الاستعلام الأصلي مثل "fast car"

        الإرجاع:
            str: الاستعلام الموسّع مثل "fast car quick rapid automobile vehicle"
                 أو الأصلي إذا لم تُوجَد مرادفات.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        خطوات التنفيذ:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. نقسّم إلى كلمات
        2. لكل كلمة نجد مرادفاتها
        3. ندمج بدون تكرار
        """
        if not WORDNET_AVAILABLE or not query.strip():
            return query

        words = query.lower().split()
        all_terms: List[str] = list(words)  # نبدأ بالكلمات الأصلية
        added_terms: Set[str] = set(words)  # لتتبع ما أضفناه (لتجنب التكرار)

        for word in words:
            # نتجاوز الكلمات التي لا نريد توسيعها
            if self._should_skip(word):
                continue

            # نجد مرادفات الكلمة
            synonyms = self._get_synonyms(word)

            # نُضيف المرادفات الجديدة فقط (لا تكرار)
            new_synonyms = [s for s in synonyms if s not in added_terms]

            for syn in new_synonyms[: self.max_synonyms]:
                all_terms.append(syn)
                added_terms.add(syn)
                logger.debug(f"[SynonymExpander] {word!r} → +{syn!r}")

        expanded = " ".join(all_terms)

        if expanded != query:
            logger.info(f"[SynonymExpander] قبل: {query!r}\n" f"  بعد:  {expanded!r}")

        return expanded

    def get_synonyms_for_word(self, word: str) -> List[str]:
        """
        يُرجع مرادفات كلمة واحدة.
        دالة عامة للاستخدام من الخارج.

        مثال:
            get_synonyms_for_word("car")
            → ["automobile", "vehicle", "auto"]
        """
        return self._get_synonyms(word)

    # ─────────────────────────────────────────────────────────
    # دوال مساعدة خاصة
    # ─────────────────────────────────────────────────────────

    def _get_synonyms(self, word: str) -> List[str]:
        """
        يجلب مرادفات كلمة من WordNet مع فلترة ذكية.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        كيف يعمل WordNet داخلياً؟
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        WordNet يُنظّم الكلمات في "Synsets":
            synsets("car") يُرجع:
            - Synset('car.n.01') = {car, auto, automobile, machine, motorcar}
            - Synset('car.n.02') = {car, railcar, railway_car, railroad_car}
            ...

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        لماذا نفلتر lemmas تنتهي بـ 'ing'؟
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        WordNet أحياناً يضع معنى خاطئ أولاً:
            "fast" → Synset 0 = "الصيام" (اسم)
                     فيُرجع "fasting" كمرادف!

        لكن المستخدم يقصد "fast" = "سريع" (صفة).

        الحل: إذا الكلمة الأصلية لا تنتهي بـ 'ing'
               ولا تنتهي بـ 'ed'، نتجاهل أي lemma
               ينتهي بـ 'ing' (لأنه على الأرجح شكل فعلي خاطئ).

        مثال:
            fast  → نتجاهل "fasting" ✓ (fast لا تنتهي بـ ing)
            drive → نتجاهل "driving" ✓ (drive لا تنتهي بـ ing)
            car   → "automobile" مقبول ✓ (لا ينتهي بـ ing)
        """
        if not WORDNET_AVAILABLE:
            return []

        synonyms: List[str] = []
        seen: Set[str] = {word.lower()}  # لتجنب التكرار
        word_lower = word.lower()

        try:
            # نأخذ أول 3 synsets فقط (لتجنب المعاني البعيدة)
            for synset in wordnet.synsets(word)[:3]:
                for lemma in synset.lemmas():
                    name = lemma.name().lower()

                    # ① تجاهل الكلمات المركّبة (motor_vehicle)
                    if "_" in name:
                        continue

                    # ② تجاهل المكررات والكلمة الأصلية
                    if name in seen:
                        continue

                    # ③ تجاهل الكلمات الطويلة جداً (أكثر من 12 حرف)
                    if len(name) > 12:
                        continue

                    # ④ الفلتر الرئيسي: تجاهل lemmas تنتهي بـ 'ing'
                    # إذا الكلمة الأصلية لا تنتهي بـ 'ing'
                    # هذا يمنع "fast" → "fasting"
                    if (
                        name.endswith("ing")
                        and not word_lower.endswith("ing")
                        and not word_lower.endswith("e")
                    ):
                        continue

                    synonyms.append(name)
                    seen.add(name)

        except Exception as e:
            logger.debug(f"[SynonymExpander] خطأ مع كلمة {word!r}: {e}")

        return synonyms

    def _should_skip(self, word: str) -> bool:
        """
        يُحدّد هل نتجاوز هذه الكلمة بدون توسيع.

        نتجاوز إذا:
            - قصيرة جداً (≤2 حرف)
            - رقم
            - في قائمة الكلمات المُستثناة
        """
        if len(word) <= 2:
            return True
        if word.isdigit():
            return True
        if word.lower() in self._SKIP_WORDS:
            return True
        return False

    @property
    def is_available(self) -> bool:
        """هل WordNet متاح وجاهز؟"""
        return WORDNET_AVAILABLE


# =============================================================
# Singleton Pattern
# =============================================================

_synonym_expander_instance: Optional[SynonymExpander] = None


def get_synonym_expander(max_synonyms: int = 3) -> SynonymExpander:
    """يُرجع النسخة الوحيدة من SynonymExpander (Singleton)."""
    global _synonym_expander_instance
    if _synonym_expander_instance is None:
        _synonym_expander_instance = SynonymExpander(max_synonyms=max_synonyms)
    return _synonym_expander_instance
