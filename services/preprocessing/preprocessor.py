"""
services/preprocessing/preprocessor.py
=======================================
المنطق الأساسي لمعالجة النصوص في نظام IR.

هذا الملف يحتوي على الكود الفعلي للمعالجة فقط.
لا يعرف شيئاً عن HTTP أو FastAPI — هذا قصدي.

المبدأ: "Separation of Concerns"
  - هذا الملف  ← يعرف كيف يعالج نصاً
  - app.py      ← يعرف كيف يستقبل HTTP ويستدعي هذا الملف

فائدة هذا الفصل:
  1. قابل للاختبار بسهولة (لا تحتاج server لاختباره)
  2. قابل للاستخدام من أي مكان (script, API, test)
  3. سهل التعديل دون لمس الـ API
"""

import re
import string
import time
from typing import List, Tuple

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet


# =============================================================
# التحقق من وجود بيانات NLTK عند تحميل الملف
# =============================================================

def _ensure_nltk_data() -> None:
    """
    يتحقق من وجود بيانات NLTK المطلوبة ويحملها إن لم تكن موجودة.
    يُستدعى مرة واحدة عند بدء تشغيل الخدمة.
    """
    required = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for resource in required:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                print(f"[Preprocessing] تحميل NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

_ensure_nltk_data()


# =============================================================
# الـ Preprocessor Class — قلب الخدمة
# =============================================================

class TextPreprocessor:
    """
    معالج النصوص الأساسي لنظام IR.

    التصميم: Stateless (بدون حالة داخلية تتغير)
    كل استدعاء لـ process() مستقل تماماً عن الاستدعاءات الأخرى.
    هذا يجعله آمناً للاستخدام في بيئات متعددة الخيوط (multi-threaded).

    مثال الاستخدام:
        preprocessor = TextPreprocessor()
        tokens, steps = preprocessor.process(
            text="Running dogs jumped over fences!",
            apply_stemming=True,
            remove_stopwords=True,
        )
        # tokens = ["run", "dog", "jump", "fenc"]
    """

    def __init__(self) -> None:
        # نبني هذه الكائنات مرة واحدة فقط عند إنشاء الـ preprocessor
        # لأن بناءها مكلف — خاصةً WordNetLemmatizer
        self._stemmer     = PorterStemmer()
        self._lemmatizer  = WordNetLemmatizer()

        # نخزّن stopwords في set لأن البحث فيه O(1) بدلاً من O(n) في list
        self._stopwords_en = set(stopwords.words("english"))
        self._stopwords_ar = set(stopwords.words("arabic"))

    # ----------------------------------------------------------
    # الدالة الرئيسية
    # ----------------------------------------------------------

    def process(
        self,
        text: str,
        language: str = "english",
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        apply_stemming: bool = True,
        apply_lemmatization: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        تعالج نصاً وتُرجع (tokens, steps_applied).

        المعاملات:
            text                : النص الأصلي
            language            : لغة النص ("english" أو "arabic")
            lowercase           : تحويل لحروف صغيرة
            remove_punctuation  : حذف علامات الترقيم
            remove_stopwords    : حذف الكلمات الوظيفية
            apply_stemming      : تطبيق Porter Stemming
            apply_lemmatization : تطبيق WordNet Lemmatization

        الإرجاع:
            Tuple[List[str], List[str]]
                - القائمة الأولى: التوكنز النظيفة
                - القائمة الثانية: أسماء الخطوات المُطبَّقة
        """
        steps_applied: List[str] = []
        current_text = text

        # --- الخطوة 1: Lowercase ---
        if lowercase:
            current_text = current_text.lower()
            steps_applied.append("lowercase")

        # --- الخطوة 2: حذف علامات الترقيم ---
        if remove_punctuation:
            current_text = self._remove_punctuation(current_text)
            steps_applied.append("remove_punctuation")

        # --- الخطوة 3: Tokenization ---
        # word_tokenize أذكى من split() — يتعامل مع الاختصارات مثل "don't"
        tokens = word_tokenize(current_text)
        steps_applied.append("tokenization")

        # --- الخطوة 4: حذف الرموز المتبقية وأرقام منفردة ---
        tokens = self._remove_non_alpha(tokens)

        # --- الخطوة 5: حذف Stopwords ---
        if remove_stopwords:
            tokens = self._remove_stopwords(tokens, language)
            steps_applied.append("remove_stopwords")

        # --- الخطوة 6: Stemming أو Lemmatization (واحد فقط) ---
        # ⚠️ لا تطبّق كليهما معاً — ينتج عنه توكنز مشوّهة
        if apply_stemming and not apply_lemmatization:
            tokens = self._apply_stemming(tokens)
            steps_applied.append("stemming")
        elif apply_lemmatization and not apply_stemming:
            tokens = self._apply_lemmatization(tokens)
            steps_applied.append("lemmatization")

        # --- الخطوة 7: حذف التوكنز القصيرة جداً (حرف واحد) ---
        tokens = [t for t in tokens if len(t) > 1]

        return tokens, steps_applied

    def process_batch(
        self,
        texts: List[str],
        **kwargs,
    ) -> List[Tuple[List[str], List[str]]]:
        """
        معالجة قائمة من النصوص دفعةً واحدة.
        kwargs تُمرَّر مباشرةً إلى process().

        لماذا هذه الدالة موجودة؟
        عند فهرسة 200,000 وثيقة، استدعاء process()
        مرة لكل وثيقة أسرع من 200,000 HTTP request.
        """
        return [self.process(text, **kwargs) for text in texts]

    # ----------------------------------------------------------
    # دوال مساعدة خاصة (Private Helper Methods)
    # ----------------------------------------------------------

    def _remove_punctuation(self, text: str) -> str:
        """
        يحذف علامات الترقيم من النص.

        نستخدم regex بدلاً من string.punctuation لأن:
        - string.punctuation لا يشمل علامات عربية
        - regex أكثر مرونة وأسرع على النصوص الطويلة
        """
        # يحذف كل ما ليس حرفاً أو رقماً أو مسافة
        return re.sub(r"[^\w\s]", " ", text)

    def _remove_non_alpha(self, tokens: List[str]) -> List[str]:
        """
        يحذف التوكنز التي تحتوي على أرقام فقط أو رموز خاصة.

        مثال: ["hello", "123", "world", "42nd"] → ["hello", "world", "nd"]
        ملاحظة: "42nd" تصبح "nd" بعد حذف الأرقام — هذا سلوك مقبول في IR.
        """
        return [token for token in tokens if token.isalpha()]

    def _remove_stopwords(
        self, tokens: List[str], language: str
    ) -> List[str]:
        """
        يحذف الكلمات الوظيفية (stopwords).

        الكلمات الوظيفية هي كلمات شائعة جداً لا تضيف معنى للبحث:
        "the", "is", "in", "a", "an", "of", "to" ...

        لماذا نحذفها؟
        - تُقلّل حجم الفهرس بشكل كبير
        - تُحسّن جودة البحث (تقليل الضوضاء)
        - تُسرّع عمليات الحساب
        """
        stop_set = (
            self._stopwords_ar
            if language == "arabic"
            else self._stopwords_en
        )
        return [token for token in tokens if token not in stop_set]

    def _apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        يطبّق Porter Stemming على كل توكن.

        Stemming = تقليل الكلمة لجذرها بقواعد إملائية بسيطة.
        سريع جداً لكن النتيجة قد لا تكون كلمة حقيقية.

        أمثلة:
            "running"  → "run"
            "studies"  → "studi"   ← ليست كلمة حقيقية لكنها تعمل في IR
            "happiness"→ "happi"
            "fences"   → "fenc"
        """
        return [self._stemmer.stem(token) for token in tokens]

    def _apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        يطبّق WordNet Lemmatization على كل توكن.

        Lemmatization = تحويل الكلمة لشكلها المعجمي الصحيح.
        أبطأ من Stemming لكن النتيجة دائماً كلمة حقيقية.

        يحتاج POS tags لنتيجة أدق:
            "running" (verb)  → "run"
            "better" (adj)    → "good"   ← يفهم السياق
            "studies" (noun)  → "study"
            "studies" (verb)  → "study"
        """
        # نحصل على POS tags لكل التوكنز دفعةً واحدة (أسرع)
        pos_tagged = pos_tag(tokens)
        return [
            self._lemmatizer.lemmatize(
                token, pos=self._get_wordnet_pos(tag)
            )
            for token, tag in pos_tagged
        ]

    @staticmethod
    def _get_wordnet_pos(treebank_tag: str) -> str:
        """
        يحوّل POS tag من صيغة Treebank إلى صيغة WordNet.

        Treebank tags: "JJ" (adj), "NN" (noun), "VB" (verb), "RB" (adv)
        WordNet tags : wordnet.ADJ, wordnet.NOUN, wordnet.VERB, wordnet.ADV

        إذا لم يُعرَف النوع، نفترض أنه اسم (NOUN) — الأكثر شيوعاً.
        """
        tag_map = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        # نأخذ الحرف الأول من الـ tag فقط: "JJR" → "J"
        return tag_map.get(treebank_tag[0].upper(), wordnet.NOUN)


# =============================================================
# Singleton — نسخة واحدة مشتركة في كل الخدمة
# =============================================================
# لماذا Singleton؟
# بناء TextPreprocessor يُحمّل بيانات NLTK — عملية مكلفة.
# نريد أن تحدث مرة واحدة عند بدء التشغيل، ليس عند كل طلب.

_preprocessor_instance: TextPreprocessor | None = None


def get_preprocessor() -> TextPreprocessor:
    """
    يُرجع النسخة الوحيدة من TextPreprocessor (Singleton pattern).
    FastAPI يستخدمه كـ Dependency Injection.
    """
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = TextPreprocessor()
    return _preprocessor_instance
