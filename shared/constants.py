"""
shared/constants.py
===================
ثوابت النظام المشتركة بين جميع الخدمات.

القاعدة: أي رقم أو مسار ثابت يُستخدم في أكثر من مكان
يجب أن يُعرَّف هنا فقط، ويُستورد من هنا في كل مكان آخر.

⚠️ لا تكتب أرقام Ports أو مسارات مباشرةً في أي ملف آخر.
"""

# =============================================================
# منافذ الخدمات (Service Ports)
# كل خدمة لها منفذ خاص بها لا يتشارك مع أي خدمة أخرى
# =============================================================

GATEWAY_PORT            = 8000  # نقطة الدخول الوحيدة للنظام
PREPROCESSING_PORT      = 8001  # خدمة المعالجة المسبقة
INDEXING_PORT           = 8002  # خدمة الفهرسة
RETRIEVAL_PORT          = 8003  # خدمة الاسترجاع
QUERY_REFINEMENT_PORT   = 8004  # خدمة تحسين الاستعلامات
EVALUATION_PORT         = 8005  # خدمة التقييم والترتيب
UI_PORT                 = 8501  # واجهة Streamlit
CLUSTERING_PORT         = 8006  # خدمة التجميع

# =============================================================
# عناوين الخدمات الداخلية (Internal Service URLs)
# يستخدمها Gateway وأي خدمة تحتاج تستدعي خدمة أخرى
# =============================================================

BASE_HOST = "http://localhost"

PREPROCESSING_URL       = f"{BASE_HOST}:{PREPROCESSING_PORT}"
INDEXING_URL            = f"{BASE_HOST}:{INDEXING_PORT}"
RETRIEVAL_URL           = f"{BASE_HOST}:{RETRIEVAL_PORT}"
QUERY_REFINEMENT_URL    = f"{BASE_HOST}:{QUERY_REFINEMENT_PORT}"
EVALUATION_URL          = f"{BASE_HOST}:{EVALUATION_PORT}"
CLUSTERING_URL          = f"{BASE_HOST}:{CLUSTERING_PORT}"


# =============================================================
# مسارات البيانات (Data Paths)
# مبنية بشكل نسبي من جذر المشروع
# =============================================================

import os

# جذر المشروع = المجلد الذي يحتوي على مجلد shared
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
DATASETS_DIR   = os.path.join(DATA_DIR, "datasets")
INDEXES_DIR    = os.path.join(DATA_DIR, "indexes")
MODELS_DIR     = os.path.join(DATA_DIR, "models")


# =============================================================
# إعدادات المعالجة المسبقة (Preprocessing Settings)
# قيم افتراضية مدروسة — يمكن تجاوزها في كل طلب
# =============================================================

DEFAULT_LANGUAGE            = "english"
DEFAULT_APPLY_STEMMING      = True
DEFAULT_APPLY_LEMMATIZATION = False   # لا يعمل مع Stemming في نفس الوقت
DEFAULT_REMOVE_STOPWORDS    = True
DEFAULT_REMOVE_PUNCTUATION  = True
DEFAULT_LOWERCASE           = True


# =============================================================
# إعدادات BM25
# قيم k1 و b المُجرَّبة والموصى بها في أبحاث IR
# =============================================================

BM25_DEFAULT_K1 = 1.5   # يتحكم في تشبع تكرار المصطلح
BM25_DEFAULT_B  = 0.75  # يتحكم في تطبيع طول الوثيقة


# =============================================================
# إعدادات الاسترجاع
# =============================================================

DEFAULT_TOP_K = 10   # عدد النتائج الافتراضية المُرجَعة
MAX_TOP_K     = 100  # الحد الأقصى المسموح به


# =============================================================
# إعدادات التقييم
# =============================================================

EVAL_K_VALUES = [5, 10, 20]  # قيم K المستخدمة في Precision@K


# =============================================================
# رسائل الأخطاء الشائعة (للاتساق بين الخدمات)
# =============================================================

ERR_EMPTY_TEXT      = "النص المُدخَل فارغ أو يحتوي على مسافات فقط"
ERR_EMPTY_QUERY     = "الاستعلام فارغ"
ERR_INVALID_TOP_K   = f"قيمة top_k يجب أن تكون بين 1 و {MAX_TOP_K}"
ERR_INDEX_NOT_BUILT = "الفهرس غير مبني — يرجى بناء الفهرس أولاً"
ERR_DATASET_UNKNOWN = "اسم مجموعة البيانات غير معروف"
