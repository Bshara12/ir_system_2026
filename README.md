#  IR System 2026 — Information Retrieval Search Engine

> نظام استرجاع معلومات متكامل مبني على مبادئ **Service-Oriented Architecture (SOA)**،  
> يدعم خمسة نماذج استرجاع ويبحث في أكثر من **523,000 وثيقة** من مجموعة بيانات Quora.

---

## الفريق

| المطور | الاسم | المسؤولية الرئيسية |
|--------|-------|-------------------|
| المطور الأول | بشاره غانم الحاتم | Indexing Service + Vector Store + Shared Layer |
| المطور الثاني | فراس حاتم الحاتم | Retrieval Service + Query Refinement + Clustering |
| المطورة الثالثة | صبا نادر عشعوش | Gateway + Evaluation + Streamlit UI + Agent (Bonus) |
| المطورة الرابعة | هبه علي عيسى | Preprocessor + Document Store + Dataset Adapters |

**المشرف النظري:** د. أبي صندوق  
**المشرفون العمليون:** م. مروة الداية | م. سليمى المحايري  
**جامعة دمشق — كلية الهندسة المعلوماتية | 2025/2026**

---

## فهرس المحتويات

- [نظرة عامة](#نظرة-عامة)
- [بنية النظام](#بنية-النظام)
- [هيكل الملفات](#هيكل-الملفات)
- [الخدمات والبورتات](#الخدمات-والبورتات)
- [متطلبات التشغيل](#متطلبات-التشغيل)
- [التثبيت](#التثبيت)
- [بناء الفهارس](#بناء-الفهارس)
- [تشغيل النظام](#تشغيل-النظام)
- [نماذج الاسترجاع](#نماذج-الاسترجاع)
- [الميزات الإضافية](#الميزات-الإضافية)
- [الاختبارات](#الاختبارات)
- [API Reference](#api-reference)
- [مجموعة البيانات](#مجموعة-البيانات)

---

## نظرة عامة

**IR System 2026** هو محرك بحث يستقبل استعلاماً نصياً ويبحث في مجموعة بيانات Quora باستخدام خمسة نماذج استرجاع:

| النموذج | النوع | طريقة المطابقة |
|---------|-------|----------------|
| **TF-IDF** | Lexical | Cosine Similarity |
| **BM25** | Probabilistic | BM25Okapi Scoring |
| **Embedding** | Semantic | FAISS Inner Product |
| **Hybrid Parallel** | Fusion | Reciprocal Rank Fusion (RRF) |
| **Hybrid Serial** | Pipeline | BM25 Filter → Embedding Rerank |

---

## بنية النظام

```
┌─────────────────────────────────────────────────────────────────┐
│                    IR System 2026 — SOA Architecture            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [ المستخدم]                                                  │
│         │                                                        │
│         ▼  HTTP                                                  │
│   [  Streamlit UI  :8501]                                      │
│         │                                                        │
│         ▼  REST API                                              │
│   [  Gateway Service  :8000]  ◄── نقطة الدخول الوحيدة        │
│         │                                                        │
│    ┌────┴────────────────────────────────────┐                  │
│    ▼                                         ▼                  │
│ [ Retrieval :8003]              [ Evaluation :8005]         │
│    │                                         ▼                  │
│    ├──► [  Preprocessing :8001]   [ Clustering :8006]       │
│    │                                                             │
│    ├──► [  Query Refinement :8004]                            │
│    │                                                             │
│    └──► [ Indexing :8002]                                     │
│              │                                                   │
│              ▼                                                   │
│        [ Indexes on Disk]                                      │
│         TF-IDF (.pkl/.npz)                                       │
│         BM25   (.pkl)                                            │
│         FAISS  (.faiss/.npy)                                     │
│         SQLite (.db) ← الوثائق الكاملة                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## هيكل الملفات

```
ir_system_2026/
│
├──  services/                    # الخدمات المستقلة (SOA)
│   │
│   ├──  preprocessing/           # خدمة المعالجة المسبقة (Port 8001)
│   │   ├── app.py                  # FastAPI entry point
│   │   ├── preprocessor.py         # TextPreprocessor الرئيسي [هبه]
│   │   ├── tokenizer.py            # Tokenization logic [بشاره]
│   │   ├── stemmer.py              # Porter Stemmer wrapper [بشاره]
│   │   └── tests/
│   │       └── test_preprocessor.py
│   │
│   ├──  indexing/                # خدمة الفهرسة (Port 8002)
│   │   ├── app.py                  # FastAPI entry point
│   │   ├── tfidf_indexer.py        # TF-IDF Index Builder [بشاره]
│   │   ├── bm25_indexer.py         # BM25 Index Builder [بشاره]
│   │   ├── embedding_indexer.py    # FAISS + SentenceTransformer [بشاره]
│   │   ├── hybrid_indexer.py       # Orchestrator (BM25 + Embedding) [بشاره]
│   │   ├── inverted_index.py       # Inverted Index + Posting Lists [بشاره]
│   │   ├── vector_store.py         # Facade فوق EmbeddingIndexer [بشاره]
│   │   ├── document_store.py       # SQLite Document Storage [هبه]
│   │   ├── ir_datasets_adapter.py  # Adapter لـ ir-datasets library [هبه]
│   │   ├── dataset_loader.py       # Strategy Pattern لتحميل البيانات [هبه]
│   │   └── tests/                  # 142+ unit tests
│   │       ├── test_tfidf_indexer.py
│   │       ├── test_bm25_indexer.py
│   │       ├── test_embedding_indexer.py
│   │       ├── test_vector_store.py
│   │       ├── test_inverted_index.py
│   │       ├── test_document_store.py
│   │       ├── test_dataset_loader.py
│   │       ├── test_hybrid_indexer.py
│   │       └── test_indexing_app.py
│   │
│   ├──  retrieval/               # خدمة الاسترجاع (Port 8003)
│   │   ├── app.py                  # FastAPI + routing بين النماذج [فراس]
│   │   ├── tfidf_retriever.py      # TF-IDF + Cosine Similarity [فراس]
│   │   ├── bm25_retriever.py       # BM25 + dynamic k1/b [فراس]
│   │   ├── embedding_retriever.py  # Semantic Search via VectorStore [فراس]
│   │   ├── hybrid_parallel.py      # RRF Fusion (3 نماذج) [فراس]
│   │   ├── hybrid_serial.py        # Two-Stage Pipeline [فراس]
│   │   └── tests/
│   │       └── test_retrieval.py
│   │
│   ├──  query_refinement/        # خدمة تحسين الاستعلامات (Port 8004)
│   │   ├── app.py                  # FastAPI entry point [فراس]
│   │   ├── spell_corrector.py      # pyspellchecker + Edit Distance [فراس]
│   │   ├── synonym_expander.py     # NLTK WordNet Expansion [فراس]
│   │   ├── query_history.py        # Session History Storage [هبه]
│   │   ├── suggestion_engine.py    # Prefix Match + Suggestions [هبه]
│   │   └── tests/
│   │       └── test_query_refinement.py
│   │
│   ├──  ranking_evaluation/      # خدمة التقييم (Port 8005)
│   │   ├── app.py                  # FastAPI entry point [صبا]
│   │   ├── metrics.py              # MAP, Recall, P@K, nDCG من الصفر [صبا]
│   │   ├── evaluator.py            # تحميل qrels + استدعاء Retrieval [صبا]
│   │   └── ranker.py               # Ranker wrapper [صبا]
│   │
│   ├──  clustering/              # خدمة التجميع (Port 8006) — ميزة إضافية
│   │   ├── app.py                  # FastAPI entry point [فراس]
│   │   ├── clusterer.py            # K-Means + LSA + Silhouette [فراس]
│   │   └── tests/
│   │       └── test_clustering.py
│   │
│   └──  gateway/                 # خدمة البوابة (Port 8000)
│       ├── app.py                  # FastAPI entry point [صبا]
│       ├── router.py               # Route definitions [صبا]
│       └── service_client.py       # HTTP Client للخدمات [صبا]
│
├──  shared/                      # الطبقة المشتركة
│   ├── constants.py                # بورتات + مسارات + إعدادات [بشاره]
│   ├── models.py                   # Pydantic Data Contracts [بشاره]
│   └── utils.py                    # دوال مشتركة [بشاره]
│
├──  ui/                          # واجهة المستخدم (Port 8501)
│   ├── main.py                     # Streamlit entry point [صبا]
│   ├──  pages/
│   │   ├── search.py               # صفحة البحث الرئيسية [صبا]
│   │   ├── evaluation.py           # صفحة التقييم [صبا]
│   │   ├── clustering.py           # صفحة التجميع [صبا]
│   │   └── settings.py             # الإعدادات [صبا]
│   └──  components/
│       ├── result_card.py          # بطاقة عرض النتيجة [صبا]
│       └── metrics_chart.py        # رسوم بيانية للتقييم [صبا]
│
├──  scripts/                     # سكريبتات المساعدة
│   ├── build_indexes_quora.py      # بناء كل الفهارس [بشاره]
│   ├── download_datasets.py        # تحميل مجموعات البيانات
│   ├── verify_indexes.py           # التحقق من صحة الفهارس [بشاره]
│   ├── verify_embedding.py         # التحقق من Embedding Index [بشاره]
│   └── verify_dataset_loader.py    # التحقق من تحميل البيانات [بشاره]
│
├──  data/                        # البيانات (غير مرفوعة على Git)
│   ├──  datasets/
│   │   └──  quora/
│   │       ├── corpus.jsonl        # 523K+ وثيقة
│   │       ├── queries.jsonl       # استعلامات التقييم
│   │       └── qrels.jsonl         # Relevance Judgments
│   └──  indexes/
│       └──  quora/
│           ├── tfidf_vectorizer.pkl
│           ├── tfidf_matrix.npz
│           ├── bm25_model.pkl
│           ├── bm25_tokens.pkl
│           ├── embedding_index.faiss
│           ├── embedding_vectors.npy
│           ├── embedding_metadata.json
│           └── documents.db        ← SQLite (الوثائق الكاملة)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## الخدمات والبورتات

| الخدمة | البورت | الوصف | Swagger |
|--------|--------|-------|---------|
| Gateway | `8000` | نقطة الدخول الوحيدة | http://localhost:8000/docs |
| Preprocessing | `8001` | تنظيف النصوص | http://localhost:8001/docs |
| Indexing | `8002` | بناء وإدارة الفهارس | http://localhost:8002/docs |
| Retrieval | `8003` | محرك البحث | http://localhost:8003/docs |
| Query Refinement | `8004` | تحسين الاستعلامات | http://localhost:8004/docs |
| Evaluation | `8005` | قياس جودة النتائج | http://localhost:8005/docs |
| Clustering | `8006` | تجميع الوثائق | http://localhost:8006/docs |
| Streamlit UI | `8501` | واجهة المستخدم | http://localhost:8501 |

---

## متطلبات التشغيل

- Python **3.10+**
- pip
- ~4 GB RAM (لتحميل الفهارس)
- الفهارس المبنية مسبقاً في `data/indexes/quora/`

---

## التثبيت

```bash
# 1. استنساخ المشروع
git clone https://github.com/Bshara12/ir_system_2026.git
cd ir_system_2026

# 2. إنشاء بيئة افتراضية
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# 3. تثبيت المكتبات
pip install -r requirements.txt

# 4. تحميل NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

---

## بناء الفهارس

>  **هذه الخطوة تُنفَّذ مرة واحدة فقط.** الفهارس تُحفظ على القرص ولا تُعاد عند كل تشغيل.

```bash
# بناء TF-IDF + BM25 + Inverted Index + Document Store (محلياً)
python scripts/build_indexes_quora.py

# ملاحظة: Embedding Index بُني على Google Colab بـ GPU
# يجب نسخ ملفات data/indexes/quora/ من Google Drive قبل التشغيل
```

**الملفات المطلوبة قبل التشغيل:**
```
data/indexes/quora/
├── tfidf_vectorizer.pkl      ← TF-IDF model
├── tfidf_matrix.npz          ← TF-IDF matrix (523K × 50K)
├── bm25_model.pkl            ← BM25 model
├── bm25_tokens.pkl           ← Tokenized documents
├── embedding_index.faiss     ← FAISS index (523K × 384)
├── embedding_vectors.npy     ← Raw vectors
├── embedding_metadata.json   ← Document metadata
└── documents.db              ← SQLite (نصوص كاملة)
```

---

## تشغيل النظام

افتح **8 terminals** أو استخدم سكريبت التشغيل:

```bash
# Terminal 1 — Preprocessing Service
uvicorn services.preprocessing.app:app --port 8001 --reload

# Terminal 2 — Indexing Service
uvicorn services.indexing.app:app --port 8002 --reload

# Terminal 3 — Retrieval Service
uvicorn services.retrieval.app:app --port 8003 --reload

# Terminal 4 — Query Refinement Service
uvicorn services.query_refinement.app:app --port 8004 --reload

# Terminal 5 — Evaluation Service
uvicorn services.ranking_evaluation.app:app --port 8005 --reload

# Terminal 6 — Clustering Service
uvicorn services.clustering.app:app --port 8006 --reload

# Terminal 7 — Gateway Service
uvicorn services.gateway.app:app --port 8000 --reload

# Terminal 8 — Streamlit UI
streamlit run ui/main.py
```

**التحقق من عمل النظام:**
```bash
# فحص Gateway
curl http://localhost:8000/health

# فحص كل الخدمات دفعة واحدة
curl http://localhost:8000/services/health
```

---

## نماذج الاسترجاع

### TF-IDF
```python
# مثال على استدعاء البحث
POST http://localhost:8003/search
{
  "query": "cloud storage solutions",
  "dataset": "dataset1",
  "model": "tfidf",
  "top_k": 10
}
```

### BM25 مع تعديل المعاملات
```python
POST http://localhost:8003/search
{
  "query": "cloud storage solutions",
  "dataset": "dataset1",
  "model": "bm25",
  "top_k": 10,
  "bm25_k1": 1.5,   # تحكم في تشبع التكرار (0.0 - 3.0)
  "bm25_b": 0.75    # تحكم في تطبيع الطول  (0.0 - 1.0)
}
```

### Hybrid Parallel (RRF)
```python
# يُشغّل TF-IDF + BM25 + Embedding بالتوازي
# يدمج النتائج بـ Reciprocal Rank Fusion (k=60)
POST http://localhost:8003/search
{
  "query": "machine learning algorithms",
  "dataset": "dataset1",
  "model": "hybrid_parallel",
  "top_k": 10
}
```

### Hybrid Serial (Two-Stage)
```python
# المرحلة 1: BM25 يُرجع أفضل 100 مرشح
# المرحلة 2: Embedding يُعيد ترتيب الـ 100
POST http://localhost:8003/search
{
  "query": "machine learning algorithms",
  "dataset": "dataset1",
  "model": "hybrid_serial",
  "top_k": 10
}
```

---

## الميزات الإضافية

### ١. Vector Store (Use Vector Stores — #11)
**المطور:** بشاره | **الملف:** `services/indexing/vector_store.py`

```python
from services.indexing.vector_store import VectorStore

store = VectorStore()
store.load("quora")

# بحث دلالي
results = store.search("fever treatment options", k=10)
# يجد: "antipyretic medication" حتى لو الكلمات مختلفة تماماً
```

**الأداة المستخدمة:** FAISS IndexFlatIP — مكتبة Meta للبحث في المتجهات بسرعة عالية.

---

### ٢. Document Clustering (#15)
**المطور:** فراس | **الملف:** `services/clustering/clusterer.py`

```bash
# تجميع كل وثائق dataset
POST http://localhost:8006/cluster/dataset
{"dataset_name": "quora", "n_clusters": 10}

# تجميع نتائج بحث محددة
POST http://localhost:8006/cluster/results
{"doc_ids": ["d1", "d2", ...], "n_clusters": 3}

# إيجاد أفضل عدد clusters تلقائياً
POST http://localhost:8006/cluster/optimal-k
{"dataset_name": "quora", "max_k": 20}
```

**الخوارزمية:** TF-IDF → TruncatedSVD (LSA, 100 بُعد) → L2 Normalize → K-Means → Silhouette Score

---

### ٣. Agent (#18)
**المطورة:** صبا | **الملف:** `services/gateway/`

نظام Agent يُتيح للمستخدم التفاعل مع محرك البحث بلغة طبيعية عبر واجهة شبيهة بالـ Chat.

---

## الاختبارات

```bash
# تشغيل كل الاختبارات
pytest

# اختبار خدمة محددة
pytest services/indexing/tests/
pytest services/retrieval/tests/
pytest services/query_refinement/tests/
pytest services/clustering/tests/

# مع تفاصيل
pytest -v

# مع نسبة التغطية
pytest --cov=services --cov-report=html
```

**نتائج الاختبارات:**
```
services/indexing/tests/         142+ tests ✅
services/preprocessing/tests/     20+ tests ✅
services/retrieval/tests/          50+ tests ✅
services/query_refinement/tests/   25+ tests ✅
services/clustering/tests/         25+ tests ✅
```

---

## API Reference

### Gateway (Port 8000)

| Method | Endpoint | الوصف |
|--------|----------|-------|
| `GET` | `/health` | فحص صحة Gateway |
| `GET` | `/services/health` | فحص صحة كل الخدمات |
| `POST` | `/search` | البحث الرئيسي |
| `POST` | `/evaluate` | تشغيل التقييم |
| `POST` | `/cluster` | تجميع الوثائق |
| `POST` | `/refine` | تحسين الاستعلام |

### Retrieval (Port 8003)

| Method | Endpoint | الوصف |
|--------|----------|-------|
| `POST` | `/search` | بحث بنموذج محدد |
| `GET` | `/models` | قائمة النماذج المتاحة |
| `GET` | `/health` | فحص الصحة + حالة الفهارس |

### Evaluation (Port 8005)

| Method | Endpoint | الوصف |
|--------|----------|-------|
| `POST` | `/evaluate/demo` | تقييم تجريبي فوري |
| `POST` | `/evaluate/dataset` | تقييم حقيقي على qrels |
| `GET` | `/metrics/supported` | المقاييس المدعومة |

### Clustering (Port 8006)

| Method | Endpoint | الوصف |
|--------|----------|-------|
| `POST` | `/cluster/dataset` | تجميع وثائق dataset |
| `POST` | `/cluster/results` | تجميع نتائج بحث |
| `POST` | `/cluster/optimal-k` | إيجاد أفضل K |

---

## مجموعة البيانات

### Quora Duplicate Questions (الأساسية)

```python
# تحميل عبر ir-datasets
import ir_datasets
dataset = ir_datasets.load("quora/train")

# الإحصاءات
# Documents: 523,000+
# Queries: متاحة في queries.jsonl
# Qrels: متاحة في qrels.jsonl
```

**روابط مفيدة:**
- [ir-datasets — Quora](https://ir-datasets.com/quora.html)
- [HuggingFace — all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [FAISS Documentation](https://faiss.ai)

---

## المكتبات الرئيسية

```txt
fastapi>=0.104.0          # REST API framework
uvicorn>=0.24.0           # ASGI server
streamlit>=1.28.0         # واجهة المستخدم
pydantic>=2.0.0           # Data validation
httpx>=0.25.0             # Async HTTP client

scikit-learn>=1.3.0       # TF-IDF, K-Means, SVD
rank-bm25>=0.2.2          # BM25Okapi
sentence-transformers>=2.2.0  # Embedding model
faiss-cpu>=1.7.4          # Vector search

nltk>=3.8.0               # NLP tools
pyspellchecker>=0.7.0     # Spell correction
ir-datasets>=0.5.0        # Dataset loading

numpy>=1.24.0
scipy>=1.11.0
```

---

## Design Patterns المستخدمة

| Pattern | الملف | الهدف |
|---------|-------|-------|
| **Facade** | `vector_store.py` | تبسيط واجهة FAISS |
| **Adapter** | `ir_datasets_adapter.py` | توحيد تنسيق البيانات |
| **Strategy** | `dataset_loader.py` | تبديل مصدر البيانات |
| **Singleton** | `get_preprocessor()` | تحميل النماذج مرة واحدة |
| **Repository** | `document_store.py` | فصل التخزين عن المنطق |
| **Orchestrator** | `hybrid_indexer.py` | تنسيق خدمات الفهرسة |

---

## .gitignore الرئيسي

```gitignore
# البيانات الضخمة
data/indexes/
data/datasets/

# Python
__pycache__/
*.pyc
*.pyo
venv/
.env

# NLTK
nltk_data/

# IDE
.vscode/
.idea/
```

---

## المصادر

1. Manning et al. (2008). *Introduction to Information Retrieval*. Cambridge.
2. Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*.
3. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
4. Cormack et al. (2009). *Reciprocal rank fusion outperforms condorcet*. SIGIR.
5. Johnson et al. (2019). *Billion-scale similarity search with GPUs*. IEEE.

---

<div align="center">

**IR System 2026** — جامعة دمشق | كلية الهندسة المعلوماتية

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-orange.svg)](https://faiss.ai)

</div>
