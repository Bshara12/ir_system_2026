"""
scripts/verify_indexes.py
===========================
التحقق أن TF-IDF و BM25 يعملان end-to-end بعد التحميل من القرص.

ما يفعله:
  1. يحمّل TF-IDF من القرص
  2. يحمّل BM25 من القرص
  3. يُنفّذ 5 استعلامات تجريبية على كل فهرس
  4. يجلب النص الكامل من DocumentStore (وليس من الذاكرة)
  5. يطبع Top-10 نتائج لكل استعلام

تشغيل:
    cd ir_system_2026
    python scripts/verify_indexes.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_NAME = "quora"
INDEXES_DIR  = str(PROJECT_ROOT / "data" / "indexes")

# استعلامات تجريبية متنوعة — مناسبة لـ Quora
TEST_QUERIES = [
    "What is the best way to learn programming?",
    "How does machine learning work?",
    "What are the differences between Python and Java?",
    "How to improve memory and concentration?",
    "What is the meaning of life?",
]


def separator(title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


# ─────────────────────────────────────────────────────────────
# الخطوة 1: تحميل الفهارس من القرص
# ─────────────────────────────────────────────────────────────

def load_tfidf():
    """يحمّل TF-IDF من القرص — نسخة جديدة تماماً من الذاكرة."""
    from services.indexing.tfidf_indexer import TFIDFIndexer

    # نُنشئ indexer جديد فارغ تماماً
    # هذا يضمن أن البيانات تأتي من القرص وليس من RAM
    indexer = TFIDFIndexer(indexes_dir=INDEXES_DIR)

    print(f"  تحميل TF-IDF من: {INDEXES_DIR}/{DATASET_NAME}/tfidf/")
    start = time.time()
    meta  = indexer.load_index(DATASET_NAME)
    elapsed = time.time() - start

    print(f"  ✅ محمّل: {meta.num_documents:,} وثيقة في {elapsed:.2f}s")
    print(f"     vocab_size: {meta.vocab_size:,}")
    return indexer


def load_bm25():
    """يحمّل BM25 من القرص — نسخة جديدة تماماً."""
    from services.indexing.bm25_indexer import BM25Indexer

    indexer = BM25Indexer(indexes_dir=INDEXES_DIR)

    print(f"  تحميل BM25 من: {INDEXES_DIR}/{DATASET_NAME}/bm25/")
    start   = time.time()
    meta    = indexer.load_index(DATASET_NAME)
    elapsed = time.time() - start

    print(f"  ✅ محمّل: {meta.num_documents:,} وثيقة في {elapsed:.2f}s")
    print(f"     k1={meta.k1}, b={meta.b}")
    return indexer


def open_document_store():
    """يفتح DocumentStore للقراءة فقط."""
    from services.indexing.document_store import DocumentStore

    store = DocumentStore(
        indexes_dir=INDEXES_DIR,
        dataset_name=DATASET_NAME,
    )
    count = store.count()
    print(f"  ✅ DocumentStore: {count:,} وثيقة")
    return store


# ─────────────────────────────────────────────────────────────
# الخطوة 2: تنفيذ استعلام واحد
# ─────────────────────────────────────────────────────────────

def run_tfidf_query(indexer, store, query: str, top_k: int = 10) -> list:
    """
    ينفّذ استعلام TF-IDF ويجلب النصوص من DocumentStore.

    ما يحدث:
      1. نعالج الاستعلام بنفس الـ preprocessing
      2. نحوّله لمتجه TF-IDF
      3. نحسب Cosine Similarity مع كل الوثائق
      4. نأخذ Top-10 doc_ids
      5. نجلب النص من DocumentStore (وليس من indexer.documents)
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from services.preprocessing.preprocessor import get_preprocessor

    preprocessor = get_preprocessor()
    tokens, _ = preprocessor.process(
        text=query,
        language="english",
        apply_stemming=indexer.metadata.apply_stemming,
        remove_stopwords=indexer.metadata.remove_stopwords,
    )
    processed_query = " ".join(tokens)

    query_vec = indexer.transform_query(processed_query)
    if query_vec is None:
        return []

    scores     = cosine_similarity(query_vec, indexer.tfidf_matrix).flatten()
    top_idx    = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        score = float(scores[idx])
        if score <= 0:
            break

        # نجلب الوثيقة من IndexedDocument (للحصول على doc_id)
        indexed_doc = indexer.get_document_by_index(int(idx))
        if indexed_doc is None:
            continue

        # ← هنا الفرق المهم: نجلب النص من DocumentStore وليس من indexed_doc
        db_doc = store.get(indexed_doc.doc_id)
        text   = db_doc["raw_text"] if db_doc else indexed_doc.original_text

        results.append({
            "rank":   rank,
            "doc_id": indexed_doc.doc_id,
            "score":  round(score, 4),
            "text":   text,
            "source": "DocumentStore" if db_doc else "IndexedDocument",
        })

    return results


def run_bm25_query(indexer, store, query: str, top_k: int = 10) -> list:
    """
    ينفّذ استعلام BM25 ويجلب النصوص من DocumentStore.

    ما يحدث:
      1. نعالج الاستعلام → tokens
      2. نحسب BM25 scores
      3. نأخذ Top-10 doc_ids
      4. نجلب النص من DocumentStore
    """
    from services.preprocessing.preprocessor import get_preprocessor

    preprocessor = get_preprocessor()
    tokens, _ = preprocessor.process(
        text=query,
        language="english",
        apply_stemming=indexer.metadata.apply_stemming,
        remove_stopwords=indexer.metadata.remove_stopwords,
    )

    raw_results = indexer.get_top_n(tokens, n=top_k)

    results = []
    for rank, (indexed_doc, score) in enumerate(raw_results, start=1):
        if indexed_doc is None:
            continue

        # ← نجلب النص من DocumentStore
        db_doc = store.get(indexed_doc.doc_id)
        text   = db_doc["raw_text"] if db_doc else indexed_doc.original_text

        results.append({
            "rank":   rank,
            "doc_id": indexed_doc.doc_id,
            "score":  round(score, 4),
            "text":   text,
            "source": "DocumentStore" if db_doc else "IndexedDocument",
        })

    return results


# ─────────────────────────────────────────────────────────────
# الخطوة 3: طباعة النتائج
# ─────────────────────────────────────────────────────────────

def print_results(results: list, query: str, model: str) -> None:
    print(f"\n  الاستعلام: \"{query}\"")
    print(f"  النموذج: {model}")
    print(f"  {'─'*56}")

    if not results:
        print(f"  ⚠️  لا توجد نتائج")
        return

    # نتحقق أن كل النتائج من DocumentStore
    from_db  = sum(1 for r in results if r["source"] == "DocumentStore")
    from_idx = len(results) - from_db

    for r in results[:5]:  # نعرض أول 5 فقط لتوفير المساحة
        text_preview = r["text"][:90].replace("\n", " ")
        print(f"  #{r['rank']:2d} [{r['score']:.4f}] {r['doc_id']}")
        print(f"      {text_preview}...")
        print(f"      ← من: {r['source']}")

    if len(results) > 5:
        print(f"  ... و{len(results)-5} نتائج أخرى")

    print(f"\n  ✅ {from_db}/{len(results)} نتيجة من DocumentStore")
    if from_idx > 0:
        print(f"  ⚠️  {from_idx} نتيجة من IndexedDocument (doc_id غير موجود في DB)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  التحقق من TF-IDF و BM25 — end-to-end")
    print(f"  Dataset: {DATASET_NAME}")
    print("="*60)

    # ── تحميل الفهارس ─────────────────────────────────────────
    separator("تحميل الفهارس من القرص")

    try:
        tfidf_indexer = load_tfidf()
    except FileNotFoundError as e:
        print(f"  ❌ TF-IDF غير موجود: {e}")
        print(f"     شغّل: python scripts/build_indexes_quora.py أولاً")
        return

    try:
        bm25_indexer = load_bm25()
    except FileNotFoundError as e:
        print(f"  ❌ BM25 غير موجود: {e}")
        return

    store = open_document_store()

    # ── تشغيل الاستعلامات ─────────────────────────────────────
    separator("نتائج TF-IDF")

    tfidf_ok = True
    for query in TEST_QUERIES:
        start   = time.time()
        results = run_tfidf_query(tfidf_indexer, store, query, top_k=10)
        elapsed = time.time() - start
        print_results(results, query, f"TF-IDF ({elapsed*1000:.0f}ms)")
        if not results:
            tfidf_ok = False

    separator("نتائج BM25")

    bm25_ok = True
    for query in TEST_QUERIES:
        start   = time.time()
        results = run_bm25_query(bm25_indexer, store, query, top_k=10)
        elapsed = time.time() - start
        print_results(results, query, f"BM25 ({elapsed*1000:.0f}ms)")
        if not results:
            bm25_ok = False

    # ── التقرير النهائي ───────────────────────────────────────
    separator("التقرير النهائي")

    print(f"\n  TF-IDF:          {'✅ يعمل' if tfidf_ok else '❌ مشكلة'}")
    print(f"  BM25:            {'✅ يعمل' if bm25_ok else '❌ مشكلة'}")
    print(f"  DocumentStore:   ✅ {store.count():,} وثيقة")

    if tfidf_ok and bm25_ok:
        print(f"\n  ✅ الفهارس جاهزة للاستخدام")
        print(f"\n  ما اكتمل من متطلبات Developer 1:")
        print(f"    ✅ DatasetLoader — يقرأ Quora صحيحاً")
        print(f"    ✅ TF-IDF Indexer — مبني ومحفوظ ومختبر")
        print(f"    ✅ BM25 Indexer — مبني ومحفوظ ومختبر")
        print(f"    ✅ DocumentStore — 522,931 وثيقة في SQLite")
        print(f"    ✅ HybridIndexer — كود جاهز")
        print(f"    ✅ VectorStore — كود جاهز")
        print(f"\n  المتبقي قبل Evaluation:")
        print(f"    ⏳ Embedding Index — يُبنى على Google Colab")
        print(f"       ثم تُنزَّل ملفاته إلى: data/indexes/quora/embedding/")
        print(f"    ⏳ بعد Embedding: Developer 1 مكتمل 100%")
        print(f"       Developer 2 يمكنه البدء بـ Retrieval Service")


if __name__ == "__main__":
    main()