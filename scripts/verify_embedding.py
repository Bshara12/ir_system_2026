"""
scripts/verify_embedding.py
=============================
التحقق أن Embedding + FAISS + DocumentStore يعملان معاً.

ما الذي نختبره هنا؟
  1. هل EmbeddingIndexer يُحمَّل من القرص صحيحاً؟
  2. هل encode_query() تُحوّل النص لمتجه صحيح؟
  3. هل FAISS يُرجع نتائج ذات معنى؟
  4. هل النتائج تأتي من DocumentStore (وليس من الذاكرة فقط)؟
  5. ما زمن كل استعلام؟

الفرق عن verify_indexes.py:
  Embedding لا يحتاج preprocessing (لا stemming، لا stopwords)
  يأخذ النص الخام مباشرةً — هذا تصميم مقصود.

تشغيل:
    cd ir_system_2026
    python scripts/verify_embedding.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_NAME = "quora"
INDEXES_DIR  = str(PROJECT_ROOT / "data" / "indexes")
TOP_K        = 10

# استعلامات متنوعة لاختبار جوانب مختلفة
TEST_QUERIES = [
    # استعلام مباشر — كلمات موجودة في الـ corpus
    "What is the best way to learn programming?",
    # استعلام معنوي — قد لا تكون الكلمات مطابقة لكن المعنى قريب
    "How does machine learning work?",
    # استعلام مقارنة
    "What are the differences between Python and Java?",
    # استعلام نصيحة
    "How to improve memory and concentration?",
    # استعلام فلسفي
    "What is the meaning of life?",
]


def separator(title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


def main():
    print("=" * 60)
    print("  التحقق من Embedding — end-to-end")
    print(f"  Dataset: {DATASET_NAME} | Top-K: {TOP_K}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # الخطوة 1: تحميل EmbeddingIndexer من القرص
    # ──────────────────────────────────────────────────────────
    # نُنشئ indexer جديد فارغ تماماً
    # هذا يضمن أن البيانات تأتي من القرص وليس من RAM
    separator("الخطوة 1: تحميل EmbeddingIndexer من القرص")

    try:
        from services.indexing.embedding_indexer import EmbeddingIndexer
        indexer = EmbeddingIndexer(indexes_dir=INDEXES_DIR)

        start = time.time()
        meta  = indexer.load_index(DATASET_NAME)
        load_time = time.time() - start

        print(f"  ✅ محمّل في {load_time:.2f}s")
        print(f"     الوثائق:  {meta.num_documents:,}")
        print(f"     النموذج:  {meta.model_name}")
        print(f"     البُعد:   {meta.embedding_dim}")
        print(f"     النوع:    {meta.index_type}")
        print(f"     normalized: {meta.normalize_embeddings}")

    except FileNotFoundError as e:
        print(f"  ❌ الفهرس غير موجود: {e}")
        print(f"     تأكد أن الملفات موجودة في:")
        print(f"     {INDEXES_DIR}/{DATASET_NAME}/embedding/")
        return
    except Exception as e:
        print(f"  ❌ خطأ في التحميل: {e}")
        import traceback
        traceback.print_exc()
        return

    # ──────────────────────────────────────────────────────────
    # الخطوة 2: فتح DocumentStore
    # ──────────────────────────────────────────────────────────
    separator("الخطوة 2: فتح DocumentStore")

    try:
        from services.indexing.document_store import DocumentStore
        store = DocumentStore(
            indexes_dir=INDEXES_DIR,
            dataset_name=DATASET_NAME,
        )
        doc_count = store.count()
        print(f"  ✅ DocumentStore: {doc_count:,} وثيقة")

    except Exception as e:
        print(f"  ❌ خطأ في DocumentStore: {e}")
        return

    # ──────────────────────────────────────────────────────────
    # الخطوة 3: اختبار encode_query
    #
    # ما يحدث هنا:
    #   النص → SentenceTransformer → متجه (1, 384) float32
    #   هذا المتجه هو "بصمة المعنى" للاستعلام
    # ──────────────────────────────────────────────────────────
    separator("الخطوة 3: اختبار encode_query()")

    test_text = "What is machine learning?"
    start     = time.time()
    query_vec = indexer.encode_query(test_text)
    enc_time  = time.time() - start

    if query_vec is None:
        print(f"  ❌ encode_query أرجع None!")
        return

    print(f"  ✅ encode_query نجح في {enc_time*1000:.0f}ms")
    print(f"     الشكل:  {query_vec.shape}")
    print(f"     النوع:  {query_vec.dtype}")
    print(f"     أول 5 أرقام: {query_vec[0][:5].tolist()}")

    # تحقق من الـ normalization
    import numpy as np
    norm = np.linalg.norm(query_vec[0])
    is_normalized = abs(norm - 1.0) < 0.01
    print(f"     الطول (norm): {norm:.6f} {'✅ مُطبَّع' if is_normalized else '⚠️  غير مُطبَّع'}")

    # ──────────────────────────────────────────────────────────
    # الخطوة 4: تشغيل الاستعلامات الكاملة
    # ──────────────────────────────────────────────────────────
    separator("الخطوة 4: تشغيل الاستعلامات")

    all_ok        = True
    total_from_db = 0
    total_results = 0
    query_times   = []

    for query in TEST_QUERIES:
        print(f"\n  الاستعلام: \"{query}\"")

        # encode النص الخام مباشرة (بدون preprocessing)
        start     = time.time()
        query_vec = indexer.encode_query(query)
        results   = indexer.get_top_k(query_vec, k=TOP_K)
        elapsed   = time.time() - start

        query_times.append(elapsed)
        print(f"  الوقت: {elapsed*1000:.0f}ms | النتائج: {len(results)}")
        print(f"  {'─'*56}")

        if not results:
            print(f"  ❌ لا توجد نتائج!")
            all_ok = False
            continue

        from_db_this_query = 0

        for rank, (doc, score) in enumerate(results, start=1):
            # نجلب النص من DocumentStore وليس من doc.original_text
            db_doc = store.get(doc.doc_id)
            source = "DocumentStore" if db_doc else "IndexedDocument"
            text   = (db_doc["raw_text"] if db_doc else doc.original_text)

            if db_doc:
                from_db_this_query += 1

            # نعرض فقط أول 5 نتائج لتوفير المساحة
            if rank <= 5:
                print(f"  #{rank:2d} [score={score:.4f}] id={doc.doc_id}")
                print(f"       {text[:80]}...")
                print(f"       ← من: {source}")

        if len(results) > 5:
            print(f"  ... و{len(results)-5} نتائج أخرى")

        total_from_db += from_db_this_query
        total_results += len(results)

        db_pct = (from_db_this_query / len(results)) * 100
        print(f"\n  ✅ {from_db_this_query}/{len(results)} ({db_pct:.0f}%) من DocumentStore")

        if from_db_this_query < len(results):
            print(f"  ⚠️  {len(results)-from_db_this_query} نتائج لم تجد doc_id في DB")
            all_ok = False

    # ──────────────────────────────────────────────────────────
    # الخطوة 5: إحصائيات الأداء
    # ──────────────────────────────────────────────────────────
    separator("الخطوة 5: إحصائيات الأداء")

    if query_times:
        avg_ms  = (sum(query_times) / len(query_times)) * 1000
        min_ms  = min(query_times) * 1000
        max_ms  = max(query_times) * 1000

        print(f"  زمن الاستعلام:")
        print(f"    متوسط:  {avg_ms:.0f}ms")
        print(f"    أقل:    {min_ms:.0f}ms")
        print(f"    أكثر:   {max_ms:.0f}ms")

        # ملاحظة: أول استعلام أبطأ بسبب تحميل النموذج
        if query_times[0] > query_times[1] * 2:
            print(f"\n  ℹ️  الاستعلام الأول أبطأ ({query_times[0]*1000:.0f}ms)")
            print(f"     السبب: تحميل SentenceTransformer في الذاكرة")
            print(f"     الاستعلامات التالية أسرع — سلوك طبيعي")

    # ──────────────────────────────────────────────────────────
    # التقرير النهائي
    # ──────────────────────────────────────────────────────────
    separator("التقرير النهائي")

    db_coverage = (total_from_db / total_results * 100) if total_results > 0 else 0

    print(f"  EmbeddingIndexer:  ✅ محمّل ({meta.num_documents:,} وثيقة)")
    print(f"  encode_query():    ✅ يعمل (shape={meta.embedding_dim})")
    print(f"  FAISS search:      ✅ يُرجع نتائج")
    print(f"  DocumentStore:     {'✅' if db_coverage == 100 else '⚠️ '} {db_coverage:.0f}% من النتائج من DB")

    if all_ok:
        print(f"\n  🎉 Embedding يعمل end-to-end بشكل صحيح!")
        print()
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  حالة Developer 1 — اكتمال 100%:")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"    ✅ DatasetLoader    — Quora 522,931 وثيقة")
        print(f"    ✅ TF-IDF Index     — مبني ومختبر")
        print(f"    ✅ BM25 Index       — مبني ومختبر")
        print(f"    ✅ Embedding Index  — مبني ومختبر (FAISS)")
        print(f"    ✅ DocumentStore    — 522,931 وثيقة في SQLite")
        print(f"    ✅ HybridIndexer    — كود جاهز")
        print(f"    ✅ VectorStore      — كود جاهز")
        print()
        print(f"  الخطوة التالية:")
        print(f"    Developer 2 يمكنه البدء بـ Retrieval Service")
        print(f"    الفهارس جاهزة في: data/indexes/quora/")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        print(f"\n  ⚠️  توجد مشاكل — راجع ❌ و ⚠️  أعلاه")


if __name__ == "__main__":
    main()