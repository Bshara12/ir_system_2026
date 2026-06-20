"""
scripts/build_indexes_quora.py
================================
بناء TF-IDF و BM25 على Quora كاملة (522,931 وثيقة)
مع ملء SQLite DocumentStore تلقائياً.

تشغيل:
    cd ir_system_2026
    python scripts/build_indexes_quora.py

ما يفعله هذا السكريبت:
  1. يبني TF-IDF Index على Quora كاملة
  2. يبني BM25 Index على Quora كاملة
  3. يملأ SQLite DocumentStore تلقائياً أثناء كل بناء
  4. يحفظ كل الفهارس على القرص

ملاحظة:
  Embedding Index سيُبنى منفصلاً على Google Colab
  ثم تُنزَّل ملفاته إلى data/indexes/quora/embedding/
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_tfidf(dataset_name: str) -> bool:
    print("\n" + "="*55)
    print(f"[1/2] بناء TF-IDF على '{dataset_name}'")
    print("="*55)

    try:
        from services.indexing.tfidf_indexer import TFIDFIndexer
        from services.indexing.document_store import DocumentStore

        indexer = TFIDFIndexer()

        start = time.time()
        meta = indexer.build_index(dataset_name=dataset_name)
        elapsed = time.time() - start

        print(f"\n✅ TF-IDF مبني: {meta.num_documents:,} وثيقة في {elapsed:.1f}s")

        # ── ملء DocumentStore ──
        print("\n[TF-IDF] ملء SQLite DocumentStore...")
        store = DocumentStore(
            indexes_dir=str(PROJECT_ROOT / "data" / "indexes"),
            dataset_name=dataset_name,
        )
        added = store.add_batch(indexer.documents)
        print(f"✅ DocumentStore: {added:,} وثيقة مُضافة")

        # ── حفظ الفهرس ──
        print("\n[TF-IDF] حفظ الفهرس...")
        indexer.save_index(dataset_name)
        print("✅ TF-IDF محفوظ")

        return True

    except Exception as e:
        print(f"❌ فشل TF-IDF: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_bm25(dataset_name: str) -> bool:
    print("\n" + "="*55)
    print(f"[2/2] بناء BM25 على '{dataset_name}'")
    print("="*55)

    try:
        from services.indexing.bm25_indexer import BM25Indexer
        from services.indexing.document_store import DocumentStore

        indexer = BM25Indexer()

        start = time.time()
        meta = indexer.build_index(dataset_name=dataset_name)
        elapsed = time.time() - start

        print(f"\n✅ BM25 مبني: {meta.num_documents:,} وثيقة في {elapsed:.1f}s")

        # ── ملء DocumentStore (INSERT OR REPLACE — آمن إذا موجود) ──
        print("\n[BM25] تحديث SQLite DocumentStore...")
        store = DocumentStore(
            indexes_dir=str(PROJECT_ROOT / "data" / "indexes"),
            dataset_name=dataset_name,
        )
        added = store.add_batch(indexer.documents)
        print(f"✅ DocumentStore: {added:,} وثيقة (INSERT OR REPLACE)")

        # ── حفظ الفهرس ──
        print("\n[BM25] حفظ الفهرس...")
        indexer.save_index(dataset_name)
        print("✅ BM25 محفوظ")

        return True

    except Exception as e:
        print(f"❌ فشل BM25: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_results(dataset_name: str) -> None:
    print("\n" + "="*55)
    print("التحقق النهائي")
    print("="*55)

    from services.indexing.document_store import DocumentStore
    from pathlib import Path

    indexes_dir = PROJECT_ROOT / "data" / "indexes"

    # فحص DocumentStore
    store = DocumentStore(str(indexes_dir), dataset_name)
    count = store.count()
    print(f"  SQLite DocumentStore: {count:,} وثيقة")

    # فحص الملفات
    for index_type in ["tfidf", "bm25"]:
        index_dir = indexes_dir / dataset_name / index_type
        if index_dir.exists():
            files = list(index_dir.iterdir())
            total_mb = sum(f.stat().st_size for f in files) / (1024**2)
            print(f"  {index_type}/: {len(files)} ملف، {total_mb:.1f} MB")
        else:
            print(f"  {index_type}/: ❌ غير موجود")

    # اختبار get() من DocumentStore
    if count > 0:
        # نجلب أول doc بشكل عشوائي
        with store._connect() as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents LIMIT 1"
            ).fetchone()
        if row:
            doc = store.get(row["doc_id"])
            print(f"\n  اختبار get():")
            print(f"    doc_id   = {doc['doc_id']}")
            print(f"    title    = {doc['title']}")
            print(f"    raw_text = {doc['raw_text'][:80]}...")
            print(f"  ✅ DocumentStore يعمل صحيحاً")


def main():
    dataset_name = "quora"

    print("="*55)
    print(f"بناء فهارس TF-IDF و BM25 على: {dataset_name}")
    print(f"الوثائق: 522,931")
    print("="*55)

    start_total = time.time()

    ok1 = build_tfidf(dataset_name)
    ok2 = build_bm25(dataset_name) if ok1 else False

    if ok1 and ok2:
        verify_results(dataset_name)

    elapsed = time.time() - start_total
    print(f"\n{'='*55}")
    print(f"{'✅ اكتمل' if ok1 and ok2 else '❌ فشل'} في {elapsed/60:.1f} دقيقة")
    print("="*55)

    if ok1 and ok2:
        print("\nالخطوة التالية:")
        print("  ← بناء Embedding Index على Google Colab")
        print("  ← ثم تنزيل الملفات إلى: data/indexes/quora/embedding/")


if __name__ == "__main__":
    main()