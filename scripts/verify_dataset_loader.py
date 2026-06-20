"""
scripts/verify_dataset_loader.py
==================================
سكريبت التحقق من Dataset Loader مع Quora.

ما يفعله هذا السكريبت:
  1. يقرأ أول 5 وثائق من quora corpus.jsonl
  2. يتحقق أن doc_id و text و title تُقرأ صحيحاً
  3. يقرأ أول 3 queries ويتحقق منها
  4. يقرأ أول 3 qrels ويتحقق منها
  5. يُظهر تقرير واضح بالنتائج

تشغيل:
  cd ir_system_2026
  python scripts/verify_dataset_loader.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def separator(title: str) -> None:
    print(f"\n{'━'*55}")
    print(f"  {title}")
    print(f"{'━'*55}")


def check_mark(condition: bool, message: str) -> bool:
    icon = "✅" if condition else "❌"
    print(f"  {icon}  {message}")
    return condition


def main():
    print("=" * 55)
    print("  التحقق من Dataset Loader مع Quora")
    print("=" * 55)

    # ─── الخطوة 1: استيراد DatasetLoader ───────────────────
    separator("الخطوة 1: استيراد المكونات")
    try:
        from services.indexing.dataset_loader import DatasetLoader
        from services.indexing.ir_datasets_adapter import Document, Query, Qrel
        check_mark(True, "DatasetLoader يُستورد بنجاح")
    except ImportError as e:
        check_mark(False, f"فشل الاستيراد: {e}")
        print("\n❌ توقف: تأكد من وجود جميع الملفات")
        sys.exit(1)

    # ─── الخطوة 2: إنشاء DatasetLoader ─────────────────────
    separator("الخطوة 2: إنشاء DatasetLoader")
    try:
        loader = DatasetLoader()
        check_mark(True, "DatasetLoader أُنشئ بنجاح")
        print(f"  ℹ️  مجلد الداتا: {loader.datasets_dir}")
    except Exception as e:
        check_mark(False, f"فشل الإنشاء: {e}")
        sys.exit(1)

    # ─── الخطوة 3: التحقق من وجود quora ────────────────────
    separator("الخطوة 3: التحقق من وجود quora")
    
    exists = loader.dataset_exists("quora")
    check_mark(exists, "quora موجود في DatasetLoader")
    
    if not exists:
        corpus_path = loader.datasets_dir / "quora" / "corpus.jsonl"
        print(f"\n  ❌ المسار المتوقع: {corpus_path}")
        print(f"  ❌ موجود؟ {corpus_path.exists()}")
        print("\n  الحل: تأكد أن corpus.jsonl موجود في data/datasets/quora/")
        sys.exit(1)

    # ─── الخطوة 4: قراءة أول 5 وثائق ───────────────────────
    separator("الخطوة 4: قراءة أول 5 وثائق من corpus")
    
    docs = []
    errors = []
    
    try:
        for doc in loader.stream_documents("quora", max_docs=5):
            docs.append(doc)
    except Exception as e:
        errors.append(str(e))

    check_mark(len(docs) > 0, f"قُرئت {len(docs)} وثائق")
    check_mark(len(errors) == 0, f"لا أخطاء في القراءة")

    if docs:
        # تحقق من محتوى أول وثيقة
        first = docs[0]
        
        print(f"\n  أول وثيقة:")
        print(f"    doc_id : {repr(first.doc_id)}")
        print(f"    text   : {repr(first.text[:80])}...")
        print(f"    title  : {repr(first.title)}")
        
        has_doc_id = bool(first.doc_id and first.doc_id.strip())
        has_text   = bool(first.text and first.text.strip())
        
        check_mark(has_doc_id, "doc_id غير فارغ")
        check_mark(has_text,   "text غير فارغ")
        check_mark(
            isinstance(first, Document),
            "النوع صحيح (Document)"
        )

        # تحقق من كل الوثائق الخمس
        all_have_id   = all(bool(d.doc_id and d.doc_id.strip()) for d in docs)
        all_have_text = all(bool(d.text and d.text.strip()) for d in docs)
        
        check_mark(all_have_id,   "كل الوثائق لها doc_id")
        check_mark(all_have_text, "كل الوثائق لها text")

    # ─── الخطوة 5: قراءة queries ────────────────────────────
    separator("الخطوة 5: قراءة أول 3 queries")
    
    queries = []
    try:
        for q in loader.stream_queries("quora"):
            queries.append(q)
            if len(queries) >= 3:
                break
    except FileNotFoundError as e:
        check_mark(False, f"queries.jsonl غير موجود: {e}")
    except Exception as e:
        check_mark(False, f"خطأ في قراءة queries: {e}")
    
    if queries:
        check_mark(True, f"قُرئت {len(queries)} استعلامات")
        first_q = queries[0]
        print(f"\n  أول استعلام:")
        print(f"    query_id : {repr(first_q.query_id)}")
        print(f"    text     : {repr(first_q.text)}")
        check_mark(bool(first_q.query_id), "query_id غير فارغ")
        check_mark(bool(first_q.text),     "text غير فارغ")

    # ─── الخطوة 6: قراءة qrels ──────────────────────────────
    separator("الخطوة 6: قراءة أول 3 qrels")
    
    qrels = []
    try:
        for qr in loader.stream_qrels("quora"):
            qrels.append(qr)
            if len(qrels) >= 3:
                break
    except FileNotFoundError as e:
        check_mark(False, f"qrels.jsonl غير موجود: {e}")
    except Exception as e:
        check_mark(False, f"خطأ في قراءة qrels: {e}")
    
    if qrels:
        check_mark(True, f"قُرئت {len(qrels)} qrels")
        first_qr = qrels[0]
        print(f"\n  أول qrel:")
        print(f"    query_id  : {repr(first_qr.query_id)}")
        print(f"    doc_id    : {repr(first_qr.doc_id)}")
        print(f"    relevance : {first_qr.relevance}")
        check_mark(bool(first_qr.query_id), "query_id غير فارغ")
        check_mark(bool(first_qr.doc_id),   "doc_id غير فارغ")

    # ─── الخطوة 7: اختبار get_full_text ─────────────────────
    separator("الخطوة 7: اختبار get_full_text()")
    
    if docs:
        first = docs[0]
        try:
            full_text = first.get_full_text()
            check_mark(bool(full_text), "get_full_text() يُرجع نصاً")
            print(f"\n  get_full_text() preview:")
            print(f"    {repr(full_text[:120])}...")
            
            # إذا كان لديه title، يجب أن يكون مدمجاً
            if first.title:
                check_mark(
                    first.title in full_text or first.text in full_text,
                    "full_text يدمج title + text"
                )
        except AttributeError:
            check_mark(False, "get_full_text() غير موجود في Document")
            print("  ← هذا يحتاج إضافة هذه الدالة لـ Document class")

    # ─── الخطوة 8: اختبار الأداء ────────────────────────────
    separator("الخطوة 8: اختبار الأداء (أول 1000 وثيقة)")
    
    import time
    start = time.time()
    count = 0
    
    try:
        for doc in loader.stream_documents("quora", max_docs=1000):
            count += 1
    except Exception as e:
        check_mark(False, f"خطأ في stream: {e}")
    
    elapsed = time.time() - start
    rate = count / elapsed if elapsed > 0 else 0
    
    check_mark(count == 1000, f"قُرئت 1000 وثيقة")
    print(f"  ⏱️  الوقت: {elapsed:.2f} ثانية")
    print(f"  ⚡ السرعة: {rate:,.0f} وثيقة/ثانية")
    
    # تقدير وقت quora كاملة
    estimated_full = 522_931 / rate if rate > 0 else 0
    print(f"  📊 تقدير وقت Quora كاملة (522K): ~{estimated_full:.0f} ثانية ({estimated_full/60:.1f} دقيقة)")

    # ─── الخلاصة النهائية ────────────────────────────────────
    separator("الخلاصة النهائية")
    
    all_passed = (
        len(docs) > 0
        and len(errors) == 0
        and bool(docs[0].doc_id)
        and bool(docs[0].text)
        and len(queries) > 0
        and len(qrels) > 0
    )
    
    if all_passed:
        print("  🎉 DatasetLoader يعمل بشكل صحيح مع Quora!")
        print()
        print("  الخطوة التالية:")
        print("  ← يمكن الانتقال لربط DocumentStore مع الـ indexers")
        print("  ← ثم بناء TF-IDF و BM25 على Quora")
    else:
        print("  ⚠️  يوجد مشاكل تحتاج معالجة قبل المتابعة")
        print("  ← راجع ❌ أعلاه وأصلحها أولاً")


if __name__ == "__main__":
    main()