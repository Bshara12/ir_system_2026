"""
scripts/download_datasets.py
==============================
تحميل الداتاسيت المطلوبة للمشروع.

الداتاسيت المختارة:
  1. beir/quora      ← 522,931 وثيقة (يوافق شرط 200K+)
  2. beir/trec-covid ← 171,332 وثيقة (داتاسيت علمية متخصصة)

كيف يعمل هذا الملف؟
  يستخدم ir_datasets لتحميل الداتاسيت
  ثم يحوّلها لصيغة JSONL التي يفهمها مشروعنا

تشغيل:
  python scripts/download_datasets.py
"""

import json
import sys
from pathlib import Path

# نضيف جذر المشروع للـ path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────
# إعدادات التحميل
# ─────────────────────────────────────────────────────────────

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

# الداتاسيت المطلوبة مع إعداداتها
DATASETS_CONFIG = {
    "quora": {
        "ir_dataset_id": "beir/quora/test",
        "output_dir":    "quora",
        "max_docs":      None,    # نحمّل كل الوثائق (522,931)
        "description":   "Quora Question Pairs — 522K وثيقة",
    },
    "trec-covid": {
        "ir_dataset_id": "beir/trec-covid",
        "output_dir":    "trec-covid",
        "max_docs":      None,    # نحمّل كل الوثائق (171K)
        "description":   "TREC-COVID — 171K وثيقة علمية",
    },
}


# ─────────────────────────────────────────────────────────────
# دوال التحميل
# ─────────────────────────────────────────────────────────────

def download_corpus(dataset_id: str, output_path: Path, max_docs=None):
    """
    يحمّل corpus (الوثائق) من ir_datasets ويحفظها كـ JSONL.

    ما هو JSONL؟
    ─────────────
    JSON Lines: كل سطر = وثيقة JSON مستقلة
    مثال:
      {"doc_id": "1", "title": "Cloud", "text": "Cloud storage..."}
      {"doc_id": "2", "title": "AI", "text": "Artificial intelligence..."}

    لماذا JSONL وليس JSON عادي؟
    ────────────────────────────
    JSON عادي: يحتاج تحميل الملف كله في الذاكرة
    JSONL: نقرأ سطراً سطراً = موفّر للذاكرة مع 500K وثيقة
    """
    try:
        import ir_datasets
    except ImportError:
        print("❌ ir_datasets غير مثبّت. شغّل: pip install ir-datasets")
        return False

    print(f"  تحميل corpus من: {dataset_id}")

    try:
        dataset = ir_datasets.load(dataset_id)
    except Exception as e:
        print(f"  ❌ فشل التحميل: {e}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in dataset.docs_iter():
            # نستخرج الحقول المشتركة بين الداتاسيت المختلفة
            doc_data = {
                "doc_id": getattr(doc, "doc_id", getattr(doc, "_id", str(count))),
                "title":  getattr(doc, "title",  ""),
                "text":   getattr(doc, "text",   ""),
            }
            f.write(json.dumps(doc_data, ensure_ascii=False) + "\n")
            count += 1

            if count % 50_000 == 0:
                print(f"  ✓ {count:,} وثيقة...")

            if max_docs and count >= max_docs:
                break

    print(f"  ✅ corpus: {count:,} وثيقة محفوظة في {output_path}")
    return True


def download_queries(dataset_id: str, output_path: Path):
    """يحمّل queries ويحفظها كـ JSONL."""
    try:
        import ir_datasets
        dataset = ir_datasets.load(dataset_id)
    except Exception as e:
        print(f"  ⚠️  لا توجد queries: {e}")
        return False

    if not dataset.has_queries():
        print("  ⚠️  هذه الداتاسيت لا تحتوي queries")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for query in dataset.queries_iter():
            q_data = {
                "query_id": query.query_id,
                "text":     query.text,
            }
            f.write(json.dumps(q_data, ensure_ascii=False) + "\n")
            count += 1

    print(f"  ✅ queries: {count:,} استعلام محفوظ")
    return True


def download_qrels(dataset_id: str, output_path: Path):
    """
    يحمّل qrels ويحفظها كـ JSONL.

    ما هو qrels؟
    ─────────────
    qrels = Query Relevance Judgments
    لكل استعلام: قائمة الوثائق الصحيحة ودرجة صلتها

    مثال:
      query_id=1, doc_id=D001, relevance=2  ← صلة عالية
      query_id=1, doc_id=D045, relevance=1  ← صلة متوسطة
      query_id=1, doc_id=D099, relevance=0  ← غير ذات صلة

    درجات الصلة:
      0 = غير ذات صلة
      1 = ذات صلة جزئية
      2 = ذات صلة عالية
    """
    try:
        import ir_datasets
        dataset = ir_datasets.load(dataset_id)
    except Exception as e:
        print(f"  ⚠️  لا توجد qrels: {e}")
        return False

    if not dataset.has_qrels():
        print("  ⚠️  هذه الداتاسيت لا تحتوي qrels")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for qrel in dataset.qrels_iter():
            qrel_data = {
                "query_id":  qrel.query_id,
                "doc_id":    qrel.doc_id,
                "relevance": qrel.relevance,
            }
            f.write(json.dumps(qrel_data, ensure_ascii=False) + "\n")
            count += 1

    print(f"  ✅ qrels: {count:,} حكم محفوظ")
    return True


# ─────────────────────────────────────────────────────────────
# التنفيذ الرئيسي
# ─────────────────────────────────────────────────────────────

def download_dataset(name: str, config: dict) -> bool:
    """يحمّل داتاسيت كاملة (corpus + queries + qrels)."""
    print(f"\n{'━'*55}")
    print(f"تحميل: {name} — {config['description']}")
    print(f"{'━'*55}")

    ds_id      = config["ir_dataset_id"]
    output_dir = DATASETS_DIR / config["output_dir"]
    max_docs   = config.get("max_docs")

    # فحص ما إذا كان corpus موجوداً بالفعل
    corpus_path = output_dir / "corpus.jsonl"
    if corpus_path.exists():
        existing = sum(1 for _ in open(corpus_path, encoding="utf-8"))
        print(f"  ℹ️  corpus موجود: {existing:,} وثيقة")

        if existing >= 200_000 or (max_docs and existing >= max_docs):
            print("  ✅ لا حاجة لإعادة التحميل")
            # نتحقق فقط من queries و qrels
        else:
            print(f"  ⚠️  الحجم غير كافٍ ({existing:,}) — إعادة التحميل")
            corpus_path.unlink()

    if not corpus_path.exists():
        success = download_corpus(ds_id, corpus_path, max_docs)
        if not success:
            return False

    # تحميل queries
    queries_path = output_dir / "queries.jsonl"
    if not queries_path.exists():
        download_queries(ds_id, queries_path)

    # تحميل qrels
    qrels_path = output_dir / "qrels.jsonl"
    if not qrels_path.exists():
        download_qrels(ds_id, qrels_path)

    # ملخص نهائي
    print(f"\n  📂 الملفات في {output_dir}:")
    for f in output_dir.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        lines   = sum(1 for _ in open(f, encoding="utf-8"))
        print(f"    {f.name}: {lines:,} سطر ({size_mb:.1f} MB)")

    return True


def main():
    print("=" * 55)
    print("تحميل الداتاسيت — مشروع IR 2026")
    print("=" * 55)

    # ماذا تريد تحميل؟
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "quora":
        configs = {"quora": DATASETS_CONFIG["quora"]}
    elif target == "trec-covid":
        configs = {"trec-covid": DATASETS_CONFIG["trec-covid"]}
    else:
        configs = DATASETS_CONFIG

    for name, config in configs.items():
        download_dataset(name, config)

    print(f"\n{'='*55}")
    print("✅ اكتمل التحميل")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()