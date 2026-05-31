"""
services/indexing/inverted_index.py
=====================================
فهرس مقلوب كلاسيكي (Inverted Index) مع Boolean Retrieval.

═══════════════════════════════════════════════════
ما هو الـ Inverted Index؟
═══════════════════════════════════════════════════

الفهرس المقلوب هو هيكل البيانات الأساسي في IR.
يُقلب العلاقة من:

  Document → [term1, term2, term3, ...]   ← الطبيعي

إلى:

  term → [doc1, doc3, doc7, ...]          ← المقلوب

هذا يجعل البحث O(1) بدلاً من O(N×L):
  بدون فهرس: ابحث في 200,000 وثيقة كل مرة
  مع فهرس: ابحث في قائمة الـ postings للمصطلح فقط

═══════════════════════════════════════════════════
Posting List — ما هي؟
═══════════════════════════════════════════════════

لكل مصطلح، نحفظ:
  - doc_id    : معرّف الوثيقة
  - frequency : كم مرة ظهر المصطلح فيها
  - positions : في أي مواضع (للـ phrase matching)

مثال:
  "cloud" → [
    Posting(doc_id="d1", freq=2, positions=[0, 5]),
    Posting(doc_id="d3", freq=1, positions=[3]),
  ]

═══════════════════════════════════════════════════
الفرق عن TF-IDF وBM25
═══════════════════════════════════════════════════

Inverted Index : "أين توجد الكلمة؟" → قائمة وثائق
TF-IDF        : "ما وزن الكلمة في الوثيقة؟" → رقم
BM25          : "ما درجة صلة الوثيقة بالاستعلام؟" → رقم

الـ Inverted Index هو الأساس الذي يبنى عليه TF-IDF وBM25
في الأنظمة الكبيرة كـ Elasticsearch وLucene.

═══════════════════════════════════════════════════
لماذا نبنيه منفصلاً رغم وجود TF-IDF وBM25؟
═══════════════════════════════════════════════════

1. Boolean Retrieval: "cloud AND storage AND NOT python"
   TF-IDF وBM25 لا يدعمان Boolean queries مباشرة.

2. Phrase matching: "cloud storage" كعبارة متصلة
   يحتاج positions — غير متوفر في TF-IDF/BM25.

3. Transparency: يُتيح رؤية الـ posting lists مباشرة
   مفيد للتدريس والتقرير.

4. المتطلب الأصلي ينص صراحةً على "Inverted Index".
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.constants import INDEXES_DIR
from services.indexing.dataset_loader import DatasetLoader, Document, get_dataset_loader


# =============================================================
# نماذج البيانات
# =============================================================

@dataclass
class Posting:
    """
    إدخال واحد في الـ Posting List لمصطلح معيّن.

    يحتوي:
      doc_id    : معرّف الوثيقة
      frequency : عدد مرات ظهور المصطلح في هذه الوثيقة
      positions : قائمة مواضع المصطلح (للـ phrase matching)
    """
    doc_id: str
    frequency: int
    positions: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Posting":
        return cls(**data)


@dataclass
class InvertedIndexMetadata:
    """بيانات وصفية لفهرس الـ Inverted Index."""
    dataset_name: str
    num_documents: int
    vocab_size: int               # عدد المصطلحات الفريدة
    total_postings: int           # إجمالي عدد الـ postings
    build_time_seconds: float
    build_timestamp: str
    apply_stemming: bool
    remove_stopwords: bool
    language: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "InvertedIndexMetadata":
        return cls(**data)


# =============================================================
# InvertedIndex — الكلاس الرئيسي
# =============================================================

class InvertedIndex:
    """
    فهرس مقلوب كلاسيكي مع Boolean Retrieval.

    الهيكل الداخلي:
      _index: Dict[term, List[Posting]]
      مثال: {
        "cloud": [Posting("d1",2,[0,5]), Posting("d3",1,[3])],
        "storag": [Posting("d1",1,[1]), Posting("d2",2,[0,4])],
        ...
      }

    ما يُنتجه للنظام:
      - Boolean AND/OR/NOT queries
      - posting_list(term) → قائمة الوثائق
      - document_frequency(term) → df لمصطلح
      - term_statistics() → إحصائيات المفردة
    """

    _INDEX_FILE    = "inverted_index.json"
    _METADATA_FILE = "inverted_metadata.json"
    _DOCMAP_FILE   = "inverted_docmap.json"

    def __init__(
        self,
        indexes_dir: str = INDEXES_DIR,
        dataset_loader: Optional[DatasetLoader] = None,
    ) -> None:
        self.indexes_dir = Path(indexes_dir)
        self._loader     = dataset_loader or get_dataset_loader()

        # الهيكل الرئيسي: term → List[Posting]
        self._index: Dict[str, List[Posting]] = {}

        # خريطة: doc_id → نص الوثيقة الأصلي (للعرض)
        self._doc_store: Dict[str, str] = {}

        # مجموعة كل الوثائق (للـ NOT operation)
        self._all_doc_ids: Set[str] = set()

        self.metadata: Optional[InvertedIndexMetadata] = None

    # ----------------------------------------------------------
    # بناء الفهرس
    # ----------------------------------------------------------

    def build_index(
        self,
        dataset_name: str,
        apply_stemming: bool = True,
        remove_stopwords: bool = True,
        language: str = "english",
        max_docs: Optional[int] = None,
        store_positions: bool = True,
    ) -> InvertedIndexMetadata:
        """
        يبني الفهرس المقلوب من مجموعة بيانات.

        الخطوات:
          1. تحميل الوثائق
          2. معالجة كل وثيقة → tokens
          3. لكل token: أضف Posting للـ index
          4. احفظ الإحصائيات

        store_positions:
          True  → أبطأ وأكبر لكن يدعم phrase matching
          False → أسرع وأصغر، Boolean فقط
        """
        print(f"\n{'='*55}")
        print(f"[InvertedIndex] بدء بناء الفهرس: '{dataset_name}'")
        print(f"{'='*55}")
        start_time = time.time()

        # ── تحميل الوثائق ─────────────────────────────────────
        print("[InvertedIndex] تحميل الوثائق...")
        raw_docs = self._loader.load_all(dataset_name, max_docs=max_docs)

        if not raw_docs:
            raise ValueError(f"مجموعة البيانات '{dataset_name}' فارغة.")

        print(f"[InvertedIndex]   ✓ {len(raw_docs):,} وثيقة")

        # ── المعالجة المسبقة ──────────────────────────────────
        print("[InvertedIndex] معالجة الوثائق...")
        from services.preprocessing.preprocessor import get_preprocessor
        preprocessor = get_preprocessor()

        # بناء الفهرس
        index: Dict[str, List[Posting]] = defaultdict(list)
        doc_store: Dict[str, str] = {}
        all_doc_ids: Set[str] = set()

        total = len(raw_docs)
        report_every = max(1, total // 10)

        for i, doc in enumerate(raw_docs):
            full_text = doc.get_full_text()
            tokens, _ = preprocessor.process(
                text=full_text,
                language=language,
                apply_stemming=apply_stemming,
                remove_stopwords=remove_stopwords,
            )

            # تخزين النص الأصلي
            doc_store[doc.doc_id] = doc.text[:200]  # أول 200 حرف
            all_doc_ids.add(doc.doc_id)

            # بناء posting لكل token
            # نحسب frequency و positions في خطوة واحدة
            term_positions: Dict[str, List[int]] = defaultdict(list)
            for position, token in enumerate(tokens):
                term_positions[token].append(position)

            for term, positions in term_positions.items():
                posting = Posting(
                    doc_id=doc.doc_id,
                    frequency=len(positions),
                    positions=positions if store_positions else [],
                )
                index[term].append(posting)

            if (i + 1) % report_every == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                print(f"[InvertedIndex]   {i+1:,}/{total:,} ({pct:.0f}%)", end="\r")

        print()

        # ترتيب الـ postings بالـ doc_id (للـ merge السريع)
        for term in index:
            index[term].sort(key=lambda p: p.doc_id)

        self._index      = dict(index)
        self._doc_store  = doc_store
        self._all_doc_ids = all_doc_ids

        # حساب إجمالي الـ postings
        total_postings = sum(len(v) for v in self._index.values())
        build_time = time.time() - start_time

        import datetime
        self.metadata = InvertedIndexMetadata(
            dataset_name=dataset_name,
            num_documents=len(raw_docs),
            vocab_size=len(self._index),
            total_postings=total_postings,
            build_time_seconds=round(build_time, 2),
            build_timestamp=datetime.datetime.now().isoformat(),
            apply_stemming=apply_stemming,
            remove_stopwords=remove_stopwords,
            language=language,
        )

        print(f"[InvertedIndex]   ✓ حجم المفردة: {len(self._index):,} مصطلح")
        print(f"[InvertedIndex]   ✓ إجمالي الـ postings: {total_postings:,}")
        print(f"[InvertedIndex] ✅ اكتمل في {build_time:.2f} ثانية")
        print(f"{'='*55}\n")

        return self.metadata

    # ----------------------------------------------------------
    # Boolean Retrieval
    # ----------------------------------------------------------

    def search_and(self, terms: List[str]) -> List[str]:
        """
        Boolean AND: يُرجع الوثائق التي تحتوي كل المصطلحات.

        خوارزمية المزج (Merge Algorithm):
          نبدأ بأصغر posting list (أقل df) ثم نتقاطع مع الأكبر.
          هذا يُقلل عدد العمليات — تحسين كلاسيكي في IR.

        مثال:
          search_and(["cloud", "storage"])
          → الوثائق التي تحتوي "cloud" AND "storage"
        """
        self._check_built()
        if not terms:
            return []

        # احصل على مجموعات الـ doc_ids لكل مصطلح
        sets = []
        for term in terms:
            postings = self._index.get(term, [])
            sets.append({p.doc_id for p in postings})

        if not sets:
            return []

        # تقاطع كل المجموعات (AND)
        # نبدأ بالأصغر لتحسين الأداء
        sets.sort(key=len)
        result = sets[0]
        for s in sets[1:]:
            result = result & s
            if not result:  # تحسين مبكر: إذا فرغت نتوقف
                break

        return sorted(result)

    def search_or(self, terms: List[str]) -> List[str]:
        """
        Boolean OR: يُرجع الوثائق التي تحتوي أي مصطلح.

        مثال:
          search_or(["cloud", "AI"])
          → الوثائق التي تحتوي "cloud" OR "AI"
        """
        self._check_built()
        if not terms:
            return []

        result: Set[str] = set()
        for term in terms:
            postings = self._index.get(term, [])
            result |= {p.doc_id for p in postings}

        return sorted(result)

    def search_not(self, term: str) -> List[str]:
        """
        Boolean NOT: يُرجع الوثائق التي لا تحتوي المصطلح.

        مثال:
          search_not("python")
          → كل الوثائق ما عدا التي تحتوي "python"
        """
        self._check_built()
        term_docs = {p.doc_id for p in self._index.get(term, [])}
        return sorted(self._all_doc_ids - term_docs)

    def search_and_not(
        self, include_terms: List[str], exclude_terms: List[str]
    ) -> List[str]:
        """
        AND NOT: تحتوي include_terms ولا تحتوي exclude_terms.

        مثال:
          search_and_not(["cloud"], ["python"])
          → وثائق تحتوي "cloud" لكن لا تحتوي "python"
        """
        include_docs = set(self.search_and(include_terms))
        exclude_docs = set(self.search_or(exclude_terms))
        return sorted(include_docs - exclude_docs)

    # ----------------------------------------------------------
    # دوال الاستعلام
    # ----------------------------------------------------------

    def get_posting_list(self, term: str) -> List[Posting]:
        """
        يُرجع الـ posting list لمصطلح معيّن.
        يُستخدم للفحص والتصحيح والتقرير.
        """
        self._check_built()
        return self._index.get(term, [])

    def document_frequency(self, term: str) -> int:
        """
        يُرجع عدد الوثائق التي تحتوي المصطلح (df).
        نفس القيمة المستخدمة في حساب IDF.
        """
        return len(self._index.get(term, []))

    def term_frequency(self, term: str, doc_id: str) -> int:
        """
        يُرجع تكرار المصطلح في وثيقة محددة (tf).
        """
        for posting in self._index.get(term, []):
            if posting.doc_id == doc_id:
                return posting.frequency
        return 0

    def get_top_terms_by_df(self, n: int = 20) -> List[tuple]:
        """
        يُرجع أكثر N مصطلحاً انتشاراً (أعلى df).
        مفيد لفحص جودة الـ preprocessing — المصطلحات ذات df العالي
        يجب أن تكون ذات معنى (لا stopwords).
        """
        self._check_built()
        term_dfs = [
            (term, len(postings))
            for term, postings in self._index.items()
        ]
        return sorted(term_dfs, key=lambda x: x[1], reverse=True)[:n]

    def get_stats(self) -> Dict:
        """إحصائيات الفهرس للعرض في الـ UI."""
        if not self.is_built():
            return {"status": "not_built"}
        return {
            "status": "ready",
            "num_documents": self.metadata.num_documents,
            "vocab_size": self.metadata.vocab_size,
            "total_postings": self.metadata.total_postings,
            "avg_postings_per_term": round(
                self.metadata.total_postings / max(1, self.metadata.vocab_size), 2
            ),
            "build_time_seconds": self.metadata.build_time_seconds,
        }

    # ----------------------------------------------------------
    # حفظ وتحميل
    # ----------------------------------------------------------

    def save_index(self, dataset_name: str) -> Path:
        """
        يحفظ الفهرس في 3 ملفات:
          inverted_index.json    ← الفهرس الكامل
          inverted_metadata.json ← البيانات الوصفية
          inverted_docmap.json   ← خريطة doc_id → نص
        """
        self._check_built()

        index_dir = self.indexes_dir / dataset_name / "inverted"
        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"[InvertedIndex] حفظ الفهرس في: {index_dir}")

        # 1. حفظ الفهرس
        index_data = {
            term: [p.to_dict() for p in postings]
            for term, postings in self._index.items()
        }
        index_path = index_dir / self._INDEX_FILE
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False)
        print(f"[InvertedIndex]   ✓ index: "
              f"{index_path.stat().st_size / 1024 / 1024:.2f} MB")

        # 2. حفظ metadata
        with open(index_dir / self._METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, ensure_ascii=False, indent=2)

        # 3. حفظ doc store
        with open(index_dir / self._DOCMAP_FILE, "w", encoding="utf-8") as f:
            json.dump(list(self._all_doc_ids), f, ensure_ascii=False)

        print(f"[InvertedIndex] ✅ حُفظ بنجاح")
        return index_dir

    def load_index(self, dataset_name: str) -> InvertedIndexMetadata:
        """يحمّل الفهرس من القرص."""
        index_dir = self.indexes_dir / dataset_name / "inverted"
        if not index_dir.exists():
            raise FileNotFoundError(
                f"الفهرس غير موجود: {index_dir}\n"
                f"شغّل build_index('{dataset_name}') أولاً."
            )

        print(f"[InvertedIndex] تحميل من: {index_dir}")
        start = time.time()

        with open(index_dir / self._INDEX_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        self._index = {
            term: [Posting.from_dict(p) for p in postings]
            for term, postings in raw.items()
        }

        with open(index_dir / self._METADATA_FILE, encoding="utf-8") as f:
            self.metadata = InvertedIndexMetadata.from_dict(json.load(f))

        with open(index_dir / self._DOCMAP_FILE, encoding="utf-8") as f:
            self._all_doc_ids = set(json.load(f))

        print(f"[InvertedIndex]   ✓ {len(self._index):,} مصطلح في {time.time()-start:.3f}s")
        return self.metadata

    def is_built(self) -> bool:
        return len(self._index) > 0

    def is_saved(self, dataset_name: str) -> bool:
        d = self.indexes_dir / dataset_name / "inverted"
        return (d / self._INDEX_FILE).exists()

    def _check_built(self) -> None:
        if not self.is_built():
            raise RuntimeError(
                "الفهرس غير مبني. شغّل build_index() أولاً."
            )


# =============================================================
# Singleton
# =============================================================

_inverted_instances: Dict[str, InvertedIndex] = {}


def get_inverted_index(dataset_name: Optional[str] = None) -> InvertedIndex:
    global _inverted_instances
    key = dataset_name or "__default__"
    if key not in _inverted_instances:
        idx = InvertedIndex()
        if dataset_name and idx.is_saved(dataset_name):
            idx.load_index(dataset_name)
        _inverted_instances[key] = idx
    return _inverted_instances[key]