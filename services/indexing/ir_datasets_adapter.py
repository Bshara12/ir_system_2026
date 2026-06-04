"""
services/indexing/ir_datasets_adapter.py
=========================================
مسؤول عن تحميل Datasets من مكتبة ir_datasets وتحويلها
إلى الصيغ الداخلية للنظام (Document, Query, Qrel).

ما هو ir_datasets؟
═══════════════════
مكتبة Python توفر واجهة موحدة لمئات من مجموعات بيانات IR.
بدلاً من تحميل ملفات ZIP يدوياً، تقول فقط:
    ds = ir_datasets.load("beir/trec-covid")
    for doc in ds.docs_iter():
        print(doc.doc_id, doc.text)

لماذا نبني adapter منفصلاً؟
═══════════════════════════
مبدأ SOLID — Single Responsibility:
  - DatasetLoader يعرف كيف يقرأ ملفات محلية
  - IrDatasetsAdapter يعرف كيف يتعامل مع مكتبة ir_datasets
  - الفصل يسمح باستبدال أحدهما دون المساس بالآخر

مثال حقيقي:
    adapter = IrDatasetsAdapter()
    
    # تحميل وثائق msmarco
    for doc in adapter.stream_documents("msmarco-passage/dev/small"):
        print(doc.doc_id, doc.text[:50])
    
    # تحميل استعلامات msmarco
    for query in adapter.stream_queries("msmarco-passage/dev/small"):
        print(query.query_id, query.text)
    
    # تحميل qrels msmarco
    for qrel in adapter.stream_qrels("msmarco-passage/dev/small"):
        print(qrel.query_id, qrel.doc_id, qrel.relevance)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================
# نماذج البيانات (Data Models)
# نستخدم dataclass لأنها:
#   - واضحة وسريعة
#   - تدعم asdict() للتحويل لـ JSON
#   - تُعرّف الحقول بشكل صريح
# =============================================================

@dataclass
class Document:
    """
    وثيقة واحدة من مجموعة البيانات.
    
    مثال:
        doc = Document(doc_id="d1", text="Cloud storage...", title="Cloud")
        full = doc.get_full_text()  # "Cloud Cloud storage..."
    """
    doc_id: str
    text: str
    title: Optional[str] = None

    def get_full_text(self) -> str:
        """
        يدمج العنوان مع النص للفهرسة.
        
        لماذا ندمجهما؟
        في IR يُسمى هذا "title boosting" — العنوان يحمل
        الكلمات المفتاحية الأهم فنضعها في البداية.
        """
        if self.title:
            return f"{self.title} {self.text}"
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        """تحويل لـ dict — مفيد للحفظ كـ JSONL."""
        return asdict(self)


@dataclass
class Query:
    """
    استعلام واحد من testing data.
    
    ما هو Query في IR؟
    ═══════════════════
    هو السؤال أو الجملة التي يبحث عنها المستخدم.
    مثال من msmarco: "how long does it take to boil an egg"
    
    في التقييم نستخدم queries لاختبار النظام:
        result = search(query.text)
        precision = evaluate(result, qrels[query.query_id])
    """
    query_id: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Qrel:
    """
    Relevance Judgment — حكم على مدى ملاءمة وثيقة لاستعلام.
    
    ما هو Qrel؟
    ══════════
    qrel = query relevance — يُخبرنا:
    "هذه الوثيقة (doc_id) ذات صلة بهذا الاستعلام (query_id)
     بدرجة (relevance)"
    
    درجات الـ relevance تختلف بين datasets:
      - Binary:  0 = غير ذات صلة، 1 = ذات صلة
      - Graded:  0 = غير ذات صلة، 1 = ذات صلة جزئياً، 2 = ذات صلة جداً
      - TREC:    0, 1, 2, 3 (مقياس من 4 درجات)
    
    مثال:
        query: "covid treatment"
        qrel:  doc_id="d42", relevance=2  ← وثيقة عالية الصلة
        qrel:  doc_id="d99", relevance=0  ← وثيقة غير ذات صلة
    
    يُستخدم في حساب MAP, nDCG, Recall, Precision@10
    """
    query_id: str
    doc_id: str
    relevance: int
    iteration: str = "0"  # بعض datasets تستخدم iterations

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================
# Registry الـ Datasets المدعومة
# =============================================================

# كل dataset له:
#   ir_datasets_id: الاسم في مكتبة ir_datasets
#   friendly_name:  الاسم المُعرض في الواجهة
#   domain:         مجال الـ dataset
#   doc_count:      عدد الوثائق التقريبي
#   query_count:    عدد الاستعلامات
#   has_qrels:      هل يحتوي qrels؟
#   notes:          ملاحظات مهمة

SUPPORTED_DATASETS: Dict[str, Dict[str, Any]] = {
    # ─────────────────────────────────────────────────────────
    # Dataset 1: MS MARCO Passage Retrieval
    # الأشهر في IR — 8.8M مقطع، qrels حقيقية من Bing
    # ─────────────────────────────────────────────────────────
    "msmarco": {
        "ir_datasets_id": "msmarco-passage/dev/small",
        "friendly_name":  "MS MARCO Passage Retrieval",
        "domain":         "General Web Search",
        "doc_count":      8_841_823,
        "query_count":    6_980,
        "qrel_count":     7_437,
        "has_qrels":      True,
        "size_gb":        3.8,
        "notes": (
            "الأشهر في أبحاث IR. استعلامات حقيقية من Bing. "
            "qrels متفرقة (sparse) — كل استعلام له 1-3 وثائق ذات صلة فقط."
        ),
    },

    # ─────────────────────────────────────────────────────────
    # Dataset 2: TREC-COVID (BEIR Benchmark)
    # أبحاث COVID-19 الطبية — dense qrels ممتازة للتقييم
    # ─────────────────────────────────────────────────────────
    "trec-covid": {
        "ir_datasets_id": "beir/trec-covid",
        "friendly_name":  "TREC-COVID (BEIR)",
        "domain":         "Biomedical / COVID-19 Research",
        "doc_count":      171_332,
        "query_count":    50,
        "qrel_count":     69_318,
        "has_qrels":      True,
        "size_gb":        0.22,
        "notes": (
            "171K وثيقة طبية (قريب من 200K). "
            "50 استعلام طبي لكن كل استعلام له ~1,386 qrel (dense). "
            "أفضل من msmarco للتقييم بسبب كثافة الـ qrels."
        ),
    },

    # ─────────────────────────────────────────────────────────
    # Datasets إضافية للمقارنة (للفريق فقط)
    # ─────────────────────────────────────────────────────────
    "dbpedia": {
        "ir_datasets_id": "beir/dbpedia-entity",
        "friendly_name":  "DBpedia Entity",
        "domain":         "Entity Retrieval (Wikipedia)",
        "doc_count":      4_635_922,
        "query_count":    400,
        "qrel_count":     49_000,
        "has_qrels":      True,
        "size_gb":        30.0,
        "notes": "ضخم جداً — 4.6M وثيقة، حجم ~30GB.",
    },
    "fever": {
        "ir_datasets_id": "beir/fever",
        "friendly_name":  "FEVER (Fact Verification)",
        "domain":         "Fact Checking",
        "doc_count":      5_416_568,
        "query_count":    6_666,
        "qrel_count":     185_445,
        "has_qrels":      True,
        "size_gb":        12.0,
        "notes": "ضخم جداً — 5.4M وثيقة.",
    },
}

# اسم الـ dataset الافتراضي للاختبارات
DEFAULT_DATASET_1 = "msmarco"
DEFAULT_DATASET_2 = "trec-covid"


# =============================================================
# IrDatasetsAdapter — المحوّل الرئيسي
# =============================================================

class IrDatasetsAdapter:
    """
    يُحوّل بيانات ir_datasets إلى Document/Query/Qrel.

    مبدأ التصميم (Adapter Pattern):
    ════════════════════════════════
    ir_datasets يُرجع objects خاصة به (مثل GenericDoc).
    نظامنا يتوقع Document/Query/Qrel.
    الـ Adapter هو الجسر بينهما.

    ┌──────────────┐    adapter    ┌──────────────┐
    │ ir_datasets  │ ─────────────▶│   نظامنا      │
    │  GenericDoc  │               │   Document   │
    │  GenericQuery│               │   Query      │
    │  TrecQrel    │               │   Qrel       │
    └──────────────┘               └──────────────┘

    الاستخدام:
        adapter = IrDatasetsAdapter()
        for doc in adapter.stream_documents("msmarco"):
            index.add(doc)
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """
        cache_dir: مجلد لتخزين الـ datasets المحمّلة.
        إذا لم يُحدد، يستخدم ir_datasets المجلد الافتراضي (~/.ir_datasets)
        """
        self.cache_dir = cache_dir
        self._loaded: Dict[str, Any] = {}  # cache للـ datasets المفتوحة

    # ----------------------------------------------------------
    # التحقق من الدعم
    # ----------------------------------------------------------

    def is_supported(self, dataset_name: str) -> bool:
        """هل هذا الاسم مدعوم في الـ registry؟"""
        return dataset_name in SUPPORTED_DATASETS

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """يُرجع معلومات الـ dataset من الـ registry."""
        return SUPPORTED_DATASETS.get(dataset_name)

    def list_supported_datasets(self) -> List[str]:
        """يُرجع قائمة بجميع الـ datasets المدعومة."""
        return list(SUPPORTED_DATASETS.keys())

    # ----------------------------------------------------------
    # تحميل الـ Dataset
    # ----------------------------------------------------------

    def _load_dataset(self, dataset_name: str) -> Any:
        """
        يفتح الـ dataset من ir_datasets.
        يُخزن في cache لتجنب إعادة الفتح.

        ماذا يحدث عند أول استدعاء؟
        → ir_datasets يتحقق من وجود الملفات محلياً
        → إذا لم تُوجد، يبدأ التحميل تلقائياً من الإنترنت
        → يخزنها في ~/.ir_datasets/
        """
        if dataset_name in self._loaded:
            return self._loaded[dataset_name]

        info = self.get_dataset_info(dataset_name)
        if info is None:
            raise ValueError(
                f"'{dataset_name}' غير مدعوم. "
                f"الـ datasets المدعومة: {self.list_supported_datasets()}"
            )

        try:
            import ir_datasets
        except ImportError:
            raise ImportError(
                "مكتبة ir_datasets غير مثبتة.\n"
                "شغّل: pip install ir-datasets"
            )

        ir_id = info["ir_datasets_id"]
        logger.info(f"[IrDatasetsAdapter] فتح dataset: {ir_id}")

        try:
            ds = ir_datasets.load(ir_id)
            self._loaded[dataset_name] = ds
            return ds
        except Exception as e:
            raise RuntimeError(
                f"فشل تحميل '{ir_id}' من ir_datasets.\n"
                f"السبب: {e}\n"
                f"تأكد من اتصال الإنترنت أو وجود الملفات محلياً."
            )

    # ----------------------------------------------------------
    # تدفق الوثائق
    # ----------------------------------------------------------

    def stream_documents(
        self,
        dataset_name: str,
        max_docs: Optional[int] = None,
    ) -> Generator[Document, None, None]:
        """
        يقرأ وثائق الـ dataset بطريقة تدفقية (Lazy Loading).

        لماذا Lazy Loading؟
        ═══════════════════
        msmarco لديه 8.8M وثيقة.
        لو حمّلنا كلها في RAM = ~15GB من الذاكرة.
        بالتدفق نقرأ وثيقة واحدة في كل مرة = أقل من 10MB في الذاكرة.

        مثال:
            for doc in adapter.stream_documents("msmarco", max_docs=1000):
                bm25_indexer.add_document(doc)
        """
        ds = self._load_dataset(dataset_name)

        if not hasattr(ds, 'docs_iter'):
            raise AttributeError(
                f"'{dataset_name}' لا يحتوي على وثائق (docs_iter)."
            )

        count = 0
        for raw_doc in ds.docs_iter():
            doc = self._convert_document(raw_doc)
            if doc is not None:
                yield doc
                count += 1
                if count % 100_000 == 0:
                    logger.info(f"[IrDatasetsAdapter] قُرئ {count:,} وثيقة...")
                if max_docs and count >= max_docs:
                    break

        logger.info(f"[IrDatasetsAdapter] اكتمل قراءة {count:,} وثيقة من '{dataset_name}'")

    def stream_queries(
        self,
        dataset_name: str,
    ) -> Generator[Query, None, None]:
        """
        يقرأ استعلامات الـ dataset.

        من يستخدم هذه الدالة؟
        ═════════════════════
        - Retrieval Service (Dev2): لتنفيذ البحث على كل استعلام
        - Evaluation Service (Dev3): لحساب MAP/nDCG على كل استعلام
        """
        ds = self._load_dataset(dataset_name)

        if not hasattr(ds, 'queries_iter'):
            raise AttributeError(
                f"'{dataset_name}' لا يحتوي على استعلامات (queries_iter)."
            )

        for raw_query in ds.queries_iter():
            query = self._convert_query(raw_query)
            if query is not None:
                yield query

    def stream_qrels(
        self,
        dataset_name: str,
    ) -> Generator[Qrel, None, None]:
        """
        يقرأ الـ qrels (relevance judgments).

        من يستخدم هذه الدالة؟
        ═════════════════════
        Evaluation Service (Dev3) فقط:
            retrieved = retrieval_service.search(query)
            relevant  = [q for q in qrels if q.query_id == query_id]
            precision = len(set(r.doc_id for r in retrieved[:10]) &
                           set(q.doc_id for q in relevant if q.relevance > 0)) / 10
        """
        ds = self._load_dataset(dataset_name)

        if not hasattr(ds, 'qrels_iter'):
            raise AttributeError(
                f"'{dataset_name}' لا يحتوي على qrels (qrels_iter)."
            )

        for raw_qrel in ds.qrels_iter():
            qrel = self._convert_qrel(raw_qrel)
            if qrel is not None:
                yield qrel

    # ----------------------------------------------------------
    # دوال التحويل (Private)
    # ----------------------------------------------------------

    def _convert_document(self, raw_doc: Any) -> Optional[Document]:
        """
        يُحوّل وثيقة ir_datasets إلى Document الداخلي.

        لماذا هذا التعقيد؟
        ═══════════════════
        كل dataset في ir_datasets لها schema مختلف:
          msmarco: (doc_id, url, title, body)
          beir/trec-covid: (doc_id, text, title, metadata)
          cranfield: (doc_id, title, text)

        نعالج الحالات المختلفة بشكل موحد.
        """
        if raw_doc is None:
            return None

        # doc_id — موجود في كل الـ datasets
        doc_id = str(getattr(raw_doc, 'doc_id', None) or
                     getattr(raw_doc, 'id', None) or "")
        if not doc_id:
            return None

        # النص — يختلف اسم الحقل بين datasets
        text = (
            getattr(raw_doc, 'text', None) or      # beir datasets
            getattr(raw_doc, 'body', None) or      # msmarco
            getattr(raw_doc, 'passage', None) or   # بعض datasets
            getattr(raw_doc, 'contents', None) or  # cranfield
            ""
        )

        # العنوان — اختياري
        title = (
            getattr(raw_doc, 'title', None) or
            getattr(raw_doc, 'heading', None) or
            None
        )

        # نتجاوز الوثائق الفارغة
        if not str(text).strip():
            return None

        return Document(
            doc_id=doc_id,
            text=str(text),
            title=str(title) if title else None,
        )

    def _convert_query(self, raw_query: Any) -> Optional[Query]:
        """يُحوّل استعلام ir_datasets إلى Query الداخلي."""
        if raw_query is None:
            return None

        query_id = str(
            getattr(raw_query, 'query_id', None) or
            getattr(raw_query, 'id', None) or ""
        )
        text = str(
            getattr(raw_query, 'text', None) or
            getattr(raw_query, 'description', None) or
            getattr(raw_query, 'title', None) or
            ""
        )

        if not query_id or not text.strip():
            return None

        return Query(query_id=query_id, text=text)

    def _convert_qrel(self, raw_qrel: Any) -> Optional[Qrel]:
        """يُحوّل qrel من ir_datasets إلى Qrel الداخلي."""
        if raw_qrel is None:
            return None

        query_id = str(
            getattr(raw_qrel, 'query_id', None) or ""
        )
        doc_id = str(
            getattr(raw_qrel, 'doc_id', None) or ""
        )
        relevance = int(
            getattr(raw_qrel, 'relevance', 0) or 0
        )

        if not query_id or not doc_id:
            return None

        return Qrel(
            query_id=query_id,
            doc_id=doc_id,
            relevance=relevance,
        )

    # ----------------------------------------------------------
    # حفظ البيانات محلياً (للعمل Offline)
    # ----------------------------------------------------------

    def save_to_jsonl(
        self,
        dataset_name: str,
        output_dir: str,
        max_docs: Optional[int] = None,
        save_queries: bool = True,
        save_qrels: bool = True,
    ) -> Dict[str, str]:
        """
        يُنزّل dataset من ir_datasets ويحفظه كملفات JSONL محلية.

        لماذا نحفظ محلياً؟
        ═══════════════════
        1. نعمل offline بعد التحميل الأول
        2. أسرع بكثير من ir_datasets عند الفهرسة
        3. يمكن مشاركته مع أعضاء الفريق بدون تحميل مرة أخرى
        4. DatasetLoader الحالي يقرأ JSONL ← يعمل مباشرة

        البنية المحفوظة:
            output_dir/
              corpus.jsonl     ← الوثائق
              queries.jsonl    ← الاستعلامات
              qrels.jsonl      ← الـ qrels

        مثال:
            paths = adapter.save_to_jsonl(
                "trec-covid",
                "data/datasets/trec-covid",
                max_docs=None,     # كل الوثائق
                save_queries=True,
                save_qrels=True,
            )
            print(paths)
            # {
            #   "corpus":  "data/datasets/trec-covid/corpus.jsonl",
            #   "queries": "data/datasets/trec-covid/queries.jsonl",
            #   "qrels":   "data/datasets/trec-covid/qrels.jsonl",
            # }
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        saved_paths: Dict[str, str] = {}

        # ── 1. حفظ الوثائق ──────────────────────────────────
        corpus_path = out / "corpus.jsonl"
        logger.info(f"[Adapter] حفظ الوثائق في: {corpus_path}")
        doc_count = 0
        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc in self.stream_documents(dataset_name, max_docs=max_docs):
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
                doc_count += 1
        logger.info(f"[Adapter] حُفظ {doc_count:,} وثيقة")
        saved_paths["corpus"] = str(corpus_path)

        # ── 2. حفظ الاستعلامات ──────────────────────────────
        if save_queries:
            try:
                queries_path = out / "queries.jsonl"
                query_count = 0
                with open(queries_path, "w", encoding="utf-8") as f:
                    for query in self.stream_queries(dataset_name):
                        f.write(json.dumps(query.to_dict(), ensure_ascii=False) + "\n")
                        query_count += 1
                logger.info(f"[Adapter] حُفظ {query_count:,} استعلام")
                saved_paths["queries"] = str(queries_path)
            except AttributeError as e:
                logger.warning(f"[Adapter] لا توجد queries: {e}")

        # ── 3. حفظ الـ Qrels ────────────────────────────────
        if save_qrels:
            try:
                qrels_path = out / "qrels.jsonl"
                qrel_count = 0
                with open(qrels_path, "w", encoding="utf-8") as f:
                    for qrel in self.stream_qrels(dataset_name):
                        f.write(json.dumps(qrel.to_dict(), ensure_ascii=False) + "\n")
                        qrel_count += 1
                logger.info(f"[Adapter] حُفظ {qrel_count:,} qrel")
                saved_paths["qrels"] = str(qrels_path)
            except AttributeError as e:
                logger.warning(f"[Adapter] لا توجد qrels: {e}")

        return saved_paths


# =============================================================
# Singleton
# =============================================================

_adapter_instance: Optional[IrDatasetsAdapter] = None


def get_ir_datasets_adapter() -> IrDatasetsAdapter:
    """
    يُرجع النسخة الوحيدة من IrDatasetsAdapter.
    يُستخدم كـ Dependency في FastAPI أو في الـ indexers.
    """
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = IrDatasetsAdapter()
    return _adapter_instance