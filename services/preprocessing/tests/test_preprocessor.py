"""
services/preprocessing/tests/test_preprocessor.py
==================================================
اختبارات وحدة (Unit Tests) لخدمة المعالجة المسبقة.

تشغيل الاختبارات:
    cd ir_system_2026
    python -m pytest services/preprocessing/tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

import pytest
from fastapi.testclient import TestClient

from services.preprocessing.preprocessor import TextPreprocessor
from services.preprocessing.app import app

client = TestClient(app)


# =============================================================
# Fixtures — إعداد مشترك لكل الاختبارات
# =============================================================

@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """نسخة من TextPreprocessor للاختبارات المباشرة."""
    return TextPreprocessor()


# =============================================================
# اختبارات TextPreprocessor (وحدة)
# =============================================================

class TestTextPreprocessor:
    """اختبارات المنطق الأساسي بدون HTTP."""

    def test_lowercase(self, preprocessor: TextPreprocessor) -> None:
        tokens, _ = preprocessor.process(
            "HELLO WORLD",
            apply_stemming=False,
            remove_stopwords=False,
        )
        assert all(t.islower() for t in tokens)

    def test_stopword_removal(self, preprocessor: TextPreprocessor) -> None:
        tokens, _ = preprocessor.process(
            "the dog jumped over the fence",
            remove_stopwords=True,
            apply_stemming=False,
        )
        # "the" و "over" كلمات وظيفية يجب حذفها
        assert "the" not in tokens
        assert "over" not in tokens
        assert "dog" in tokens

    def test_stemming_reduces_words(self, preprocessor: TextPreprocessor) -> None:
        tokens, steps = preprocessor.process(
            "running dogs jumped",
            apply_stemming=True,
            remove_stopwords=False,
        )
        assert "run" in tokens      # running → run
        assert "dog" in tokens      # dogs → dog
        assert "jump" in tokens     # jumped → jump
        assert "stemming" in steps

    def test_lemmatization_produces_real_words(
        self, preprocessor: TextPreprocessor
    ) -> None:
        tokens, steps = preprocessor.process(
            "running dogs jumped",
            apply_stemming=False,
            apply_lemmatization=True,
            remove_stopwords=False,
        )
        # lemmatization يُنتج كلمات حقيقية
        assert "run" in tokens
        assert "lemmatization" in steps

    def test_empty_tokens_after_stopword_removal(
        self, preprocessor: TextPreprocessor
    ) -> None:
        # نص يحتوي فقط على stopwords
        tokens, _ = preprocessor.process(
            "the is a an",
            remove_stopwords=True,
            apply_stemming=False,
        )
        assert tokens == []

    def test_batch_processing_length(
        self, preprocessor: TextPreprocessor
    ) -> None:
        texts = ["hello world", "foo bar baz", "test case"]
        results = preprocessor.process_batch(texts)
        # يجب أن يكون عدد النتائج مساوياً لعدد النصوص
        assert len(results) == len(texts)

    def test_ir_consistency_rule(
        self, preprocessor: TextPreprocessor
    ) -> None:
        """
        القاعدة الأهم في IR:
        معالجة الوثيقة ومعالجة الاستعلام يجب أن تُنتجا نفس التوكنز
        للكلمات المتشابهة.
        """
        # كلمة في وثيقة
        doc_tokens, _ = preprocessor.process(
            "Running dogs",
            apply_stemming=True,
            remove_stopwords=False,
        )
        # نفس الكلمة في استعلام
        query_tokens, _ = preprocessor.process(
            "run dog",
            apply_stemming=True,
            remove_stopwords=False,
        )
        # يجب أن تتطابق النتائج
        assert set(doc_tokens) == set(query_tokens)


# =============================================================
# اختبارات الـ API (Integration)
# =============================================================

class TestPreprocessAPI:
    """اختبارات HTTP endpoints."""

    def test_health_endpoint(self) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_preprocess_basic(self) -> None:
        r = client.post("/preprocess", json={
            "text": "Running dogs jumped!",
            "apply_stemming": True,
            "remove_stopwords": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert "tokens" in data
        assert len(data["tokens"]) > 0
        assert data["token_count"] == len(data["tokens"])

    def test_preprocess_empty_text_returns_422(self) -> None:
        r = client.post("/preprocess", json={"text": "  "})
        assert r.status_code == 422

    def test_stemming_and_lemmatization_together_returns_422(self) -> None:
        r = client.post("/preprocess", json={
            "text": "hello world",
            "apply_stemming": True,
            "apply_lemmatization": True,
        })
        assert r.status_code == 422

    def test_batch_preprocess(self) -> None:
        r = client.post("/preprocess/batch", json={
            "texts": ["hello world", "foo bar"],
            "apply_stemming": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["total_processed"] == 2
        assert len(data["results"]) == 2

    def test_response_contains_steps(self) -> None:
        r = client.post("/preprocess", json={
            "text": "Hello World",
            "apply_stemming": True,
            "remove_stopwords": True,
        })
        data = r.json()
        steps = data["steps_applied"]
        assert "lowercase" in steps
        assert "stemming" in steps
        assert "remove_stopwords" in steps
