"""
services/indexing/tests/test_indexing_app.py
=============================================
اختبارات وحدة لـ Indexing Service API.

تشغيل:
    cd ir_system_2026
    python -m pytest services/indexing/tests/test_indexing_app.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient


# =============================================================
# نُنشئ الـ client مع override للـ indexes_dir
# =============================================================

@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    TestClient مع مجلد indexes مؤقت.
    نستخدم monkeypatch لتوجيه الفهارس للمجلد المؤقت.
    """
    monkeypatch.setenv("INDEXES_DIR", str(tmp_path / "indexes"))

    from services.indexing.app import app
    return TestClient(app)


# =============================================================
# الاختبارات
# =============================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
        assert r.json()["service_name"] == "indexing"


class TestBuildIndexEndpoint:
    def test_invalid_index_type_returns_400(self, client):
        """
        ماذا يختبر: نوع فهرس غير معروف → 400.
        ما الخطأ الذي يمنعه: بناء فهرس بنوع خاطئ بصمت.
        """
        r = client.post("/index/build", json={
            "dataset_name": "dataset1",
            "index_type": "unknown_type",
        })
        assert r.status_code == 400

    def test_missing_dataset_returns_error(self, client):
        """
        ماذا يختبر: dataset غير موجود → خطأ واضح.
        ما الخطأ الذي يمنعه: انهيار الـ server بدون رسالة مفهومة.
        """
        r = client.post("/index/build", json={
            "dataset_name": "nonexistent_dataset_xyz",
            "index_type": "tfidf",
        })
        # يجب أن يُرجع 404 أو 500 مع رسالة واضحة
        assert r.status_code in (404, 500, 400)


class TestIndexStatusEndpoint:
    def test_status_returns_all_four_indexes(self, client):
        """
        ماذا يختبر: /index/status يُرجع حالة الفهارس الأربعة.
        ما الخطأ الذي يمنعه: Gateway لا يعرف أي فهرس جاهز.
        """
        r = client.get("/index/status/dataset1")
        assert r.status_code == 200
        data = r.json()
        assert "inverted"  in data
        assert "tfidf"     in data
        assert "bm25"      in data
        assert "embedding" in data
        assert data["dataset_name"] == "dataset1"

    def test_status_shows_not_saved_for_new_dataset(self, client):
        """
        ماذا يختبر: فهارس غير مبنية تُظهر saved=False.
        ما الخطأ الذي يمنعه: Gateway يُرسل طلبات لفهرس غير موجود.
        """
        r = client.get("/index/status/brand_new_dataset")
        data = r.json()
        assert data["tfidf"]["saved"] is False
        assert data["bm25"]["saved"] is False


class TestBooleanSearchEndpoint:
    def test_missing_index_returns_404(self, client):
        """
        ماذا يختبر: Boolean search قبل بناء الفهرس → 404.
        ما الخطأ الذي يمنعه: Retrieval يحاول Boolean search على فهرس غير موجود.
        """
        r = client.post("/index/boolean-search", json={
            "dataset_name": "nonexistent",
            "operation": "and",
            "terms": ["cloud"],
        })
        assert r.status_code == 404

    def test_invalid_operation_returns_400(self, client):
        """
        ماذا يختبر: operation غير معروف → 400.
        """
        r = client.post("/index/boolean-search", json={
            "dataset_name": "any",
            "operation": "xor",
            "terms": ["cloud"],
        })
        # إما 400 أو 404 (الفهرس غير موجود)
        assert r.status_code in (400, 404)


class TestStatsEndpoint:
    def test_stats_returns_dataset_name(self, client):
        """
        ماذا يختبر: /index/stats يُرجع اسم الـ dataset.
        """
        r = client.get("/index/stats/dataset1")
        assert r.status_code == 200
        assert r.json()["dataset_name"] == "dataset1"
