"""
services/gateway/service_client.py
=================================
أدوات مساعدة لإجراء استدعاءات HTTP بين الخدمات الداخلية.
Provides small wrappers around httpx for consistent error handling.
"""

from typing import Any, Tuple
import logging

import httpx

logger = logging.getLogger(__name__)


async def get_json(url: str, timeout: float = 5.0) -> Tuple[int, Any]:
	"""GET request and return (status_code, parsed_json).

	يحاول إرجاع JSON أو يرفع الاستثناء للمتصل للتعامل معه.
	"""
	async with httpx.AsyncClient(timeout=timeout) as client:
		resp = await client.get(url)
		try:
			data = resp.json()
		except Exception:
			data = resp.text
		return resp.status_code, data


async def post_json(url: str, json_data: dict, timeout: float = 10.0) -> Tuple[int, Any]:
	"""POST request with JSON payload and return (status_code, parsed_json)."""
	async with httpx.AsyncClient(timeout=timeout) as client:
		resp = await client.post(url, json=json_data)
		try:
			data = resp.json()
		except Exception:
			data = resp.text
		return resp.status_code, data


async def check_service_health(name: str, base_url: str) -> dict:
	"""Check a service's /health endpoint and return a serializable dict.

	Returns a dict compatible with `shared.models.ServiceStatus` fields.
	"""
	url = f"{base_url}/health"
	try:
		status_code, data = await get_json(url, timeout=3.0)
		if status_code == 200:
			# Ensure minimal shape
			return {
				"service_name": name,
				"status": "healthy",
				"version": data.get("version", "1.0.0") if isinstance(data, dict) else "1.0.0",
				"details": data.get("details", {}) if isinstance(data, dict) else {},
			}
		else:
			return {"service_name": name, "status": "unhealthy", "details": {"http_status": status_code, "response": data}}
	except Exception as e:
		logger.debug(f"check_service_health {name} failed: {e}")
		return {"service_name": name, "status": "unhealthy", "details": {"error": str(e)}}

