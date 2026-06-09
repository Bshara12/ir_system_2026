"""
services/gateway/app.py
========================
FastAPI application for the Gateway Service (port 8000).

Exposes:
- GET /health
- GET /services/health
- POST /search  -> forwards to Retrieval Service

Run:
	uvicorn services.gateway.app:app --port 8000 --reload
"""

import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.gateway.router import router
from shared.constants import GATEWAY_PORT


app = FastAPI(title="Gateway Service", version="1.0.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
	import uvicorn

	print(f"[Gateway] starting on port {GATEWAY_PORT}")
	uvicorn.run("services.gateway.app:app", host="0.0.0.0", port=GATEWAY_PORT, reload=True)

