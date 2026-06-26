"""FastAPI app for Ranking & Evaluation service (core metrics).

Provides a health endpoint and a demo evaluation endpoint that uses the
in-memory sample data. This module intentionally keeps logic simple — the
core metric functions live in `metrics.py` and a thin wrapper in
`evaluator.py`.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

from shared.models import ServiceStatus, ErrorResponse, RetrievalModel
from shared.constants import EVALUATION_PORT

from .evaluator import evaluate_demo, evaluate_dataset


class DatasetEvaluationRequest(BaseModel):
	dataset_name: str = Field(default="quora", description="Dataset folder name under data/datasets")
	model: RetrievalModel = Field(default=RetrievalModel.BM25)
	top_k: int = Field(default=10, ge=1, le=100)
	max_queries: int = Field(default=5, ge=1, le=100)
	bm25_k1: float = Field(default=1.5, ge=0.0)
	bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
	apply_refinement: bool = Field(default=False)


app = FastAPI(
	title="Ranking Evaluation Service",
	description="خدمة تقييم نظام استرجاع المعلومات باستخدام Precision, Recall, MAP, nDCG.",
	version="1.0.0",
)


@app.get("/health", response_model=ServiceStatus, tags=["System"])
async def health_check() -> ServiceStatus:
	"""Return the health status of the evaluation service."""
	return ServiceStatus(service_name="evaluation", status="healthy", version="1.0.0", details={"port": EVALUATION_PORT})


@app.post("/evaluate/demo", tags=["Evaluation"])
async def evaluate_demo_endpoint():
	"""Run demo evaluation on in-memory sample data and return metrics.

	This endpoint is intended for local testing and demo purposes because
	the project currently has no queries or qrels files on disk.
	"""
	results = evaluate_demo()
	return results


@app.post(
	"/evaluate/dataset",
	tags=["Evaluation"],
	responses={
		404: {"model": ErrorResponse, "description": "Queries or qrels file not found."},
		503: {"model": ErrorResponse, "description": "Retrieval Service unavailable."},
		400: {"model": ErrorResponse, "description": "Invalid evaluation request or no valid queries."},
	},
)
async def evaluate_dataset_endpoint(request: DatasetEvaluationRequest):
	"""Run dataset evaluation using the Retrieval Service and qrels from disk."""
	try:
		results = await evaluate_dataset(
			dataset_name=request.dataset_name,
			model=request.model,
			top_k=request.top_k,
			max_queries=request.max_queries,
			bm25_k1=request.bm25_k1,
			bm25_b=request.bm25_b,
			apply_refinement=request.apply_refinement,
		)
		return results
	except FileNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except httpx.ConnectError:
		raise HTTPException(
			status_code=503,
			detail="Retrieval Service is unavailable. Ensure services.retrieval is running on port 8003.",
		)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal evaluation error: {str(e)}")
