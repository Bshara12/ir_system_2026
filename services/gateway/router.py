from typing import List

from fastapi import APIRouter, HTTPException

from shared.models import (
	ServiceStatus,
	RetrievalRequest,
	RetrievalResponse,
	ErrorResponse,
)
from shared.constants import (
	PREPROCESSING_URL,
	INDEXING_URL,
	RETRIEVAL_URL,
	QUERY_REFINEMENT_URL,
	EVALUATION_URL,
	GATEWAY_PORT,
)

from .service_client import check_service_health, post_json


router = APIRouter()


@router.get("/health", response_model=ServiceStatus, tags=["System"])
async def health_check() -> ServiceStatus:
	"""حالة الـ Gateway نفسها."""
	return ServiceStatus(service_name="gateway", status="healthy", version="1.0.0", details={"port": GATEWAY_PORT})


@router.get("/services/health", response_model=List[ServiceStatus], tags=["System"])
async def services_health() -> List[ServiceStatus]:
	"""يتحقق من حالة الخدمات الداخلية ويُرجع قائمة بالحالات."""
	services = [
		("preprocessing", PREPROCESSING_URL),
		("indexing", INDEXING_URL),
		("retrieval", RETRIEVAL_URL),
		("query_refinement", QUERY_REFINEMENT_URL),
		("evaluation", EVALUATION_URL),
	]
	results = []
	for name, url in services:
		info = await check_service_health(name, url)
		results.append(ServiceStatus(**info))
	return results


@router.post(
	"/search",
	response_model=RetrievalResponse,
	tags=["Gateway"],
	responses={503: {"model": ErrorResponse}},
)
async def search(request: RetrievalRequest) -> RetrievalResponse:
	"""Forward the search request to Retrieval Service and return its response.

	يَستقبل نفس نموذج `RetrievalRequest` ثم يعيد `RetrievalResponse`.
	"""
	try:
		status_code, data = await post_json(f"{RETRIEVAL_URL}/search", json_data=request.model_dump(mode="json"))
	except Exception as e:
		raise HTTPException(status_code=503, detail=f"Cannot reach Retrieval service: {e}")

	if status_code == 200:
		return RetrievalResponse(**data)
	else:
		# إعادة الخطأ كما هو من الخدمة الخلفية
		raise HTTPException(status_code=status_code, detail=data)


@router.post("/evaluate/demo", tags=["Evaluation"])
async def evaluate_demo():
	"""Forward demo evaluation request to Evaluation Service.

	يستقبل طلب التقييم التجريبي من الواجهة الأمامية ويرسله لخدمة التقييم.
	"""
	try:
		status_code, data = await post_json(f"{EVALUATION_URL}/evaluate/demo", json_data={})
	except Exception as e:
		raise HTTPException(status_code=503, detail=f"Cannot reach Evaluation service: {e}")

	if status_code == 200:
		return data
	else:
		raise HTTPException(status_code=status_code, detail=data)


@router.post("/evaluate/dataset", tags=["Evaluation"])
async def evaluate_dataset(request: dict):
	"""Forward real dataset evaluation request to Evaluation Service."""
	try:
		status_code, data = await post_json(
			f"{EVALUATION_URL}/evaluate/dataset",
			json_data=request,
			timeout=180.0,
		)
	except Exception as e:
		raise HTTPException(status_code=503, detail=f"Cannot reach Evaluation service: {e}")

	if status_code == 200:
		return data
	else:
		raise HTTPException(status_code=status_code, detail=data)

