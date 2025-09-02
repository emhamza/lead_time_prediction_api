from fastapi import APIRouter, HTTPException, Depends
import time
from services.prediction import SurvivalPredictionService
from api.models import (
    SurvivalPredictionRequest,
    SurvivalPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from api.dependencies import get_prediction_service

router = APIRouter()


@router.post("/predict", response_model=SurvivalPredictionResponse)
async def predict_survival(
        request: SurvivalPredictionRequest,
        prediction_service: SurvivalPredictionService = Depends(get_prediction_service)
):
    """
    Predict survival percentiles and survival curve for a single order.

    Returns:
    - p50: 50th percentile survival time
    - p90: 90th percentile survival time
    - survival_curve: List of time-probability points
    - risk_score: Relative risk assessment
    - event_probability: Probability of event occurring
    """
    try:
        result = prediction_service.predict_survival(request.dict())
        return SurvivalPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_survival(
        batch_request: BatchPredictionRequest,
        prediction_service: SurvivalPredictionService = Depends(get_prediction_service)
):
    """
    Predict survival for multiple orders in batch mode.
    """
    start_time = time.time()

    try:
        predictions = []
        for request in batch_request.requests:
            result = prediction_service.predict_survival(request.dict())
            predictions.append(SurvivalPredictionResponse(**result))

        processing_time = time.time() - start_time

        return BatchPredictionResponse(
            predictions=predictions,
            processing_time=processing_time,
            total_processed=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "survival-analysis-api"}