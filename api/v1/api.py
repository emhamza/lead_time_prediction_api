from fastapi import APIRouter, HTTPException, Depends
import time

from predict.prediction_lookup import  get_prediction_by_po_id
from train.re_train import retrain_vendor_model
from train.train import train_vendor_model
from src.auth import get_current_user
from src.auth import authenticate_user, create_access_token
from datetime import timedelta
from src.schemas import TokenResponse, LoginRequest
from predict.predict import predict_vendor_model
from pydantic import BaseModel
from typing import Optional

router=APIRouter()


@router.post("/login", response_model=TokenResponse)
async def login(form_data: LoginRequest):
    """
    JSON-based login endpoint for custom clients.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token_expires = timedelta(hours=2)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/train/{vendor_id}")
async def train_vendor(vendor_id: str, user: dict = Depends(get_current_user)):
    """
    Train a survival model for a specific vendor_id.
    Saves the trained model and training columns.
    """
    start_time = time.time()
    try:
        model_info = train_vendor_model(vendor_id)  # <-- returns dict now

        processing_time = time.time() - start_time
        return {
            "status": "success",
            "vendor_id": vendor_id,
            "mlflow_run_id": model_info["mlflow_run_id"],
            "mlflow_model_uri": model_info["mlflow_model_uri"],
            "processing_time": processing_time,
            "message": f"Training completed for vendor {vendor_id}"
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/pred/{vendor_id}")
async def predict_vendor(vendor_id: str, user:dict = Depends(get_current_user)):
    """
        Predict survival probabilities for a specific vendor_id.
        Saves predictions in predictions/v1/{vendor_id}.json
    """
    try:
        results = predict_vendor_model(vendor_id)
        return results
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/prediction/{po_id}")
async def get_prediction_by_po(po_id: str, user:dict = Depends(get_current_user)):
    try:
        result = get_prediction_by_po_id(po_id)
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prediction: {str(e)}")



class RetrainParams(BaseModel):
    n_estimators: Optional[int] = 200
    min_samples_split: Optional[int] = 10
    min_samples_leaf: Optional[int] = 15
    max_features: Optional[str] = None
    random_state: Optional[int] = 42

@router.post("/re-train/{vendor_id}")
async def retrain_vendor(vendor_id: str, params: RetrainParams, user: dict = Depends(get_current_user)):
    start_time = time.time()
    try:
        custom_params = {k: v for k, v in params.dict().items() if v is not None}

        model_info = retrain_vendor_model(vendor_id, custom_params=custom_params)

        processing_time = time.time() - start_time
        return {
            "status": "success",
            "vendor_id": vendor_id,
            "mlflow_run_id": model_info["mlflow_run_id"],
            "mlflow_model_uri": model_info["mlflow_model_uri"],
            "processing_time": processing_time,
            "message": f"Re-training completed for vendor {vendor_id} with parameters {custom_params}"
        }

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")