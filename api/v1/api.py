from fastapi import APIRouter, HTTPException
import time
from train.train import train_vendor_model

router=APIRouter()


#new api endpoint
@router.post("/train/{vendor_id}")
async def train_vendor(vendor_id: str):
    """
    Train a survival model for a specific vendor_id.
    Saves the trained model and training columns.
    """
    start_time = time.time()
    try:
        model_path, n_rows = train_vendor_model(vendor_id)

        processing_time = time.time() - start_time
        return {
            "status": "success",
            "vendor_id": vendor_id,
            "rows_used": n_rows,
            "model_path": model_path,
            # "training_columns_path": cols_path,
            "processing_time": processing_time,
            "message": f"Training completed for vendor {vendor_id}"
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")