from fastapi import APIRouter, HTTPException, Depends
import time
from train.train import train_vendor_model
from src.auth import get_current_user
from fastapi.security import OAuth2PasswordRequestForm
from src.auth import authenticate_user, create_access_token
from datetime import timedelta
from src.schemas import TokenResponse, LoginRequest

router=APIRouter()


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token endpoint for Swagger UI.
    Accepts form data with username and password.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login-json", response_model=TokenResponse)
async def login_json(form_data: LoginRequest):
    """
    JSON-based login endpoint for custom clients.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token_expires = timedelta(minutes=30)
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
        model_path, n_rows = train_vendor_model(vendor_id)

        processing_time = time.time() - start_time
        return {
            "status": "success",
            "vendor_id": vendor_id,
            "rows_used": n_rows,
            "model_path": model_path,
            "processing_time": processing_time,
            "message": f"Training completed for vendor {vendor_id}"
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")