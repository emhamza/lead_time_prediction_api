from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.api import router as api_router
from mlflow.tracking import MlflowClient
import mlflow
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from utils.logging import logger


app = FastAPI(
    title="Survival Analysis API",
    description="API for predicting survival probabilities and training vendor-specific models",
    version="0.2.0",
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1", tags=["auth"])
app.include_router(api_router, prefix="/api/v1", tags=["ML"])


@app.on_event("startup")
async def startup_event():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if exp is None:
        exp_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"✅ Created MLflow experiment: {MLFLOW_EXPERIMENT_NAME} (id={exp_id})")
    else:
        logger.info(f"ℹ️ MLflow experiment already exists: {MLFLOW_EXPERIMENT_NAME} (id={exp.experiment_id})")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@app.get("/")
async def root():
    return {
        "message": "Survival Analysis API",
        "version": "0.2.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True
    )
