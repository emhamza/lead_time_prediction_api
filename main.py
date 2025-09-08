from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.api import router as api_router
from utils.logging import setup_logging


setup_logging()

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
