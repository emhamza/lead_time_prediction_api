from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.api import router as api_router
from api.v1.auth_endpoints import router as auth_router
from utils.logging import setup_logging

# ----------------------------
# Setup logging
# ----------------------------
setup_logging()

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI(
    title="Survival Analysis API",
    description="API for predicting survival probabilities and training vendor-specific models",
    version="0.2.0",
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
)

# ----------------------------
# Add CORS middleware
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Include API routers
# ----------------------------
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(api_router, prefix="/api/v1", tags=["ML"])

# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
async def root():
    return {
        "message": "Survival Analysis API",
        "version": "0.2.0",
        "docs": "/docs"
    }

# ----------------------------
# Entry point for uvicorn
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",   # hardcoded host
        port=8000,          # hardcoded port
        reload=True         # good for dev, set False in prod
    )
