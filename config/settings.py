import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = Field(default="Survival Analysis API", description="API title")
    API_DESCRIPTION: str = Field(default="API for predicting survival probabilities and percentiles",
                                 description="API description")
    API_VERSION: str = Field(default="0.1.0", description="API version")

    # Model paths
    MODEL_PATH: str = Field(default="random_survival_forest.joblib", description="Path to trained model")
    TRAINING_COLUMNS_PATH: str = Field(default="training_columns.joblib", description="Path to training columns")

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()