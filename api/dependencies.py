from functools import lru_cache
from services.prediction import SurvivalPredictionService
from configApi.settings import settings

# @lru_cache()
def get_prediction_service() -> SurvivalPredictionService:
    """Get the prediction service (singleton)"""
    return SurvivalPredictionService(
        model_path=settings.MODEL_PATH,
        training_columns_path=settings.TRAINING_COLUMNS_PATH
    )