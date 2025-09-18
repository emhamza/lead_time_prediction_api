import mlflow
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import logging

from src.preprocessing import DataPreprocessor
from src.load import load_data
from src.config import TEST_FILE, MLFLOW_EXPERIMENT_NAME
from dbLogic.mongo_utils import save_prediction_to_mongo, load_predictions_from_mongo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_vendor_model(vendor_id: int) -> Dict[int, Any]:
    try:
        logger.info(f"Loading test data for vendor_id={vendor_id}")
        df = load_data(TEST_FILE)
        vendor_df = df[df["fulfiller_id"] == vendor_id].copy()

        if vendor_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No test data found for vendor_id={vendor_id}"
            )

        if "po_id" not in vendor_df.columns:
            vendor_df["po_id"] = vendor_df.index.astype(int)
            logger.warning("po_id column not found. Using row indices as po_id.")

        incoming_po_ids = set(vendor_df["po_id"].astype(int).unique())
        logger.info(f"Found {len(incoming_po_ids)} po_ids in incoming data for vendor_id={vendor_id}")

        existing_predictions_df = load_predictions_from_mongo(vendor_id)

        if existing_predictions_df.empty:
            logger.info(f"🆕 No existing predictions found for vendor_id={vendor_id}. Processing all po_ids.")
            po_ids_to_process = incoming_po_ids
            prediction_mode = "new_vendor"
        else:
            existing_po_ids = set(existing_predictions_df["po_id"].astype(int).unique())
            po_ids_to_process = incoming_po_ids - existing_po_ids

            logger.info(f"📋 Existing predictions found for vendor_id={vendor_id}")
            logger.info(f"   Existing po_ids: {len(existing_po_ids)}")
            logger.info(f"   Incoming po_ids: {len(incoming_po_ids)}")
            logger.info(f"   New po_ids to process: {len(po_ids_to_process)}")

            if not po_ids_to_process:
                return _create_response(
                    status="no_new_predictions_needed",
                    vendor_id=vendor_id,
                    message=f"All {len(incoming_po_ids)} po_ids already have predictions",
                    metadata={
                        "existing_po_count": len(existing_po_ids),
                        "incoming_po_count": len(incoming_po_ids),
                        "new_po_count": 0
                    }
                )

            prediction_mode = "append_to_existing"

        vendor_df_filtered = vendor_df[vendor_df["po_id"].isin(po_ids_to_process)].copy()

        logger.info(f"🔄 Processing {len(vendor_df_filtered)} records for vendor_id={vendor_id}")

        model = _load_vendor_model(vendor_id)

        new_predictions = _generate_predictions(vendor_df_filtered, model)

        summary = _create_summary_statistics(vendor_id, new_predictions, model.unique_times_)

        prediction_result = {
            "predictions": new_predictions,
            "summary": summary,
            "metadata": {
                "prediction_timestamp": pd.Timestamp.now().isoformat(),
                "prediction_mode": prediction_mode,
                "new_predictions_count": len(new_predictions),
                "existing_predictions_count": len(existing_predictions_df) if not existing_predictions_df.empty else 0,
                "total_po_ids_after_update": len(new_predictions) + (
                    len(existing_predictions_df) if not existing_predictions_df.empty else 0)
            }
        }

        save_prediction_to_mongo(prediction_result, vendor_id)

        logger.info(
            f"✅ Successfully generated and saved {len(new_predictions)} new predictions for vendor_id={vendor_id}")

        return _create_response(
            status="success",
            vendor_id=vendor_id,
            message=f"Generated predictions for {len(new_predictions)} new po_ids",
            predictions=new_predictions,
            summary=summary,
            metadata=prediction_result["metadata"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error for vendor_id={vendor_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for vendor_id={vendor_id}: {str(e)}"
        )


def _load_vendor_model(vendor_id: int):
    """Load the most recent trained model for the specified vendor."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if experiment is None:
            raise HTTPException(
                status_code=404,
                detail=f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found"
            )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.vendor_id = '{vendor_id}'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for vendor_id={vendor_id}"
            )

        run = runs[0]
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        logger.info(f"📦 Loading model from MLflow (run_id={run_id[:8]}...)")
        model = mlflow.sklearn.load_model(model_uri)

        model._mlflow_run_id = run_id

        return model

    except Exception as e:
        logger.error(f"Failed to load model for vendor_id={vendor_id}: {int(e)}")
        raise


def _generate_predictions(vendor_df: pd.DataFrame, model) -> List[Dict[int, Any]]:
    try:
        preprocessor = DataPreprocessor()
        X, y, processed_df = preprocessor.preprocess_data(vendor_df)

        logger.info(f"📊 Preprocessed features shape: {X.shape}")

        logger.info("🔮 Generating survival predictions...")
        survival_functions = model.predict_survival_function(X)
        event_times = model.unique_times_

        predictions = []
        for idx, (survival_fn, po_id) in enumerate(zip(survival_functions, vendor_df["po_id"])):
            try:
                survival_probs = survival_fn(event_times)

                p50_time = _calculate_percentile_survival_time(event_times, survival_probs, 0.5)
                p90_time = _calculate_percentile_survival_time(event_times, survival_probs, 0.1)

                survival_curve = [
                    {"time": float(time), "survival_probability": round(float(prob), 2)}
                    for time, prob in zip(event_times, survival_probs)
                ]

                risk_score = float(1 - survival_probs[-1]) if len(survival_probs) > 0 else None

                prediction_row = {
                    "po_id": str(po_id),
                    "p50_survival_time": p50_time,
                    "p90_survival_time": p90_time,
                    "survival_curve": survival_curve,
                    "risk_score": risk_score,
                    "model_version": "v1",
                    "model_id": getattr(model, '_mlflow_run_id', 'unknown')
                }
                predictions.append(prediction_row)

            except Exception as e:
                logger.error(f"Error processing po_id {po_id}: {str(e)}")
                continue

        logger.info(f"✅ Generated {len(predictions)} predictions successfully")
        return predictions

    except Exception as e:
        logger.error(f"Failed to generate predictions: {str(e)}")
        raise


def _calculate_percentile_survival_time(event_times: np.ndarray, survival_probs: np.ndarray,
                                        target_probability: float) -> Optional[float]:
    try:
        idx = np.where(survival_probs <= target_probability)[0]
        if len(idx) == 0:
            return None
        return float(event_times[idx[0]])
    except Exception as e:
        logger.warning(f"Could not calculate percentile survival time: {e}")
        return None


def _create_summary_statistics(vendor_id: str, predictions: List[Dict[str, Any]], event_times: np.ndarray) -> Dict[
    str, Any]:
    try:
        p50_times = [p["p50_survival_time"] for p in predictions if p["p50_survival_time"] is not None]
        p90_times = [p["p90_survival_time"] for p in predictions if p["p90_survival_time"] is not None]
        risk_scores = [p["risk_score"] for p in predictions if p["risk_score"] is not None]

        return {
            "vendor_id": vendor_id,
            "total_predictions": len(predictions),
            "p50_statistics": _calculate_stats(p50_times),
            "p90_statistics": _calculate_stats(p90_times),
            "risk_score_statistics": _calculate_stats(risk_scores),
            "event_time_range": {
                "min_time": float(event_times.min()) if len(event_times) > 0 else None,
                "max_time": float(event_times.max()) if len(event_times) > 0 else None,
                "time_points": len(event_times)
            }
        }
    except Exception as e:
        return {"vendor_id": vendor_id, "total_predictions": len(predictions), "error": str(e)}


def _calculate_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None
        }

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values))
    }


def _create_response(status: str, vendor_id: int, message: str, **kwargs) -> Dict[str, Any]:
    response = {
        "status": status,
        "vendor_id": vendor_id,
        "message": message,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    response.update(kwargs)
    return response