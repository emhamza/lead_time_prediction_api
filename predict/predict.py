import json
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Dict, Any
from fastapi import HTTPException

from src.preprocessing import DataPreprocessor
from src.load import load_data
from src.config import TEST_FILE, MLFLOW_EXPERIMENT_NAME
from utils.logging import logger


def predict_vendor_model(vendor_id: str) -> Dict[str, Any]:
    try:
        # -------------------------
        # 🔹 Fetch latest model from MLflow
        # -------------------------
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if experiment is None:
            raise HTTPException(status_code=404, detail="MLflow experiment not found. Please train first.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.vendor_id = '{vendor_id}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise HTTPException(status_code=404, detail=f"No MLflow run found for vendor_id={vendor_id}")

        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"➡️ Loading latest model for vendor {vendor_id} from MLflow (run_id={run_id})")

        model = mlflow.sklearn.load_model(model_uri)

        # -------------------------
        # 🔹 Load test data
        # -------------------------
        df = load_data(TEST_FILE)
        vendor_df = df[df["vendor_id"] == vendor_id].copy()

        if vendor_df.empty:
            raise HTTPException(status_code=404, detail=f"No test data found for vendor_id={vendor_id}")

        if "PO_ID" not in vendor_df.columns:
            vendor_df["PO_ID"] = vendor_df.index.astype(str)
            logger.warning("⚠️ PO_ID column not found. Using row indices as PO_ID.")

        logger.info(f"➡️ Making predictions for vendor_id={vendor_id} | Test samples: {len(vendor_df)}")

        # -------------------------
        # 🔹 Preprocess
        # -------------------------
        preprocessor = DataPreprocessor()
        X, y, processed_df = preprocessor.preprocess_data(vendor_df)
        logger.info(f"➡️ Preprocessed features shape: {X.shape}")

        # -------------------------
        # 🔹 Generate predictions
        # -------------------------
        logger.info("➡️ Generating survival predictions...")
        survival_functions = model.predict_survival_function(X)
        event_times = model.unique_times_

        predictions = []
        for survival_fn, po_id in zip(survival_functions, vendor_df["PO_ID"]):
            survival_probs = survival_fn(event_times)

            p50_time = calculate_percentile_survival_time(event_times, survival_probs, 0.5)
            p90_time = calculate_percentile_survival_time(event_times, survival_probs, 0.1)

            survival_curve = [
                {"time": float(time), "survival_probability": round(float(prob), 2)}
                for time, prob in zip(event_times, survival_probs)
            ]

            predictions.append({
                "PO_ID": str(po_id),
                "p50_survival_time": p50_time,
                "p90_survival_time": p90_time,
                "survival_curve": survival_curve,
                "risk_score": float(1 - survival_probs[-1]) if len(survival_probs) > 0 else None
            })

        # -------------------------
        # 🔹 Summary statistics
        # -------------------------
        p50_times = [p["p50_survival_time"] for p in predictions if p["p50_survival_time"] is not None]
        p90_times = [p["p90_survival_time"] for p in predictions if p["p90_survival_time"] is not None]

        summary = {
            "vendor_id": vendor_id,
            "total_predictions": len(predictions),
            "p50_statistics": stats_summary(p50_times),
            "p90_statistics": stats_summary(p90_times),
            "event_time_range": {
                "min_time": float(event_times.min()),
                "max_time": float(event_times.max())
            }
        }

        # -------------------------
        # 🔹 Save prediction results locally
        # -------------------------
        prediction_result = {
            "predictions": predictions,
            "summary": summary,
            "metadata": {
                "model_uri": model_uri,
                "mlflow_run_id": run_id,
                "prediction_timestamp": pd.Timestamp.now().isoformat(),
                "n_features": X.shape[1],
                "n_time_points": len(event_times)
            }
        }

        predictions_dir = "predictions/v1"
        os.makedirs(predictions_dir, exist_ok=True)
        prediction_path = os.path.join(predictions_dir, f"{vendor_id}.json")

        with open(prediction_path, "w") as f:
            json.dump(prediction_result, f, indent=4)

        logger.info(f"✅ Predictions completed and saved: {prediction_path}")
        return {
            "predictions": predictions,
            "summary": summary,
            "prediction_path": prediction_path,
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# -------------------------
# 🔹 Helpers
# -------------------------
def calculate_percentile_survival_time(event_times: np.ndarray, survival_probs: np.ndarray,
                                       target_probability: float) -> float:
    try:
        idx = np.where(survival_probs <= target_probability)[0]
        return float(event_times[idx[0]]) if len(idx) > 0 else None
    except Exception:
        return None


def stats_summary(values):
    return {
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "min": float(np.min(values)) if values else None,
        "max": float(np.max(values)) if values else None,
    }
