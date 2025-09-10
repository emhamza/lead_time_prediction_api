import mlflow
import numpy as np
import pandas as pd
from typing import Dict, Any
from fastapi import HTTPException

from src.preprocessing import DataPreprocessor
from src.load import load_data
from src.config import TEST_FILE, MLFLOW_EXPERIMENT_NAME
from dbLogic.mongo_utils import save_prediction_to_mongo, load_predictions_from_mongo


def predict_vendor_model(vendor_id: str) -> Dict[str, Any]:
    try:
        # check for the predictions
        existing_df = load_predictions_from_mongo(vendor_id)
        if not existing_df.empty:
            print(f"🟡 PREDICTION IS ALREADY AVAILABLE for vendor_id={vendor_id}")
            return {
                "status": "already_exists",
                "vendor_id": vendor_id,
                "message": "Prediction already exists in MongoDB"
            }

        df = load_data(TEST_FILE)
        vendor_df = df[df["vendor_id"] == vendor_id].copy()

        # proceed with the prediction logic if not found
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if experiment is None:
            raise HTTPException(status_code=404, detail=f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found in MLflow")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.vendor_id = '{vendor_id}'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            raise HTTPException(status_code=404, detail=f"No MLflow run found for vendor_id={vendor_id}")

        run = runs[0]
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        print(f"➡️ Loading model from MLflow (run_id={run_id}, vendor_id={vendor_id})")
        model = mlflow.sklearn.load_model(model_uri)

        if vendor_df.empty:
            raise HTTPException(status_code=404, detail=f"No test data found for vendor_id={vendor_id}")

        if "PO_ID" not in vendor_df.columns:
            vendor_df["PO_ID"] = vendor_df.index.astype(str)
            print("⚠️ Warning: PO_ID column not found. Using row indices as PO_ID.")

        print(f"➡️ Making predictions for vendor_id={vendor_id} | Test samples: {len(vendor_df)}")

        preprocessor = DataPreprocessor()
        X, y, processed_df = preprocessor.preprocess_data(vendor_df)

        print(f"➡️ Preprocessed features shape: {X.shape}")

        print("➡️ Generating survival predictions...")
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

            prediction_row = {
                "PO_ID": str(po_id),
                "p50_survival_time": p50_time,
                "p90_survival_time": p90_time,
                "survival_curve": survival_curve,
                "risk_score": float(1 - survival_probs[-1]) if len(survival_probs) > 0 else None,
                "model_version": "v1",
                "model_id": run_id
            }
            predictions.append(prediction_row)

        p50_times = [p["p50_survival_time"] for p in predictions if p["p50_survival_time"] is not None]
        p90_times = [p["p90_survival_time"] for p in predictions if p["p90_survival_time"] is not None]

        summary = {
            "vendor_id": vendor_id,
            "total_predictions": len(predictions),
            "p50_statistics": {
                "mean": float(np.mean(p50_times)) if p50_times else None,
                "median": float(np.median(p50_times)) if p50_times else None,
                "std": float(np.std(p50_times)) if p50_times else None,
                "min": float(np.min(p50_times)) if p50_times else None,
                "max": float(np.max(p50_times)) if p50_times else None,
            },
            "p90_statistics": {
                "mean": float(np.mean(p90_times)) if p90_times else None,
                "median": float(np.median(p90_times)) if p90_times else None,
                "std": float(np.std(p90_times)) if p90_times else None,
                "min": float(np.min(p90_times)) if p90_times else None,
                "max": float(np.max(p90_times)) if p90_times else None,
            },
            "event_time_range": {
                "min_time": float(event_times.min()),
                "max_time": float(event_times.max()),
            },
        }

        prediction_result = {
            "predictions": predictions,
            "summary": summary,
            "metadata": {
                "prediction_timestamp": pd.Timestamp.now().isoformat(),
                "n_features": X.shape[1],
                "n_time_points": len(event_times),
            },
        }

        save_prediction_to_mongo(prediction_result, vendor_id)
        print(f"✅ Predictions completed and saved for vendor_id={vendor_id}")
        print(f"✅ Generated predictions for {len(predictions)} records")

        return {
            "predictions": predictions,
            "summary": summary,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def calculate_percentile_survival_time(event_times: np.ndarray, survival_probs: np.ndarray, target_probability: float) -> float:
    """Calculate the time at which survival probability reaches the target percentile."""
    try:
        idx = np.where(survival_probs <= target_probability)[0]
        if len(idx) == 0:
            return None
        return float(event_times[idx[0]])
    except Exception as e:
        print(f"Warning: Could not calculate percentile survival time: {e}")
        return None
