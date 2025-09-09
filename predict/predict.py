import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from fastapi import HTTPException

from src.preprocessing import DataPreprocessor
from src.load import load_data
from src.config import TEST_FILE
from dbLogic.mongo_utils import save_prediction_to_mongo


def predict_vendor_model(vendor_id: str) -> Dict[str, Any]:
    try:
        model_path = os.path.join("artifacts/v1", f"{vendor_id}.joblib")

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for vendor_id={vendor_id}. Please train the model first."
            )

        model = joblib.load(model_path)

        df = load_data(TEST_FILE)
        vendor_df = df[df["vendor_id"] == vendor_id].copy()

        if vendor_df.empty:
            raise HTTPException(status_code=404, detail=f"No test data found for vendor_id={vendor_id}")

        if "PO_ID" not in vendor_df.columns:
            # If PO_ID doesn't exist, create one using index
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

        for idx, (survival_fn, po_id) in enumerate(zip(survival_functions, vendor_df["PO_ID"])):
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
                "risk_score": float(1 - survival_probs[-1]) if len(survival_probs) > 0 else None
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
                "max": float(np.max(p50_times)) if p50_times else None
            },
            "p90_statistics": {
                "mean": float(np.mean(p90_times)) if p90_times else None,
                "median": float(np.median(p90_times)) if p90_times else None,
                "std": float(np.std(p90_times)) if p90_times else None,
                "min": float(np.min(p90_times)) if p90_times else None,
                "max": float(np.max(p90_times)) if p90_times else None
            },
            "event_time_range": {
                "min_time": float(event_times.min()),
                "max_time": float(event_times.max())
            }
        }

        prediction_result = {
            "predictions": predictions,
            "summary": summary,
            "metadata": {
                "model_path": model_path,
                "prediction_timestamp": pd.Timestamp.now().isoformat(),
                "n_features": X.shape[1],
                "n_time_points": len(event_times)
            }
        }

        save_prediction_to_mongo(prediction_result, vendor_id)
        print(f"✅ Predictions completed and saved")
        print(f"✅ Generated predictions for {len(predictions)} records")

        return {
            "predictions": predictions,
            "summary": summary,
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def calculate_percentile_survival_time(event_times: np.ndarray, survival_probs: np.ndarray,
                                       target_probability: float) -> float:
    """
    Calculate the time at which survival probability reaches the target percentile.

    Args:
        event_times: Array of time points
        survival_probs: Array of survival probabilities
        target_probability: Target survival probability (e.g., 0.5 for median)

    Returns:
        Time at which survival probability reaches target, or None if not reached
    """
    try:
        idx = np.where(survival_probs <= target_probability)[0]

        if len(idx) == 0:
            return None

        return float(event_times[idx[0]])

    except Exception as e:
        print(f"Warning: Could not calculate percentile survival time: {e}")
        return None