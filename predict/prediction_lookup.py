from fastapi import HTTPException
from typing import Dict, Any
import logging

from predict.predict import predict_vendor_model
from src.load import load_data
from src.config import TEST_FILE
from dbLogic.mongo_utils import load_predictions_from_mongo, save_prediction_to_mongo

logger = logging.getLogger(__name__)

def get_prediction_by_po_id(po_id: str) -> Dict[str, Any]:
    logger.info(f"🔎 Looking up prediction for PO_ID={po_id}")

    # Step 1: Load test data and find vendor_id
    df = load_data(TEST_FILE)
    df["PO_ID"] = df["PO_ID"].astype(str)
    row = df[df["PO_ID"] == po_id]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No record found for PO_ID={po_id} in test data"
        )

    vendor_id = row["vendor_id"].iloc[0]
    logger.info(f"📦 Found vendor_id={vendor_id} for PO_ID={po_id}")

    # Step 2: Try loading saved predictions
    predictions_df = load_predictions_from_mongo(vendor_id)
    predictions_df["PO_ID"] = predictions_df["PO_ID"].astype(str)
    prediction_row = predictions_df[predictions_df["PO_ID"] == po_id]

    if not prediction_row.empty:
        logger.info(f"✅ Found cached prediction for PO_ID={po_id}")
        prediction_result = prediction_row.to_dict(orient="records")[0]
    else:
        logger.info(f"⚙️ No cached prediction for PO_ID={po_id}, generating new one")
        try:
            results = predict_vendor_model(vendor_id)
            return results
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {
        "prediction": prediction_result
    }
