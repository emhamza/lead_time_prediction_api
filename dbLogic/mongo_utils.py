from datetime import datetime
import pandas as pd
from .connect import get_collection
from src.config import DATA_COLLECTION_NAME, PRED_COLLECTION_NAME

def load_dataset_from_mongo(collection_name=DATA_COLLECTION_NAME):
    collection = get_collection(collection_name)
    df = pd.DataFrame(list(collection.find({})))
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    return df


def save_prediction_to_mongo(
    prediction_results: dict,
    vendor_id: str,
    collection_name: str = PRED_COLLECTION_NAME
):
    collection = get_collection(collection_name)

    prediction_document = {
        "vendor_id": vendor_id,
        "prediction": prediction_results,
        "saved_at": datetime.utcnow()
    }

    result = collection.replace_one(
        {"vendor_id": vendor_id},
        prediction_document,
        upsert=True
    )

    if result.matched_count > 0:
        print(f"🔄 Replaced existing prediction for vendor {vendor_id} in '{collection_name}'")
    else:
        print(f"🆕 Inserted new prediction for vendor {vendor_id} into '{collection_name}'")

def load_predictions_from_mongo(
        collection_name=PRED_COLLECTION_NAME
):
    collection = get_collection(collection_name)
    documents = list(collection.find({}))

    records = []
    for doc in documents:
        vendor_id = doc.get("vendor_id")
        saved_at = doc.get("saved_at")
        prediction_list = doc.get("prediction", {}).get("predictions", [])
        for pred in prediction_list:
            record = {
                "vendor_id": vendor_id,
                "saved_at": saved_at,
                "PO_ID": pred.get("PO_ID"),
                "p50_survival_time": pred.get("p50_survival_time"),
                "p90_survival_time": pred.get("p90_survival_time"),
                "risk_score": pred.get("risk_score"),
                "model_version": pred.get("model_version"),
                "model_id": pred.get("model_id"),
                "survival_curve": pred.get("survival_curve")
            }
            records.append(record)
    df = pd.DataFrame(records)
    return df
