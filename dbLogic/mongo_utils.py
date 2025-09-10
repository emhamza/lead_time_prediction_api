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

    existing_doc = collection.find_one({"vendor_id": vendor_id})
    new_predictions = prediction_results.get("predictions", [])

    if existing_doc:
        existing_predictions = existing_doc.get("prediction", {}).get("predictions", [])
        merged_predictions = existing_predictions + new_predictions

        updated_document = {
            "vendor_id": vendor_id,
            "prediction": {"predictions": merged_predictions},
            "saved_at": datetime.utcnow()
        }

        result = collection.replace_one({"vendor_id": vendor_id}, updated_document)
        print(f"🔄 Appended {len(new_predictions)} new predictions for vendor {vendor_id} in '{collection_name}'")

    else:
        prediction_document = {
            "vendor_id": vendor_id,
            "prediction": {"predictions": new_predictions},
            "saved_at": datetime.utcnow()
        }

        collection.insert_one(prediction_document)
        print(f"🆕 Inserted new prediction document for vendor {vendor_id} into '{collection_name}'")

def load_predictions_from_mongo(
    vendor_id: str,
    collection_name: str = PRED_COLLECTION_NAME
) -> pd.DataFrame:
    collection = get_collection(collection_name)
    document = collection.find_one({"vendor_id": vendor_id})

    if not document:
        return pd.DataFrame()

    saved_at = document.get("saved_at")
    prediction_list = document.get("prediction", {}).get("predictions", [])

    records = []
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

    return pd.DataFrame(records)
