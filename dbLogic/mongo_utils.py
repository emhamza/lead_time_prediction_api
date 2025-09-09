from datetime import datetime
import pandas as pd
from .connect import get_collection
from dotenv import load_dotenv
import os

load_dotenv()

DATA_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_DATA")
PRED_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_PRED")

def load_dataset_from_mongo(collection_name=DATA_COLLECTION_NAME):
    collection = get_collection(collection_name)
    df = pd.DataFrame(list(collection.find({})))
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    return df


MODEL_ID = "model_v1"
def save_prediction_to_mongo(
    prediction_results: dict,
    vendor_id: str,
    collection_name: str = PRED_COLLECTION_NAME
):
    collection = get_collection(collection_name)

    prediction_document = {
        "vendor_id": vendor_id,
        "model_id": MODEL_ID,
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