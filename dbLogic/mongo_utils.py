from pymongo import MongoClient
from datetime import datetime

def save_prediction_to_mongo(
        prediction_results: dict,
        vendor_id:str,
        uri: str = "mongodb://localhost:27017/",
        db_name: str = 'minted',
        collection_name: str = 'WISMOPred_v1'
):
    client = MongoClient(uri)
    collection = client[db_name][collection_name]

    prediction_document = {
        "vendor_id": vendor_id,
        "prediction": prediction_results,
        "saved_at": datetime.utcnow(),
    }

    result = collection.replace_one(
        {"vendor_id": vendor_id},
        prediction_document,
        upsert=True
    )

    if result.matched_count > 0:
        print(f"🔄 Existing prediction for vendor {vendor_id} replaced in collection '{collection_name}'.")
    else:
        print(f"🆕 New prediction for vendor {vendor_id} inserted into collection '{collection_name}'.")