from pymongo import MongoClient
import pandas as pd

def load_dataset_from_mongo(
        uri='mongodb://localhost:27017/',
        db_name="minted",
        collection_name="WISMO",
):
    client = MongoClient(uri)
    collection = client[db_name][collection_name]
    df = pd.DataFrame(list(collection.find({})))
    df.drop(columns=['_id'], inplace=True)
    return df