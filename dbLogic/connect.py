from pymongo import MongoClient
from src.config import MONGO_DB, MONGO_URI

_client = MongoClient(MONGO_URI)
_db = _client[MONGO_DB]

def get_collection(collection_name: str):
    return _db[collection_name]
