import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

_client = MongoClient(MONGO_URI)
_db = _client[MONGO_DB]

def get_collection(collection_name: str):
    return _db[collection_name]
