import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient
from utils.logging import logger
from dotenv import load_dotenv

load_dotenv()

DATA_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_DATA")
PRED_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_PRED")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_FILE = os.path.join(PROJECT_ROOT, "data", "consolidated_test.csv")

RSF_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}


CUTOFF_DATE = pd.Timestamp('2025-04-01')
TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURES = [
    'fulfiller_id',
    'po_id',
    'total_orders',
    'routing_lane_id',
    'fast_track_orders',
    'delivered',
    'ship_method',
    'lead_time_days',
    'is_working',
    'total_items',
    'is_delivered'  # New feature for the event
]


# -------------------------
# ⚡ MLflow Configuration
# -------------------------
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
MLFLOW_EXPERIMENT_NAME = "LeadTimePrediction_MongoDB"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()
exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if exp is None:
    exp_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"✅ Created MLflow experiment: {MLFLOW_EXPERIMENT_NAME} (id={exp_id})")
else:
    logger.info(f"ℹ️ MLflow experiment already exists: {MLFLOW_EXPERIMENT_NAME} (id={exp.experiment_id})")

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)