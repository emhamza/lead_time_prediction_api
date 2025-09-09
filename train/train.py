import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from src.preprocessing import DataPreprocessor
from dbLogic.mongo_utils import load_dataset_from_mongo
from src.config import MLFLOW_EXPERIMENT_NAME


def train_vendor_model(vendor_id: str):
    """
    Train a Random Survival Forest (RSF) model for a given vendor_id.
    Logs the trained model to MLflow and appends test data to a consolidated test file.

    Returns:
        model_info (dict): MLflow model URI, run_id, and local path
    """

    df = load_dataset_from_mongo()

    if "vendor_id" not in df.columns:
        raise ValueError("Dataset does not contain 'vendor_id' column. Please check preprocessing step.")

    vendor_df = df[df["vendor_id"] == vendor_id].copy()

    if vendor_df.empty:
        raise ValueError(f"No data found for vendor_id={vendor_id}")

    print(f"➡️ Training model for vendor_id={vendor_id} | Rows: {len(vendor_df)}")

    # -------------------------
    # Split train/test
    # -------------------------
    train_df, test_df = train_test_split(vendor_df, test_size=0.2, random_state=42)

    consolidated_test_path = "data/consolidated_test.csv"
    os.makedirs(os.path.dirname(consolidated_test_path), exist_ok=True)

    if os.path.exists(consolidated_test_path):
        existing_test = pd.read_csv(consolidated_test_path)
        existing_test = existing_test[existing_test["vendor_id"] != vendor_id]
        combined_test = pd.concat([existing_test, test_df], ignore_index=True)
    else:
        combined_test = test_df

    combined_test.to_csv(consolidated_test_path, index=False)
    print(f"✅ Test data for vendor {vendor_id} added to consolidated test file: {consolidated_test_path}")

    # -------------------------
    # Preprocess
    # -------------------------
    preprocessor = DataPreprocessor()
    print("➡️ Preprocessing training data...")
    X_train, y_train, _ = preprocessor.preprocess_data(train_df)

    test_preprocessor = DataPreprocessor()
    print("➡️ Preprocessing testing data...")
    X_test, y_test, _ = test_preprocessor.preprocess_data(test_df)

    # -------------------------
    # Train model
    # -------------------------
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=42
    )

    print(f"➡️ Fitting RSF model for vendor {vendor_id}...")
    rsf.fit(X_train, y_train)
    print(f"✅ Model trained for vendor {vendor_id}")

    # -------------------------
    # Log to MLflow
    # -------------------------
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"train_vendor_{vendor_id}") as run:
        mlflow.log_param("vendor_id", vendor_id)
        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("n_test_rows", len(test_df))
        mlflow.log_param("total_rows", len(vendor_df))
        mlflow.log_param("n_features", X_train.shape[1])

        # Optionally log metrics (simple example)
        mlflow.log_metric("train_size", len(train_df))
        mlflow.log_metric("test_size", len(test_df))

        # Log model
        mlflow.sklearn.log_model(rsf, artifact_path="model")

        print(f"✅ Model logged to MLflow (run_id={run.info.run_id})")

        model_info = {
            "mlflow_run_id": run.info.run_id,
            "mlflow_model_uri": f"runs:/{run.info.run_id}/model",
        }

    return model_info
