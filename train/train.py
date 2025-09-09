import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from src.load import load_data
from src.preprocessing import DataPreprocessor
from src.config import DATA_FILE, RSF_PARAMS, TEST_FILE, TEST_SIZE, RANDOM_STATE
from utils.logging import logger


def train_vendor_model(vendor_id: str):
    """
    Train a Random Survival Forest (RSF) model for a given vendor_id.
    Saves the trained model, appends test data to consolidated file, and logs to MLflow.

    Returns:
        model_path (str): path to saved RSF model
        n_rows (int): number of rows used for training
    """

    # Load dataset
    df = load_data(DATA_FILE)
    if "vendor_id" not in df.columns:
        raise ValueError("Dataset does not contain 'vendor_id' column.")

    vendor_df = df[df["vendor_id"] == vendor_id].copy()
    if vendor_df.empty:
        raise ValueError(f"No data found for vendor_id={vendor_id}")

    logger.info(f"➡️ Training model for vendor_id={vendor_id} | Rows: {len(vendor_df)}")

    # Train-test split
    train_df, test_df = train_test_split(vendor_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Consolidated test file update
    os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)
    if os.path.exists(TEST_FILE):
        existing_test = pd.read_csv(TEST_FILE)
        existing_test = existing_test[existing_test["vendor_id"] != vendor_id]
        combined_test = pd.concat([existing_test, test_df], ignore_index=True)
    else:
        combined_test = test_df
    combined_test.to_csv(TEST_FILE, index=False)
    logger.info(f"✅ Test data for vendor {vendor_id} appended to {TEST_FILE}")

    # Preprocessing
    preprocessor = DataPreprocessor()
    X_train, y_train, _ = preprocessor.preprocess_data(train_df)

    test_preprocessor = DataPreprocessor()
    X_test, y_test, _ = test_preprocessor.preprocess_data(test_df)

    # Model init
    rsf = RandomSurvivalForest(**RSF_PARAMS)

    # 🔹 Start MLflow run
    with mlflow.start_run(run_name=f"vendor_{vendor_id}"):
        mlflow.log_param("vendor_id", vendor_id)
        mlflow.log_params(RSF_PARAMS)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        logger.info(f"➡️ Fitting RSF model for vendor {vendor_id}...")
        rsf.fit(X_train, y_train)
        logger.info(f"✅ Model trained for vendor {vendor_id}")

        # Evaluate
        train_score = rsf.score(X_train, y_train)
        test_score = rsf.score(X_test, y_test)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        logger.info(f"📊 Scores | Train: {train_score:.4f}, Test: {test_score:.4f}")

        # Save & log model
        model_path = f"artifacts/v1/{vendor_id}.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(rsf, model_path)
        logger.info(f"✅ Model saved locally: {model_path}")

        mlflow.sklearn.log_model(rsf, artifact_path="model")
        mlflow.log_artifact(model_path)
        logger.info("✅ Model logged to MLflow")

    return model_path, len(vendor_df)
