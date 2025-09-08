import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from src.load import load_data
from src.preprocessing import DataPreprocessor
from src.config import DATA_FILE, RSF_PARAMS, TEST_SIZE, RANDOM_STATE


def train_vendor_model(vendor_id: str):
    """
    Train a Random Survival Forest (RSF) model for a given vendor_id.
    Saves the trained model and training columns to disk, logs to MLflow.

    Returns:
        model_path (str): path to saved RSF model
        n_rows (int): number of rows used for training
    """
    print(f"➡️ Loading dataset from: {DATA_FILE}")
    df = load_data(DATA_FILE)

    if "vendor_id" not in df.columns:
        raise ValueError("Dataset does not contain 'vendor_id' column. Please check preprocessing step.")

    vendor_df = df[df["vendor_id"] == vendor_id]

    if vendor_df.empty:
        raise ValueError(f"No data found for vendor_id={vendor_id}")

    print(f"➡️ Training model for vendor_id={vendor_id} | Rows: {len(vendor_df)}")

    preprocessor = DataPreprocessor()
    X, y, processed_df = preprocessor.preprocess_data(vendor_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"➡️ Initializing RSF model with params: {RSF_PARAMS}")
    rsf = RandomSurvivalForest(**RSF_PARAMS)

    with mlflow.start_run(run_name=f"vendor_{vendor_id}"):
        # 🔹 Log training parameters
        mlflow.log_param("vendor_id", vendor_id)
        mlflow.log_params(RSF_PARAMS)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        print(f"➡️ Fitting RSF model for vendor {vendor_id}...")
        rsf.fit(X_train, y_train)
        print(f"✅ Model trained for vendor {vendor_id}")

        # 🔹 Evaluate and log metrics
        train_score = rsf.score(X_train, y_train)
        test_score = rsf.score(X_test, y_test)
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("test_score", test_score)
        print(f"📊 Scores logged | Train: {train_score:.4f}, Test: {test_score:.4f}")

        # 🔹 Save model locally and log to MLflow
        model_path = f"artifacts/v1/{vendor_id}.joblib"
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(rsf, model_path)
        print(f"✅ Model saved locally: {model_path}")

        mlflow.sklearn.log_model(rsf, artifact_path="model")
        mlflow.log_artifact(model_path)
        print(f"✅ Model logged to MLflow under experiment")

    return model_path, len(vendor_df)
