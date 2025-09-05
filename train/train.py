import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from dataLogic.loader import load_data, save_training_columns
from dataLogic.preprocessor import DataPreprocessor
from config import DATA_FILE

def train_vendor_model(vendor_id: str):
    """
    Train a Random Survival Forest (RSF) model for a given vendor_id.
    Saves the trained model and training columns to disk.

    Returns:
        model_path (str): path to saved RSF model
        cols_path (str): path to saved training columns
        n_rows (int): number of rows used for training
    """

    # ----------------------------
    # 1. Load dataset
    # ----------------------------
    df = load_data(DATA_FILE)

    if "vendor_id" not in df.columns:
        raise ValueError("Dataset does not contain 'vendor_id' column. Please check preprocessing step.")

    # ----------------------------
    # 2. Filter dataset for vendor_id
    # ----------------------------
    vendor_df = df[df["vendor_id"] == vendor_id]

    if vendor_df.empty:
        raise ValueError(f"No data found for vendor_id={vendor_id}")

    print(f"➡️ Training model for vendor_id={vendor_id} | Rows: {len(vendor_df)}")

    # ----------------------------
    # 3. Preprocess data
    # ----------------------------
    preprocessor = DataPreprocessor()
    X, y, processed_df = preprocessor.preprocess_data(vendor_df)

    # Save training columns
    cols_path = f"vendorModels/{vendor_id}_training_column.joblib"
    os.makedirs("vendorModels", exist_ok=True)
    save_training_columns(preprocessor.training_columns, cols_path)

    # ----------------------------
    # 4. Train RSF model
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # ----------------------------
    # 5. Save model
    # ----------------------------
    model_path = f"vendorModels/{vendor_id}.joblib"
    joblib.dump(rsf, model_path)
    print(f"✅ Model saved: {model_path}")

    return model_path, cols_path, len(vendor_df)