import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from src.load import load_data
from src.preprocessing import DataPreprocessor
from src.config import DATA_FILE


def train_vendor_model(vendor_id: str):
    """
    Train a Random Survival Forest (RSF) model for a given vendor_id.
    Saves the trained model and appends test data to a consolidated test file.

    Returns:
        model_path (str): path to saved RSF model
        n_rows (int): number of rows used for training
    """

    df = load_data(DATA_FILE)

    if "vendor_id" not in df.columns:
        raise ValueError("Dataset does not contain 'vendor_id' column. Please check preprocessing step.")

    vendor_df = df[df["vendor_id"] == vendor_id].copy()

    if vendor_df.empty:
        raise ValueError(f"No data found for vendor_id={vendor_id}")

    print(f"➡️ Training model for vendor_id={vendor_id} | Rows: {len(vendor_df)}")

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

    preprocessor = DataPreprocessor()

    print("➡️ Preprocessing training data...")
    X_train, y_train, _ = preprocessor.preprocess_data(train_df)

    test_preprocessor = DataPreprocessor()
    print("➡️ Preprocessing testing data...")
    X_test, y_test, _ = test_preprocessor.preprocess_data(test_df)

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

    model_path = f"artifacts/v1/{vendor_id}.joblib"
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rsf, model_path)
    print(f"✅ Model saved: {model_path}")

    return model_path, len(vendor_df)