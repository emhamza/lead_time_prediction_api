import os
import joblib
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from src.load import load_data
from src.preprocessing import DataPreprocessor
from src.config import DATA_FILE

def train_vendor_model(vendor_id: str):
    """
    Train a Random Survival Forest (RSF) model for a given vendor_id.
    Saves the trained model and training columns to disk.

    Returns:
        model_path (str): path to saved RSF model
        cols_path (str): path to saved training columns
        n_rows (int): number of rows used for training
    """

    df = load_data(DATA_FILE)

    if "vendor_id" not in df.columns:
        raise ValueError("Dataset does not contain 'vendor_id' column. Please check preprocessing step.")

    vendor_df = df[df["vendor_id"] == vendor_id]

    if vendor_df.empty:
        raise ValueError(f"No data found for vendor_id={vendor_id}")

    print(f"➡️ Training model for vendor_id={vendor_id} | Rows: {len(vendor_df)}")

    preprocessor = DataPreprocessor()
    X, y, processed_df = preprocessor.preprocess_data(vendor_df)

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

    model_path = f"artifacts/v1/{vendor_id}.joblib"
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rsf, model_path)
    print(f"✅ Model saved: {model_path}")

    return model_path, len(vendor_df)