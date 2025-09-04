from sklearn.model_selection import train_test_split
from dataLogic.loader import load_data, save_training_columns, save_test_data
from dataLogic.preprocessor import DataPreprocessor
from models.rsf_model import RSFModel
from utils.evaluation import evaluate_model
from config import TEST_SIZE, RANDOM_STATE
import numpy as np
from dataLogic.vendor_utils import assign_vendor_ids
import pandas as pd
import sklearn
import sksurv

def train_survival_model():
    """Main function to train and evaluate the survival model."""

    # STEP 1: Load data
    df = load_data()

    # Step 2: Assign vendor IDs
    # df = assign_vendor_ids(df)
    # df.to_csv('dataset_with_vendors.csv', index=False)

    # STEP 2: Preprocess
    preprocessor = DataPreprocessor()
    X, y, df_processed = preprocessor.preprocess_data(df)

    # STEP 3: Save training columns
    save_training_columns(preprocessor.training_columns)

    # STEP 4: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Data hygiene for infinities
    if hasattr(X_train, "values"):
        try:
            any_inf = np.isinf(X_train.values).any()
        except Exception:
            any_inf = False
    else:
        any_inf = np.isinf(X_train).any()

    if any_inf:
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(0)

    # STEP 5: Save test data
    test_indices = X_test.index
    test_data_original = df_processed.loc[test_indices].copy()
    save_test_data(test_data_original)
    print(f"Test data saved. Shape")

    # STEP 6: Train model
    model = RSFModel()
    model.train(X_train, y_train)

    # Verify training success
    assert (
        hasattr(model.model, "unique_times_")
        or hasattr(model.model, "event_times_")
    ), "Model training failed: missing fitted attributes."

    # STEP 8: Evaluate
    evaluation_results = evaluate_model(model, X_train, y_train, X_test, y_test)


    # STEP 9: Save model
    model.save()
    print("Model training and saving process complete ✅")
    print("=" * 60)

    return model, evaluation_results


if __name__ == "__main__":
    trained_model, results = train_survival_model()
