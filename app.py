from sklearn.model_selection import train_test_split
from dataLogic.loader import load_data, save_training_columns, save_test_data
from dataLogic.preprocessor import DataPreprocessor
from models.rsf_model import RSFModel
from utils.evaluation import evaluate_model
from config import TEST_SIZE, RANDOM_STATE
import numpy as np
import pandas as pd
import sklearn
import sksurv

def train_survival_model():
    """Main function to train and evaluate the survival model."""

    print("=" * 60)
    print("🚀 Starting survival model training pipeline...")
    print(f"🔎 Versions -> scikit-survival: {sksurv.__version__}, "
          f"scikit-learn: {sklearn.__version__}, pandas: {pd.__version__}, numpy: {np.__version__}")

    # STEP 1: Load data
    print("=" * 60)
    print("STEP 1: Loading data...")
    df = load_data()
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # STEP 2: Preprocess
    print("=" * 60)
    print("STEP 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y, df_processed = preprocessor.preprocess_data(df)
    print("Preprocessing complete ✅")
    print(f"X type: {type(X)}, shape: {X.shape}")
    print(f"y type: {type(y)}, dtype: {getattr(y, 'dtype', None)}, shape: {getattr(y,'shape',None)}")
    print(f"y dtype names: {getattr(getattr(y, 'dtype', None), 'names', None)}")
    print(f"df_processed type: {type(df_processed)}, shape: {df_processed.shape}")
    print("-" * 60)

    # STEP 3: Save training columns
    print("STEP 3: Saving training columns...")
    save_training_columns(preprocessor.training_columns)
    print(f"Training columns saved ({len(preprocessor.training_columns)} features).")

    # STEP 4: Split
    print("=" * 60)
    print("STEP 4: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"y_train sample: {y_train[:5]}")

    # Data hygiene for infinities
    if hasattr(X_train, "values"):
        try:
            any_inf = np.isinf(X_train.values).any()
        except Exception:
            any_inf = False
    else:
        any_inf = np.isinf(X_train).any()

    if any_inf:
        print("⚠️ Warning: X_train contains infinite values. Cleaning data...")
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(0)
        print("Infinite values replaced.")

    # STEP 5: Save test data
    print("=" * 60)
    print("STEP 5: Saving test data...")
    test_indices = X_test.index
    test_data_original = df_processed.loc[test_indices].copy()
    save_test_data(test_data_original)
    print(f"Test data saved. Shape: {test_data_original.shape}")

    # STEP 6: Train model
    print("=" * 60)
    print("STEP 6: Training model...")
    model = RSFModel()
    model.train(X_train, y_train)

    # DEBUG: Verify training success
    print("[DEBUG] Model type after training:", type(model.model))
    if model.model is not None:
        unique_times = getattr(model.model, "unique_times_", None)
        event_times = getattr(model.model, "event_times_", None)
        print(f"[DEBUG] unique_times_: {unique_times}")
        print(f"[DEBUG] event_times_: {event_times}")
    else:
        print("[DEBUG] ❌ model.model is None! Training likely failed.")

    print("Model training finished ✅")
    print("-" * 60)

    # STEP 7: Verify training success (assert fitted attributes)
    print("STEP 7: Verifying training success...")
    has_unique = hasattr(model.model, "unique_times_")
    has_event = hasattr(model.model, "event_times_")
    print(f"Model has unique_times_: {has_unique}")
    print(f"Model has event_times_: {has_event}")
    assert (has_unique or has_event), \
        "❌ Model is not trained properly (missing unique_times_/event_times_). " \
        "Check your y_train format and scikit-survival version."
    print("Assertion passed ✅")
    print("=" * 60)

    # STEP 8: Evaluate
    print("STEP 8: Evaluating model...")
    evaluation_results = evaluate_model(model, X_train, y_train, X_test, y_test)
    print("Evaluation complete ✅")
    print(f"Evaluation results: {evaluation_results}")

    # STEP 9: Save model
    print("=" * 60)
    print("STEP 9: Saving trained model...")
    model.save()
    print("Model training and saving process complete ✅")
    print("=" * 60)

    return model, evaluation_results


if __name__ == "__main__":
    trained_model, results = train_survival_model()
