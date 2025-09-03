from sklearn.model_selection import train_test_split
from dataLogic.loader import load_data, save_training_columns, save_test_data
from dataLogic.preprocessor import DataPreprocessor
from models.rsf_model import RSFModel
from utils.evaluation import evaluate_model
from config import TEST_SIZE, RANDOM_STATE
import numpy as np


def train_survival_model():
    """Main function to train and evaluate the survival model."""

    # Load data
    df = load_data()

    # Preprocess data
    preprocessor = DataPreprocessor()
    # Unpack the returned tuple correctly
    X, y, df_processed = preprocessor.preprocess_data(df)

    print("Preprocessing successful. Data types returned:")
    print(f"X is of type: {type(X)}")
    print(f"y is of type: {type(y)}")
    print(f"df_processed is of type: {type(df_processed)}")
    print("-" * 30)

    # Save training columns
    save_training_columns(preprocessor.training_columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if np.isinf(X_train).any().any():
        print("Warning: X_train contains infinite values. Cleaning data...")
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train.fillna(0, inplace=True)  # Or any other appropriate imputation method
        print("Infinite values replaced.")


    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("y_train sample:", y_train[:5])

    # Save test data
    test_indices = X_test.index
    test_data_original = df_processed.loc[test_indices].copy()
    save_test_data(test_data_original)

    # Train model
    model = RSFModel()
    model.train(X_train, y_train)

    # Confirm training succeeded
    assert hasattr(model.model, "event_times_"), "Model is not trained properly"

    # Evaluate model
    evaluation_results = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save model
    model.save()
    print("Model training and saving process complete.")

    return model, evaluation_results


if __name__ == "__main__":
    trained_model, results = train_survival_model()