from sklearn.model_selection import train_test_split
from dataLogic.loader import load_data, save_training_columns, save_test_data
from dataLogic.preprocessor import DataPreprocessor
from models.rsf_model import RSFModel
from utils.evaluation import evaluate_model
from config import TEST_SIZE, RANDOM_STATE


def train_survival_model():
    """Main function to train and evaluate the survival model."""

    # Load data
    df = load_data()

    # Preprocess data
    preprocessor = DataPreprocessor()
    X, y, df_processed = preprocessor.preprocess_data(df)

    # Save training columns
    save_training_columns(preprocessor.training_columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Save test data
    test_indices = X_test.index
    test_data_original = df_processed.loc[test_indices].copy()
    save_test_data(test_data_original)

    # Train model
    model = RSFModel()
    model.train(X_train, y_train)

    # Evaluate model
    evaluation_results = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save model
    model.save()

    return model, evaluation_results


if __name__ == "__main__":
    trained_model, results = train_survival_model()