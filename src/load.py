import pandas as pd
import joblib
from src.config import DATA_FILE

def load_data(file_path=DATA_FILE):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise

def save_training_columns(columns, file_path='training_columns.joblib'):
    """Save training columns for later use."""
    joblib.dump(columns, file_path)
    print(f"Training columns saved.")

def load_training_columns(file_path='training_columns.joblib'):
    """Load training columns."""
    return joblib.load(file_path)

def save_test_data(test_data, file_path='test.csv'):
    """Save test data to CSV."""
    test_data.to_csv(file_path, index=False)
    print(f"Test data saved.")