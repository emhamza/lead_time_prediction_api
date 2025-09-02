from sksurv.ensemble import RandomSurvivalForest
import joblib
from config import RSF_PARAMS, MODEL_FILE


class RSFModel:
    """Random Survival Forest model wrapper."""

    def __init__(self, params=None):
        self.params = params or RSF_PARAMS
        self.model = RandomSurvivalForest(**self.params)

    def train(self, X_train, y_train):
        """Train the model."""
        print("Training Random Survival Forest Model...")
        self.model.fit(X_train, y_train)
        print("Random Survival Forest model training complete.")
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def score(self, X, y):
        """Calculate concordance index."""
        return self.model.score(X, y)

    def save(self, file_path=MODEL_FILE):
        """Save the trained model."""
        joblib.dump(self.model, file_path)
        print(f"Model saved as {file_path}")

    @staticmethod
    def load(file_path=MODEL_FILE):
        """Load a trained model."""
        model = RSFModel()
        model.model = joblib.load(file_path)
        return model