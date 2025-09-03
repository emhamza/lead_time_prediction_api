from sksurv.ensemble import RandomSurvivalForest
import joblib

class RSFModel:
    def __init__(self):
        """Initializes the Random Survival Forest model."""
        # Check for correct initialization
        print("Initializing Random Survival Forest model...")
        self.model = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
        print("Random Survival Forest model initialized.")

    def train(self, X_train, y_train):
        """Train the model."""
        print("Training Random Survival Forest Model...")
        # Add a check to confirm the model object is what we expect before training
        if not isinstance(self.model, RandomSurvivalForest):
            print("Error: self.model is not a RandomSurvivalForest instance.")
            return self

        try:
            self.model.fit(X_train, y_train)
            print("Random Survival Forest model training complete.")
            # Verify the attribute immediately after fitting
            if hasattr(self.model, "event_times_"):
                print("Model successfully trained. 'event_times_' attribute found.")
            else:
                print("Warning: Model trained but 'event_times_' attribute is missing.")
        except Exception as e:
            print(f"An exception occurred during model training: {e}")
            # Re-raise the exception to provide a full traceback
            raise

        return self

    def save(self, path="trained_model.joblib"):
        """Saves the trained model."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path="trained_model.joblib"):
        """Loads a trained model."""
        model_instance = RSFModel()
        model_instance.model = joblib.load(path)
        return model_instance