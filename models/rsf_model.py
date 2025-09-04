# rsf_model.py
from sksurv.ensemble import RandomSurvivalForest
import joblib

class RSFModel:
    def __init__(self):
        """Initializes the Random Survival Forest model."""
        self.model = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X_train, y_train):
        """Train the underlying Random Survival Forest."""
        self.model.fit(X_train, y_train)
        print("[RSFModel.train] Training complete ✅")
        # if hasattr(self.model, "unique_times_"):
        #     print(f"[RSFModel.train] unique_times_ count: {len(self.model.unique_times_)}")
        # else:
        #     print("[RSFModel.train] ⚠️ Warning: unique_times_ missing after training")
        return self

    def save(self, path="trained_model.joblib", model_path="rsf_model.joblib"):
        """
        Save the wrapper and the underlying model separately.
        - path: path to save the RSFModel wrapper
        - model_path: path to save the underlying RandomSurvivalForest
        """
        if self.model is None:
            raise ValueError("Cannot save: underlying model is None")
        joblib.dump(self.model, model_path)

        # Save wrapper without the model to prevent double pickling issues
        wrapper_copy = RSFModel.__new__(RSFModel)
        wrapper_copy.model = None  # model will be restored on load
        joblib.dump(wrapper_copy, path)
        print(f"[RSFModel.save] Wrapper saved ✅")

    @staticmethod
    def load(path="trained_model.joblib", model_path="rsf_model.joblib"):
        """
        Load trained wrapper and restore underlying model.
        """
        wrapper = joblib.load(path)

        underlying_model = joblib.load(model_path)
        wrapper.model = underlying_model
        return wrapper

    def __getattr__(self, attr):
        """Delegate attribute access to underlying RandomSurvivalForest."""
        model = self.__dict__.get("model", None)
        if model is None:
            raise AttributeError(f"Underlying model is None. Cannot access '{attr}'")
        if hasattr(model, attr):
            return getattr(model, attr)
        raise AttributeError(f"'RSFModel' object has no attribute '{attr}'")
