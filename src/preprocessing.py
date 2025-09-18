import pandas as pd
from src.config import FEATURES, CUTOFF_DATE
from sksurv.util import Surv
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    """Handles data preprocessing for survival analysis."""

    def __init__(self):
        self.training_columns = None
        self.label_encoders = {}

    def preprocess_data(self, df):
        """Preprocess the data for survival analysis."""

        df = df.drop_duplicates()

        df = self._process_datetime_features(df)

        df = self._create_derived_features(df)

        y = self._create_survival_target(df)

        X = self._prepare_features(df)
        print("  ✅ Data Preprocessing completed.")


        return X, y, df

    def _process_datetime_features(self, df):
        """Process datetime features."""
        print("  ➡️  Processing datetime columns...")
        df['Fulfiller_Acknowledgement_DateTime'] = pd.to_datetime(df['fulfiller_ack_date'])

        df['Fulfiller_Acknowledgement_Day'] = df['Fulfiller_Acknowledgement_DateTime'].dt.day
        df['Fulfiller_Acknowledgement_Month'] = df['Fulfiller_Acknowledgement_DateTime'].dt.month
        df['Fulfiller_Acknowledgement_Year'] = df['Fulfiller_Acknowledgement_DateTime'].dt.year

        print("  ✅ Datetime processing complete.")
        return df

    def _create_derived_features(self, df):
        """Create derived features."""
        print("  ➡️  Creating derived features...")

        df['is_delivered'] = (df['delivered'] > 0).astype(int)

        print("  ✅ Derived features created.")
        return df

    def _create_survival_target(self, df):
        """Create survival analysis target array (structured dtype)."""
        print("  ➡️  Creating survival target array...")

        # 'is_delivered' is the event, 1 for delivery, 0 for censored.
        event_mask = (df['is_delivered'] == 1).astype(bool)

        # 'lead_time_days' is the observed time to the event or censoring.
        observed_time = df['lead_time_days']

        y = Surv.from_arrays(event=event_mask, time=observed_time.astype(float))
        print("  ✅ Survival target array created.")
        return y

    def _prepare_features(self, df):
        """Prepare features for modeling using Label Encoding instead of One-Hot."""
        X = df[FEATURES].copy()

        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))  # fit + transform
            self.label_encoders[col] = le  # save encoder for later use

        self.training_columns = X.columns.tolist()

        print("  ✅ Features label encoded.")
        return X
