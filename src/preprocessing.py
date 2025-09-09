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
        df['Order_Creation_DateTime'] = pd.to_datetime(df['Order_Creation_DateTime'])
        df['Acknowledgement_DateTime'] = pd.to_datetime(df['Acknowledgement_DateTime'])

        df['Order_Creation_Day'] = df['Order_Creation_DateTime'].dt.day
        df['Order_Creation_Month'] = df['Order_Creation_DateTime'].dt.month
        df['Order_Creation_Year'] = df['Order_Creation_DateTime'].dt.year

        df['Acknowledgement_Day'] = df['Acknowledgement_DateTime'].dt.day
        df['Acknowledgement_Month'] = df['Acknowledgement_DateTime'].dt.month
        df['Acknowledgement_Year'] = df['Acknowledgement_DateTime'].dt.year

        print("  ✅ Datetime processing complete.")
        return df

    def _create_derived_features(self, df):
        """Create derived features."""
        print("  ➡️  Creating derived features...")

        df['Time_to_Acknowledge'] = (
            df['Acknowledgement_DateTime'] - df['Order_Creation_DateTime']
        ).dt.days

        df['is_low_lead_time'] = (df['Lead_Time'] <= 4.5).astype(int)

        df['Censoring_Time'] = df['Lead_Time'].fillna(
            (CUTOFF_DATE - df['Order_Creation_DateTime']).dt.days
        )

        print("  ✅ Derived features created.")
        return df

    def _create_survival_target(self, df):
        """Create survival analysis target array (structured dtype)."""
        event_mask = df['Lead_Time'].notna()
        observed_time = df['Censoring_Time']

        y = Surv.from_arrays(event=event_mask.astype(bool),
                             time=observed_time.astype(float))

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
