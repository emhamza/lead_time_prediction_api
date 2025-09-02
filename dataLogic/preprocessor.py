import pandas as pd
from sksurv.util import Surv
from config import FEATURES, CUTOFF_DATE


class DataPreprocessor:
    """Handles data preprocessing for survival analysis."""

    def __init__(self):
        self.training_columns = None

    def preprocess_data(self, df):
        """Preprocess the data for survival analysis."""
        # Drop duplicates
        df = df.drop_duplicates()

        # Convert datetime columns
        df = self._process_datetime_features(df)

        # Create derived features
        df = self._create_derived_features(df)

        # Create survival target
        y = self._create_survival_target(df)

        # Prepare features
        X = self._prepare_features(df)

        return X, y, df

    def _process_datetime_features(self, df):
        """Process datetime features."""
        df['Order_Creation_DateTime'] = pd.to_datetime(df['Order_Creation_DateTime'])
        df['Acknowledgement_DateTime'] = pd.to_datetime(df['Acknowledgement_DateTime'])

        # Extract date components
        df['Order_Creation_Day'] = df['Order_Creation_DateTime'].dt.day
        df['Order_Creation_Month'] = df['Order_Creation_DateTime'].dt.month
        df['Order_Creation_Year'] = df['Order_Creation_DateTime'].dt.year

        df['Acknowledgement_Day'] = df['Acknowledgement_DateTime'].dt.day
        df['Acknowledgement_Month'] = df['Acknowledgement_DateTime'].dt.month
        df['Acknowledgement_Year'] = df['Acknowledgement_DateTime'].dt.year

        return df

    def _create_derived_features(self, df):
        """Create derived features."""
        # Time to acknowledge
        df['Time_to_Acknowledge'] = (
                df['Acknowledgement_DateTime'] - df['Order_Creation_DateTime']
        ).dt.days

        # Low lead time flag
        df['is_low_lead_time'] = (df['Lead_Time'] <= 4.5).astype(int)

        # Censoring time
        df['Censoring_Time'] = df['Lead_Time'].fillna(
            (CUTOFF_DATE - df['Order_Creation_DateTime']).dt.days
        )

        return df

    def _create_survival_target(self, df):
        """Create survival analysis target array."""
        event_mask = df['Lead_Time'].notna()
        observed_time = df['Censoring_Time']
        return Surv.from_arrays(event=event_mask, time=observed_time)

    def _prepare_features(self, df):
        """Prepare features for modeling."""
        X = df[FEATURES]
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Store training columns for consistency in prediction
        self.training_columns = X_encoded.columns.tolist()

        return X_encoded