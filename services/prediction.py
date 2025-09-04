import pandas as pd
import numpy as np
import joblib
from typing import List
import logging
from models.rsf_model import RSFModel

logger = logging.getLogger(__name__)


class SurvivalPredictionService:
    def __init__(self, model_path: str, training_columns_path: str):
        print("[Init] Initializing SurvivalPredictionService...")
        self.model = self._load_model(model_path)
        self.training_columns = self._load_training_columns(training_columns_path)
        self._initialize_encoder()
        print("[Init] Initialization complete ✅")

    def _load_model(self, model_path: str):
        """Load the trained RSF model"""
        print(f"[Load Model] Attempting to load model from {model_path}")
        try:
            model = RSFModel.load(model_path)
            print("[Load Model] Model loaded successfully ✅")
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"[Load Model] ❌ Error: {e}")
            logger.error(f"Error loading model: {e}")
            raise

    def _load_training_columns(self, training_columns_path: str):
        """Load the training columns"""
        print(f"[Load Columns] Attempting to load training columns from {training_columns_path}")
        try:
            columns = joblib.load(training_columns_path)
            print(f"[Load Columns] Loaded {len(columns)} training columns ✅")
            logger.info("Training columns loaded successfully")
            return columns
        except Exception as e:
            print(f"[Load Columns] ❌ Error: {e}")
            logger.error(f"Error loading training columns: {e}")
            raise

    def _initialize_encoder(self):
        """Initialize encoder info"""
        print("[Encoder] Initializing categorical encoder...")
        self.categorical_columns = [
            col for col in self.training_columns
            if any(x in col for x in ['_', '']) and col not in [
                'Order_Quantity', 'Order_Volume', 'Order_Weight',
                'Fulfiller_Throughput', 'Total_Backlog_Ack',
                'Current_Backlog', 'Relative_Queue_Position',
                'Estimated_Processing_Rate', 'Days_in_Queue',
                'Recent_Shipments', 'Lead_Time_Trend',
                'Time_to_Acknowledge', 'is_low_lead_time'
            ]
        ]
        print(f"[Encoder] Found {len(self.categorical_columns)} categorical columns")
        print("[Encoder] Encoder initialized ✅")

    def _preprocess_request(self, request_data: dict) -> pd.DataFrame:
        """Preprocess a single prediction request"""
        print("[Preprocess] Starting preprocessing request data...")
        data = request_data.copy()

        # Convert datetime strings to datetime objects
        if isinstance(data['Order_Creation_DateTime'], str):
            data['Order_Creation_DateTime'] = pd.to_datetime(data['Order_Creation_DateTime'])
            print("[Preprocess] Converted Order_Creation_DateTime to datetime")
        if isinstance(data['Acknowledgement_DateTime'], str):
            data['Acknowledgement_DateTime'] = pd.to_datetime(data['Acknowledgement_DateTime'])
            print("[Preprocess] Converted Acknowledgement_DateTime to datetime")

        # Derived features
        data['Order_Creation_Day'] = data['Order_Creation_DateTime'].day
        data['Order_Creation_Month'] = data['Order_Creation_DateTime'].month
        data['Order_Creation_Year'] = data['Order_Creation_DateTime'].year

        data['Acknowledgement_Day'] = data['Acknowledgement_DateTime'].day
        data['Acknowledgement_Month'] = data['Acknowledgement_DateTime'].month
        data['Acknowledgement_Year'] = data['Acknowledgement_DateTime'].year

        data['Time_to_Acknowledge'] = (data['Acknowledgement_DateTime'] - data['Order_Creation_DateTime']).days
        data['is_low_lead_time'] = 0  # default

        print(f"[Preprocess] Derived features created -> Time_to_Acknowledge={data['Time_to_Acknowledge']}")

        # Convert to DataFrame
        df = pd.DataFrame([data])
        print(f"[Preprocess] Data converted to DataFrame with shape {df.shape}")

        # One-hot encode
        df_encoded = pd.get_dummies(df, drop_first=True)
        print(f"[Preprocess] One-hot encoded DataFrame shape: {df_encoded.shape}")

        # Add missing columns
        missing_columns = [col for col in self.training_columns if col not in df_encoded.columns]
        if missing_columns:
            print(f"[Preprocess] Adding {len(missing_columns)} missing columns")
            missing_data = pd.DataFrame(0, index=df_encoded.index, columns=missing_columns)
            df_encoded = pd.concat([df_encoded, missing_data], axis=1)

        # Reorder
        df_encoded = df_encoded[self.training_columns].copy()
        print(f"[Preprocess] Final encoded DataFrame shape: {df_encoded.shape}")
        print("[Preprocess] Preprocessing complete ✅")

        return df_encoded

    def predict_survival(self, request_data: dict) -> dict:
        """Predict survival for a single request"""
        print("[Predict] Starting prediction flow...")
        try:
            # Preprocess request
            features = self._preprocess_request(request_data)
            print("[Predict] Features prepared for prediction ✅")

            # Retrieve event times safely
            print("[Predict] Retrieving event times from underlying model...")
            if hasattr(self.model, "unique_times_"):
                event_times = self.model.unique_times_
                print(f"[Predict] Found unique_times_ with length {len(event_times)}")
            elif hasattr(self.model, "event_times_"):
                event_times = self.model.event_times_
                print(f"[Predict] Found event_times_ with length {len(event_times)}")
            else:
                raise ValueError("Model appears to be untrained. Neither unique_times_ nor event_times_ found.")

            # Predict survival function
            survival_func = self.model.predict_survival_function(features, return_array=True)
            print(f"[Predict] Survival function calculated, length={len(survival_func[0])}")

            # Calculate percentiles
            percentiles = self._calculate_percentiles(survival_func[0], event_times)
            print(
                f"[Predict] Percentiles calculated -> P50={percentiles['p50']}, P90={percentiles['p90']}, Mean={percentiles['mean']}")

            # Create survival curve
            survival_curve = self._create_survival_curve(survival_func[0], event_times)
            print(f"[Predict] Survival curve created with {len(survival_curve)} points")

            # Risk score
            risk_score = -percentiles['mean']
            print(f"[Predict] Risk score calculated -> {risk_score}")

            # Event probability
            event_probability = 1 - survival_func[0][-1]
            print(f"[Predict] Event probability calculated -> {event_probability}")

            print("[Predict] Prediction successful ✅")
            return {
                'percentiles': percentiles,
                'survival_curve': survival_curve,
                'risk_score': risk_score,
                'event_probability': event_probability,
                'success': True,
                'message': 'Prediction successful'
            }

        except Exception as e:
            print(f"[Predict] ❌ Prediction error: {e}")
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'message': f'Prediction failed: {str(e)}',
                'percentiles': {'p50': 0, 'p90': 0, 'mean': 0},
                'survival_curve': [],
                'risk_score': 0,
                'event_probability': 0
            }

    def _calculate_percentiles(self, survival_probs: np.array, event_times: np.array) -> dict:
        """Calculate survival time percentiles"""
        print("[Percentiles] Calculating percentiles...")
        p50_time = self._find_survival_time(survival_probs, event_times, 0.5)
        p90_time = self._find_survival_time(survival_probs, event_times, 0.1)
        mean_survival = np.trapz(survival_probs, event_times)
        print(f"[Percentiles] Done -> P50={p50_time}, P90={p90_time}, Mean={mean_survival}")
        return {'p50': float(p50_time), 'p90': float(p90_time), 'mean': float(mean_survival)}

    def _find_survival_time(self, survival_probs: np.array, event_times: np.array, threshold: float) -> float:
        """Find survival time crossing a threshold"""
        for i, prob in enumerate(survival_probs):
            if prob <= threshold:
                print(f"[Find Time] Threshold {threshold} crossed at index {i}, time={event_times[i]}")
                return event_times[i]
        print(f"[Find Time] Threshold {threshold} never crossed, returning max time={event_times[-1]}")
        return event_times[-1]

    def _create_survival_curve(self, survival_probs: np.array, event_times: np.array) -> List[dict]:
        """Create survival curve points"""
        print("[Curve] Creating survival curve points...")
        curve_points = []
        for i in range(0, len(event_times), max(1, len(event_times) // 100)):
            curve_points.append({'time': float(event_times[i]), 'probability': float(survival_probs[i])})
        print(f"[Curve] Created {len(curve_points)} points ✅")
        return curve_points
