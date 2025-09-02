import pandas as pd
import numpy as np
import joblib
from typing import List
import logging

logger = logging.getLogger(__name__)


class SurvivalPredictionService:
    def __init__(self, model_path: str, training_columns_path: str):
        self.model = self._load_model(model_path)
        self.training_columns = self._load_training_columns(training_columns_path)
        self._initialize_encoder()

    def _load_model(self, model_path: str):
        """Load the trained RSF model"""
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_training_columns(self, training_columns_path: str):
        """Load the training columns"""
        try:
            columns = joblib.load(training_columns_path)
            logger.info(f"Training columns loaded successfully")
            return columns
        except Exception as e:
            logger.error(f"Error loading training columns: {e}")
            raise

    def _initialize_encoder(self):
        """Initialize the one-hot encoder based on training columns"""
        # Extract categorical columns from training columns
        self.categorical_columns = [col for col in self.training_columns
                                    if any(x in col for x in ['_', '']) and col not in [
                                        'Order_Quantity', 'Order_Volume', 'Order_Weight',
                                        'Fulfiller_Throughput', 'Total_Backlog_Ack',
                                        'Current_Backlog', 'Relative_Queue_Position',
                                        'Estimated_Processing_Rate', 'Days_in_Queue',
                                        'Recent_Shipments', 'Lead_Time_Trend',
                                        'Time_to_Acknowledge', 'is_low_lead_time'
                                    ]]

    def _preprocess_request(self, request_data: dict) -> pd.DataFrame:
        """Preprocess a single prediction request"""
        # Create a copy of the request data
        data = request_data.copy()

        # Convert datetime strings to datetime objects if needed
        if isinstance(data['Order_Creation_DateTime'], str):
            data['Order_Creation_DateTime'] = pd.to_datetime(data['Order_Creation_DateTime'])
        if isinstance(data['Acknowledgement_DateTime'], str):
            data['Acknowledgement_DateTime'] = pd.to_datetime(data['Acknowledgement_DateTime'])

        # Create derived features (same as training)
        data['Order_Creation_Day'] = data['Order_Creation_DateTime'].day
        data['Order_Creation_Month'] = data['Order_Creation_DateTime'].month
        data['Order_Creation_Year'] = data['Order_Creation_DateTime'].year

        data['Acknowledgement_Day'] = data['Acknowledgement_DateTime'].day
        data['Acknowledgement_Month'] = data['Acknowledgement_DateTime'].month
        data['Acknowledgement_Year'] = data['Acknowledgement_DateTime'].year

        data['Time_to_Acknowledge'] = (data['Acknowledgement_DateTime'] - data['Order_Creation_DateTime']).days

        # For prediction, we don't have Lead_Time, so we can't create is_low_lead_time
        # We'll set it to a default value or skip if not in training columns
        data['is_low_lead_time'] = 0  # Default value

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Ensure all training columns are present
        for col in self.training_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reorder columns to match training
        df_encoded = df_encoded[self.training_columns]

        return df_encoded

    def predict_survival(self, request_data: dict) -> dict:
        """Predict survival for a single request"""
        try:
            # Preprocess the request
            features = self._preprocess_request(request_data)

            # Get survival function
            survival_func = self.model.predict_survival_function(features, return_array=True)

            # Get event times from the model
            event_times = self.model.event_times_

            # Calculate percentiles
            percentiles = self._calculate_percentiles(survival_func[0], event_times)

            # Create survival curve
            survival_curve = self._create_survival_curve(survival_func[0], event_times)

            # Calculate risk score (negative of mean survival time)
            risk_score = -percentiles['mean']

            # Probability of event (1 - survival probability at max time)
            event_probability = 1 - survival_func[0][-1]

            return {
                'percentiles': percentiles,
                'survival_curve': survival_curve,
                'risk_score': risk_score,
                'event_probability': event_probability,
                'success': True,
                'message': 'Prediction successful'
            }

        except Exception as e:
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
        # Find times where survival probability crosses percentiles
        p50_time = self._find_survival_time(survival_probs, event_times, 0.5)
        p90_time = self._find_survival_time(survival_probs, event_times, 0.1)

        # Calculate mean survival time (area under survival curve)
        mean_survival = np.trapz(survival_probs, event_times)

        return {
            'p50': float(p50_time),
            'p90': float(p90_time),
            'mean': float(mean_survival)
        }

    def _find_survival_time(self, survival_probs: np.array, event_times: np.array, threshold: float) -> float:
        """Find time when survival probability drops below threshold"""
        for i, prob in enumerate(survival_probs):
            if prob <= threshold:
                return event_times[i]
        return event_times[-1]  # Return max time if threshold not reached

    def _create_survival_curve(self, survival_probs: np.array, event_times: np.array) -> List[dict]:
        """Create survival curve data points"""
        # Sample points for the curve (every 10th point for efficiency)
        curve_points = []
        for i in range(0, len(event_times), max(1, len(event_times) // 100)):
            curve_points.append({
                'time': float(event_times[i]),
                'probability': float(survival_probs[i])
            })
        return curve_points