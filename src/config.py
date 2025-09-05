# config.py
import pandas as pd
import os

# Base directory for the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "dataset_with_vendors.csv")


# Model parameters
RSF_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

# Data processing
CUTOFF_DATE = pd.Timestamp('2025-04-01')
TEST_SIZE = 0.2
RANDOM_STATE = 42


# Features list
FEATURES = [
    "Order_Quantity", "Order_Volume", "Order_Weight", "Priority_Flag",
    "Fulfiller_ID", "Routing_Lane_ID", "Fulfiller_Throughput", "Total_Backlog_Ack",
    "Current_Backlog", "Relative_Queue_Position", "Estimated_Processing_Rate",
    "Days_in_Queue", "Day_of_Week", "Day_of_Month", "Month", "Season",
    "Peak_Season", "Demand_Surge", "Recent_Shipments", "Lead_Time_Trend",
    "Geography", "Carrier", "Product_Category", "Order_Creation_Day",
    "Order_Creation_Month", "Order_Creation_Year", "Acknowledgement_Day",
    "Acknowledgement_Month", "Acknowledgement_Year", "Time_to_Acknowledge",
    "is_low_lead_time"
]
# Configuration (paths, constants)
# ...existing code...
