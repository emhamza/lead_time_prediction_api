import requests
import pandas as pd
import json
from datetime import datetime

# API endpoint
url = "http://localhost:8001/api/v1/predict"


def load_test_data(file_path='test.csv'):
    """Load test data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} records from {file_path}")
        return df
    except FileNotFoundError:
        print(f"❌ File {file_path} not found")
        return None
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None


def prepare_payload_from_row(row):
    """Convert a DataFrame row to API payload format"""
    payload = {
        "Order_Quantity": float(row.get('Order_Quantity', 0)),
        "Order_Volume": float(row.get('Order_Volume', 0)),
        "Order_Weight": float(row.get('Order_Weight', 0)),
        "Priority_Flag": str(row.get('Priority_Flag', '')),
        "Fulfiller_ID": str(row.get('Fulfiller_ID', '')),
        "Routing_Lane_ID": str(row.get('Routing_Lane_ID', '')),
        "Fulfiller_Throughput": float(row.get('Fulfiller_Throughput', 0)),
        "Total_Backlog_Ack": float(row.get('Total_Backlog_Ack', 0)),
        "Current_Backlog": float(row.get('Current_Backlog', 0)),
        "Relative_Queue_Position": float(row.get('Relative_Queue_Position', 0)),
        "Estimated_Processing_Rate": float(row.get('Estimated_Processing_Rate', 0)),
        "Days_in_Queue": float(row.get('Days_in_Queue', 0)),
        "Day_of_Week": int(row.get('Day_of_Week', 1)),
        "Day_of_Month": int(row.get('Day_of_Month', 1)),
        "Month": int(row.get('Month', 1)),
        "Season": str(row.get('Season', '')),
        "Peak_Season": bool(row.get('Peak_Season', False)),
        "Demand_Surge": bool(row.get('Demand_Surge', False)),
        "Recent_Shipments": float(row.get('Recent_Shipments', 0)),
        "Lead_Time_Trend": float(row.get('Lead_Time_Trend', 0)),
        "Geography": str(row.get('Geography', '')),
        "Carrier": str(row.get('Carrier', '')),
        "Product_Category": str(row.get('Product_Category', '')),
        "Order_Creation_DateTime": str(row.get('Order_Creation_DateTime', '2024-01-01T00:00:00')),
        "Acknowledgement_DateTime": str(row.get('Acknowledgement_DateTime', '2024-01-01T00:00:00'))
    }
    return payload


def make_prediction(payload):
    """Make prediction request to API"""
    try:
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"HTTP Error {response.status_code}: {response.text}"

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Make sure the server is running!"
    except requests.exceptions.Timeout:
        return None, "Request timed out. The server might be busy."
    except Exception as e:
        return None, f"Unexpected error: {e}"


def main():
    # Load test data
    test_df = load_test_data('test.csv')
    if test_df is None:
        return

    print(f"\n📊 Found {len(test_df)} test records")

    # Ask user how many records to process
    try:
        num_records = int(input("How many records do you want to process? (Enter 0 for all): "))
        if num_records == 0 or num_records > len(test_df):
            num_records = len(test_df)
    except:
        num_records = min(5, len(test_df))  # Default to 5 records

    print(f"🔧 Processing {num_records} records...\n")

    successful_predictions = 0
    failed_predictions = 0

    # Process each record
    for i, (index, row) in enumerate(test_df.head(num_records).iterrows()):
        print(f"📦 Processing record {i + 1}/{num_records} (Index: {index})")

        # Prepare payload
        payload = prepare_payload_from_row(row)

        # Make prediction
        result, error = make_prediction(payload)

        if result and result.get('success', False):
            successful_predictions += 1
            print("✅ Prediction Successful!")
            print(f"   P50: {result['percentiles']['p50']:.2f} days")
            print(f"   P90: {result['percentiles']['p90']:.2f} days")
            print(f"   Event Probability: {result['event_probability']:.2%}")
            print(f"   Risk Score: {result['risk_score']:.2f}")
        else:
            failed_predictions += 1
            print(f"❌ Prediction Failed: {error}")

        print("-" * 50)

    # Summary
    print(f"\n📈 Summary:")
    print(f"✅ Successful predictions: {successful_predictions}")
    print(f"❌ Failed predictions: {failed_predictions}")
    print(f"📊 Success rate: {(successful_predictions / num_records * 100):.1f}%")


def batch_predict():
    """Alternative: Use batch prediction endpoint if available"""
    test_df = load_test_data('test.csv')
    if test_df is None:
        return

    # Prepare batch payload
    batch_payload = []
    for _, row in test_df.head(10).iterrows():  # Limit to 10 for demo
        batch_payload.append(prepare_payload_from_row(row))

    batch_url = "http://localhost:8001/api/v1/predict/batch"

    try:
        response = requests.post(batch_url, json={"requests": batch_payload}, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful!")
            print(f"📊 Processed {result['total_processed']} records")
            print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")

            for i, prediction in enumerate(result['predictions']):
                if prediction['success']:
                    print(f"\nRecord {i + 1}:")
                    print(f"  P50: {prediction['percentiles']['p50']:.2f} days")
                    print(f"  Event Probability: {prediction['event_probability']:.2%}")
                else:
                    print(f"\nRecord {i + 1}: ❌ Failed - {prediction['message']}")

        else:
            print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Batch prediction error: {e}")


if __name__ == "__main__":
    print("🚀 Survival Analysis API Tester")
    print("=" * 50)

    # Choose mode
    print("1. Single record predictions")
    print("2. Batch predictions")

    try:
        choice = int(input("Choose mode (1 or 2): "))
        if choice == 2:
            batch_predict()
        else:
            main()
    except:
        print("⚠️  Invalid choice, defaulting to single record mode")
        main()