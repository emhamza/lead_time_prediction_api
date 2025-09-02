import requests

# API endpoint
url = "http://localhost:8001/api/v1/predict"

# Sample request data
payload = {
    "Order_Quantity": 10.0,
    "Order_Volume": 5.5,
    "Order_Weight": 2.3,
    "Priority_Flag": "HIGH",
    "Fulfiller_ID": "F184",
    "Routing_Lane_ID": "LANE456",
    "Fulfiller_Throughput": 100.0,
    "Total_Backlog_Ack": 50.0,
    "Current_Backlog": 25.0,
    "Relative_Queue_Position": 0.3,
    "Estimated_Processing_Rate": 10.0,
    "Days_in_Queue": 2.0,
    "Day_of_Week": 3,
    "Day_of_Month": 15,
    "Month": 6,
    "Season": "Summer",
    "Peak_Season": False,
    "Demand_Surge": False,
    "Recent_Shipments": 100.0,
    "Lead_Time_Trend": 1.2,
    "Geography": "North America",
    "Carrier": "UPS",
    "Product_Category": "Electronics",
    "Order_Creation_DateTime": "2024-01-15T10:00:00",
    "Acknowledgement_DateTime": "2024-01-15T10:30:00"
}

try:
    # Make the request
    response = requests.post(url, json=payload)

    # Check if successful
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction Successful!")
        print(f"📊 P50 Survival Time: {result['percentiles']['p50']} days")
        print(f"📊 P90 Survival Time: {result['percentiles']['p90']} days")
        print(f"📊 Mean Survival Time: {result['percentiles']['mean']} days")
        print(f"⚠️  Event Probability: {result['event_probability']:.2%}")
        print(f"🔴 Risk Score: {result['risk_score']:.2f}")

        # Show first few survival curve points
        print("\n📈 Survival Curve (first 5 points):")
        for point in result['survival_curve'][:5]:
            print(f"   Time {point['time']}: {point['probability']:.2%} survival")

    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to the API. Make sure the server is running!")
except Exception as e:
    print(f"❌ Unexpected error: {e}")