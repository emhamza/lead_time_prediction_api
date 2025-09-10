# Lead Time Prediction API

This project implements a **Survival Analysis API** for training and predicting vendor-specific machine learning models using Random Survival Forest (RSF).

---

## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn main:app --reload
```

### Run MLFlow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

**API Access:**
- Swagger Docs: http://127.0.0.1:8000/docs
- Root Endpoint: http://127.0.0.1:8000/

---

## 📌 API Endpoints

### 1. Train Model
`POST /api/v1/train/{vendor_id}`

Trains an RSF model for the specified vendor.

**Example Request:**
```http
POST /api/v1/train/VENDOR_A
```

**Example Response:**
```json
{
  "status": "success",
  "vendor_id": "VENDOR_A", 
  "rows_used": 320,
  "model_path": "artifacts/v1/VENDOR_A.joblib"
}
```

### 2. Predict Survival
`POST /api/v1/pred/{vendor_id}`

Generates survival predictions for a trained vendor model.

**Example Request:**
```http
POST /api/v1/pred/VENDOR_A
```

**Example Response:**
```json
{
  "predictions": [
    {
      "PO_ID": "12345",
      "p50_survival_time": 45.2,
      "p90_survival_time": 78.5,
      "survival_curve": [...],
      "risk_score": 0.25
    }
  ],
  "summary": {
    "vendor_id": "VENDOR_A",
    "total_predictions": 150
  },
  "prediction_path": "predictions/v1/VENDOR_A.json"
}
```

---

## 📊 Workflow

1. **Train**: Call `/train/{vendor_id}` to create vendor-specific RSF model
2. **Predict**: Call `/pred/{vendor_id}` to generate survival predictions with p50/p90 times and curves
3. **Results**: Predictions saved to `predictions/v1/{vendor_id}.json`

---

## ⚖️ License

This project is for internal use and prototyping purposes.