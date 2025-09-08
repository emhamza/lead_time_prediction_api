# Survival Analysis API

This project implements a **Survival Analysis API** for training
vendor-specific machine learning models.\
Currently, the focus is on the `/train/{vendor_id}` endpoint, which
allows you to train a Random Survival Forest (RSF) model for a given
vendor.

------------------------------------------------------------------------

## 🚀 Getting Started

### 1. Install dependencies

Make sure you have a working Python environment. Install dependencies
using:

``` bash
pip install -r requirements.txt
```

### 2. Run the API

Start the FastAPI application with:

``` bash
uvicorn main:app --reload
```

The API will be available at:

-   Swagger Docs: <http://127.0.0.1:8000/docs>
-   Root Endpoint: <http://127.0.0.1:8000/>

------------------------------------------------------------------------

## 📌 Current Endpoint

### Train Model

`POST /api/v1/train/{vendor_id}`

Trains a Random Survival Forest (RSF) model for the given vendor ID.

#### Example Request

``` http
POST /api/v1/train/VENDOR_A
```

#### Example Response

``` json
{
  "status": "success",
  "vendor_id": "VENDOR_A",
  "rows_used": 320,
  "model_path": "vendorModels/VENDOR_A.joblib",
  "training_columns_path": "vendorModels/VENDOR_A_columns.joblib"
}
```

------------------------------------------------------------------------
## 📊 Workflow

1.  Dataset with `vendor_id` column is prepared.\
2.  `/train/{vendor_id}` endpoint is called.\
3.  Data is filtered for the vendor.\
4.  Preprocessing is applied.\
5.  RSF model is trained & saved.\
6.  Training columns are saved for future inference.

------------------------------------------------------------------------

## ⚖️ License

This project is for internal use and prototyping purposes.
