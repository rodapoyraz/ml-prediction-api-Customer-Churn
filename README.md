# Customer Churn Prediction API

## Overview

This project is an end-to-end machine learning system that predicts whether a customer is likely to churn based on historical data.

It goes beyond a typical notebook by integrating:

* data preprocessing
* model training and evaluation
* a production-ready prediction pipeline
* a FastAPI-based REST API

The system allows users to send customer data and receive real-time churn predictions.

---

## Problem Statement

Customer churn is a critical business problem, especially in subscription-based services.

The goal of this project is to:

* identify customers at risk of leaving
* enable proactive retention strategies
* balance predictive performance with business impact

---

## Model Approach

### Data Processing

* Converted `TotalCharges` from string to numeric
* Handled missing values introduced during conversion
* Removed non-informative features (e.g., `customerID`)
* Encoded target variable (`Churn`) as binary

### Feature Engineering

* Numerical features: scaled using `StandardScaler`
* Categorical features: encoded using `OneHotEncoder`
* Combined using `ColumnTransformer` and `Pipeline`

### Model

* Logistic Regression
* Improved using `class_weight="balanced"` to address class imbalance

---

## Model Performance

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | ~0.73 |
| Recall (Churn)    | ~0.80 |
| Precision (Churn) | ~0.49 |

### Interpretation

The model was optimized for **recall on churn**, meaning:

* It successfully identifies most customers who will churn
* It may produce more false positives, which is acceptable in many business contexts

In real-world scenarios, missing a churner is typically more costly than incorrectly flagging a non-churner.

---

## Project Architecture

```
Raw Data → Cleaning → Preprocessing Pipeline → Model → Saved Pipeline → API → Prediction
```

### Folder Structure

```
ml-prediction-api/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   └── train.py
│
├── models/
│   └── full_pipeline.pkl
│
├── app/
│   ├── app.py
│   └── schemas.py
│
├── tests/
│   └── test_api.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Tech Stack

* Python
* Pandas, NumPy
* scikit-learn
* FastAPI
* joblib
* pytest
* uv (for dependency management)

---

## 🔌 API Usage

### Start the API

```bash
uv run uvicorn app.app:app --reload
```

### Open API docs

```
http://127.0.0.1:8000/docs
```

---

### 🔹 Endpoint: `POST /predict`

#### Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 900.0
}
```

#### Response

```json
{
  "prediction": "will_churn",
  "probability": 0.8877
}
```

---

## Testing

Run tests using:

```bash
uv run pytest
```

The test suite verifies:

* API availability
* prediction endpoint functionality
* response structure

---

## Key Design Decisions

* **Pipeline-based preprocessing** ensures consistency between training and inference
* **Class imbalance handling** improves detection of churners
* **Full pipeline serialization** simplifies deployment and API integration
* **FastAPI** enables clean, scalable API development

---

## Future Improvements

* Compare models (Random Forest, XGBoost)
* Add model explainability (SHAP / feature importance)
* Containerize with Docker
* Deploy to cloud (Render / AWS)
* Add monitoring and logging

---

## Summary

This project demonstrates the ability to:

* build a complete ML pipeline from raw data to deployment
* make informed modeling decisions based on business context
* expose machine learning models as production-ready APIs

---