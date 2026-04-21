from pathlib import Path

from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

from app.schemas import CustomerData

app = FastAPI(title="Customer Churn Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "full_pipeline.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_df = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": "will_churn" if prediction == 1 else "will_not_churn",
            "probability": round(float(probability), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
