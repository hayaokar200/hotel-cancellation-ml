from fastapi import FastAPI
import joblib
from pathlib import Path
import pandas as pd
from typing import Any, Dict
from fastapi import HTTPException
from pydantic import BaseModel

app = FastAPI(title="Hotel Cancellation Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "hotel_cancellation_model.joblib"

# Load the trained model
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

@app.get("/")
def home():
    return {
        "message": "Hotel Cancellation API is running"
    }

class BookingRequest(BaseModel):
    booking: Dict[str, Any]

@app.post("/predic")
def predict(request: BookingRequest):
    try:
        booking_df = pd.DataFrame([request.booking])

        missing = [
            feature for feature in features
            if feature not in booking_df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail={"missing_features":missing}
            )

        booking_df = booking_df[features]

        probability = float(model.predict_proba(booking_df)[0, 1])

        prediction = int(probability>=threshold)

        return {
            "cancellation_probability": round(probability,3),
            "prediction" : prediction,
            "threshold": threshold
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid booking data"
        )