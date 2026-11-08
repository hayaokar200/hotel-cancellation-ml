from fastapi import FastAPI
import joblib
from pathlib import Path
import pandas as pd
from typing import Any, Dict
from fastapi import HTTPException
from pydantic import BaseModel, model_validator
from datetime import date
import calendar

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
class BookingFields(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    deposit_type: str
    customer_type: str
    adr: float

    @model_validator(mode="after")
    def validate_booking(self):
        if self.adults < 1:
            raise ValueError("At least one adult is required")

        if self.children < 0 or self.babies < 0:
            raise ValueError("Children and babies cannot be negative")

        if self.lead_time < 0:
            raise ValueError("Lead time cannot be negative")

        if self.stays_in_weekend_nights + self.stays_in_week_nights < 1:
            raise ValueError("Stay must be at least one night")

        if self.is_repeated_guest not in (0, 1):
            raise ValueError("is_repeated_guest must be 0 or 1")

        month_names = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        if self.arrival_date_month not in month_names:
            raise ValueError("Invalid arrival month")



        return self

class BookingRequest(BaseModel):
    booking: BookingFields

@app.post("/predict")
def predict(request: BookingRequest):
    
    booking_data = request.booking.model_dump()

    booking_df = pd.DataFrame([booking_data])

    booking_df = booking_df[features]

    probability = float(model.predict_proba(booking_df)[0, 1])

    prediction = int(probability>=threshold)

    return {
        "cancellation_probability": round(probability,3),
        "prediction" : prediction,
        "threshold": threshold
    }
    