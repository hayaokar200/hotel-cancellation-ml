from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


@pytest.fixture
def valid_booking():
    """A valid request payload for the /predict endpoint."""
    return {
        "booking": {
            "hotel": "City Hotel",
            "lead_time": 30,
            "arrival_date_year": 2017,
            "arrival_date_month": "August",
            "arrival_date_week_number": 32,
            "arrival_date_day_of_month": 7,
            "stays_in_weekend_nights": 2,
            "stays_in_week_nights": 3,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "adr": 120.50,
        }
    }


def test_valid_booking_returns_probability(valid_booking):
    payload = deepcopy(valid_booking)
    response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text
    probability = response.json()["cancellation_probability"]
    assert isinstance(probability, (int, float))
    assert 0 <= probability <= 1


def test_wrong_lead_time_type_returns_422(valid_booking):
    payload = deepcopy(valid_booking)
    payload["booking"]["lead_time"] = "soon"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_zero_total_nights_returns_422(valid_booking):
    payload = deepcopy(valid_booking)
    payload["booking"]["stays_in_weekend_nights"] = 0
    payload["booking"]["stays_in_week_nights"] = 0

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
