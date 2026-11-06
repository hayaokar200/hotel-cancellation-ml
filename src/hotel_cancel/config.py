"""All project settings in one place.

Rule: if a value appears in more than one place, or you might want to change it
for an experiment, it lives here. Other files import it:

    from hotel_cancel import config
    config.THRESHOLD
"""
from pathlib import Path

# ---------- Paths ----------
# This file is src/hotel_cancel/config.py, so the project root is two folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "hotel_cancellation_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

# ---------- Target ----------
TARGET = "is_canceled"

# ---------- Columns we never use as features ----------
# Not known when the booking is created (leakage) — notebook cell 12.
LEAKY_COLUMNS = [
    "reservation_status",       # literally says Canceled / Check-Out
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
]
# Dropped for other reasons — notebook cells 12, 20, 31.
OTHER_DROPPED_COLUMNS = [
    "company",                      # 94% missing
    "agent",                        # removing it didn't change F1 (cell 32)
    "required_car_parking_spaces",  # dropped in cell 20 (TODO: write down why)
    "total_of_special_requests",    # dropped in cell 20 (TODO: write down why)
]
# Helper columns we build for splitting, never fed to the model — cell 35.
DATE_COLUMNS = ["arrival_date", "booking_date"]

# ---------- Time-based split (by booking_date) — cell 34 ----------
TRAIN_FRACTION = 0.70   # oldest 70% of bookings
VAL_FRACTION = 0.15     # next 15%; the newest 15% is the test set

# ---------- Model — cell 38 ----------
RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ---------- Decision threshold — cell 40 ----------
# Chosen on VALIDATION only. 0.4 was the best F1 in the notebook.
THRESHOLD_CANDIDATES = [0.2, 0.3, 0.4, 0.5, 0.6]
