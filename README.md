# Hotel Cancellation Prediction

Machine learning project for predicting hotel booking cancellations.

## Project contents

- `notebooks/01_data_exploration.ipynb`: exploratory analysis and model experimentation.
- `api/main.py`: FastAPI inference endpoint.
- `Dockerfile`: container configuration.
- `requirements.txt`: Python dependencies.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The original dataset (`data/hotel_bookings.csv`) and trained model (`models/hotel_cancellation_model.joblib`) are intentionally excluded from Git. Copy them from your local backup if you need to run the existing API.

## Run the API

After restoring the trained model:

```bash
uvicorn api.main:app --reload
```

Visit `/docs` on the running server for interactive API documentation. The current prediction endpoint is `POST /predic`.

## Status

Work in progress. Next steps: verify reproducible model training, evaluate classification metrics, and document model performance.
