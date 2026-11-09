# Hotel Cancellation Prediction

Predicts whether a hotel booking will be cancelled, using only information that is available
**at the moment the booking is made**. Built as an end-to-end project: reproducible training
pipeline, honest time-based evaluation, and a typed FastAPI service.

## Results

| Model | Val F1 | Test F1 | Test precision | Test recall |
|---|---|---|---|---|
| Random Forest | 0.733 | 0.596 | 48% | 78% |
| **LightGBM (chosen)** | **0.741** | **0.611** | **51%** | **76%** |

In plain words: when the model flags a booking as likely to cancel, it is right about half the
time, and it catches about 3 out of 4 real cancellations.

Decision threshold: **0.4**, chosen on the validation set. All experiments are logged in
[`reports/experiments.md`](reports/experiments.md).

## Approach

- **Time-based split.** Bookings are sorted by booking date (arrival date − lead time) and split
  70 / 15 / 15 into train, validation and test. The split happens at a date cutoff, so bookings
  from the same day never land on both sides. A random split gave a much higher F1 (0.76), but it
  lets the model train on bookings from the future, so it overstates real performance.
- **No leakage.** Columns that are only known after the booking is made (`reservation_status`,
  `reservation_status_date`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`)
  are dropped.
- **One pipeline.** Preprocessing (median imputation + scaling for numbers, one-hot encoding for
  categories) and the classifier are a single scikit-learn `Pipeline`. The preprocessing is fitted
  on training data only, and the API uses exactly the same transformations as training.
- **Test set used once.** Models were compared and the threshold was chosen on validation only.
  The model choice was written down before the final test run. Test evaluation is switched off by
  default (`EVALUATE_TEST` in `config.py`) so experiments can't accidentally use it.
- **Model as a setting.** The classifier is chosen in `config.py` (`MODEL_NAME = "rf"` or `"lgb"`),
  so a new experiment changes one thing and nothing else.

## Limitations

- **Drift.** Performance drops by about 0.13 F1 from validation to test for both models. The
  cancellation rate falls from about 40% in validation to about 30% in test, so the data itself
  changes over time.
- **Where the dataset ends.** Arrivals end on 2017-08-31, so the most recent bookings can only
  have short lead times. Part of the measured drift may come from this cut-off rather than from
  real behaviour changes.
- **Threshold.** The threshold was tuned on validation (40% cancellations); it may not be the best
  cutoff for the test period (30% cancellations). It is not re-tuned on test on purpose.
- **Features to question.** `country` is reportedly less reliable for cancelled bookings, and
  `arrival_date_year` does not generalize to future years.
- **Tests need a trained model.** The API loads `models/hotel_cancellation_model.joblib` on
  startup. That file is not in Git, so the tests only run after training locally.

## Project structure

```
src/hotel_cancel/
  config.py      # all settings: paths, dropped columns, split fractions, model params
  data.py        # load → drop columns → add dates → time-based split → X / y
  features.py    # preprocessing (ColumnTransformer)
  evaluate.py    # metrics and threshold selection
  train.py       # one command: rebuild the model, save it and the metrics
api/main.py      # FastAPI service with a typed request schema
tests/           # API tests (pytest)
reports/         # metrics.json, experiments.md
notebooks/       # exploration and original experiments
```

## How to run

**1. Set up** (Python 3.14):

```bash
git clone https://github.com/hayaokar200/hotel-cancellation-ml.git
cd hotel-cancellation-ml
python3.14 -m venv ml-env
source ml-env/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

**2. Get the data.** Download *Hotel booking demand* from Kaggle and save it as
`data/hotel_bookings.csv` (the data is not stored in Git).

**3. Train:**

```bash
python -m hotel_cancel.train
```

This saves the model to `models/hotel_cancellation_model.joblib`, writes `reports/metrics.json`
and appends the run to `reports/experiments.md`.

**4. Run the API:**

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and send a booking to `POST /predict`. The response contains the
cancellation probability, the 0/1 prediction and the threshold. Invalid input (wrong type,
missing field, impossible values such as zero nights) returns a `422` that names the problem field.

**5. Run the tests:**

```bash
pytest
```
