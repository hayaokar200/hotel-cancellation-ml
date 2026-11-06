# Week 1 — ML revision notes

> Answer in 3–5 sentences each, without looking anything up.
> "Not sure because…" is a valid answer.

## 1. Why did I drop `reservation_status` and `booking_changes`?

In order o prevent Data Leakage, reservation_status and booking_changes values are not avilable during the prediciton so it should not be used to train the model.

## 2. Why did F1 fall from 0.76 (random split) to 0.60 (time-based split)? Which number would I tell a hotel manager?

because when I was using the random split the model actually trained on future bookings, which was also tested on it so basically he was tested on data that the model already saw (almost same features)

## 3. Why must the threshold be chosen on validation, not test?

because the test data should never be seen by the model, it is used only on the last step to mesure the model performance

## 4. Test precision is 0.48. What does that mean for a hotel using the model?

that means that when the model predicts one booking as cancelled, 0.48% it's correct

## 5. Why are the imputer and scaler inside the Pipeline instead of applied to the whole dataset first?

because the model learns some values from the imputer and scaler, so if we applied on the full dataset, it would be trained on test and validation data which is not allowed

---

## Things I'm unsure about / want to revisit

- Q2 and Q5: rewrite in my own words after Friday's explanation

---

## Follow-up questions (from my tutor)

1. Q1: `reservation_status` is even worse than "not available yet". What are its values, and why does that make it almost the same as the target?
   Check-Out,Canceled,No-Show. it is worst beacause it is telling if the booking is cancalled or not which is the prediction value.
2. Q2: My random split had F1 0.76. Give one reason why it was too optimistic.
   it was trained on future data
3. Q3: If I tried 5 thresholds on the test set and kept the best, why would the reported score be too high?
   because we will be testing the model on data it knows
4. Q4: Test recall is 0.78. In one sentence, what does it mean for the hotel?
   out of all the actuall cancelled booking the model can predict 78% of them
5. Q5: My API sends a raw booking straight to `model.predict_proba`. Where does the scaling happen, and why is that good?
   the model fits the steps of the train preprocessing using the pipeline on the new data.
