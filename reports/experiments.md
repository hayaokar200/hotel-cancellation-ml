## Experiment - 2026-11-08 16:33

- **Model:** rf
- **Parameters:** `{'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1}`
- **Threshold:** `0.4`
- **threshold:** 0.4000
- **precision:** 0.6461
- **recall:** 0.8471
- **f1:** 0.7331

---

## Experiment - 2026-11-08 16:35

- **Model:** lgb
- **Parameters:** `{'random_state': 42}`
- **Threshold:** `0.4`
- **threshold:** 0.4000
- **precision:** 0.6558
- **recall:** 0.8510
- **f1:** 0.7408

---

Decision: LightGBM — higher val F1 (0.741 vs 0.733), with both precision and recall up, using default params. The gap is small and from one validation window, so it’s not a definitive win; seed/time-window check is a possible follow-up.

## Experiment - 2026-11-08 16:45

- **Model:** lgb
- **Parameters:** `{'random_state': 42}`
- **Threshold:** `0.4`
- **threshold:** 0.4000
- **precision:** 0.6558
- **recall:** 0.8510
- **f1:** 0.7408

---

Final test result: F1 0.611 (precision 51%, recall 76%)

## Experiment - 2026-11-09 11:45

- **Model:** lgb
- **Parameters:** `{'random_state': 42}`
- **threshold:** 0.4000
- **precision:** 0.6558
- **recall:** 0.8510
- **f1:** 0.7408
- **roc_auc:** 0.8844
- **average_precision:** 0.8573

---
