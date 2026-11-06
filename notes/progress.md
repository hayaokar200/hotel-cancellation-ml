# Learning Progress & Tutor Handover

> **For Claude (new session):** Read this file first and continue as my tutor.
> Give me ONE concrete task per session (what to do, what "done" looks like),
> check my understanding with 2–3 questions afterwards, and review my code.
> Keep it short — never dump the whole plan on me. Update this file at the
> end of each session. Also read `notes/week1.md` (my answers) and the code in `src/`.
> When I'm stuck, give hints and questions first; write worked examples only if I ask.

## Goal
- Software engineer (3 yrs) with a university AI background → **AI Engineer** role
  (LLM apps, RAG, agents) within 12 weeks.
- Build GitHub portfolio projects along the way; one LinkedIn post every Friday.
- Basics revision is capped at ~2 weeks, then go deep.

## Where things live
| Where | What |
|---|---|
| `notes/weekN.md` | My learning notes and answers (one file per week) |
| `notes/progress.md` | This handover: status and next steps (Claude updates it at the end of each session) |
| Google Sheet "AI Engineer 12-Week Tracker" | Daily schedule + Status, LinkedIn calendar, Job tracker (I update it) |

## Weekly rhythm (~12h/week)
| Day | Time | Session |
|---|---|---|
| Mon | 1.5h | Learn (concepts + notes + 3 self-check questions) |
| Wed | 1.5h | Practice (small focused experiment) |
| Fri | 1h | Reflect + LinkedIn post |
| Sat | 4h | Build |
| Sun | 4h | Test, document, GitHub |

Dates are guides. Unfinished tasks carry forward, and work/life comes first.

## 12-week roadmap (adjusted from the original spreadsheet)
| Wk | Starts | Topic | Milestone |
|---|---|---|---|
| 1 | 2026-11-02 | ML revision | Hotel cancellation project cleaned up, reproducible, honest README |
| 2 | 2026-11-09 | DL refresh | Rebuild logistic regression in PyTorch → small NN; compare with baseline |
| 3 | 2026-11-16 | Transformers & LLM basics | Tokenization/embeddings notebook |
| 4 | 2026-11-23 | LLM app engineering | Structured-output FastAPI (typed schema, validation, errors) |
| 5 | 2026-11-30 | RAG basics | Capstone v1: ingest + ask + cite. **Deploy a rough v1 early** |
| 6 | 2026-12-07 | RAG retrieval quality | Eval harness: 30+ labeled questions, compare 2 setups |
| 7 | 2026-12-14 | RAG reliability | Failure/injection tests, tracing. **Start applying (~2/week)** |
| 8 | 2026-12-21 | Agents (LangGraph) | Booking ops assistant, read-only tools. *Holiday week: lighter / buffer* |
| 9 | 2026-12-28 | MCP & safe actions | Approval workflow for write tools. *Holiday week: lighter / buffer* |
| 10 | 2027-01-04 | Cloud (pick ONE) | Deploy capstone to the chosen cloud with budget guardrails |
| 11 | 2027-01-11 | Production polish | CI, logging, latency/cost numbers |
| 12 | 2027-01-18 | Capstone refinement | Improve the weakest part, demo, docs |

Plan changes agreed with my tutor:
- Keep my software skills short (FastAPI, Docker, tests); spend the saved time on RAG eval and agents.
- Deploy early (week 5–6) and keep redeploying; use one cloud, not AWS and Azure.
- Start job applications around week 7.

## Current status: Week 1 (ML revision)

### Done
- [x] Mon/Wed: ML revision through the hotel cancellation notebook
- [x] Reviewed the project with my tutor (see "Project review" below)
- [x] `ml-env/` added to `.gitignore`; notebook snapshot committed
- [x] Split `requirements.txt` (runtime, pinned) and `requirements-dev.txt` (notebook/test tools)
- [x] Pinned versions to match `ml-env`; Dockerfile now uses `python:3.14-slim`
- [x] Fri 2026-11-06: model rebuilt with scikit-learn 1.9.1; API works (`/predic` returns a probability)
- [x] Fri: tracker redesigned and moved to Google Sheets (I update Status there daily)
- [x] Fri: answered the 5 questions in `notes/week1.md`, reviewed with my tutor

### Week-1 notes review (Fri)
- Solid: leakage (Q1), keeping the test set for the final measurement (Q3).
- Fixed: precision 0.48 means **48%**, not 0.48%.
- Q2 and Q5 rewritten and all 5 follow-ups answered: **week-1 concepts understood → ready for DL.**
- Small things to keep sharp (quick check on Monday):
  - Q2: the 0.76 → 0.60 drop has a second cause, **drift** (data changed over time). Tell the manager **0.60**.
  - Q3: picking the threshold on test = fitting a decision to those exact rows, so the score is optimistic.
  - Q5: at prediction time the Pipeline **transforms** with the statistics learned in `fit`. It does **not** refit.
    This avoids training/serving skew.

### Fri 2026-11-06 (evening): skeleton started — **RESUME EXACTLY HERE**
How we work on the skeleton: I write it myself; Claude gives hints, not code, then reviews.
Skeleton = function names + type hints + docstring (with notebook cell number) + `...` body.
Exception: `train.main()` contains real calls.

Status of `src/hotel_cancel/`:
| File | Status |
|---|---|
| `__init__.py` | ✅ empty, done |
| `config.py` | ✅ written by Claude as a worked example. **TODO (me):** write why `required_car_parking_spaces` and `total_of_special_requests` were dropped |
| `data.py` | ✅ written by Claude as a worked example: `load_raw → drop_unused_columns → add_dates → temporal_split → split_xy` |
| `features.py` | ✅ mine, reviewed: `get_numerical_features`, `get_categorical_features`, `build_preprocessor(X)` (pass **X_train**) |
| `evaluate.py` | 🟡 mine: `get_eval_metrics(y_true, y_prob, threshold)`, `get_best_threshold(y_true, y_prob, thresholds=config.THRESHOLD_CANDIDATES)`. **Leftover:** docstring of `get_best_threshold` still says `getEvalMetrics` + typos; add "on validation data"; spacing in its signature |
| `train.py` | 🟡 **in progress (mine)** — `main()` steps 1–5 written |

`train.py` review (to fix first thing):
- **Bug:** `evaluate.build_preprocessor(X_train)` → must be `features.build_preprocessor(X_train)`.
- Typo `preprocesor`; comment style `# Step 1: ...`; step numbering differs from the hint list (fine).

`train.py` — what's left (hints were given, I try alone):
6. Build the full model = preprocessor + classifier (`config.RF_PARAMS`). Maybe a helper `build_model(...)`.
7. Fit on train.
8. Probabilities on **validation** → `evaluate.get_best_threshold`.
9. Probabilities on **test** → `evaluate.get_eval_metrics` **once**.
10. Save the model: the dict keys must match `api/main.py` (`model`, `threshold`, `features`).
    Create `models/` first if missing (see notebook cell 48).
11. Save `reports/metrics.json` (decide: test metrics + validation F1 + chosen threshold?). Create `reports/` first.
- Bonus question to think about: retrain on train + validation before the final test? Trade-off?
- Not answered yet: none — just continue writing steps 6–11.

After the skeleton is reviewed:
- Set up `pyproject.toml` + `pip install -e .` (so `python -m hotel_cancel.train` works with the `src/` layout).
  Until then, imports work with `PYTHONPATH=src`.
- Fill in the function bodies by copying logic from the notebook; run `train.py`; compare
  metrics with the notebook (val F1 ≈ 0.733, test F1 ≈ 0.60 at threshold 0.4).

### Next (in order)
1. **Sat (build, 4h)**: finish `train.py` skeleton → review → `pyproject.toml` → fill in bodies →
   `train.py` rebuilds the model with one command; then add a LightGBM/XGBoost experiment;
   log every experiment in `reports/experiments.md`.
2. **Before publishing LinkedIn**: fix Q4 in `week1.md` (48%, not 0.48%). The post draft is ready
   (Claude gave an example); **publish only when the repo is clean** (Sunday if ready, otherwise later).
3. **Sun (validate, 4h)**: typed Pydantic request + fix `/predic` → `/predict`; add ROC-AUC/PR-AUC;
   compare train vs validation F1 (overfitting); read about `TimeSeriesSplit`; 2–3 tests; README with results + limitations.

### Target structure
```
data/                 # gitignored; README explains the Kaggle download
notebooks/01_eda.ipynb, 02_experiments.ipynb
src/hotel_cancel/     # config.py, data.py, features.py, train.py, evaluate.py
api/main.py
models/               # gitignored
reports/metrics.json, experiments.md
tests/
notes/                # this file + weekly notes
```
Workflow rules: freeze the split and metric first; change one thing per experiment;
score on validation; log it; look at the test set once at the end.

## Project review (hotel cancellation)
Strengths: time-based split built from `booking_date`; found that the random split
overstated performance (F1 0.76 → 0.60); drift analysis; written feature-availability assumptions.

Open issues:
- Test set was evaluated several times (cells 17, 23, 24, 29, 41).
- Part of the "drift" may come from where the dataset ends (arrivals end 2017-08-31, so
  the latest bookings can only have short lead times). Document it as a limitation.
- Question `arrival_date_year` (doesn't help with unseen future years) and `country`
  (reportedly unreliable for cancelled bookings; check the cancellation rate per country).
- Threshold 0.4 was chosen on validation (40% cancellations) but the test set has 30%.

## Week-1 self-check (ready for DL if I can answer most)
- Data leakage: why drop post-booking features?
- Random vs time-based split: why the F1 drop?
- Why choose the threshold on validation, not test?
- Precision 0.48 on test: what does that mean for the hotel?
- Why fit the imputer and scaler inside the Pipeline?
- Is the drift real, or a result of where the dataset ends?
- What does logistic regression output, and what loss does it minimize?
- When would I report PR-AUC instead of accuracy?

## Lessons learned
- **Pin whatever the model artifact depends on** (scikit-learn, pandas, numpy, joblib).
  A pickle saved by one scikit-learn version may not load with another (I saw this error myself).
  Notebook-only tools can stay unpinned.
- **Don't commit model binaries to git.** Commit `train.py` + metrics; share the
  model through a GitHub Release or Hugging Face Hub if needed.

## Setting up on a new PC
```bash
git clone https://github.com/hayaokar200/hotel-cancellation-ml.git
cd hotel-cancellation-ml
python3.14 -m venv ml-env && source ml-env/bin/activate
pip install -r requirements-dev.txt
# copy the dataset to data/hotel_bookings.csv (not in git; Kaggle "hotel booking demand")
```
