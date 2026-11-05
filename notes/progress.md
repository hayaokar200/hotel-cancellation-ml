# Learning Progress & Tutor Handover

> **For Claude (new session):** Read this file first and continue as my tutor.
> Give me ONE concrete task per session (what to do, what "done" looks like),
> check my understanding with 2–3 questions afterwards, and review my code.
> Keep it short — never dump the whole plan on me. Update this file at the
> end of each session.

## Goal
- Software engineer (3 yrs) with a university AI background → **AI Engineer** role
  (LLM apps, RAG, agents) within 12 weeks.
- Build GitHub portfolio projects along the way; one LinkedIn post every Friday.
- Basics revision is capped at ~2 weeks, then go deep.

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

### Next (in order)
1. **Rebuild the model**: run the notebook with the current environment (the old
   `.joblib` fails to load because it was made with an older scikit-learn:
   `_RemainderColsList` error). Then check the API with `uvicorn api.main:app --reload` → `/docs`.
2. **Skeleton**: create `src/hotel_cancel/` files with function names + one-line docstrings only.
3. **Notes** in `notes/week1.md` (no looking things up):
   - Why did I drop `reservation_status` and `booking_changes`?
   - Why did F1 fall from 0.76 (random split) to 0.60 (time-based split), and which number would I tell a hotel manager?
4. **Fri**: answer the week-1 self-check table + LinkedIn post about the 0.76 → 0.60 finding.
5. **Sat (build, 4h)**: move the code into `src/`; `train.py` rebuilds the model with one command;
   add a LightGBM/XGBoost experiment; log every experiment in `reports/experiments.md`.
6. **Sun (validate, 4h)**: typed Pydantic request + fix `/predic` → `/predict`; add ROC-AUC/PR-AUC;
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
