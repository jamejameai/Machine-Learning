# CIIS School Intelligence AI

CIIS School Intelligence AI is an intelligent extension for the CIIS School Management System. It transforms student attendance, assessment performance, teacher assignments, and enrollment records into actionable insights, explainable early-warning signals, operational forecasts, and prioritized school recommendations.

---

## Architecture Overview

```
CIIS School Management (PHP / CodeIgniter 3)
                   │  Internal Authenticated API
                   ▼
       CIIS School Intelligence AI (FastAPI)
 ┌─────────────────┬─────────────────┬─────────────────┐
 │   ML Pipeline   │ Analytics Engine│ Recommendation  │
 │ • Feature Eng.  │ • Subjects      │ • Prioritized   │
 │ • Logistic Reg. │ • Workload      │ • Explainable   │
 │ • Random Forest │ • Forecasts     │ • Action Tracks │
 └─────────────────┴─────────────────┴─────────────────┘
                   │
           SQLite / MySQL DB
    (ciis_ai_* additive schema)
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone or navigate to the directory
cd /Users/macbook/Machine-Learning

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Demo / Sample School Data

```bash
python main.py --demo-data
```
Generates 300 student records, 30 teachers, 160 classes, 34,080 attendance records, and 17,040 assessment grades with 100% data quality validation.

### 3. Train Machine Learning Models

```bash
python main.py --train
```
Runs the full training pipeline:
- Feature extraction with leakage protection
- 5-fold Stratified Cross Validation
- Baseline: Logistic Regression
- Advanced: Random Forest Classifier
- Model comparison & acceptance criteria verification (F1 ≥ 0.70)
- Serializes versioned artifacts to `./models/`

### 4. Run the Intelligence Service & Web Dashboard

```bash
python main.py
```
- Web Dashboard: [http://localhost:8000/](http://localhost:8000/)
- Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## Running Automated Tests

```bash
source venv/bin/activate
pytest tests/ -v
```
All **71 test cases** pass, covering:
1. Synthetic school data generation & quality verification
2. Feature engineering & leakage prevention
3. Risk label definition & class balance
4. Baseline Logistic Regression & Random Forest training
5. Model versioning, serialization & artifact loading
6. Explainable inference & supportive language checks
7. Class & subject performance analytics
8. Teacher workload & ethical non-ranking analytics
9. Operational forecasting & assumptions disclosure
10. Prioritized recommendation engine & deduplication
11. REST API endpoints & error handling
12. Additive database schema (`ciis_ai_*`) & migrations

---

## API Endpoints

| Category | Endpoint | Method | Description |
|---|---|---|---|
| **Health** | `/health`, `/api/v1/health` | GET | Readiness & model status |
| **Data** | `/api/v1/data/summary` | GET | Source data metrics & tables |
| **Data** | `/api/v1/data/quality` | GET | Quality audit & completeness scores |
| **Predictions** | `/api/v1/students/predictions` | GET | Early warning student risk list |
| **Predictions** | `/api/v1/students/predictions/{id}` | GET | Student risk breakdown & contributing factors |
| **Class Analytics** | `/api/v1/classes/analytics/subjects` | GET | Subject performance & pass rates |
| **Class Analytics** | `/api/v1/classes/analytics/struggling`| GET | Identified learning gap subjects |
| **Teacher Support**| `/api/v1/teachers/support/workload` | GET | Teacher student ratios & sections |
| **Teacher Support**| `/api/v1/teachers/support/outcomes` | GET | Contextualized subject support patterns |
| **Forecasting** | `/api/v1/school/forecasts` | GET | Enrollment & retention projections |
| **Recommendations**| `/api/v1/recommendations` | GET | Prioritized actionable improvement items |
| **Recommendations**| `/api/v1/recommendations/{id}` | PATCH | Update recommendation status & outcomes |
| **Models** | `/api/v1/models/status` | GET | Active models, F1 scores & metrics |
| **Dashboard** | `/api/v1/dashboard/overview` | GET | Executive overview metrics |
| **UI** | `/` | GET | President's Intelligence Dashboard |

---

## Security & Ethics Safeguards

1. **Least-Privilege & Pseudonymous IDs**: Inference uses internal IDs; full student names and identities are never leaked into logs or model weights.
2. **Supportive Language Guarantee**: Predictions label students as *"Needs Academic Support"* or *"On Track"*. Derogatory labels (`failing`, `loser`, `bad student`) are strictly prevented.
3. **No Automated High-Stakes Decisions**: AI predictions inform advisory staff; they never automatically alter grades, sanctions, or enrollment statuses.
4. **Ethical Teacher Analytics**: Explicit disclaimer and architectural guardrail preventing automated ranking of teachers by raw student test scores.
5. **No Data Leakage**: Temporal and stratified validation splits ensure future outcomes never leak into training features.
