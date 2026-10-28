# CIIS School Intelligence AI

CIIS School Intelligence AI is an intelligent extension for the CIIS School Management System. It transforms student attendance, assessment performance, teacher assignments, and enrollment records into actionable insights, explainable early-warning signals, operational forecasts, prioritized school recommendations, and teacher instructional improvement diagnostics.

---

## Architecture Overview

```
CIIS School Management (PHP / CodeIgniter 3)
                   │  Internal Authenticated API
                   ▼
       CIIS School Intelligence AI (FastAPI)
 ┌─────────────────┬─────────────────┬─────────────────┐
 │   ML Pipeline   │ Analytics Engine│ Recommendation  │
 │ • Student Risk  │ • Subjects      │ • Prioritized   │
 │ • Teacher ML    │ • Workload      │ • Explainable   │
 │ • Feature Eng.  │ • Forecasts     │ • Action Tracks │
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
Runs the full training pipeline for both:
1. **Student Academic Risk Models**: Logistic Regression baseline & Random Forest classifier (5-fold Stratified CV).
2. **Teacher Support & Instructional Improvement Models**: Evaluates instructional load and classroom variance patterns, selecting the best model (F1 ≥ 0.70, AUC 1.0) and serializing artifacts to `./models/`.

### 4. Run the Intelligence Service & Web Dashboard

```bash
python main.py
```
- Web Dashboard: [http://localhost:8000/](http://localhost:8000/)
- Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## Teacher Support & Instructional Improvement Machine Learning

The Teacher Support ML Engine learns from classroom dynamics and faculty workload patterns to recommend targeted instructional support, professional development, and workload rebalancing.

### How the Model Learns:
- **Multi-Prep Load**: Number of distinct subjects taught simultaneously by the teacher.
- **Score Dispersion**: Standard deviation of student scores within the teacher's sections (detects bimodal understanding or struggling subgroups).
- **Section Attendance**: Average student attendance rate in classes taught by the teacher.
- **Grade Trajectory**: Difference between recent exam scores vs. midterm benchmarks.
- **Assessment Cadence**: Grading velocity and assignment counts per section.
- **Section Volume**: Total student count and class load across academic terms.

### What it Shows to Improve:
- **Primary Focus Area**: Pinpoints the exact area needing attention (`Differentiated Instruction`, `Curriculum Pacing & Attendance`, `Multi-Prep Workload Relief`, `Formative Assessment Cadence`, or `Mentor Candidate`).
- **Concrete Actionable Steps**: Step-by-step instructional actions for the teacher and administrative support actions.
- **Recommended PD Workshop**: Targeted training module matched to the teacher's unique classroom diagnostic.
- **Target Measurable Outcome**: Concrete, quantifiable goal (e.g., reduce section variance < 12 points, recover attendance > 90%).

---

## Running Automated Tests

```bash
source venv/bin/activate
pytest tests/ -v
```
All **78 test cases** pass, covering:
1. Synthetic school data generation & quality verification
2. Student feature engineering & leakage prevention
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
13. **Teacher ML Pipeline**: Feature engineering, support labels, RF/LR training, improvement plan generation, ethical language enforcement, and teacher ML REST API endpoints.

---

## API Endpoints

| Category | Endpoint | Method | Description |
|---|---|---|---|
| **Health** | `/health`, `/api/v1/health` | GET | Readiness & model status |
| **Data** | `/api/v1/data/summary` | GET | Source data metrics & tables |
| **Data** | `/api/v1/data/quality` | GET | Quality audit & completeness scores |
| **Predictions** | `/api/v1/students/predictions` | GET | Early warning student risk list |
| **Predictions** | `/api/v1/students/predictions/{id}` | GET | Student risk breakdown & contributing factors |
| **Teacher ML** | `/api/v1/teachers/ml/predictions` | GET | Teacher ML support priorities & instructional diagnostics |
| **Teacher ML** | `/api/v1/teachers/ml/predictions/{id}` | GET | Individual teacher diagnostic factors, concrete improvement actions, PD workshop, and target outcomes |
| **Teacher ML** | `/api/v1/teachers/ml/improvement-areas` | GET | Aggregated focus areas, priority breakdown & recommended PD catalog |
| **Teacher Support**| `/api/v1/teachers/support/workload` | GET | Teacher student ratios & sections |
| **Teacher Support**| `/api/v1/teachers/support/outcomes` | GET | Contextualized subject support patterns |
| **Class Analytics** | `/api/v1/classes/analytics/subjects` | GET | Subject performance & pass rates |
| **Class Analytics** | `/api/v1/classes/analytics/struggling`| GET | Identified learning gap subjects |
| **Forecasting** | `/api/v1/school/forecasts` | GET | Enrollment & retention projections |
| **Recommendations**| `/api/v1/recommendations` | GET | Prioritized actionable improvement items |
| **Recommendations**| `/api/v1/recommendations/{id}` | PATCH | Update recommendation status & outcomes |
| **Models** | `/api/v1/models/status` | GET | Active models, F1 scores & metrics |
| **Dashboard** | `/api/v1/dashboard/overview` | GET | Executive overview metrics & teacher support counts |
| **UI** | `/` | GET | President's Intelligence Dashboard with Teacher Action Modal |

---

## Security & Ethics Safeguards

1. **Least-Privilege & Pseudonymous IDs**: Inference uses internal IDs; full student names and identities are never leaked into logs or model weights.
2. **Supportive Language Guarantee**: Predictions label students as *"Needs Academic Support"* or *"On Track"*. Derogatory labels (`failing`, `loser`, `bad student`) are strictly prevented.
3. **No Automated High-Stakes Decisions**: AI predictions inform advisory staff; they never automatically alter grades, sanctions, or enrollment statuses.
4. **Ethical Teacher Analytics**: Explicit disclaimer and architectural guardrail preventing automated ranking of teachers by raw student test scores. ML focuses exclusively on institutional and pedagogical support ("Review Recommended", "Targeted Support", "Mentor Candidate").
5. **No Data Leakage**: Temporal and stratified validation splits ensure future outcomes never leak into training features.
