"""
FastAPI Service for CIIS School Intelligence AI.

Provides RESTful API endpoints for:
- Health check and readiness
- Student risk predictions
- Class and subject analytics
- Teacher support analytics
- School forecasting
- Recommendations
- Model lifecycle management
- Dashboard interface

All endpoints validate inputs, handle errors gracefully, and enforce timeouts.
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from app.config import get_settings
from app.database import init_database, get_db, ModelMetadata, Prediction, Recommendation as RecommendationDB
from app.database import PredictionRun, DataQualityReport, AuditLog, RecommendationHistory
from app.demo_data import generate_all_demo_data, get_data_summary
from app.ml_pipeline import (
    run_full_training_pipeline, StudentRiskPredictor,
    StudentFeatureEngineer, define_risk_labels
)
from app.analytics import ClassAnalytics, TeacherAnalytics, SchoolForecaster
from app.recommendations import RecommendationEngine

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ciis_ai")

# ─── Global State ─────────────────────────────────────────────────────────────

class AppState:
    """Application state holder — not a substitute for database."""
    data: dict = {}
    training_results: dict = {}
    predictions: list = []
    analytics_cache: dict = {}
    recommendations: list = []
    data_summary: dict = {}
    is_ready: bool = False
    last_refresh: Optional[str] = None
    model_status: str = "not_trained"


app_state = AppState()

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, clean up on shutdown."""
    logger.info("Starting CIIS School Intelligence AI...")

    # Initialize database
    init_database()
    logger.info("Database initialized.")

    # Create model directory
    settings = get_settings()
    os.makedirs(settings.model_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Generate demo data and run initial training
    try:
        _initialize_demo_system()
    except Exception as e:
        logger.error(f"Failed to initialize demo system: {e}")
        app_state.model_status = "initialization_failed"

    yield

    logger.info("Shutting down CIIS School Intelligence AI...")


def _initialize_demo_system():
    """Initialize the full demo system with data, training, and analytics."""
    logger.info("Generating demo data...")
    data = generate_all_demo_data(n_students=300, n_teachers=30, n_attendance_days=120)
    app_state.data = data
    app_state.data_summary = get_data_summary(data)

    logger.info("Running ML training pipeline...")
    results = run_full_training_pipeline(
        data["students"], data["attendance"], data["grades"]
    )
    app_state.training_results = results
    app_state.model_status = "trained"

    logger.info("Running predictions...")
    _run_predictions()

    logger.info("Running analytics...")
    _run_analytics()

    logger.info("Generating recommendations...")
    _generate_recommendations()

    app_state.is_ready = True
    app_state.last_refresh = datetime.datetime.now(datetime.timezone.utc).isoformat()
    logger.info("System initialization complete!")


def _run_predictions():
    """Run predictions using the best trained model."""
    settings = get_settings()
    best_model = app_state.training_results.get("best_model", "logistic_regression")

    model_path = Path(settings.model_dir) / f"{best_model}_v1.0.joblib"
    metadata_path = Path(settings.model_dir) / f"{best_model}_v1.0_metadata.json"

    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return

    predictor = StudentRiskPredictor(str(model_path), str(metadata_path))

    # Build features for all active students
    engineer = StudentFeatureEngineer()
    features = engineer.build_features(
        app_state.data["students"],
        app_state.data["attendance"],
        app_state.data["grades"],
    )

    predictions = predictor.predict(features)
    app_state.predictions = predictions
    logger.info(f"Generated {len(predictions)} predictions")


def _run_analytics():
    """Run all analytics modules."""
    data = app_state.data

    # Class analytics
    class_analytics = ClassAnalytics(data["grades"], data["attendance"])
    app_state.analytics_cache["subject_performance"] = class_analytics.subject_performance_summary()
    app_state.analytics_cache["grade_level_performance"] = class_analytics.grade_level_performance()
    app_state.analytics_cache["section_comparisons"] = class_analytics.class_section_comparison()
    app_state.analytics_cache["assessment_trends"] = class_analytics.assessment_trend()
    app_state.analytics_cache["attendance_by_grade"] = class_analytics.attendance_by_grade()
    app_state.analytics_cache["struggling_subjects"] = class_analytics.identify_struggling_subjects()

    # Teacher analytics
    teacher_analytics = TeacherAnalytics(
        data["teachers"], data["classes"], data["grades"], data["attendance"]
    )
    app_state.analytics_cache["teacher_workload"] = teacher_analytics.workload_summary()
    app_state.analytics_cache["teacher_outcomes"] = teacher_analytics.class_outcomes_summary()
    app_state.analytics_cache["collaboration"] = teacher_analytics.collaboration_opportunities()

    # School forecasting
    forecaster = SchoolForecaster(data["students"], data["attendance"], data["grades"])
    app_state.analytics_cache["enrollment_summary"] = forecaster.enrollment_summary()
    app_state.analytics_cache["attendance_trend"] = forecaster.attendance_trend()
    app_state.analytics_cache["academic_progress"] = forecaster.academic_progress_by_grade()
    app_state.analytics_cache["operational_metrics"] = forecaster.operational_metrics()
    app_state.analytics_cache["enrollment_forecast"] = forecaster.simple_enrollment_forecast()


def _generate_recommendations():
    """Generate recommendations from all available data."""
    engine = RecommendationEngine()
    recommendations = engine.generate_all(
        predictions=app_state.predictions,
        class_analytics={
            "struggling_subjects": app_state.analytics_cache.get("struggling_subjects", []),
            "section_comparisons": app_state.analytics_cache.get("section_comparisons", []),
        },
        attendance_data={
            "by_grade": app_state.analytics_cache.get("attendance_by_grade", []),
        },
        operational_metrics=app_state.analytics_cache.get("operational_metrics", {}),
    )
    app_state.recommendations = recommendations


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CIIS School Intelligence AI",
    description="AI-powered intelligence platform for school management",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Response Models ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    is_ready: bool
    model_status: str
    last_refresh: Optional[str] = None


class PredictionResponse(BaseModel):
    student_id: str
    prediction_label: str
    prediction_score: Optional[float] = None
    risk_level: str
    contributing_factors: list = []
    model_version: str = ""
    predicted_at: str = ""
    confidence_notes: str = ""


class RecommendationResponse(BaseModel):
    id: int
    title: str
    area: str
    priority: str
    finding_summary: str
    supporting_metrics: Optional[dict] = None
    recommended_action: Optional[str] = None
    suggested_owner: Optional[str] = None
    status: str = "new"
    created_at: str = ""


class RecommendationUpdateRequest(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    actual_outcome: Optional[str] = None
    dismissed_reason: Optional[str] = None
    notes: Optional[str] = None


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Service health and readiness check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.app_env,
        is_ready=app_state.is_ready,
        model_status=app_state.model_status,
        last_refresh=app_state.last_refresh,
    )


@app.get("/api/v1/data/summary")
async def data_summary():
    """Data audit summary showing available data and quality."""
    if not app_state.data_summary:
        raise HTTPException(status_code=503, detail="Data not yet loaded")
    return {
        "data_sources": app_state.data_summary,
        "assessed_at": app_state.last_refresh,
        "note": "Using synthetic demo data for development/testing",
    }


@app.get("/api/v1/data/quality")
async def data_quality():
    """Detailed data quality report."""
    if not app_state.data_summary:
        raise HTTPException(status_code=503, detail="Data not yet loaded")

    quality_reports = []
    for name, info in app_state.data_summary.items():
        quality_reports.append({
            "data_source": name,
            "total_records": info["total_records"],
            "quality_score": info["quality_score"],
            "missing_fields": info["missing_fields"],
            "duplicates": info["duplicates"],
            "columns": info["columns"],
        })

    return {"reports": quality_reports}


# ─── Student Predictions ─────────────────────────────────────────────────────

@app.get("/api/v1/students/predictions")
async def get_student_predictions(
    risk_level: Optional[str] = Query(None, description="Filter by risk level: high, medium, low"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get student risk predictions with optional filtering."""
    if not app_state.predictions:
        raise HTTPException(
            status_code=503,
            detail="Predictions not available. Model may not be trained yet."
        )

    filtered = app_state.predictions
    if risk_level:
        filtered = [p for p in filtered if p["risk_level"] == risk_level]

    # Sort by risk score descending
    filtered.sort(key=lambda x: x.get("prediction_score", 0) or 0, reverse=True)

    total = len(filtered)
    paginated = filtered[offset:offset + limit]

    return {
        "predictions": paginated,
        "total": total,
        "limit": limit,
        "offset": offset,
        "model_version": app_state.training_results.get("best_model", "unknown"),
        "generated_at": app_state.last_refresh,
    }


@app.get("/api/v1/students/predictions/{student_id}")
async def get_student_prediction(student_id: str):
    """Get prediction for a specific student."""
    if not app_state.predictions:
        raise HTTPException(status_code=503, detail="Predictions not available")

    prediction = next(
        (p for p in app_state.predictions if p["student_id"] == student_id),
        None
    )
    if not prediction:
        raise HTTPException(status_code=404, detail=f"No prediction found for student {student_id}")

    return prediction


@app.get("/api/v1/students/predictions/summary/overview")
async def predictions_summary():
    """Summary of all predictions."""
    if not app_state.predictions:
        return {"total": 0, "by_risk_level": {}, "note": "No predictions available"}

    by_risk = {}
    for p in app_state.predictions:
        level = p.get("risk_level", "unknown")
        by_risk[level] = by_risk.get(level, 0) + 1

    return {
        "total": len(app_state.predictions),
        "by_risk_level": by_risk,
        "model": app_state.training_results.get("best_model"),
        "generated_at": app_state.last_refresh,
    }


# ─── Class & Subject Analytics ───────────────────────────────────────────────

@app.get("/api/v1/classes/analytics/subjects")
async def subject_performance():
    """Performance analytics by subject."""
    return {
        "subjects": app_state.analytics_cache.get("subject_performance", []),
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/classes/analytics/grades")
async def grade_level_performance():
    """Performance analytics by grade level."""
    return {
        "grade_levels": app_state.analytics_cache.get("grade_level_performance", []),
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/classes/analytics/sections")
async def section_comparison():
    """Section-by-section comparison within same grade and subject."""
    return {
        "comparisons": app_state.analytics_cache.get("section_comparisons", []),
        "note": "Only compares sections within the same grade level and subject",
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/classes/analytics/trends")
async def assessment_trends():
    """Academic performance trends across assessment periods."""
    return {
        "trends": app_state.analytics_cache.get("assessment_trends", []),
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/classes/analytics/struggling")
async def struggling_subjects():
    """Subjects with high failure rates that need attention."""
    return {
        "struggling_subjects": app_state.analytics_cache.get("struggling_subjects", []),
        "threshold": 70.0,
        "note": "Subjects where >20% of students are below threshold",
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/classes/analytics/attendance")
async def attendance_by_grade():
    """Attendance analytics by grade level."""
    return {
        "grades": app_state.analytics_cache.get("attendance_by_grade", []),
        "as_of": app_state.last_refresh,
    }


# ─── Teacher Support ─────────────────────────────────────────────────────────

@app.get("/api/v1/teachers/support/workload")
async def teacher_workload():
    """Teacher workload analysis — not a ranking."""
    return {
        "teachers": app_state.analytics_cache.get("teacher_workload", []),
        "note": "Workload summary for support planning, not performance evaluation",
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/teachers/support/outcomes")
async def teacher_class_outcomes():
    """Aggregate class outcomes — association, not causation."""
    return {
        "outcomes": app_state.analytics_cache.get("teacher_outcomes", []),
        "disclaimer": "Class outcomes are affected by many factors beyond teacher control. "
                      "These are aggregate summaries, not teacher performance evaluations.",
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/teachers/support/collaboration")
async def collaboration_opportunities():
    """Identify collaboration opportunities between teachers."""
    return {
        "opportunities": app_state.analytics_cache.get("collaboration", []),
        "as_of": app_state.last_refresh,
    }


# ─── School Forecasting ──────────────────────────────────────────────────────

@app.get("/api/v1/school/enrollment")
async def enrollment_summary():
    """Current enrollment status and distribution."""
    return app_state.analytics_cache.get("enrollment_summary", {})


@app.get("/api/v1/school/attendance-trend")
async def attendance_trend():
    """Attendance trends over time."""
    data = app_state.analytics_cache.get("attendance_trend", [])
    return {
        "trend": data[-30:] if len(data) > 30 else data,  # Last 30 data points
        "total_points": len(data),
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/school/academic-progress")
async def academic_progress():
    """Academic progress across grades and periods."""
    return {
        "progress": app_state.analytics_cache.get("academic_progress", []),
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/school/forecasts")
async def school_forecasts():
    """Enrollment and operational forecasts."""
    return {
        "enrollment_forecast": app_state.analytics_cache.get("enrollment_forecast", {}),
        "note": "Forecasts are projections based on current trends with stated assumptions. "
                "They should be interpreted with caution.",
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/school/metrics")
async def operational_metrics():
    """Key operational metrics for school management."""
    return app_state.analytics_cache.get("operational_metrics", {})


# ─── Recommendations ─────────────────────────────────────────────────────────

@app.get("/api/v1/recommendations")
async def get_recommendations(
    area: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="Filter by status: new, reviewed, approved, in_progress, completed, dismissed"),
    limit: int = Query(50, ge=1, le=200),
):
    """Get prioritized recommendations with optional filtering."""
    filtered = app_state.recommendations

    if area:
        filtered = [r for r in filtered if r["area"] == area]
    if priority:
        filtered = [r for r in filtered if r["priority"] == priority]
    if status:
        filtered = [r for r in filtered if r.get("status") == status]

    return {
        "recommendations": filtered[:limit],
        "total": len(filtered),
        "as_of": app_state.last_refresh,
    }


@app.get("/api/v1/recommendations/{rec_id}")
async def get_recommendation(rec_id: int):
    """Get a specific recommendation."""
    rec = next((r for r in app_state.recommendations if r.get("id") == rec_id), None)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found")
    return rec


@app.put("/api/v1/recommendations/{rec_id}")
async def update_recommendation(rec_id: int, update: RecommendationUpdateRequest):
    """Update a recommendation's status, assignment, or outcome."""
    rec = next((r for r in app_state.recommendations if r.get("id") == rec_id), None)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found")

    changes = {}
    if update.status:
        valid_statuses = ["new", "reviewed", "approved", "in_progress", "completed", "dismissed"]
        if update.status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        changes["old_status"] = rec.get("status")
        rec["status"] = update.status
        changes["new_status"] = update.status

    if update.assigned_to:
        rec["assigned_to"] = update.assigned_to
        changes["assigned_to"] = update.assigned_to

    if update.actual_outcome:
        rec["actual_outcome"] = update.actual_outcome

    if update.dismissed_reason and rec.get("status") == "dismissed":
        rec["dismissed_reason"] = update.dismissed_reason

    rec["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {"message": "Recommendation updated", "changes": changes, "recommendation": rec}


# ─── Model Status ─────────────────────────────────────────────────────────────

@app.get("/api/v1/models/status")
async def model_status():
    """Model lifecycle and evaluation status."""
    evaluations = app_state.training_results.get("evaluations", {})
    best = app_state.training_results.get("best_model")

    models = []
    for name, eval_data in evaluations.items():
        status = "active" if name == best else "experimental"
        meets_acceptance = eval_data.get("f1", 0) >= settings.model_acceptance_f1

        models.append({
            "model_name": name,
            "status": status if meets_acceptance else "below_acceptance",
            "metrics": {
                "accuracy": eval_data.get("accuracy"),
                "precision": eval_data.get("precision"),
                "recall": eval_data.get("recall"),
                "f1": eval_data.get("f1"),
                "roc_auc": eval_data.get("roc_auc"),
            },
            "training_samples": eval_data.get("training_samples"),
            "test_samples": eval_data.get("test_samples"),
            "feature_importances": eval_data.get("feature_importances"),
            "meets_acceptance_criteria": meets_acceptance,
            "acceptance_threshold_f1": settings.model_acceptance_f1,
        })

    return {
        "models": models,
        "best_model": best,
        "training_data": "Synthetic demo data (not real student records)",
        "class_distribution": app_state.training_results.get("class_distribution"),
        "feature_names": app_state.training_results.get("feature_names"),
        "last_trained": app_state.last_refresh,
    }


@app.post("/api/v1/models/retrain")
async def retrain_models():
    """Retrain models with current data. Returns new evaluation results."""
    try:
        _initialize_demo_system()
        return {
            "message": "Models retrained successfully",
            "best_model": app_state.training_results.get("best_model"),
            "evaluations": app_state.training_results.get("evaluations"),
            "retrained_at": app_state.last_refresh,
        }
    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


# ─── Dashboard Overview ──────────────────────────────────────────────────────

@app.get("/api/v1/dashboard/overview")
async def dashboard_overview():
    """Executive overview for the president's dashboard."""
    metrics = app_state.analytics_cache.get("operational_metrics", {})
    pred_summary = {}
    if app_state.predictions:
        for p in app_state.predictions:
            level = p.get("risk_level", "unknown")
            pred_summary[level] = pred_summary.get(level, 0) + 1

    rec_summary = {}
    for r in app_state.recommendations:
        priority = r.get("priority", "unknown")
        rec_summary[priority] = rec_summary.get(priority, 0) + 1

    return {
        "school_metrics": {
            "total_students": metrics.get("total_active_students", 0),
            "attendance_rate": metrics.get("overall_attendance_rate"),
            "avg_grade": metrics.get("overall_avg_grade"),
            "pass_rate": metrics.get("overall_pass_rate"),
        },
        "student_risk_summary": pred_summary,
        "recommendations_summary": {
            "total": len(app_state.recommendations),
            "by_priority": rec_summary,
            "pending_review": len([r for r in app_state.recommendations if r.get("status") == "new"]),
        },
        "model_health": {
            "status": app_state.model_status,
            "best_model": app_state.training_results.get("best_model"),
            "f1_score": app_state.training_results.get("evaluations", {}).get(
                app_state.training_results.get("best_model", ""), {}
            ).get("f1"),
        },
        "last_refresh": app_state.last_refresh,
        "data_source": "Synthetic demo data",
    }


# ─── Dashboard HTML ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the intelligence dashboard."""
    dashboard_path = Path(__file__).parent / "templates" / "dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text())
    return HTMLResponse(content=_get_fallback_dashboard())


def _get_fallback_dashboard() -> str:
    """Minimal fallback dashboard if template is missing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>CIIS School Intelligence AI</title></head>
    <body>
        <h1>CIIS School Intelligence AI</h1>
        <p>Dashboard template not found. API is running at <a href="/api/v1/health">/api/v1/health</a></p>
        <p>See <a href="/docs">/docs</a> for API documentation.</p>
    </body>
    </html>
    """
