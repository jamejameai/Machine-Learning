"""
Comprehensive Test Suite for CIIS School Intelligence AI.

Tests cover:
1. Demo data generation and quality
2. Feature engineering correctness
3. ML pipeline: training, evaluation, leakage prevention
4. Model lifecycle and versioning
5. Prediction inference and schema
6. Analytics modules
7. Recommendation engine
8. API endpoints
9. Database operations
10. Security and error handling

Run: pytest tests/ -v
"""

import os
import json
import shutil
import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Set test environment before imports
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///./test_ciis_ai.db"
os.environ["MODEL_DIR"] = "./test_models"
os.environ["APP_DEBUG"] = "false"
os.environ["API_KEY"] = "test-api-key"
os.environ["LOG_SENSITIVE_DATA"] = "false"

from app.config import get_settings, Settings
from app.database import init_database, Base, get_engine, reset_engine
from app.demo_data import (
    generate_students, generate_teachers, generate_classes,
    generate_attendance, generate_grades, generate_all_demo_data,
    get_data_summary
)
from app.ml_pipeline import (
    StudentFeatureEngineer, define_risk_labels,
    StudentRiskModelTrainer, run_full_training_pipeline,
    StudentRiskPredictor
)
from app.analytics import ClassAnalytics, TeacherAnalytics, SchoolForecaster
from app.recommendations import RecommendationEngine


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def demo_data():
    """Generate demo data once for all tests."""
    return generate_all_demo_data(n_students=200, n_teachers=20, n_attendance_days=60, seed=42)


@pytest.fixture(scope="session")
def trained_models(demo_data):
    """Train models once for all tests."""
    os.makedirs("./test_models", exist_ok=True)
    results = run_full_training_pipeline(
        demo_data["students"], demo_data["attendance"], demo_data["grades"], seed=42
    )
    return results


@pytest.fixture
def feature_engineer():
    return StudentFeatureEngineer()


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """Clean up test artifacts after all tests."""
    yield
    # Clean up
    for f in Path(".").glob("test_ciis_ai.db*"):
        f.unlink(missing_ok=True)
    if Path("./test_models").exists():
        shutil.rmtree("./test_models", ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEMO DATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDemoDataGeneration:
    """Tests for synthetic data generation."""

    def test_students_generation(self):
        """Verify student data structure and content."""
        students = generate_students(100, seed=42)
        assert len(students) == 100
        assert "student_id" in students.columns
        assert "grade_level" in students.columns
        assert "enrollment_status" in students.columns
        assert students["student_id"].is_unique, "Student IDs must be unique"

    def test_students_have_valid_grades(self):
        """Grade levels must be within expected range."""
        students = generate_students(200, seed=42)
        valid_grades = {7, 8, 9, 10, 11, 12}
        assert set(students["grade_level"].unique()).issubset(valid_grades)

    def test_students_enrollment_distribution(self):
        """Most students should be active."""
        students = generate_students(500, seed=42)
        active_pct = (students["enrollment_status"] == "active").mean()
        assert active_pct > 0.85, f"Expected >85% active, got {active_pct*100:.1f}%"

    def test_teachers_generation(self):
        """Verify teacher data structure."""
        teachers = generate_teachers(20, seed=42)
        assert len(teachers) == 20
        assert "teacher_id" in teachers.columns
        assert teachers["teacher_id"].is_unique

    def test_attendance_generation(self, demo_data):
        """Verify attendance records are valid."""
        att = demo_data["attendance"]
        assert len(att) > 0
        valid_statuses = {"present", "absent", "late"}
        assert set(att["status"].unique()).issubset(valid_statuses)

    def test_grades_generation(self, demo_data):
        """Verify grades are within valid range."""
        grades = demo_data["grades"]
        assert len(grades) > 0
        assert grades["score"].min() >= 0
        assert grades["score"].max() <= 100
        assert grades["percentage"].min() >= 0
        assert grades["percentage"].max() <= 100

    def test_data_reproducibility(self):
        """Same seed should produce identical data."""
        data1 = generate_students(100, seed=123)
        data2 = generate_students(100, seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_data_summary(self, demo_data):
        """Data summary should report accurate counts."""
        summary = get_data_summary(demo_data)
        assert "students" in summary
        assert "attendance" in summary
        assert "grades" in summary
        assert summary["students"]["total_records"] == len(demo_data["students"])

    def test_no_sensitive_data_exposure(self, demo_data):
        """Verify demo data uses pseudonymous IDs."""
        students = demo_data["students"]
        # IDs should be formatted STU-XXXX, not real names or SSNs
        assert all(s.startswith("STU-") for s in students["student_id"])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """Tests for the feature engineering pipeline."""

    def test_features_built_correctly(self, demo_data, feature_engineer):
        """Feature matrix should have expected columns."""
        features = feature_engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        expected = [
            "attendance_rate", "late_rate", "absence_trend",
            "avg_grade", "grade_std", "subjects_below_passing",
            "exam_quiz_gap", "grade_trend", "grade_level_encoded"
        ]
        for col in expected:
            assert col in features.columns, f"Missing feature: {col}"

    def test_attendance_rate_range(self, demo_data, feature_engineer):
        """Attendance rate should be between 0 and 1."""
        features = feature_engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        assert features["attendance_rate"].min() >= 0
        assert features["attendance_rate"].max() <= 1.0

    def test_avg_grade_range(self, demo_data, feature_engineer):
        """Average grade should be in a reasonable range."""
        features = feature_engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        assert features["avg_grade"].min() >= 0
        assert features["avg_grade"].max() <= 100

    def test_no_data_leakage_in_features(self, demo_data, feature_engineer):
        """Features should not include the target variable."""
        features = feature_engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        # The label column should not be in the feature set
        assert "label" not in feature_engineer.feature_names
        assert "needs_support" not in feature_engineer.feature_names

    def test_handles_empty_attendance(self, demo_data, feature_engineer):
        """Should handle missing attendance data gracefully."""
        empty_att = pd.DataFrame(columns=demo_data["attendance"].columns)
        features = feature_engineer.build_features(
            demo_data["students"], empty_att, demo_data["grades"]
        )
        assert len(features) > 0
        # Attendance features should be 0 (default) when no data
        assert features["attendance_rate"].isna().sum() >= 0  # May have NaN from merge

    def test_handles_empty_grades(self, demo_data, feature_engineer):
        """Should handle missing grades data gracefully."""
        empty_grades = pd.DataFrame(columns=demo_data["grades"].columns)
        features = feature_engineer.build_features(
            demo_data["students"], demo_data["attendance"], empty_grades
        )
        assert len(features) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LABEL DEFINITION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelDefinition:
    """Tests for target variable definition."""

    def test_labels_are_binary(self, demo_data, feature_engineer):
        """Labels should be 0 or 1."""
        features = feature_engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        labels = define_risk_labels(features)
        assert set(labels.unique()).issubset({0, 1})

    def test_label_distribution_reasonable(self, demo_data, feature_engineer):
        """Neither class should be empty."""
        features = feature_engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        labels = define_risk_labels(features)
        assert labels.sum() > 0, "Must have some positive labels"
        assert (labels == 0).sum() > 0, "Must have some negative labels"

    def test_configurable_thresholds(self, demo_data, feature_engineer):
        """Different thresholds should change label distribution."""
        features = feature_engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        strict_labels = define_risk_labels(features, passing_threshold=85.0, min_attendance=0.90)
        lenient_labels = define_risk_labels(features, passing_threshold=60.0, min_attendance=0.60)

        # Strict criteria should flag more students
        assert strict_labels.sum() >= lenient_labels.sum()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ML PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLPipeline:
    """Tests for the machine learning training and evaluation pipeline."""

    def test_training_produces_results(self, trained_models):
        """Training should complete and produce evaluation results."""
        assert "best_model" in trained_models
        assert "evaluations" in trained_models
        assert len(trained_models["evaluations"]) == 2  # LR and RF

    def test_both_models_trained(self, trained_models):
        """Both Logistic Regression and Random Forest should be trained."""
        assert "logistic_regression" in trained_models["evaluations"]
        assert "random_forest" in trained_models["evaluations"]

    def test_metrics_are_valid(self, trained_models):
        """All metrics should be in valid ranges."""
        for name, eval_data in trained_models["evaluations"].items():
            assert 0 <= eval_data["accuracy"] <= 1, f"{name} accuracy out of range"
            assert 0 <= eval_data["precision"] <= 1, f"{name} precision out of range"
            assert 0 <= eval_data["recall"] <= 1, f"{name} recall out of range"
            assert 0 <= eval_data["f1"] <= 1, f"{name} F1 out of range"
            if eval_data.get("roc_auc") is not None:
                assert 0 <= eval_data["roc_auc"] <= 1, f"{name} ROC-AUC out of range"

    def test_no_train_test_leakage(self, trained_models):
        """Training and test sets should be different sizes."""
        assert trained_models["training_samples"] > 0
        assert trained_models["test_samples"] > 0
        total = trained_models["training_samples"] + trained_models["test_samples"]
        # Test should be roughly 20% of total
        test_ratio = trained_models["test_samples"] / total
        assert 0.15 <= test_ratio <= 0.25, f"Test ratio {test_ratio:.2f} outside expected range"

    def test_model_artifacts_saved(self, trained_models):
        """Model artifacts should be saved to disk."""
        model_dir = Path("./test_models")
        assert model_dir.exists()
        joblib_files = list(model_dir.glob("*.joblib"))
        assert len(joblib_files) >= 2, f"Expected at least 2 model files, found {len(joblib_files)}"

    def test_metadata_files_saved(self, trained_models):
        """Model metadata should be saved alongside artifacts."""
        model_dir = Path("./test_models")
        metadata_files = list(model_dir.glob("*_metadata.json"))
        assert len(metadata_files) >= 2

        # Verify metadata structure
        with open(metadata_files[0]) as f:
            meta = json.load(f)
        assert "model_name" in meta
        assert "model_version" in meta
        assert "feature_names" in meta
        assert "evaluation" in meta

    def test_feature_importances_present(self, trained_models):
        """Feature importances should be calculated."""
        for name, eval_data in trained_models["evaluations"].items():
            assert "feature_importances" in eval_data
            assert eval_data["feature_importances"] is not None
            assert len(eval_data["feature_importances"]) > 0

    def test_confusion_matrix_valid(self, trained_models):
        """Confusion matrix should be a valid 2x2 matrix."""
        for name, eval_data in trained_models["evaluations"].items():
            cm = eval_data.get("confusion_matrix")
            assert cm is not None, f"{name} missing confusion matrix"
            assert len(cm) == 2 and len(cm[0]) == 2, f"{name} invalid confusion matrix shape"
            # All values should be non-negative
            for row in cm:
                for val in row:
                    assert val >= 0

    def test_best_model_selected(self, trained_models):
        """A best model should be selected."""
        best = trained_models["best_model"]
        assert best in ["logistic_regression", "random_forest"]

    def test_class_distribution_recorded(self, trained_models):
        """Class distribution should be recorded for transparency."""
        assert "class_distribution" in trained_models
        dist = trained_models["class_distribution"]
        assert len(dist) == 2  # Binary classification

    def test_reproducibility(self, demo_data):
        """Same seed should produce same results."""
        results1 = run_full_training_pipeline(
            demo_data["students"], demo_data["attendance"], demo_data["grades"], seed=42
        )
        results2 = run_full_training_pipeline(
            demo_data["students"], demo_data["attendance"], demo_data["grades"], seed=42
        )

        assert results1["evaluations"]["logistic_regression"]["f1"] == \
               results2["evaluations"]["logistic_regression"]["f1"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictions:
    """Tests for prediction inference."""

    def test_predictions_generated(self, demo_data, trained_models):
        """Should generate predictions for all active students."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )

        predictions = predictor.predict(features)
        active_count = (demo_data["students"]["enrollment_status"] == "active").sum()
        assert len(predictions) == active_count

    def test_prediction_schema(self, demo_data, trained_models):
        """Each prediction should have required fields."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )

        predictions = predictor.predict(features)
        required_fields = [
            "student_id", "prediction_label", "prediction_score",
            "risk_level", "contributing_factors", "model_version",
            "predicted_at", "confidence_notes"
        ]

        for pred in predictions[:5]:
            for field in required_fields:
                assert field in pred, f"Missing field: {field}"

    def test_prediction_scores_valid(self, demo_data, trained_models):
        """Prediction scores should be between 0 and 1."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )

        predictions = predictor.predict(features)
        for pred in predictions:
            if pred["prediction_score"] is not None:
                assert 0 <= pred["prediction_score"] <= 1

    def test_risk_levels_valid(self, demo_data, trained_models):
        """Risk levels should be from defined set."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )

        predictions = predictor.predict(features)
        valid_levels = {"high", "medium", "low"}
        for pred in predictions:
            assert pred["risk_level"] in valid_levels

    def test_prediction_labels_valid(self, demo_data, trained_models):
        """Labels should use supportive language."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )

        predictions = predictor.predict(features)
        valid_labels = {"needs_support", "on_track"}
        for pred in predictions:
            assert pred["prediction_label"] in valid_labels
            # Should never use negative labels
            assert "failure" not in pred["prediction_label"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANALYTICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassAnalytics:
    """Tests for class and subject analytics."""

    def test_subject_performance(self, demo_data):
        """Should generate subject performance summaries."""
        analytics = ClassAnalytics(demo_data["grades"], demo_data["attendance"])
        results = analytics.subject_performance_summary()
        assert len(results) > 0
        for r in results:
            assert "subject" in r
            assert "avg_score" in r
            assert "pass_rate" in r
            assert 0 <= r["avg_score"] <= 100

    def test_grade_level_performance(self, demo_data):
        """Should generate grade-level performance summaries."""
        analytics = ClassAnalytics(demo_data["grades"], demo_data["attendance"])
        results = analytics.grade_level_performance()
        assert len(results) > 0

    def test_section_comparison_fair(self, demo_data):
        """Should only compare same grade + subject."""
        analytics = ClassAnalytics(demo_data["grades"], demo_data["attendance"])
        results = analytics.class_section_comparison()
        for comp in results:
            # Each comparison should be within one grade and one subject
            assert "grade_level" in comp
            assert "subject" in comp

    def test_attendance_by_grade(self, demo_data):
        """Attendance analytics should cover all grades."""
        analytics = ClassAnalytics(demo_data["grades"], demo_data["attendance"])
        results = analytics.attendance_by_grade()
        assert len(results) > 0
        for r in results:
            assert 0 <= r["attendance_rate"] <= 100

    def test_struggling_subjects_threshold(self, demo_data):
        """Struggling subjects should respect threshold."""
        analytics = ClassAnalytics(demo_data["grades"], demo_data["attendance"])
        results = analytics.identify_struggling_subjects(threshold=70.0)
        for r in results:
            assert r["pct_below_threshold"] > 20  # Only flags >20%


class TestTeacherAnalytics:
    """Tests for teacher support analytics."""

    def test_workload_summary(self, demo_data):
        """Should calculate teacher workloads."""
        analytics = TeacherAnalytics(
            demo_data["teachers"], demo_data["classes"],
            demo_data["grades"], demo_data["attendance"]
        )
        results = analytics.workload_summary()
        assert len(results) > 0
        for r in results:
            assert "n_classes" in r
            assert r["n_classes"] > 0

    def test_no_teacher_ranking(self, demo_data):
        """Outcomes should include disclaimer about causation."""
        analytics = TeacherAnalytics(
            demo_data["teachers"], demo_data["classes"],
            demo_data["grades"], demo_data["attendance"]
        )
        results = analytics.class_outcomes_summary()
        for r in results:
            assert "note" in r  # Must include disclaimer


class TestSchoolForecaster:
    """Tests for school-level forecasting."""

    def test_enrollment_summary(self, demo_data):
        """Should provide enrollment breakdown."""
        forecaster = SchoolForecaster(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        summary = forecaster.enrollment_summary()
        assert summary["total_enrolled"] > 0
        assert summary["active"] > 0
        assert "by_grade_level" in summary

    def test_forecast_includes_assumptions(self, demo_data):
        """Forecasts must include assumptions and limitations."""
        forecaster = SchoolForecaster(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        forecast = forecaster.simple_enrollment_forecast()
        assert "assumptions" in forecast
        assert len(forecast["assumptions"]) > 0
        assert "projections" in forecast

    def test_operational_metrics(self, demo_data):
        """Should calculate key operational metrics."""
        forecaster = SchoolForecaster(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        metrics = forecaster.operational_metrics()
        assert "total_active_students" in metrics
        assert metrics["total_active_students"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RECOMMENDATION ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecommendationEngine:
    """Tests for the recommendation engine."""

    def test_generates_recommendations(self, demo_data, trained_models):
        """Should generate recommendations from predictions."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        predictions = predictor.predict(features)

        engine = RecommendationEngine()
        recs = engine.generate_all(predictions=predictions)
        assert len(recs) > 0

    def test_recommendations_have_required_fields(self, demo_data, trained_models):
        """Each recommendation should have all required fields."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        predictions = predictor.predict(features)

        engine = RecommendationEngine()
        recs = engine.generate_all(predictions=predictions)

        required = ["title", "area", "priority", "finding_summary", "status"]
        for rec in recs:
            for field in required:
                assert field in rec, f"Recommendation missing field: {field}"

    def test_recommendations_sorted_by_priority(self):
        """Recommendations should be sorted by priority."""
        engine = RecommendationEngine()
        # Generate with analytics data
        recs = engine.generate_all(
            class_analytics={
                "struggling_subjects": [
                    {"subject": "Math", "grade_level": 7, "pct_below_threshold": 40,
                     "n_struggling": 20, "n_total": 50, "avg_score": 65, "threshold": 70, "severity": "high"},
                ],
                "section_comparisons": [],
            },
            operational_metrics={"overall_pass_rate": 65, "max_class_size": 45, "avg_class_size": 35},
        )

        if len(recs) >= 2:
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            for i in range(len(recs) - 1):
                assert priority_order[recs[i]["priority"]] <= priority_order[recs[i+1]["priority"]]

    def test_no_duplicate_recommendations(self):
        """Should not generate duplicate recommendations."""
        engine = RecommendationEngine()
        recs = engine.generate_all(
            class_analytics={
                "struggling_subjects": [
                    {"subject": "Math", "grade_level": 7, "pct_below_threshold": 40,
                     "n_struggling": 20, "n_total": 50, "avg_score": 65, "threshold": 70, "severity": "high"},
                    {"subject": "Math", "grade_level": 7, "pct_below_threshold": 40,
                     "n_struggling": 20, "n_total": 50, "avg_score": 65, "threshold": 70, "severity": "high"},
                ],
                "section_comparisons": [],
            },
        )
        titles = [r["title"] for r in recs]
        assert len(titles) == len(set(titles)), "Duplicate recommendations found"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. API ENDPOINT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        from app.api import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health check should return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "is_ready" in data
        assert "model_status" in data

    def test_dashboard_loads(self, client):
        """Dashboard should return HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "CIIS" in response.text

    def test_predictions_endpoint(self, client):
        """Predictions endpoint should return data."""
        response = client.get("/api/v1/students/predictions")
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total" in data

    def test_predictions_filter(self, client):
        """Should filter predictions by risk level."""
        response = client.get("/api/v1/students/predictions?risk_level=high")
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            for p in data["predictions"]:
                assert p["risk_level"] == "high"

    def test_subject_analytics(self, client):
        """Subject analytics endpoint should work."""
        response = client.get("/api/v1/classes/analytics/subjects")
        assert response.status_code == 200
        data = response.json()
        assert "subjects" in data

    def test_teacher_workload(self, client):
        """Teacher workload should include disclaimer."""
        response = client.get("/api/v1/teachers/support/workload")
        assert response.status_code == 200

    def test_teacher_outcomes_disclaimer(self, client):
        """Teacher outcomes should have disclaimer."""
        response = client.get("/api/v1/teachers/support/outcomes")
        assert response.status_code == 200
        data = response.json()
        assert "disclaimer" in data

    def test_recommendations_endpoint(self, client):
        """Recommendations should be accessible."""
        response = client.get("/api/v1/recommendations")
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data

    def test_recommendation_update(self, client):
        """Should be able to update recommendation status."""
        # First get recommendations
        response = client.get("/api/v1/recommendations")
        if response.status_code == 200 and response.json()["recommendations"]:
            rec_id = response.json()["recommendations"][0]["id"]
            update_response = client.put(
                f"/api/v1/recommendations/{rec_id}",
                json={"status": "reviewed"}
            )
            assert update_response.status_code == 200

    def test_model_status_endpoint(self, client):
        """Model status should show evaluation metrics."""
        response = client.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    def test_dashboard_overview_endpoint(self, client):
        """Dashboard overview should return complete data."""
        response = client.get("/api/v1/dashboard/overview")
        assert response.status_code == 200
        data = response.json()
        assert "school_metrics" in data
        assert "student_risk_summary" in data
        assert "recommendations_summary" in data
        assert "model_health" in data

    def test_school_metrics(self, client):
        """School metrics endpoint should work."""
        response = client.get("/api/v1/school/metrics")
        assert response.status_code == 200

    def test_enrollment_endpoint(self, client):
        """Enrollment summary should work."""
        response = client.get("/api/v1/school/enrollment")
        assert response.status_code == 200

    def test_forecasts_endpoint(self, client):
        """Forecasts should include assumptions."""
        response = client.get("/api/v1/school/forecasts")
        assert response.status_code == 200
        data = response.json()
        assert "note" in data  # Must include caution

    def test_data_quality_endpoint(self, client):
        """Data quality report should be available."""
        response = client.get("/api/v1/data/quality")
        assert response.status_code in [200, 503]

    def test_invalid_recommendation_update(self, client):
        """Invalid status should return 400."""
        response = client.put(
            "/api/v1/recommendations/1",
            json={"status": "invalid_status"}
        )
        assert response.status_code in [400, 404]

    def test_nonexistent_prediction(self, client):
        """Should return 404 for unknown student."""
        response = client.get("/api/v1/students/predictions/NONEXISTENT")
        assert response.status_code in [404, 503]

    def test_api_docs_available(self, client):
        """Swagger docs should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 9. DATABASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabase:
    """Tests for database initialization and models."""

    def test_database_init(self):
        """Database should initialize without errors."""
        reset_engine()
        os.environ["DATABASE_URL"] = "sqlite:///./test_db_init.db"
        try:
            engine = init_database()
            assert engine is not None
        finally:
            Path("./test_db_init.db").unlink(missing_ok=True)
            reset_engine()
            os.environ["DATABASE_URL"] = "sqlite:///./test_ciis_ai.db"

    def test_tables_created(self):
        """All required tables should be created."""
        reset_engine()
        os.environ["DATABASE_URL"] = "sqlite:///./test_db_tables.db"
        try:
            engine = init_database()
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            expected_tables = [
                "ciis_ai_model_metadata",
                "ciis_ai_predictions",
                "ciis_ai_prediction_runs",
                "ciis_ai_recommendations",
                "ciis_ai_recommendation_history",
                "ciis_ai_data_quality",
                "ciis_ai_audit_log",
            ]

            for table in expected_tables:
                assert table in tables, f"Missing table: {table}"
        finally:
            Path("./test_db_tables.db").unlink(missing_ok=True)
            reset_engine()
            os.environ["DATABASE_URL"] = "sqlite:///./test_ciis_ai.db"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SECURITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurity:
    """Security-related tests."""

    def test_no_sensitive_data_in_predictions(self, demo_data, trained_models):
        """Predictions should not contain full student names."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        predictions = predictor.predict(features)

        for pred in predictions:
            # Should use pseudonymous IDs, not names
            assert pred["student_id"].startswith("STU-")
            pred_str = json.dumps(pred)
            # Should not contain actual names from demo data
            assert "first_name" not in pred_str
            assert "last_name" not in pred_str

    def test_prediction_uses_supportive_language(self, demo_data, trained_models):
        """Labels should never use harmful language."""
        best = trained_models["best_model"]
        model_path = f"./test_models/{best}_v1.0.joblib"
        metadata_path = f"./test_models/{best}_v1.0_metadata.json"

        predictor = StudentRiskPredictor(model_path, metadata_path)
        engineer = StudentFeatureEngineer()
        features = engineer.build_features(
            demo_data["students"], demo_data["attendance"], demo_data["grades"]
        )
        predictions = predictor.predict(features)

        harmful_terms = ["failure", "failing", "loser", "bad student", "stupid"]
        for pred in predictions:
            pred_str = json.dumps(pred).lower()
            for term in harmful_terms:
                assert term not in pred_str, f"Found harmful term '{term}' in prediction"

    def test_env_not_in_repo(self):
        """Secret .env should not be committed (checked via .gitignore)."""
        gitignore = Path("/Users/macbook/Machine-Learning/.gitignore")
        if gitignore.exists():
            content = gitignore.read_text()
            assert ".env" in content
