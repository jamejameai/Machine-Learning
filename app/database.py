"""
Database models and connection management for CIIS AI.

Uses SQLAlchemy with SQLite for demo/development and MySQL for production.
All AI-specific tables use the 'ciis_ai_' prefix to avoid conflicts with existing CIIS tables.
"""

import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime,
    Boolean, JSON, Enum as SAEnum, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import StaticPool
from app.config import get_settings

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

Base = declarative_base()


# ─── AI Model Metadata ───────────────────────────────────────────────────────

class ModelMetadata(Base):
    """Tracks ML model versions, training details, and evaluation metrics."""
    __tablename__ = "ciis_ai_model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)  # e.g., "logistic_regression", "random_forest"
    status = Column(String(20), nullable=False, default="experimental")  # experimental, validated, active, retired
    feature_names = Column(JSON, nullable=True)  # List of feature column names
    hyperparameters = Column(JSON, nullable=True)
    training_samples = Column(Integer, nullable=True)
    evaluation_metrics = Column(JSON, nullable=True)  # {"f1": 0.75, "precision": 0.8, ...}
    training_data_period = Column(String(100), nullable=True)  # e.g., "2024-01 to 2024-12"
    artifact_path = Column(String(500), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    predictions = relationship("Prediction", back_populates="model")


# ─── Prediction Records ──────────────────────────────────────────────────────

class PredictionRun(Base):
    """A batch of predictions generated at a specific time."""
    __tablename__ = "ciis_ai_prediction_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("ciis_ai_model_metadata.id"), nullable=False)
    run_type = Column(String(50), nullable=False)  # e.g., "student_risk", "enrollment_forecast"
    total_predictions = Column(Integer, default=0)
    status = Column(String(20), default="completed")  # running, completed, failed
    started_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)


class Prediction(Base):
    """Individual prediction for a student or entity."""
    __tablename__ = "ciis_ai_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("ciis_ai_model_metadata.id"), nullable=False)
    run_id = Column(Integer, ForeignKey("ciis_ai_prediction_runs.id"), nullable=True)
    entity_type = Column(String(50), nullable=False)  # "student", "class", "school"
    entity_id = Column(String(50), nullable=False)  # Pseudonymous ID
    prediction_label = Column(String(100), nullable=False)  # e.g., "needs_support", "on_track"
    prediction_score = Column(Float, nullable=True)  # Calibrated probability
    risk_level = Column(String(20), nullable=True)  # "low", "medium", "high"
    contributing_factors = Column(JSON, nullable=True)  # Feature importances
    confidence_notes = Column(Text, nullable=True)
    data_quality_flags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    model = relationship("ModelMetadata", back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_entity", "entity_type", "entity_id"),
        Index("ix_predictions_created", "created_at"),
    )


# ─── Recommendations ─────────────────────────────────────────────────────────

class Recommendation(Base):
    """Evidence-based recommendation with tracking workflow."""
    __tablename__ = "ciis_ai_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(300), nullable=False)
    area = Column(String(50), nullable=False)  # "student", "class", "subject", "teacher_support", "school_operations"
    priority = Column(String(20), nullable=False, default="medium")  # "critical", "high", "medium", "low"
    finding_summary = Column(Text, nullable=False)
    supporting_metrics = Column(JSON, nullable=True)
    time_period = Column(String(100), nullable=True)
    reason = Column(Text, nullable=True)  # Why this matters
    recommended_action = Column(Text, nullable=True)
    suggested_owner = Column(String(200), nullable=True)
    suggested_review_date = Column(DateTime, nullable=True)
    dependencies = Column(Text, nullable=True)
    expected_outcome = Column(Text, nullable=True)
    actual_outcome = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="new")  # new, reviewed, approved, in_progress, completed, dismissed
    source_type = Column(String(50), nullable=True)  # "ml_prediction", "analytics", "rule_based"
    source_id = Column(String(100), nullable=True)  # Reference to the source prediction or analysis
    assigned_to = Column(String(200), nullable=True)
    dismissed_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    history = relationship("RecommendationHistory", back_populates="recommendation", order_by="RecommendationHistory.created_at")

    __table_args__ = (
        Index("ix_recommendations_area", "area"),
        Index("ix_recommendations_status", "status"),
        Index("ix_recommendations_priority", "priority"),
    )


class RecommendationHistory(Base):
    """Audit trail for recommendation status changes."""
    __tablename__ = "ciis_ai_recommendation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(Integer, ForeignKey("ciis_ai_recommendations.id"), nullable=False)
    action = Column(String(50), nullable=False)  # e.g., "status_changed", "assigned", "outcome_recorded"
    old_value = Column(String(200), nullable=True)
    new_value = Column(String(200), nullable=True)
    changed_by = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    recommendation = relationship("Recommendation", back_populates="history")


# ─── Data Quality ─────────────────────────────────────────────────────────────

class DataQualityReport(Base):
    """Records data quality assessments for source data."""
    __tablename__ = "ciis_ai_data_quality"

    id = Column(Integer, primary_key=True, autoincrement=True)
    data_source = Column(String(100), nullable=False)  # e.g., "students", "attendance", "grades"
    total_records = Column(Integer, default=0)
    valid_records = Column(Integer, default=0)
    missing_fields = Column(JSON, nullable=True)  # {"field_name": count}
    duplicate_count = Column(Integer, default=0)
    quality_score = Column(Float, nullable=True)  # 0.0 to 1.0
    issues = Column(JSON, nullable=True)  # List of identified issues
    recommendations = Column(Text, nullable=True)
    assessed_at = Column(DateTime, default=utc_now)


# ─── Audit Log ────────────────────────────────────────────────────────────────

class AuditLog(Base):
    """Audit trail for significant AI system actions."""
    __tablename__ = "ciis_ai_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=True)
    entity_id = Column(String(50), nullable=True)
    user_id = Column(String(50), nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    __table_args__ = (
        Index("ix_audit_action", "action"),
        Index("ix_audit_created", "created_at"),
    )


# ─── Database Engine & Session ────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        if settings.database_url.startswith("sqlite"):
            _engine = create_engine(
                settings.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=settings.app_debug,
            )
        else:
            _engine = create_engine(
                settings.database_url,
                pool_size=5,
                max_overflow=10,
                echo=settings.app_debug,
            )
    return _engine


def get_session_factory():
    """Get the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db():
    """Dependency for getting database sessions."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Create all tables. Safe to call multiple times — only creates missing tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    return engine


def reset_engine():
    """Reset the engine (for testing)."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
