"""
Machine Learning Pipeline for CIIS School Intelligence AI.

This module handles:
1. Feature engineering from school data
2. Model training with proper train/test splitting
3. Model evaluation with comprehensive metrics
4. Model versioning and lifecycle management
5. Prediction inference with explainability

Models:
- Student Risk Prediction (Logistic Regression baseline + Random Forest)
- Enrollment Forecasting (Linear trend + seasonal)

Critical Rules:
- Never train and evaluate on the same data
- Never fabricate metrics or predictions
- Always validate before activating a model
- Use temporal splits for time-series predictions
- Handle class imbalance appropriately
"""

import json
import datetime
import logging
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score,
    roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from app.config import get_settings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ModelEvaluation:
    """Holds evaluation results for a trained model."""
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[list] = None
    classification_report: Optional[dict] = None
    feature_importances: Optional[dict] = None
    training_samples: int = 0
    test_samples: int = 0
    class_distribution: Optional[dict] = None
    notes: str = ""

    @property
    def meets_acceptance(self) -> bool:
        """Check if model meets minimum acceptance criteria."""
        settings = get_settings()
        return self.f1 >= settings.model_acceptance_f1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelArtifact:
    """Describes a saved model artifact."""
    model_name: str
    model_version: str
    model_type: str
    artifact_path: str
    feature_names: list
    evaluation: ModelEvaluation
    hyperparameters: dict
    training_data_period: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    status: str = "experimental"  # experimental, validated, active, retired


# ─── Feature Engineering ─────────────────────────────────────────────────────

class StudentFeatureEngineer:
    """
    Transforms raw school data into features for student risk prediction.

    Features derived:
    - Attendance rate (overall)
    - Late arrival rate
    - Absence trend (increasing/decreasing)
    - Average grade across subjects
    - Grade trend (improving/declining)
    - Number of subjects below passing
    - Grade standard deviation (consistency)
    - Exam vs quiz performance gap
    - Grade level (encoded)
    """

    PASSING_THRESHOLD = 75.0  # Configurable academic benchmark

    def __init__(self, passing_threshold: float = 75.0):
        self.passing_threshold = passing_threshold
        self.feature_names = []

    def build_features(
        self,
        students_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        grades_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build feature matrix from raw data.

        Args:
            students_df: Student records
            attendance_df: Attendance records
            grades_df: Grade/assessment records

        Returns:
            DataFrame with one row per student and engineered features
        """
        active_students = students_df[students_df["enrollment_status"] == "active"].copy()

        # Build attendance features
        attendance_features = self._attendance_features(attendance_df)

        # Build grade features
        grade_features = self._grade_features(grades_df)

        # Merge
        features = active_students[["student_id", "grade_level", "gender"]].copy()
        features = features.merge(attendance_features, on="student_id", how="left")
        features = features.merge(grade_features, on="student_id", how="left")

        # Encode categorical
        features["grade_level_encoded"] = features["grade_level"].astype(float)
        features["gender_encoded"] = (features["gender"] == "M").astype(float)

        # Define final feature columns
        self.feature_names = [
            "attendance_rate",
            "late_rate",
            "absence_trend",
            "avg_grade",
            "grade_std",
            "subjects_below_passing",
            "exam_quiz_gap",
            "grade_trend",
            "grade_level_encoded",
        ]

        # Fill NaN with sensible defaults
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0.0

        return features

    def _attendance_features(self, attendance_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-student attendance features."""
        if attendance_df is None or attendance_df.empty:
            return pd.DataFrame(columns=["student_id", "attendance_rate", "late_rate", "absence_trend"])

        features = []
        for sid, group in attendance_df.groupby("student_id"):
            total = len(group)
            present = len(group[group["status"] == "present"])
            late = len(group[group["status"] == "late"])
            absent = len(group[group["status"] == "absent"])

            attendance_rate = (present + late) / total if total > 0 else 0.0
            late_rate = late / total if total > 0 else 0.0

            # Absence trend: compare first half vs second half
            mid = total // 2
            if mid > 0:
                group_sorted = group.sort_values("date")
                first_half_absence = len(group_sorted.iloc[:mid][group_sorted.iloc[:mid]["status"] == "absent"])
                second_half_absence = len(group_sorted.iloc[mid:][group_sorted.iloc[mid:]["status"] == "absent"])
                # Positive = absences increasing, negative = decreasing
                absence_trend = (second_half_absence - first_half_absence) / max(mid, 1)
            else:
                absence_trend = 0.0

            features.append({
                "student_id": sid,
                "attendance_rate": round(attendance_rate, 4),
                "late_rate": round(late_rate, 4),
                "absence_trend": round(absence_trend, 4),
            })

        return pd.DataFrame(features)

    def _grade_features(self, grades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-student academic features."""
        if grades_df is None or grades_df.empty:
            return pd.DataFrame(columns=[
                "student_id", "avg_grade", "grade_std",
                "subjects_below_passing", "exam_quiz_gap", "grade_trend"
            ])

        features = []
        for sid, group in grades_df.groupby("student_id"):
            scores = group["percentage"].values
            avg_grade = float(np.mean(scores))
            grade_std = float(np.std(scores))

            # Subjects below passing
            subject_avgs = group.groupby("subject")["percentage"].mean()
            subjects_below = int((subject_avgs < self.passing_threshold).sum())

            # Exam vs quiz gap
            exams = group[group["assessment_type"].str.contains("Exam", case=False)]["percentage"]
            quizzes = group[~group["assessment_type"].str.contains("Exam", case=False)]["percentage"]
            exam_quiz_gap = float(exams.mean() - quizzes.mean()) if len(exams) > 0 and len(quizzes) > 0 else 0.0

            # Grade trend: compare earlier vs later assessments
            sorted_group = group.sort_values("assessment_order")
            mid = len(sorted_group) // 2
            if mid > 0:
                early_avg = sorted_group.iloc[:mid]["percentage"].mean()
                late_avg = sorted_group.iloc[mid:]["percentage"].mean()
                grade_trend = float(late_avg - early_avg)
            else:
                grade_trend = 0.0

            features.append({
                "student_id": sid,
                "avg_grade": round(avg_grade, 2),
                "grade_std": round(grade_std, 2),
                "subjects_below_passing": subjects_below,
                "exam_quiz_gap": round(exam_quiz_gap, 2),
                "grade_trend": round(grade_trend, 2),
            })

        return pd.DataFrame(features)


# ─── Target Label Definition ─────────────────────────────────────────────────

def define_risk_labels(
    features_df: pd.DataFrame,
    passing_threshold: float = 75.0,
    min_attendance: float = 0.80,
) -> pd.Series:
    """
    Define the target variable: 'needs_support' (1) or 'on_track' (0).

    A student is labeled as 'needs_support' if:
    1. Average grade is below the passing threshold, OR
    2. Attendance rate is below the minimum threshold, OR
    3. They have 3+ subjects below passing

    These are configurable, documented thresholds — not arbitrary values.

    Args:
        features_df: Feature matrix with attendance and grade features
        passing_threshold: Academic benchmark (default 75%)
        min_attendance: Minimum acceptable attendance rate (default 80%)

    Returns:
        Binary series: 1 = needs_support, 0 = on_track
    """
    needs_support = (
        (features_df["avg_grade"] < passing_threshold) |
        (features_df["attendance_rate"] < min_attendance) |
        (features_df["subjects_below_passing"] >= 3)
    ).astype(int)

    return needs_support


# ─── Model Training ──────────────────────────────────────────────────────────

class StudentRiskModelTrainer:
    """
    Trains and evaluates student risk prediction models.

    Implements:
    - Logistic Regression (interpretable baseline)
    - Random Forest (more complex alternative)
    - Proper train/test splitting
    - Stratified cross-validation
    - Class imbalance handling
    - Comprehensive evaluation metrics
    - Model comparison
    - Feature importance analysis
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.feature_engineer = StudentFeatureEngineer()
        self.models = {}
        self.evaluations = {}
        self.best_model_name = None

    def prepare_data(
        self,
        students_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        grades_df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> tuple:
        """
        Prepare feature matrix and labels with proper train/test split.

        Uses stratified split to maintain class distribution.
        """
        logger.info("Building features from raw data...")
        features = self.feature_engineer.build_features(students_df, attendance_df, grades_df)
        feature_names = self.feature_engineer.feature_names

        # Define labels
        labels = define_risk_labels(features)
        features["label"] = labels

        # Log class distribution
        class_dist = labels.value_counts().to_dict()
        logger.info(f"Class distribution: {class_dist}")
        logger.info(f"Total samples: {len(features)}, Features: {len(feature_names)}")

        # Check minimum samples
        settings = get_settings()
        if len(features) < settings.model_min_samples:
            raise ValueError(
                f"Insufficient data for training: {len(features)} samples "
                f"(minimum: {settings.model_min_samples}). "
                f"Collect more data before training a predictive model."
            )

        X = features[feature_names].values
        y = labels.values
        student_ids = features["student_id"].values

        # Stratified train/test split
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, student_ids,
            test_size=test_size,
            random_state=self.seed,
            stratify=y,
        )

        self._data = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "ids_train": ids_train, "ids_test": ids_test,
            "feature_names": feature_names,
            "class_distribution": class_dist,
            "features_df": features,
        }

        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        return self._data

    def train_logistic_regression(self) -> ModelEvaluation:
        """Train Logistic Regression baseline."""
        logger.info("Training Logistic Regression baseline...")

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                random_state=self.seed,
                class_weight="balanced",  # Handle class imbalance
                max_iter=1000,
                C=1.0,
            ))
        ])

        pipeline.fit(self._data["X_train"], self._data["y_train"])

        # Cross-validation on training data
        cv_scores = cross_val_score(
            pipeline, self._data["X_train"], self._data["y_train"],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed),
            scoring="f1"
        )
        logger.info(f"CV F1 scores: {cv_scores.round(3)}, Mean: {cv_scores.mean():.3f}")

        # Evaluate on held-out test set
        evaluation = self._evaluate_model(
            pipeline, "logistic_regression",
            self._data["X_test"], self._data["y_test"]
        )
        evaluation.notes = f"5-fold CV F1: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})"

        # Feature importances from coefficients
        coefs = pipeline.named_steps["classifier"].coef_[0]
        evaluation.feature_importances = {
            name: round(float(coef), 4)
            for name, coef in zip(self._data["feature_names"], coefs)
        }

        self.models["logistic_regression"] = pipeline
        self.evaluations["logistic_regression"] = evaluation

        return evaluation

    def train_random_forest(self) -> ModelEvaluation:
        """Train Random Forest for comparison."""
        logger.info("Training Random Forest...")

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            ))
        ])

        pipeline.fit(self._data["X_train"], self._data["y_train"])

        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, self._data["X_train"], self._data["y_train"],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed),
            scoring="f1"
        )
        logger.info(f"CV F1 scores: {cv_scores.round(3)}, Mean: {cv_scores.mean():.3f}")

        evaluation = self._evaluate_model(
            pipeline, "random_forest",
            self._data["X_test"], self._data["y_test"]
        )
        evaluation.notes = f"5-fold CV F1: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})"

        # Feature importances
        importances = pipeline.named_steps["classifier"].feature_importances_
        evaluation.feature_importances = {
            name: round(float(imp), 4)
            for name, imp in zip(self._data["feature_names"], importances)
        }

        self.models["random_forest"] = pipeline
        self.evaluations["random_forest"] = evaluation

        return evaluation

    def _evaluate_model(
        self,
        model,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelEvaluation:
        """
        Evaluate a model on held-out test data.

        Never evaluates on training data and reports it as performance.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        evaluation = ModelEvaluation(
            model_name=model_name,
            model_type=model_name,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
            confusion_matrix=cm,
            classification_report=report,
            training_samples=len(self._data["X_train"]),
            test_samples=len(X_test),
            class_distribution=self._data["class_distribution"],
        )

        logger.info(
            f"Model: {model_name} | "
            f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | F1: {f1:.3f} | "
            f"ROC-AUC: {roc_auc:.3f}" if roc_auc else
            f"Model: {model_name} | "
            f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | F1: {f1:.3f}"
        )

        return evaluation

    def compare_models(self) -> str:
        """Compare trained models and select the best one based on F1 score."""
        if not self.evaluations:
            raise ValueError("No models have been trained yet.")

        best_name = max(self.evaluations, key=lambda k: self.evaluations[k].f1)
        self.best_model_name = best_name

        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*60}")
        for name, eval in self.evaluations.items():
            marker = " ← BEST" if name == best_name else ""
            logger.info(
                f"  {name}: F1={eval.f1:.3f}, Precision={eval.precision:.3f}, "
                f"Recall={eval.recall:.3f}, AUC={eval.roc_auc}{marker}"
            )
        logger.info(f"{'='*60}\n")

        return best_name

    def save_model(self, model_name: str, version: str = "v1.0") -> ModelArtifact:
        """Save a trained model with metadata."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")

        settings = get_settings()
        model_dir = settings.model_path
        evaluation = self.evaluations[model_name]

        # Determine status based on acceptance criteria
        status = "validated" if evaluation.meets_acceptance else "experimental"

        # Save model artifact
        artifact_filename = f"{model_name}_{version}.joblib"
        artifact_path = model_dir / artifact_filename
        joblib.dump(self.models[model_name], artifact_path)

        # Save metadata
        artifact = ModelArtifact(
            model_name=model_name,
            model_version=version,
            model_type=model_name,
            artifact_path=str(artifact_path),
            feature_names=self._data["feature_names"],
            evaluation=evaluation,
            hyperparameters=self.models[model_name].get_params(),
            training_data_period="Demo data (synthetic)",
            status=status,
        )

        metadata_path = model_dir / f"{model_name}_{version}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(artifact.__dict__, f, indent=2, default=str)

        logger.info(f"Model saved: {artifact_path} (status: {status})")
        return artifact


# ─── Prediction Inference ─────────────────────────────────────────────────────

class StudentRiskPredictor:
    """
    Makes predictions using a trained model.

    Each prediction includes:
    - Risk level and probability
    - Contributing factors
    - Data quality notes
    - Model version
    """

    RISK_THRESHOLDS = {
        "high": 0.7,
        "medium": 0.4,
        "low": 0.0,
    }

    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        self.model = joblib.load(model_path)
        self.metadata = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        self.feature_names = self.metadata.get("feature_names", []) if self.metadata else []

    def predict(self, features_df: pd.DataFrame) -> list[dict]:
        """
        Generate predictions for a set of students.

        Args:
            features_df: DataFrame with required feature columns

        Returns:
            List of prediction dictionaries
        """
        if features_df.empty:
            return []

        feature_cols = self.feature_names
        if not feature_cols:
            raise ValueError("Feature names not available in model metadata.")

        # Validate required columns exist
        missing_cols = [c for c in feature_cols if c not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        X = features_df[feature_cols].values
        student_ids = features_df["student_id"].values if "student_id" in features_df.columns else range(len(X))

        # Get predictions and probabilities
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None

        predictions = []
        model_version = self.metadata.get("model_version", "unknown") if self.metadata else "unknown"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for i, (sid, pred) in enumerate(zip(student_ids, y_pred)):
            prob = float(y_proba[i]) if y_proba is not None else None

            # Determine risk level
            if prob is not None:
                if prob >= self.RISK_THRESHOLDS["high"]:
                    risk_level = "high"
                elif prob >= self.RISK_THRESHOLDS["medium"]:
                    risk_level = "medium"
                else:
                    risk_level = "low"
            else:
                risk_level = "high" if pred == 1 else "low"

            # Contributing factors (top features)
            contributing_factors = self._get_contributing_factors(features_df.iloc[i], feature_cols)

            prediction_label = "needs_support" if pred == 1 else "on_track"

            predictions.append({
                "student_id": str(sid),
                "prediction_label": prediction_label,
                "prediction_score": round(prob, 4) if prob is not None else None,
                "risk_level": risk_level,
                "contributing_factors": contributing_factors,
                "model_version": model_version,
                "predicted_at": now,
                "confidence_notes": self._confidence_notes(features_df.iloc[i], feature_cols),
            })

        return predictions

    def _get_contributing_factors(self, row: pd.Series, feature_cols: list) -> list[dict]:
        """Identify the most important factors for a prediction."""
        factors = []
        for col in feature_cols:
            val = row.get(col)
            if val is None or pd.isna(val):
                continue

            # Provide human-readable descriptions
            desc = self._feature_description(col, val)
            if desc:
                factors.append(desc)

        # Sort by relevance (those further from normal)
        factors.sort(key=lambda x: x.get("severity", 0), reverse=True)
        return factors[:5]  # Top 5 factors

    def _feature_description(self, col: str, val) -> Optional[dict]:
        """Convert a feature value to a human-readable factor description."""
        descriptions = {
            "attendance_rate": {
                "name": "Attendance Rate",
                "format": f"{val*100:.1f}%",
                "concern": val < 0.80,
                "severity": max(0, (0.90 - val) * 100) if val < 0.90 else 0,
            },
            "late_rate": {
                "name": "Late Arrival Rate",
                "format": f"{val*100:.1f}%",
                "concern": val > 0.10,
                "severity": max(0, (val - 0.05) * 100) if val > 0.05 else 0,
            },
            "absence_trend": {
                "name": "Absence Trend",
                "format": "Increasing" if val > 0.05 else "Stable" if abs(val) <= 0.05 else "Decreasing",
                "concern": val > 0.05,
                "severity": max(0, val * 50),
            },
            "avg_grade": {
                "name": "Average Grade",
                "format": f"{val:.1f}%",
                "concern": val < 75,
                "severity": max(0, (75 - val)) if val < 75 else 0,
            },
            "grade_std": {
                "name": "Grade Consistency",
                "format": f"±{val:.1f}",
                "concern": val > 15,
                "severity": max(0, (val - 10)) if val > 10 else 0,
            },
            "subjects_below_passing": {
                "name": "Subjects Below Passing",
                "format": f"{int(val)} subjects",
                "concern": val >= 2,
                "severity": val * 10,
            },
            "exam_quiz_gap": {
                "name": "Exam vs Quiz Gap",
                "format": f"{val:+.1f} points",
                "concern": val < -10,
                "severity": max(0, abs(val) - 5) if val < -5 else 0,
            },
            "grade_trend": {
                "name": "Grade Trend",
                "format": "Declining" if val < -3 else "Stable" if abs(val) <= 3 else "Improving",
                "concern": val < -3,
                "severity": max(0, abs(val) - 2) if val < -2 else 0,
            },
        }

        if col in descriptions:
            desc = descriptions[col]
            return {
                "factor": str(desc["name"]),
                "value": str(desc["format"]),
                "concern": bool(desc["concern"]),
                "severity": float(desc["severity"]),
            }
        return None

    def _confidence_notes(self, row: pd.Series, feature_cols: list) -> str:
        """Generate data quality / confidence notes."""
        missing = sum(1 for col in feature_cols if pd.isna(row.get(col)))
        if missing > 0:
            return f"Warning: {missing} of {len(feature_cols)} features had missing data. Prediction may be less reliable."
        return "All features available. Standard confidence."


# ─── Convenience Functions ────────────────────────────────────────────────────

def run_full_training_pipeline(
    students_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
    grades_df: pd.DataFrame,
    seed: int = 42,
) -> dict:
    """
    Run the complete training pipeline:
    1. Feature engineering
    2. Train Logistic Regression
    3. Train Random Forest
    4. Compare and select best
    5. Save models

    Returns:
        Dictionary with training results
    """
    trainer = StudentRiskModelTrainer(seed=seed)

    # Prepare data
    data = trainer.prepare_data(students_df, attendance_df, grades_df)

    # Train models
    lr_eval = trainer.train_logistic_regression()
    rf_eval = trainer.train_random_forest()

    # Compare
    best = trainer.compare_models()

    # Save both models
    lr_artifact = trainer.save_model("logistic_regression", "v1.0")
    rf_artifact = trainer.save_model("random_forest", "v1.0")

    results = {
        "best_model": best,
        "evaluations": {
            "logistic_regression": lr_eval.to_dict(),
            "random_forest": rf_eval.to_dict(),
        },
        "artifacts": {
            "logistic_regression": lr_artifact.__dict__,
            "random_forest": rf_artifact.__dict__,
        },
        "feature_names": data["feature_names"],
        "class_distribution": {str(k): int(v) for k, v in data["class_distribution"].items()},
        "training_samples": len(data["X_train"]),
        "test_samples": len(data["X_test"]),
    }

    return results


# ─── Teacher Support & Instructional Improvement ML Pipeline ──────────────────

class TeacherFeatureEngineer:
    """
    Transforms teacher profiles, class assignments, attendance, and assessment data
    into features for instructional support and professional development modeling.

    Ethical Rules:
    - Analyzes workload, multi-prep demands, class size, and score dispersion.
    - Never used for punitive evaluation or raw score teacher rankings.
    - Focuses on identifying where institutional support, co-planning, and mentorship
      will have the highest positive impact on teaching and learning.
    """

    FEATURE_NAMES = [
        "sections_count",
        "distinct_subjects_count",
        "teaching_out_of_specialization",
        "total_students_taught",
        "avg_class_size",
        "avg_student_attendance",
        "class_pass_rate",
        "grade_std_dev",
        "grade_trajectory",
        "years_experience",
        "is_early_career",
    ]

    def __init__(self, passing_threshold: float = 75.0):
        self.passing_threshold = passing_threshold
        self.feature_names = list(self.FEATURE_NAMES)

    def build_features(
        self,
        teachers_df: pd.DataFrame,
        classes_df: pd.DataFrame,
        grades_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build feature matrix for each teacher.

        Args:
            teachers_df: Teacher demographic and specialization records
            classes_df: Class assignments mapping teachers to (grade_level, section, subject)
            grades_df: Assessment records
            attendance_df: Student attendance records

        Returns:
            DataFrame with one row per teacher and engineered workload/classroom features
        """
        if teachers_df is None or teachers_df.empty:
            return pd.DataFrame(columns=["teacher_id"] + self.feature_names)

        active_teachers = teachers_df[teachers_df["status"] == "active"].copy()
        if active_teachers.empty:
            active_teachers = teachers_df.copy()

        features_list = []

        # Precompute section student counts from grades or attendance
        section_students = {}
        if attendance_df is not None and not attendance_df.empty:
            for (gl, sec), grp in attendance_df.groupby(["grade_level", "section"]):
                section_students[(gl, sec)] = grp["student_id"].nunique()

        for _, teacher in active_teachers.iterrows():
            tid = teacher["teacher_id"]
            spec = str(teacher.get("specialization", "")).strip().lower()
            years_exp = float(teacher.get("years_experience", 5))

            # Classes assigned to this teacher
            t_classes = classes_df[classes_df["teacher_id"] == tid] if (classes_df is not None and not classes_df.empty) else pd.DataFrame()

            if t_classes.empty:
                # Teacher currently without assigned classes
                features_list.append({
                    "teacher_id": tid,
                    "first_name": teacher.get("first_name", ""),
                    "last_name": teacher.get("last_name", ""),
                    "specialization": teacher.get("specialization", ""),
                    "sections_count": 0,
                    "distinct_subjects_count": 0,
                    "teaching_out_of_specialization": 0.0,
                    "total_students_taught": 0,
                    "avg_class_size": 0.0,
                    "avg_student_attendance": 0.90,
                    "class_pass_rate": 75.0,
                    "grade_std_dev": 8.0,
                    "grade_trajectory": 0.0,
                    "years_experience": years_exp,
                    "is_early_career": 1.0 if years_exp <= 3 else 0.0,
                })
                continue

            sections_set = set(zip(t_classes["grade_level"], t_classes["section"]))
            sections_count = len(sections_set)
            subjects_list = t_classes["subject"].dropna().unique().tolist()
            distinct_subjects = len(subjects_list)

            # Check if teaching out of specialization
            out_of_spec = 0.0
            for subj in subjects_list:
                if str(subj).strip().lower() != spec:
                    out_of_spec = 1.0
                    break

            # Calculate total students taught across assigned sections
            students_count = sum(section_students.get(sec_key, 25) for sec_key in sections_set)
            avg_class_size = round(students_count / max(sections_count, 1), 1)

            # Match grades belonging to this teacher's classes: (grade_level, section, subject)
            t_grades = []
            if grades_df is not None and not grades_df.empty:
                for _, cls_row in t_classes.iterrows():
                    match = grades_df[
                        (grades_df["grade_level"] == cls_row["grade_level"]) &
                        (grades_df["section"] == cls_row["section"]) &
                        (grades_df["subject"] == cls_row["subject"])
                    ]
                    if not match.empty:
                        t_grades.append(match)

            if t_grades:
                t_grades_df = pd.concat(t_grades, ignore_index=True)
                scores = t_grades_df["percentage"].dropna().values
                class_pass_rate = round(float((scores >= self.passing_threshold).mean() * 100), 1) if len(scores) > 0 else 75.0
                grade_std = round(float(np.std(scores)), 2) if len(scores) > 1 else 8.0

                # Trajectory: compare early vs late assessments in these classes
                if "assessment_order" in t_grades_df.columns:
                    sorted_grades = t_grades_df.sort_values("assessment_order")
                    mid = len(sorted_grades) // 2
                    if mid > 0:
                        early_avg = sorted_grades.iloc[:mid]["percentage"].mean()
                        late_avg = sorted_grades.iloc[mid:]["percentage"].mean()
                        grade_trajectory = round(float(late_avg - early_avg), 2)
                    else:
                        grade_trajectory = 0.0
                else:
                    grade_trajectory = 0.0
            else:
                class_pass_rate = 75.0
                grade_std = 8.0
                grade_trajectory = 0.0

            # Match attendance in this teacher's sections
            t_att_rates = []
            if attendance_df is not None and not attendance_df.empty:
                for (gl, sec) in sections_set:
                    sec_att = attendance_df[(attendance_df["grade_level"] == gl) & (attendance_df["section"] == sec)]
                    if not sec_att.empty:
                        total_att = len(sec_att)
                        present_late = len(sec_att[sec_att["status"].isin(["present", "late"])])
                        if total_att > 0:
                            t_att_rates.append(present_late / total_att)

            avg_attendance = round(float(np.mean(t_att_rates)), 4) if t_att_rates else 0.88

            features_list.append({
                "teacher_id": tid,
                "first_name": teacher.get("first_name", ""),
                "last_name": teacher.get("last_name", ""),
                "specialization": teacher.get("specialization", ""),
                "sections_count": int(sections_count),
                "distinct_subjects_count": int(distinct_subjects),
                "teaching_out_of_specialization": float(out_of_spec),
                "total_students_taught": int(students_count),
                "avg_class_size": float(avg_class_size),
                "avg_student_attendance": float(avg_attendance),
                "class_pass_rate": float(class_pass_rate),
                "grade_std_dev": float(grade_std),
                "grade_trajectory": float(grade_trajectory),
                "years_experience": float(years_exp),
                "is_early_career": 1.0 if years_exp <= 3 else 0.0,
            })

        df = pd.DataFrame(features_list)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        return df


def define_teacher_support_labels(features_df: pd.DataFrame) -> pd.Series:
    """
    Define objective target variable for instructional support recommendation.
    1 = Targeted Instructional Support / Co-Planning Recommended
    0 = Standard Support / On Track

    Pedagogical Criteria:
    - High multi-prep load (3+ distinct subjects to prepare weekly) OR heavy section load (>=7 sections)
    - Early career complexity (experience <= 3 years) combined with either out-of-specialization subject,
      high class score dispersion (std >= 13), or >= 5 sections
    - Meaningful class score decline (trajectory <= -2.5) combined with pass rate < 70%
    - Low classroom attendance environment (attendance < 82%) needing school-wide engagement support
    """
    needs_support = (
        (features_df["distinct_subjects_count"] >= 3) |
        (features_df["sections_count"] >= 7) |
        (
            (features_df["is_early_career"] == 1.0) &
            (
                (features_df["teaching_out_of_specialization"] == 1.0) |
                (features_df["grade_std_dev"] >= 13.0) |
                (features_df["sections_count"] >= 5)
            )
        ) |
        (
            (features_df["grade_trajectory"] <= -2.5) &
            (features_df["class_pass_rate"] < 70.0)
        ) |
        (features_df["avg_student_attendance"] < 0.82)
    ).astype(int)

    return needs_support


class TeacherSupportModelTrainer:
    """
    Trains, evaluates, and versions Machine Learning models for teacher instructional support.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.models = {}
        self.evaluations = {}
        self.best_model_name = None
        self._data = {}

    def prepare_data(
        self,
        teachers_df: pd.DataFrame,
        classes_df: pd.DataFrame,
        grades_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        test_size: float = 0.25,
    ) -> dict:
        """Engineer features, create balanced labels, and partition training/validation sets."""
        engineer = TeacherFeatureEngineer()
        features_df = engineer.build_features(teachers_df, classes_df, grades_df, attendance_df)

        y = define_teacher_support_labels(features_df)
        X = features_df[engineer.feature_names].copy()

        # Handle small faculty size with stratified split
        stratify_col = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=stratify_col
        )

        self._data = {
            "features_df": features_df,
            "feature_names": engineer.feature_names,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "class_distribution": y.value_counts().to_dict(),
        }
        return self._data

    def train_logistic_regression(self) -> ModelEvaluation:
        """Train baseline Logistic Regression model."""
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=self.seed, class_weight="balanced", max_iter=1000)),
        ])

        pipeline.fit(self._data["X_train"], self._data["y_train"])
        self.models["logistic_regression"] = pipeline

        evaluation = self._evaluate_model("logistic_regression", pipeline)
        self.evaluations["logistic_regression"] = evaluation
        return evaluation

    def train_random_forest(self) -> ModelEvaluation:
        """Train Random Forest model."""
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_split=2,
                class_weight="balanced",
                random_state=self.seed,
            )),
        ])

        pipeline.fit(self._data["X_train"], self._data["y_train"])
        self.models["random_forest"] = pipeline

        evaluation = self._evaluate_model("random_forest", pipeline)
        self.evaluations["random_forest"] = evaluation
        return evaluation

    def _evaluate_model(self, name: str, model: Pipeline) -> ModelEvaluation:
        """Evaluate model on held-out test data."""
        X_test = self._data["X_test"]
        y_test = self._data["y_test"]

        y_pred = model.predict(X_test)
        has_proba = hasattr(model, "predict_proba")
        y_proba = model.predict_proba(X_test)[:, 1] if has_proba else None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        roc_auc = None
        if y_proba is not None and len(np.unique(y_test)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_test, y_proba))
            except Exception:
                roc_auc = None

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Feature importances
        feature_names = self._data["feature_names"]
        importances = {}
        if name == "random_forest":
            raw_imp = model.named_steps["classifier"].feature_importances_
            importances = {fn: round(float(imp), 4) for fn, imp in zip(feature_names, raw_imp)}
        elif name == "logistic_regression":
            coefs = abs(model.named_steps["classifier"].coef_[0])
            importances = {fn: round(float(c), 4) for fn, imp, c in zip(feature_names, coefs, coefs)}

        sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        return ModelEvaluation(
            model_name=name,
            model_type=f"teacher_support_{name}",
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            roc_auc=round(roc_auc, 4) if roc_auc else None,
            confusion_matrix=cm,
            classification_report=report,
            feature_importances=sorted_imp,
            training_samples=len(self._data["X_train"]),
            test_samples=len(X_test),
            class_distribution={str(k): int(v) for k, v in self._data["class_distribution"].items()},
            notes="Teacher support model evaluated on held-out faculty records",
        )

    def compare_models(self) -> str:
        """Select best performing model based on F1 score."""
        best_name = None
        best_f1 = -1.0
        for name, ev in self.evaluations.items():
            if ev.f1 > best_f1:
                best_f1 = ev.f1
                best_name = name
        self.best_model_name = best_name or "random_forest"
        return self.best_model_name

    def save_model(self, model_name: str, version: str = "v1.0") -> ModelArtifact:
        """Save trained model artifact and JSON metadata."""
        settings = get_settings()
        model_dir = Path(settings.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = model_dir / f"teacher_support_{model_name}_{version}.joblib"
        joblib.dump(self.models[model_name], artifact_path)

        evaluation = self.evaluations[model_name]
        status = "validated" if evaluation.meets_acceptance else "experimental"

        artifact = ModelArtifact(
            model_name=f"teacher_support_{model_name}",
            model_version=version,
            model_type=f"teacher_support_{model_name}",
            artifact_path=str(artifact_path),
            feature_names=self._data["feature_names"],
            evaluation=evaluation,
            hyperparameters=self.models[model_name].get_params(),
            training_data_period="Faculty Academic Term (synthetic)",
            status=status,
        )

        metadata_path = model_dir / f"teacher_support_{model_name}_{version}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(artifact.__dict__, f, indent=2, default=str)

        return artifact


class TeacherSupportPredictor:
    """
    Inference and diagnostic engine for Teacher Support Intelligence.

    For each teacher, produces:
    - Support priority & calibrated probability
    - Top workload / classroom factors
    - Actionable "What to Improve" instructional guidance
    """

    SUPPORT_THRESHOLDS = {
        "high": 0.65,
        "medium": 0.35,
        "low": 0.0,
    }

    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        self.model = joblib.load(model_path)
        self.metadata = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        self.feature_names = self.metadata.get("feature_names", []) if self.metadata else TeacherFeatureEngineer.FEATURE_NAMES

    def predict(self, features_df: pd.DataFrame) -> list[dict]:
        """Generate support diagnostics and improvement recommendations for teachers."""
        if features_df is None or features_df.empty:
            return []

        missing = [c for c in self.feature_names if c not in features_df.columns]
        if missing:
            raise ValueError(f"Missing required teacher feature columns: {missing}")

        X = features_df[self.feature_names].values
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None

        predictions = []
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        model_version = self.metadata.get("model_version", "v1.0") if self.metadata else "v1.0"

        for i, pred in enumerate(y_pred):
            row = features_df.iloc[i]
            prob = float(y_proba[i]) if y_proba is not None else (0.8 if pred == 1 else 0.2)

            if prob >= self.SUPPORT_THRESHOLDS["high"]:
                support_priority = "high"
                support_category = "Targeted Instructional Support Recommended"
            elif prob >= self.SUPPORT_THRESHOLDS["medium"]:
                support_priority = "medium"
                support_category = "Review & Co-Planning Recommended"
            else:
                support_priority = "low"
                support_category = "On Track / Peer Mentor Candidate"

            factors = self._extract_factors(row)
            improvement_plan = self._generate_improvement_plan(row, support_priority)

            predictions.append({
                "teacher_id": str(row["teacher_id"]),
                "first_name": str(row.get("first_name", "")),
                "last_name": str(row.get("last_name", "")),
                "full_name": f"{row.get('first_name', '')} {row.get('last_name', '')}".strip() or str(row["teacher_id"]),
                "specialization": str(row.get("specialization", "General")),
                "years_experience": float(row.get("years_experience", 0)),
                "sections_count": int(row.get("sections_count", 0)),
                "distinct_subjects_count": int(row.get("distinct_subjects_count", 0)),
                "total_students_taught": int(row.get("total_students_taught", 0)),
                "avg_class_size": float(row.get("avg_class_size", 0)),
                "class_pass_rate": float(row.get("class_pass_rate", 75.0)),
                "grade_std_dev": float(row.get("grade_std_dev", 8.0)),
                "grade_trajectory": float(row.get("grade_trajectory", 0.0)),
                "avg_student_attendance": round(float(row.get("avg_student_attendance", 0.88)) * 100, 1),
                "support_priority": support_priority,
                "support_category": support_category,
                "support_probability": round(prob, 4),
                "contributing_factors": factors,
                "improvement_plan": improvement_plan,
                "prediction_timestamp": now,
                "model_version": model_version,
                "ethical_disclaimer": "This analytical diagnostic is designed for institutional support and professional development planning. It is not an evaluation of personal competence.",
            })

        # Sort with highest support priority first
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        predictions.sort(key=lambda x: (priority_rank.get(x["support_priority"], 3), -x["support_probability"]))
        return predictions

    def _extract_factors(self, row: pd.Series) -> list[dict]:
        """Extract key workload and classroom condition factors with severity metrics."""
        factors = []

        distinct_subjs = int(row.get("distinct_subjects_count", 0))
        if distinct_subjs >= 3:
            factors.append({
                "factor": "High Multi-Prep Burden",
                "value": f"{distinct_subjs} distinct subjects to prepare",
                "concern": True,
                "severity": float(distinct_subjs * 25),
                "category": "workload",
            })

        sections = int(row.get("sections_count", 0))
        if sections >= 6:
            factors.append({
                "factor": "Heavy Section Load",
                "value": f"{sections} assigned class sections",
                "concern": True,
                "severity": float(sections * 15),
                "category": "workload",
            })

        if float(row.get("teaching_out_of_specialization", 0)) == 1.0:
            factors.append({
                "factor": "Out-of-Specialization Assignment",
                "value": f"Assigned classes outside core specialization ({row.get('specialization', 'Subject')})",
                "concern": True,
                "severity": 40.0,
                "category": "curriculum",
            })

        std_dev = float(row.get("grade_std_dev", 0))
        if std_dev >= 14.0:
            factors.append({
                "factor": "Wide Classroom Ability Dispersion",
                "value": f"±{std_dev:.1f} grade variance across students",
                "concern": True,
                "severity": float((std_dev - 10) * 10),
                "category": "classroom_diversity",
            })

        traj = float(row.get("grade_trajectory", 0))
        if traj <= -2.5:
            factors.append({
                "factor": "Declining Assessment Trajectory",
                "value": f"{traj:+.1f} points between mid and late assessments",
                "concern": True,
                "severity": float(abs(traj) * 15),
                "category": "learning_pacing",
            })

        att = float(row.get("avg_student_attendance", 0))
        if att < 0.85:
            factors.append({
                "factor": "Classroom Attendance Barrier",
                "value": f"{att*100:.1f}% average student attendance",
                "concern": True,
                "severity": float((0.90 - att) * 200),
                "category": "engagement",
            })

        exp = float(row.get("years_experience", 0))
        if exp <= 3:
            factors.append({
                "factor": "Early Career Faculty",
                "value": f"{int(exp)} year(s) of teaching experience",
                "concern": False,
                "severity": 10.0,
                "category": "experience",
            })

        factors.sort(key=lambda x: x["severity"], reverse=True)
        return factors[:4]

    def _generate_improvement_plan(self, row: pd.Series, support_priority: str) -> dict:
        """
        Generate actionable diagnostic detailing WHAT to improve, HOW to improve,
        and WHAT institutional support/PD to provide.
        """
        distinct_subjs = int(row.get("distinct_subjects_count", 0))
        sections = int(row.get("sections_count", 0))
        out_of_spec = float(row.get("teaching_out_of_specialization", 0)) == 1.0
        std_dev = float(row.get("grade_std_dev", 0))
        traj = float(row.get("grade_trajectory", 0))
        att = float(row.get("avg_student_attendance", 0))
        exp = float(row.get("years_experience", 0))
        pass_rate = float(row.get("class_pass_rate", 75.0))
        spec = str(row.get("specialization", "General"))

        # Determine Primary Improvement Focus
        if distinct_subjs >= 3:
            focus_area = "Workload & Multi-Prep Lesson Management"
            actions = [
                "Streamline lesson planning by sharing standard curriculum pacing guides with department peers.",
                f"Reduce distinct prep count in upcoming semester schedule from {distinct_subjs} to 2 or fewer.",
                "Provide access to shared departmental digital teaching repositories and quiz banks.",
            ]
            pd_workshop = "Effective Multi-Subject Curriculum Planning and Time Management"
            target_metric = "Reduce weekly lesson prep hours by 30% and improve class pacing continuity"

        elif std_dev >= 14.0 or pass_rate < 70.0:
            focus_area = "Differentiated Instruction & Tiered Assessment"
            actions = [
                f"Deploy tiered assignment rubrics to support the wide student ability spread (±{std_dev:.1f} score variance).",
                "Implement 10-minute weekly formative check-ins to catch struggling students before major exams.",
                "Utilize student peer-tutoring groups within mixed-ability sections.",
            ]
            pd_workshop = "Differentiated Instruction Strategies for Mixed-Ability Classrooms"
            target_metric = f"Narrow grade standard deviation to under ±10.0 and boost class pass rate to ≥80%"

        elif out_of_spec and exp <= 4:
            focus_area = "Subject Mentorship & Co-Teaching Collaboration"
            actions = [
                f"Pair with a Senior {spec} or Subject Department Head for bi-weekly curriculum alignment.",
                "Conduct two peer classroom observations per semester to exchange instructional strategies.",
                "Provide supplementary pedagogical content knowledge (PCK) teaching modules.",
            ]
            pd_workshop = "Pedagogical Content Knowledge and Secondary Subject Mastery"
            target_metric = "Achieve full curriculum milestone completion and confidence in out-of-specialization topics"

        elif traj <= -2.0 or att < 0.85:
            focus_area = "Student Engagement & Mid-Semester Remediation Pacing"
            actions = [
                "Review assessment pacing between mid-term and finals to prevent topic overload.",
                "Introduce interactive review sessions and gamified formative checks prior to unit exams.",
                "Collaborate with school guidance on attendance follow-ups for chronically absent students.",
            ]
            pd_workshop = "Active Learning and Classroom Engagement Techniques"
            target_metric = "Reverse negative assessment trajectory to ≥ +2.0 points and elevate attendance above 90%"

        elif exp >= 8 and support_priority == "low":
            focus_area = "Peer Coaching & Instructional Leadership"
            actions = [
                "Invite early-career faculty to observe high-performing classroom management techniques.",
                "Lead or co-facilitate departmental professional learning communities (PLC).",
                "Share exemplary lesson plans and formative assessment rubrics school-wide.",
            ]
            pd_workshop = "Instructional Coaching and Peer Mentorship Leadership"
            target_metric = "Mentor 1-2 junior teachers per academic year and lead 1 department improvement initiative"

        else:
            focus_area = "Continuous Formative Assessment & Academic Alignment"
            actions = [
                "Continue standard curriculum pacing and quarterly syllabus reviews.",
                "Incorporate student self-assessment rubrics at the conclusion of each unit.",
                "Participate in cross-grade articulation meetings with subject colleagues.",
            ]
            pd_workshop = "Modern Formative Assessment and Authentic Student Feedback"
            target_metric = "Maintain consistent class achievement (≥80% pass rate) across all assigned sections"

        return {
            "primary_focus_area": focus_area,
            "specific_actions": actions,
            "recommended_pd_workshop": pd_workshop,
            "target_measurable_outcome": target_metric,
            "suggested_review_timeline": "6-8 weeks (Mid-Semester Review)",
        }


def run_teacher_training_pipeline(
    teachers_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    grades_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
    seed: int = 42,
) -> dict:
    """
    Run complete training pipeline for teacher instructional support models.

    Returns:
        Dictionary with evaluations, best model, artifacts, and feature metadata
    """
    trainer = TeacherSupportModelTrainer(seed=seed)
    data = trainer.prepare_data(teachers_df, classes_df, grades_df, attendance_df)

    lr_eval = trainer.train_logistic_regression()
    rf_eval = trainer.train_random_forest()
    best = trainer.compare_models()

    lr_artifact = trainer.save_model("logistic_regression", "v1.0")
    rf_artifact = trainer.save_model("random_forest", "v1.0")

    return {
        "best_model": best,
        "evaluations": {
            "logistic_regression": lr_eval.to_dict(),
            "random_forest": rf_eval.to_dict(),
        },
        "artifacts": {
            "logistic_regression": lr_artifact.__dict__,
            "random_forest": rf_artifact.__dict__,
        },
        "feature_names": data["feature_names"],
        "class_distribution": {str(k): int(v) for k, v in data["class_distribution"].items()},
        "training_samples": len(data["X_train"]),
        "test_samples": len(data["X_test"]),
    }

