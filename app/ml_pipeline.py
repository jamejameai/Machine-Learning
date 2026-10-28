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
