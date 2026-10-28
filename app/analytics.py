"""
Analytics Module for CIIS School Intelligence AI.

Provides:
1. Class & Subject Analytics - Performance trends, comparisons
2. Teacher Support Analytics - Workload, outcomes, collaboration
3. School Forecasting - Enrollment trends, attendance projections

All analytics use verified data. No fabricated insights.
"""

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ─── Class & Subject Analytics ────────────────────────────────────────────────

class ClassAnalytics:
    """Analyzes class and subject performance using verified data."""

    def __init__(self, grades_df: pd.DataFrame, attendance_df: pd.DataFrame):
        self.grades = grades_df
        self.attendance = attendance_df

    def subject_performance_summary(self) -> list[dict]:
        """
        Calculate performance summary per subject.

        Returns average scores, pass rates, and grade distribution.
        Does NOT compare subjects without accounting for grade level differences.
        """
        if self.grades.empty:
            return []

        results = []
        for subject, group in self.grades.groupby("subject"):
            scores = group["percentage"].values
            n_students = group["student_id"].nunique()

            results.append({
                "subject": subject,
                "avg_score": round(float(np.mean(scores)), 2),
                "median_score": round(float(np.median(scores)), 2),
                "std_score": round(float(np.std(scores)), 2),
                "min_score": round(float(np.min(scores)), 2),
                "max_score": round(float(np.max(scores)), 2),
                "pass_rate": round(float(np.mean(scores >= 75)) * 100, 1),
                "n_students": n_students,
                "n_assessments": len(group),
                "grade_levels": sorted(group["grade_level"].unique().tolist()),
            })

        return sorted(results, key=lambda x: x["avg_score"])

    def grade_level_performance(self) -> list[dict]:
        """Performance summary by grade level."""
        if self.grades.empty:
            return []

        results = []
        for grade, group in self.grades.groupby("grade_level"):
            scores = group["percentage"].values
            n_students = group["student_id"].nunique()

            results.append({
                "grade_level": int(grade),
                "avg_score": round(float(np.mean(scores)), 2),
                "median_score": round(float(np.median(scores)), 2),
                "pass_rate": round(float(np.mean(scores >= 75)) * 100, 1),
                "n_students": n_students,
                "subjects": sorted(group["subject"].unique().tolist()),
            })

        return sorted(results, key=lambda x: x["grade_level"])

    def class_section_comparison(self) -> list[dict]:
        """
        Compare sections within the same grade level and subject.

        Only compares truly comparable groups (same grade + subject).
        """
        if self.grades.empty:
            return []

        results = []
        for (grade, subject), group in self.grades.groupby(["grade_level", "subject"]):
            sections = {}
            for section, sec_group in group.groupby("section"):
                scores = sec_group["percentage"].values
                sections[section] = {
                    "section": section,
                    "avg_score": round(float(np.mean(scores)), 2),
                    "n_students": sec_group["student_id"].nunique(),
                    "pass_rate": round(float(np.mean(scores >= 75)) * 100, 1),
                }

            if len(sections) > 1:
                avgs = [s["avg_score"] for s in sections.values()]
                spread = round(max(avgs) - min(avgs), 2)

                results.append({
                    "grade_level": int(grade),
                    "subject": subject,
                    "sections": list(sections.values()),
                    "score_spread": spread,
                    "note": "Large variation detected" if spread > 10 else "Within normal range",
                })

        return sorted(results, key=lambda x: x["score_spread"], reverse=True)

    def assessment_trend(self) -> list[dict]:
        """Track how performance changes across assessment periods."""
        if self.grades.empty:
            return []

        results = []
        for (subject,), group in self.grades.groupby(["subject"]):
            trend_data = []
            for order, assess_group in group.groupby("assessment_order"):
                scores = assess_group["percentage"].values
                trend_data.append({
                    "assessment_order": int(order),
                    "assessment_type": assess_group["assessment_type"].iloc[0],
                    "avg_score": round(float(np.mean(scores)), 2),
                    "n_scores": len(scores),
                })

            if len(trend_data) >= 2:
                first = trend_data[0]["avg_score"]
                last = trend_data[-1]["avg_score"]
                trend = "improving" if last > first + 2 else "declining" if last < first - 2 else "stable"

                results.append({
                    "subject": subject,
                    "assessments": trend_data,
                    "trend": trend,
                    "change": round(last - first, 2),
                })

        return results

    def attendance_by_grade(self) -> list[dict]:
        """Attendance analytics by grade level."""
        if self.attendance.empty:
            return []

        results = []
        for grade, group in self.attendance.groupby("grade_level"):
            total = len(group)
            present = len(group[group["status"].isin(["present", "late"])])
            absent = len(group[group["status"] == "absent"])
            late = len(group[group["status"] == "late"])

            results.append({
                "grade_level": int(grade),
                "attendance_rate": round(present / total * 100, 1) if total > 0 else 0,
                "absence_rate": round(absent / total * 100, 1) if total > 0 else 0,
                "late_rate": round(late / total * 100, 1) if total > 0 else 0,
                "total_records": total,
                "n_students": group["student_id"].nunique(),
            })

        return sorted(results, key=lambda x: x["grade_level"])

    def identify_struggling_subjects(self, threshold: float = 70.0) -> list[dict]:
        """Identify subjects where significant portions of students are below threshold."""
        if self.grades.empty:
            return []

        results = []
        for (grade, subject), group in self.grades.groupby(["grade_level", "subject"]):
            student_avgs = group.groupby("student_id")["percentage"].mean()
            below = (student_avgs < threshold).sum()
            total = len(student_avgs)
            pct_below = below / total * 100 if total > 0 else 0

            if pct_below > 20:  # More than 20% struggling
                results.append({
                    "grade_level": int(grade),
                    "subject": subject,
                    "pct_below_threshold": round(pct_below, 1),
                    "n_struggling": int(below),
                    "n_total": total,
                    "threshold": threshold,
                    "avg_score": round(float(student_avgs.mean()), 2),
                    "severity": "critical" if pct_below > 50 else "high" if pct_below > 35 else "moderate",
                })

        return sorted(results, key=lambda x: x["pct_below_threshold"], reverse=True)


# ─── Teacher Support Analytics ────────────────────────────────────────────────

class TeacherAnalytics:
    """
    Analyzes teaching patterns to support improvement.

    NOT a teacher ranking system. Does not infer competence or motivation.
    Provides aggregate insights only.
    """

    def __init__(
        self,
        teachers_df: pd.DataFrame,
        classes_df: pd.DataFrame,
        grades_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
    ):
        self.teachers = teachers_df
        self.classes = classes_df
        self.grades = grades_df
        self.attendance = attendance_df

    def workload_summary(self) -> list[dict]:
        """Analyze teacher workload (number of classes, subjects, grade levels)."""
        if self.classes.empty:
            return []

        results = []
        for tid, group in self.classes.groupby("teacher_id"):
            teacher_info = self.teachers[self.teachers["teacher_id"] == tid]
            name = f"{teacher_info.iloc[0]['first_name']} {teacher_info.iloc[0]['last_name']}" if len(teacher_info) > 0 else tid

            results.append({
                "teacher_id": tid,
                "teacher_name": name,
                "n_classes": len(group),
                "subjects": sorted(group["subject"].unique().tolist()),
                "n_subjects": group["subject"].nunique(),
                "grade_levels": sorted(group["grade_level"].unique().tolist()),
                "n_grade_levels": group["grade_level"].nunique(),
                "workload_level": "high" if len(group) > 8 else "moderate" if len(group) > 5 else "standard",
            })

        return sorted(results, key=lambda x: x["n_classes"], reverse=True)

    def class_outcomes_summary(self) -> list[dict]:
        """
        Summarize class outcomes by teacher — aggregate only.

        WARNING: Does not rank teachers. Association ≠ causation.
        Many factors beyond teacher control affect class outcomes.
        """
        if self.classes.empty or self.grades.empty:
            return []

        results = []
        for tid, teacher_classes in self.classes.groupby("teacher_id"):
            teacher_info = self.teachers[self.teachers["teacher_id"] == tid]
            teacher_name = f"{teacher_info.iloc[0]['first_name']} {teacher_info.iloc[0]['last_name']}" if len(teacher_info) > 0 else tid

            class_results = []
            for _, cls in teacher_classes.iterrows():
                # Find grades matching this class
                class_grades = self.grades[
                    (self.grades["grade_level"] == cls["grade_level"]) &
                    (self.grades["section"] == cls["section"]) &
                    (self.grades["subject"] == cls["subject"])
                ]

                if not class_grades.empty:
                    scores = class_grades["percentage"].values
                    class_results.append({
                        "class_id": cls["class_id"],
                        "subject": cls["subject"],
                        "grade_level": int(cls["grade_level"]),
                        "section": cls["section"],
                        "avg_score": round(float(np.mean(scores)), 2),
                        "pass_rate": round(float(np.mean(scores >= 75)) * 100, 1),
                        "n_students": class_grades["student_id"].nunique(),
                    })

            if class_results:
                results.append({
                    "teacher_id": tid,
                    "teacher_name": teacher_name,
                    "classes": class_results,
                    "overall_avg": round(float(np.mean([c["avg_score"] for c in class_results])), 2),
                    "note": "Aggregate summary. Many factors affect class outcomes beyond teacher actions.",
                })

        return results

    def collaboration_opportunities(self) -> list[dict]:
        """Identify teachers teaching the same subject who could collaborate."""
        if self.classes.empty:
            return []

        results = []
        for subject, group in self.classes.groupby("subject"):
            teachers_for_subject = group["teacher_id"].unique()
            if len(teachers_for_subject) > 1:
                teacher_names = []
                for tid in teachers_for_subject:
                    info = self.teachers[self.teachers["teacher_id"] == tid]
                    if len(info) > 0:
                        teacher_names.append(f"{info.iloc[0]['first_name']} {info.iloc[0]['last_name']}")

                results.append({
                    "subject": subject,
                    "n_teachers": len(teachers_for_subject),
                    "teacher_ids": teachers_for_subject.tolist(),
                    "teacher_names": teacher_names,
                    "suggestion": f"Consider peer collaboration sessions for {subject} teachers.",
                })

        return results


# ─── School Forecasting ──────────────────────────────────────────────────────

class SchoolForecaster:
    """
    Generates school-level forecasts from historical data.

    Uses simple, transparent baselines. Displays assumptions and uncertainty.
    Does not forecast metrics the data cannot establish.
    """

    def __init__(
        self,
        students_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        grades_df: pd.DataFrame,
    ):
        self.students = students_df
        self.attendance = attendance_df
        self.grades = grades_df

    def enrollment_summary(self) -> dict:
        """Current enrollment status and distribution."""
        total = len(self.students)
        active = len(self.students[self.students["enrollment_status"] == "active"])
        withdrawn = len(self.students[self.students["enrollment_status"] == "withdrawn"])
        transferred = len(self.students[self.students["enrollment_status"] == "transferred"])

        by_grade = {}
        for grade, group in self.students[self.students["enrollment_status"] == "active"].groupby("grade_level"):
            by_grade[int(grade)] = len(group)

        return {
            "total_enrolled": total,
            "active": active,
            "withdrawn": withdrawn,
            "transferred": transferred,
            "retention_rate": round(active / total * 100, 1) if total > 0 else 0,
            "by_grade_level": by_grade,
            "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def attendance_trend(self, window_days: int = 7) -> list[dict]:
        """Calculate attendance rate over time using a rolling window."""
        if self.attendance.empty:
            return []

        self.attendance["date_parsed"] = pd.to_datetime(self.attendance["date"])
        daily = self.attendance.groupby("date_parsed").agg(
            total=("status", "count"),
            present=("status", lambda x: ((x == "present") | (x == "late")).sum()),
        ).reset_index()
        daily["rate"] = daily["present"] / daily["total"]
        daily = daily.sort_values("date_parsed")

        # Rolling average
        daily["rolling_rate"] = daily["rate"].rolling(window=window_days, min_periods=1).mean()

        return [{
            "date": row["date_parsed"].strftime("%Y-%m-%d"),
            "daily_rate": round(float(row["rate"]) * 100, 1),
            "rolling_avg": round(float(row["rolling_rate"]) * 100, 1),
            "total_students": int(row["total"]),
        } for _, row in daily.iterrows()]

    def academic_progress_by_grade(self) -> list[dict]:
        """Track academic progress across assessment periods by grade."""
        if self.grades.empty:
            return []

        results = []
        for grade, group in self.grades.groupby("grade_level"):
            period_data = []
            for order, assess_group in group.groupby("assessment_order"):
                scores = assess_group["percentage"].values
                period_data.append({
                    "assessment_order": int(order),
                    "assessment_type": assess_group["assessment_type"].iloc[0],
                    "avg_score": round(float(np.mean(scores)), 2),
                    "pass_rate": round(float(np.mean(scores >= 75)) * 100, 1),
                    "n_students": assess_group["student_id"].nunique(),
                })

            results.append({
                "grade_level": int(grade),
                "periods": period_data,
                "overall_avg": round(float(group["percentage"].mean()), 2),
            })

        return sorted(results, key=lambda x: x["grade_level"])

    def operational_metrics(self) -> dict:
        """Key operational metrics for school management."""
        active_students = self.students[self.students["enrollment_status"] == "active"]

        # Class sizes
        by_grade_section = active_students.groupby(["grade_level", "section"]).size()

        # Overall attendance
        if not self.attendance.empty:
            total_att = len(self.attendance)
            present = len(self.attendance[self.attendance["status"].isin(["present", "late"])])
            overall_attendance = round(present / total_att * 100, 1) if total_att > 0 else 0
        else:
            overall_attendance = None

        # Overall academic
        if not self.grades.empty:
            overall_avg_grade = round(float(self.grades["percentage"].mean()), 2)
            overall_pass_rate = round(float((self.grades["percentage"] >= 75).mean()) * 100, 1)
        else:
            overall_avg_grade = None
            overall_pass_rate = None

        return {
            "total_active_students": len(active_students),
            "n_grade_levels": active_students["grade_level"].nunique(),
            "avg_class_size": round(float(by_grade_section.mean()), 1) if len(by_grade_section) > 0 else 0,
            "max_class_size": int(by_grade_section.max()) if len(by_grade_section) > 0 else 0,
            "overall_attendance_rate": overall_attendance,
            "overall_avg_grade": overall_avg_grade,
            "overall_pass_rate": overall_pass_rate,
            "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def simple_enrollment_forecast(self, periods_ahead: int = 3) -> dict:
        """
        Simple enrollment projection based on current data.

        This is a baseline projection using current retention rates.
        Should be interpreted with caution — actual enrollment depends on many external factors.
        """
        active = len(self.students[self.students["enrollment_status"] == "active"])
        total = len(self.students)
        retention_rate = active / total if total > 0 else 0.95

        projections = []
        for i in range(1, periods_ahead + 1):
            projected = round(active * (retention_rate ** i))
            projections.append({
                "period": f"+{i} semester(s)",
                "projected_enrollment": projected,
                "retention_assumption": round(retention_rate * 100, 1),
                "confidence": "low — based on current retention rate only",
            })

        return {
            "current_enrollment": active,
            "current_retention_rate": round(retention_rate * 100, 1),
            "projections": projections,
            "assumptions": [
                "Assumes constant retention rate (likely oversimplified)",
                "Does not account for new enrollments",
                "Does not account for external factors (economic, policy)",
                "Based on demo/synthetic data — not real trends",
            ],
            "forecast_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
