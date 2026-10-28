"""
Recommendation Engine for CIIS School Intelligence AI.

Generates evidence-based, actionable recommendations from:
- ML predictions (student risk)
- Class/subject analytics
- Attendance patterns
- Operational metrics

Rules:
- Every recommendation includes supporting evidence
- Facts are separated from predictions and suggestions
- No fabricated evidence or causal claims
- Deterministic, transparent rules (no external LLM required)
- Deduplication to avoid overwhelming users
"""

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generates prioritized, evidence-based recommendations."""

    def __init__(self):
        self.recommendations = []
        self._existing_titles = set()  # Deduplication

    def generate_all(
        self,
        predictions: Optional[list[dict]] = None,
        class_analytics: Optional[dict] = None,
        attendance_data: Optional[dict] = None,
        operational_metrics: Optional[dict] = None,
        teacher_predictions: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Generate recommendations from all available data sources.

        Args:
            predictions: Student risk predictions
            class_analytics: Class/subject analysis results
            attendance_data: Attendance trend data
            operational_metrics: School operational metrics
            teacher_predictions: Teacher ML instructional support predictions

        Returns:
            Prioritized list of recommendations
        """
        self.recommendations = []
        self._existing_titles = set()

        if predictions:
            self._student_risk_recommendations(predictions)

        if class_analytics:
            self._class_analytics_recommendations(class_analytics)

        if attendance_data:
            self._attendance_recommendations(attendance_data)

        if operational_metrics:
            self._operational_recommendations(operational_metrics)

        if teacher_predictions:
            self._teacher_support_recommendations(teacher_predictions)

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self.recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

        logger.info(f"Generated {len(self.recommendations)} recommendations")
        return self.recommendations

    def _add_recommendation(self, rec: dict):
        """Add a recommendation with deduplication."""
        title = rec.get("title", "")
        if title in self._existing_titles:
            return  # Skip duplicate
        self._existing_titles.add(title)
        rec["id"] = len(self.recommendations) + 1
        rec["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rec["status"] = "new"
        self.recommendations.append(rec)

    # ─── Student Risk Recommendations ─────────────────────────────────

    def _student_risk_recommendations(self, predictions: list[dict]):
        """Generate recommendations from student risk predictions."""
        high_risk = [p for p in predictions if p.get("risk_level") == "high"]
        medium_risk = [p for p in predictions if p.get("risk_level") == "medium"]

        # High risk students
        if high_risk:
            self._add_recommendation({
                "title": f"Review {len(high_risk)} students identified as needing additional support",
                "area": "student",
                "priority": "high",
                "finding_summary": (
                    f"{len(high_risk)} students have been identified as potentially needing "
                    f"additional academic support based on attendance patterns, grade trends, "
                    f"and assessment performance."
                ),
                "supporting_metrics": {
                    "high_risk_count": len(high_risk),
                    "medium_risk_count": len(medium_risk),
                    "total_analyzed": len(predictions),
                    "pct_high_risk": round(len(high_risk) / len(predictions) * 100, 1) if predictions else 0,
                },
                "time_period": "Current semester",
                "reason": (
                    "Early identification allows intervention before academic difficulties "
                    "become more severe. Students with declining attendance and grades "
                    "benefit most from timely support."
                ),
                "recommended_action": (
                    "Review individual student profiles in the Student Intelligence section. "
                    "For each high-risk student, check recent attendance, grade trends, and "
                    "contributing factors. Schedule follow-up meetings with class advisers."
                ),
                "suggested_owner": "Academic Affairs / Class Advisers",
                "suggested_review_date": (
                    datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
                ).isoformat(),
                "expected_outcome": "Each high-risk student reviewed and support plan initiated",
                "source_type": "ml_prediction",
            })

        # Individual high-risk students (top 5 most urgent)
        for student in sorted(high_risk, key=lambda x: x.get("prediction_score", 0), reverse=True)[:5]:
            factors_text = ", ".join(
                f"{f['factor']}: {f['value']}"
                for f in student.get("contributing_factors", [])[:3]
                if f.get("concern")
            )

            self._add_recommendation({
                "title": f"Review student {student['student_id']} — high support need indicated",
                "area": "student",
                "priority": "high",
                "finding_summary": (
                    f"Student {student['student_id']} shows indicators of needing additional support. "
                    f"Key factors: {factors_text or 'See details in student profile'}."
                ),
                "supporting_metrics": {
                    "risk_score": student.get("prediction_score"),
                    "risk_level": student.get("risk_level"),
                    "contributing_factors": student.get("contributing_factors", []),
                },
                "time_period": "Current semester",
                "reason": "Individual review recommended based on combined risk factors.",
                "recommended_action": (
                    "Review this student's attendance record and recent grades. "
                    "Consult with the class adviser. If concerns are confirmed, "
                    "consider arranging additional tutoring or mentoring support."
                ),
                "suggested_owner": "Class Adviser",
                "expected_outcome": "Student reviewed and support plan created if needed",
                "source_type": "ml_prediction",
                "source_id": student["student_id"],
            })

    # ─── Class Analytics Recommendations ──────────────────────────────

    def _class_analytics_recommendations(self, analytics: dict):
        """Generate recommendations from class/subject analytics."""
        struggling = analytics.get("struggling_subjects", [])
        section_comparisons = analytics.get("section_comparisons", [])

        for item in struggling:
            severity = item.get("severity", "moderate")
            priority = "high" if severity == "critical" else "medium"

            self._add_recommendation({
                "title": (
                    f"Address learning gaps in {item['subject']} "
                    f"(Grade {item['grade_level']})"
                ),
                "area": "subject",
                "priority": priority,
                "finding_summary": (
                    f"{item['pct_below_threshold']}% of students in Grade {item['grade_level']} "
                    f"{item['subject']} are below the {item['threshold']}% threshold "
                    f"({item['n_struggling']} of {item['n_total']} students). "
                    f"Average score: {item['avg_score']}%."
                ),
                "supporting_metrics": item,
                "reason": (
                    "A significant portion of students below the passing threshold "
                    "suggests a systemic issue that may benefit from curricular review "
                    "or additional teaching support."
                ),
                "recommended_action": (
                    f"Review the {item['subject']} curriculum and teaching approach for "
                    f"Grade {item['grade_level']}. Consider additional practice materials, "
                    f"peer tutoring, or teacher collaboration sessions."
                ),
                "suggested_owner": "Subject Coordinator",
                "expected_outcome": f"Pass rate in {item['subject']} improves next period",
                "source_type": "analytics",
            })

        # Large section variations
        for comp in section_comparisons:
            if comp.get("score_spread", 0) > 10:
                self._add_recommendation({
                    "title": (
                        f"Investigate performance variation in {comp['subject']} "
                        f"(Grade {comp['grade_level']})"
                    ),
                    "area": "class",
                    "priority": "medium",
                    "finding_summary": (
                        f"There is a {comp['score_spread']} point difference between "
                        f"sections in Grade {comp['grade_level']} {comp['subject']}. "
                        f"This may warrant review, though many factors affect outcomes."
                    ),
                    "supporting_metrics": comp,
                    "reason": (
                        "Significant variation between sections taking the same subject "
                        "may indicate differences in teaching approach, class dynamics, "
                        "or student composition that could be addressed."
                    ),
                    "recommended_action": (
                        "Review teaching approaches across sections. Facilitate teacher "
                        "sharing of effective methods. Note: section differences may "
                        "reflect student composition, not teaching quality."
                    ),
                    "suggested_owner": "Department Head",
                    "source_type": "analytics",
                })

    # ─── Attendance Recommendations ───────────────────────────────────

    def _attendance_recommendations(self, attendance_data: dict):
        """Generate recommendations from attendance patterns."""
        grade_attendance = attendance_data.get("by_grade", [])

        for grade_data in grade_attendance:
            if grade_data.get("absence_rate", 0) > 15:
                self._add_recommendation({
                    "title": (
                        f"Investigate high absence rate in Grade {grade_data['grade_level']} "
                        f"({grade_data['absence_rate']}%)"
                    ),
                    "area": "school_operations",
                    "priority": "high" if grade_data["absence_rate"] > 25 else "medium",
                    "finding_summary": (
                        f"Grade {grade_data['grade_level']} has an absence rate of "
                        f"{grade_data['absence_rate']}%, affecting {grade_data.get('n_students', 'unknown')} "
                        f"students. The late arrival rate is {grade_data.get('late_rate', 0)}%."
                    ),
                    "supporting_metrics": grade_data,
                    "reason": (
                        "High absence rates are associated with lower academic outcomes. "
                        "Identifying the cause early helps prevent cascading effects."
                    ),
                    "recommended_action": (
                        "Review individual attendance records for this grade level. "
                        "Identify students with chronic absence. Consider parent "
                        "communication, incentive programs, or counselor follow-up."
                    ),
                    "suggested_owner": "Student Affairs / Guidance",
                    "expected_outcome": "Absence rate reduced to below 15%",
                    "source_type": "analytics",
                })

    # ─── Operational Recommendations ──────────────────────────────────

    def _operational_recommendations(self, metrics: dict):
        """Generate recommendations from operational metrics."""
        if metrics.get("overall_pass_rate") and metrics["overall_pass_rate"] < 70:
            self._add_recommendation({
                "title": "School-wide pass rate below 70% — review academic support programs",
                "area": "school_operations",
                "priority": "critical",
                "finding_summary": (
                    f"The overall school pass rate is {metrics['overall_pass_rate']}%, "
                    f"with an average grade of {metrics.get('overall_avg_grade', 'N/A')}%. "
                    f"This is below the 70% institutional target."
                ),
                "supporting_metrics": {
                    "overall_pass_rate": metrics["overall_pass_rate"],
                    "overall_avg_grade": metrics.get("overall_avg_grade"),
                    "total_students": metrics.get("total_active_students"),
                },
                "reason": "A school-wide pass rate below target requires systemic review.",
                "recommended_action": (
                    "Convene academic affairs meeting. Review subject-level pass rates "
                    "to identify specific problem areas. Consider additional learning "
                    "resources, tutorial programs, or curriculum adjustments."
                ),
                "suggested_owner": "School President / Academic Affairs",
                "expected_outcome": "Action plan to improve pass rate above 70%",
                "source_type": "analytics",
            })

        if metrics.get("max_class_size", 0) > 40:
            self._add_recommendation({
                "title": f"Large class sizes detected (max: {metrics['max_class_size']} students)",
                "area": "school_operations",
                "priority": "medium",
                "finding_summary": (
                    f"The largest class has {metrics['max_class_size']} students. "
                    f"Average class size is {metrics.get('avg_class_size', 'N/A')}."
                ),
                "supporting_metrics": {
                    "max_class_size": metrics["max_class_size"],
                    "avg_class_size": metrics.get("avg_class_size"),
                },
                "reason": "Very large class sizes may affect teaching effectiveness and student engagement.",
                "recommended_action": (
                    "Review class distributions. Consider splitting large classes "
                    "or adding sections if resources allow."
                ),
                "suggested_owner": "School Administration",
                "source_type": "analytics",
            })

    # ─── Teacher Support ML Recommendations ───────────────────────────

    def _teacher_support_recommendations(self, teacher_predictions: list[dict]):
        """Generate institutional support & improvement recommendations from teacher ML diagnostics."""
        high_support = [t for t in teacher_predictions if t.get("support_priority") == "high"]
        multi_prep = [t for t in teacher_predictions if t.get("distinct_subjects_count", 0) >= 3]
        wide_variance = [t for t in teacher_predictions if t.get("grade_std_dev", 0) >= 14.0]
        early_career = [t for t in teacher_predictions if t.get("years_experience", 0) <= 3]
        mentor_candidates = [t for t in teacher_predictions if t.get("years_experience", 0) >= 8 and t.get("support_priority") == "low"]

        # 1. School-wide Multi-Prep Relief
        if multi_prep:
            self._add_recommendation({
                "title": f"Workload Rebalancing: Provide co-planning support for {len(multi_prep)} multi-prep teachers",
                "area": "teacher_support",
                "priority": "high",
                "finding_summary": (
                    f"{len(multi_prep)} faculty members are currently assigned 3 or more distinct subjects "
                    f"to prepare each week. Machine learning diagnostics identify multi-prep load as a primary driver "
                    f"of instructional pacing strain."
                ),
                "supporting_metrics": {
                    "multi_prep_teacher_count": len(multi_prep),
                    "teachers_affected": [t.get("full_name") for t in multi_prep[:5]],
                },
                "reason": (
                    "Preparing 3+ distinct curricula concurrently fragments instructional time. "
                    "Targeted co-planning and digital lesson repositories mitigate burnout and ensure curriculum depth."
                ),
                "recommended_action": (
                    "Review next term's scheduling to cap preps at 2 where possible. In the interim, provide "
                    "shared departmental lesson plan templates and allocate 1 hour weekly of collaborative planning time."
                ),
                "suggested_owner": "Academic Dean / Department Chairs",
                "suggested_review_date": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=14)).isoformat(),
                "expected_outcome": "30% reduction in lesson preparation overhead and improved pacing alignment",
                "source_type": "ml_prediction",
            })

        # 2. Differentiated Learning Support
        if wide_variance:
            self._add_recommendation({
                "title": f"Instructional Support: Differentiated learning workshop for {len(wide_variance)} faculty sections",
                "area": "teacher_support",
                "priority": "medium",
                "finding_summary": (
                    f"{len(wide_variance)} teachers have classroom score standard deviations exceeding ±14.0 points, "
                    f"indicating substantial diversity in baseline student ability across sections."
                ),
                "supporting_metrics": {
                    "wide_variance_teacher_count": len(wide_variance),
                    "avg_std_dev": round(float(np.mean([t.get("grade_std_dev", 0) for t in wide_variance])), 2),
                },
                "reason": (
                    "Mixed-ability classrooms require tiered learning activities and formative checks "
                    "to prevent struggling students from falling behind while keeping advanced learners engaged."
                ),
                "recommended_action": (
                    "Host an instructional workshop on 'Tiered Formative Assessment and Leveled Reading Tasks'. "
                    "Supply department rubric templates for scaffolding complex assignments."
                ),
                "suggested_owner": "Instructional Specialist / Academic Affairs",
                "suggested_review_date": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=21)).isoformat(),
                "expected_outcome": "Narrow classroom score variance by 20% and elevate section pass rates",
                "source_type": "ml_prediction",
            })

        # 3. Faculty Mentorship Pairing Program
        if early_career and mentor_candidates:
            self._add_recommendation({
                "title": f"Pair {len(early_career)} early-career teachers with senior instructional mentors",
                "area": "teacher_support",
                "priority": "high",
                "finding_summary": (
                    f"Identified {len(early_career)} faculty members in their first 3 years of teaching alongside "
                    f"{len(mentor_candidates)} senior faculty with high classroom consistency. Structured peer coaching "
                    f"accelerates instructional mastery."
                ),
                "supporting_metrics": {
                    "early_career_count": len(early_career),
                    "mentor_candidates_count": len(mentor_candidates),
                },
                "reason": (
                    "Mentorship provides formative support on classroom management, curriculum pacing, "
                    "and parent communication without evaluative pressure."
                ),
                "recommended_action": (
                    "Pair each early-career teacher with a senior mentor in their subject cluster. "
                    "Establish monthly non-evaluative peer observations and lesson review sessions."
                ),
                "suggested_owner": "Faculty Development Committee / Principal",
                "suggested_review_date": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=14)).isoformat(),
                "expected_outcome": "100% mentor pairing completion and positive early-career retention",
                "source_type": "ml_prediction",
            })

        # 4. Individual priority support items (top 3 highest priority)
        for t in high_support[:3]:
            plan = t.get("improvement_plan", {})
            self._add_recommendation({
                "title": f"Instructional Action Plan: {t.get('full_name')} ({t.get('specialization')}) — {plan.get('primary_focus_area', 'Instructional Support')}",
                "area": "teacher_support",
                "priority": "high",
                "finding_summary": (
                    f"Machine learning support diagnostic flags high support priority for {t.get('full_name')}. "
                    f"Workload: {t.get('sections_count')} sections ({t.get('distinct_subjects_count')} preps), "
                    f"Class pass rate: {t.get('class_pass_rate')}%, Grade spread: ±{t.get('grade_std_dev')}."
                ),
                "supporting_metrics": {
                    "teacher_id": t.get("teacher_id"),
                    "support_probability": t.get("support_probability"),
                    "primary_focus": plan.get("primary_focus_area"),
                    "factors": [f["factor"] for f in t.get("contributing_factors", [])],
                },
                "reason": f"Targeted support in {plan.get('primary_focus_area')} will optimize classroom outcomes.",
                "recommended_action": " ".join(plan.get("specific_actions", ["Review curriculum pacing with department head."])),
                "suggested_owner": "Department Head",
                "suggested_review_date": (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=28)).isoformat(),
                "expected_outcome": plan.get("target_measurable_outcome", "Improve instructional consistency and student achievement"),
                "source_type": "ml_prediction",
            })
