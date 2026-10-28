"""
Demo/Synthetic Data Generator for CIIS School Intelligence AI.

Generates realistic but entirely synthetic school data for development and testing.
No real student data is used. All identifiers are pseudonymous.

Data Generated:
- Students with enrollment info
- Teachers with assignments
- Classes and subjects
- Attendance records
- Exam/assessment scores
- Academic periods
"""

import random
import datetime
import pandas as pd
import numpy as np
from typing import Optional


# Reproducible random state
SEED = 42


def _set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# ─── Configuration ────────────────────────────────────────────────────────────

GRADE_LEVELS = [7, 8, 9, 10, 11, 12]
SECTIONS = ["A", "B", "C"]
SUBJECTS = [
    "Mathematics", "English", "Science", "Filipino", "Social Studies",
    "Computer Science", "Physical Education", "Values Education",
    "Music & Arts", "Technology & Livelihood Education"
]
ACADEMIC_YEARS = ["2023-2024", "2024-2025", "2025-2026"]
SEMESTERS = ["First Semester", "Second Semester"]

FIRST_NAMES = [
    "Maria", "Juan", "Ana", "Carlos", "Sofia", "Miguel", "Isabella", "Gabriel",
    "Camila", "Diego", "Valentina", "Alejandro", "Lucia", "Daniel", "Elena",
    "Santiago", "Victoria", "Sebastian", "Natalia", "Mateo", "Andrea", "Nicolas",
    "Paula", "Fernando", "Catalina", "Rafael", "Mariana", "Emilio", "Daniela",
    "Antonio", "Jimena", "Oscar", "Fernanda", "Eduardo", "Regina", "Rodrigo",
    "Valeria", "Luis", "Renata", "Jorge", "Ximena", "Manuel", "Lorena",
    "Francisco", "Carolina", "Ricardo", "Adriana", "Roberto", "Monica",
    "Arturo", "Veronica", "Hector", "Diana", "Raul", "Gloria", "Jose",
    "Teresa", "Pedro", "Silvia", "Marco", "Rosa", "Alberto", "Laura",
    "Enrique", "Patricia", "Cesar", "Claudia", "Gustavo", "Alejandra",
]

LAST_NAMES = [
    "Santos", "Reyes", "Cruz", "Garcia", "Torres", "Flores", "Rivera",
    "Ramos", "Lopez", "Gonzales", "Bautista", "Mendoza", "Castillo",
    "Aquino", "Dela Cruz", "Villanueva", "Fernandez", "Martinez",
    "Rodriguez", "Hernandez", "Morales", "Diaz", "Perez", "Romero",
    "Salazar", "Aguilar", "Navarro", "Espinoza", "Guerrero", "Ortiz",
]

TEACHER_FIRST_NAMES = [
    "Mark", "Jennifer", "Robert", "Catherine", "James", "Elizabeth",
    "David", "Sarah", "Michael", "Christine", "Raymond", "Grace",
    "Benjamin", "Mary", "Peter", "Agnes", "Paul", "Teresa",
    "John", "Rachel", "Richard", "Anne", "Dennis", "Joy",
]


def generate_students(n_students: int = 300, seed: int = SEED) -> pd.DataFrame:
    """
    Generate synthetic student records.

    Each student has:
    - Pseudonymous ID (STU-XXXX)
    - Name (synthetic)
    - Grade level
    - Section
    - Enrollment status
    - Enrollment date
    - Gender (for demographic analytics only)

    Returns:
        DataFrame with student records
    """
    _set_seed(seed)

    students = []
    for i in range(1, n_students + 1):
        grade = random.choice(GRADE_LEVELS)
        section = random.choice(SECTIONS[:2] if grade >= 11 else SECTIONS)
        enrollment_year = random.choice([2022, 2023, 2024, 2025])
        enrollment_date = datetime.date(enrollment_year, random.randint(5, 8), random.randint(1, 28))

        # ~5% withdrawn, ~3% transferred
        status_roll = random.random()
        if status_roll < 0.05:
            status = "withdrawn"
        elif status_roll < 0.08:
            status = "transferred"
        else:
            status = "active"

        students.append({
            "student_id": f"STU-{i:04d}",
            "first_name": random.choice(FIRST_NAMES),
            "last_name": random.choice(LAST_NAMES),
            "grade_level": grade,
            "section": section,
            "gender": random.choice(["M", "F"]),
            "enrollment_status": status,
            "enrollment_date": enrollment_date.isoformat(),
        })

    return pd.DataFrame(students)


def generate_teachers(n_teachers: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic teacher records."""
    _set_seed(seed)

    teachers = []
    for i in range(1, n_teachers + 1):
        specialization = random.choice(SUBJECTS)
        years_exp = random.randint(1, 25)

        teachers.append({
            "teacher_id": f"TCH-{i:04d}",
            "first_name": random.choice(TEACHER_FIRST_NAMES),
            "last_name": random.choice(LAST_NAMES),
            "specialization": specialization,
            "years_experience": years_exp,
            "status": "active" if random.random() > 0.05 else "on_leave",
        })

    return pd.DataFrame(teachers)


def generate_classes(
    students_df: pd.DataFrame,
    teachers_df: pd.DataFrame,
    seed: int = SEED
) -> pd.DataFrame:
    """Generate class/section assignments."""
    _set_seed(seed)

    classes = []
    class_id = 0

    active_teachers = teachers_df[teachers_df["status"] == "active"]["teacher_id"].tolist()

    for grade in GRADE_LEVELS:
        sections_for_grade = SECTIONS[:2] if grade >= 11 else SECTIONS
        for section in sections_for_grade:
            for subject in SUBJECTS:
                class_id += 1
                teacher = random.choice(active_teachers)

                classes.append({
                    "class_id": f"CLS-{class_id:04d}",
                    "grade_level": grade,
                    "section": section,
                    "subject": subject,
                    "teacher_id": teacher,
                    "academic_year": "2025-2026",
                    "semester": "First Semester",
                    "max_capacity": random.randint(30, 45),
                })

    return pd.DataFrame(classes)


def generate_attendance(
    students_df: pd.DataFrame,
    n_days: int = 120,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Generate synthetic attendance records.

    Creates realistic patterns:
    - Most students attend regularly (>90%)
    - Some students have moderate absence (~15-25%)
    - A few students have high absence (>30%)
    - Some late arrivals
    """
    _set_seed(seed)

    active_students = students_df[students_df["enrollment_status"] == "active"]
    records = []

    # Assign attendance profiles
    student_profiles = {}
    for _, student in active_students.iterrows():
        roll = random.random()
        if roll < 0.70:
            # Regular attendee: 90-98% attendance
            profile = {"attend_rate": random.uniform(0.90, 0.98), "late_rate": random.uniform(0.02, 0.08)}
        elif roll < 0.88:
            # Moderate absence: 75-89%
            profile = {"attend_rate": random.uniform(0.75, 0.89), "late_rate": random.uniform(0.05, 0.15)}
        elif roll < 0.95:
            # Frequent absence: 60-74%
            profile = {"attend_rate": random.uniform(0.60, 0.74), "late_rate": random.uniform(0.10, 0.20)}
        else:
            # Chronic absence: 40-59%
            profile = {"attend_rate": random.uniform(0.40, 0.59), "late_rate": random.uniform(0.15, 0.30)}
        student_profiles[student["student_id"]] = profile

    # Generate daily records
    start_date = datetime.date(2025, 6, 9)  # Start of school year
    school_days = []
    current = start_date
    day_count = 0
    while day_count < n_days:
        if current.weekday() < 5:  # Monday-Friday
            school_days.append(current)
            day_count += 1
        current += datetime.timedelta(days=1)

    for _, student in active_students.iterrows():
        sid = student["student_id"]
        profile = student_profiles[sid]

        for day in school_days:
            roll = random.random()
            if roll < profile["attend_rate"]:
                if random.random() < profile["late_rate"]:
                    status = "late"
                else:
                    status = "present"
            else:
                status = "absent"

            records.append({
                "student_id": sid,
                "date": day.isoformat(),
                "status": status,
                "grade_level": student["grade_level"],
                "section": student["section"],
            })

    return pd.DataFrame(records)


def generate_grades(
    students_df: pd.DataFrame,
    seed: int = SEED
) -> pd.DataFrame:
    """
    Generate synthetic exam/assessment scores.

    Creates realistic grade distributions with correlations:
    - Students have a base ability level
    - Performance varies by subject
    - Some subjects are harder for some students
    - Grades span multiple assessment types and periods
    """
    _set_seed(seed)

    active_students = students_df[students_df["enrollment_status"] == "active"]
    records = []

    assessment_types = ["Quiz 1", "Quiz 2", "Midterm Exam", "Quiz 3", "Quiz 4", "Final Exam"]
    assessment_weights = [0.10, 0.10, 0.25, 0.10, 0.10, 0.35]

    for _, student in active_students.iterrows():
        sid = student["student_id"]
        grade_level = student["grade_level"]

        # Base ability: normally distributed
        base_ability = np.clip(np.random.normal(78, 12), 40, 100)

        for subject in SUBJECTS:
            # Subject affinity: some students are better at certain subjects
            subject_modifier = np.random.normal(0, 8)

            for i, assessment in enumerate(assessment_types):
                # Score based on base ability + subject affinity + noise
                base_score = base_ability + subject_modifier + np.random.normal(0, 5)

                # Higher assessments are slightly harder
                if "Exam" in assessment:
                    base_score -= np.random.uniform(2, 6)

                # Clip to valid range
                score = float(np.clip(base_score, 25, 100))

                # Max possible score
                max_score = 100.0

                records.append({
                    "student_id": sid,
                    "subject": subject,
                    "grade_level": grade_level,
                    "section": student["section"],
                    "assessment_type": assessment,
                    "assessment_order": i + 1,
                    "score": round(score, 1),
                    "max_score": max_score,
                    "percentage": round(score / max_score * 100, 1),
                    "academic_year": "2025-2026",
                    "semester": "First Semester",
                })

    return pd.DataFrame(records)


def generate_all_demo_data(
    n_students: int = 300,
    n_teachers: int = 30,
    n_attendance_days: int = 120,
    seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """
    Generate a complete set of synthetic school data.

    Returns:
        Dictionary of DataFrames keyed by data type
    """
    print("Generating synthetic school data for development/testing...")
    print(f"  Random seed: {seed}")

    students = generate_students(n_students, seed)
    print(f"  Students: {len(students)} records")

    teachers = generate_teachers(n_teachers, seed)
    print(f"  Teachers: {len(teachers)} records")

    classes = generate_classes(students, teachers, seed)
    print(f"  Classes: {len(classes)} records")

    attendance = generate_attendance(students, n_attendance_days, seed)
    print(f"  Attendance: {len(attendance)} records")

    grades = generate_grades(students, seed)
    print(f"  Grades: {len(grades)} records")

    data = {
        "students": students,
        "teachers": teachers,
        "classes": classes,
        "attendance": attendance,
        "grades": grades,
    }

    print("Demo data generation complete.")
    return data


def get_data_summary(data: dict[str, pd.DataFrame]) -> dict:
    """Generate a data audit summary."""
    summary = {}
    for name, df in data.items():
        missing = df.isnull().sum().to_dict()
        missing_fields = {k: int(v) for k, v in missing.items() if v > 0}

        summary[name] = {
            "total_records": len(df),
            "columns": list(df.columns),
            "missing_fields": missing_fields,
            "duplicates": int(df.duplicated().sum()),
            "quality_score": round(1.0 - (sum(missing.values()) / (len(df) * len(df.columns))), 4)
            if len(df) > 0 else 0.0,
        }

    return summary


if __name__ == "__main__":
    # Quick test
    data = generate_all_demo_data()
    summary = get_data_summary(data)
    for name, info in summary.items():
        print(f"\n{name}:")
        for k, v in info.items():
            print(f"  {k}: {v}")
