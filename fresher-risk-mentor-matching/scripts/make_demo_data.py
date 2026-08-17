"""Create small synthetic files for a local pipeline smoke test."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    n_students, n_mentors = 120, 120
    attendance = rng.normal(78, 12, n_students).clip(20, 100)
    assignments = rng.normal(65, 16, n_students).clip(0, 100)
    quizzes = rng.normal(62, 18, n_students).clip(0, 100)
    commute = rng.normal(45, 20, n_students).clip(5, 150)
    backlogs = rng.poisson(0.5, n_students)
    support = rng.normal(6, 2, n_students).clip(0, 10)
    risk_score = (
        -0.06 * attendance
        - 0.04 * assignments
        - 0.03 * quizzes
        + 0.02 * commute
        + 0.7 * backlogs
        - 0.25 * support
        + rng.normal(0, 2.2, n_students)
    )
    at_risk = (risk_score > np.quantile(risk_score, 0.7)).astype(int)
    students = pd.DataFrame(
        {
            "student_id": [f"S{i:03d}" for i in range(n_students)],
            "attendance": attendance,
            "assignment_avg": assignments,
            "quiz_avg": quizzes,
            "commute_time_min": commute,
            "prior_backlogs": backlogs,
            "peer_support_score": support,
            "at_risk": at_risk,
        }
    )

    mentors = students.drop(columns=["student_id", "at_risk"]).sample(
        n=n_mentors, random_state=42
    )
    mentors.insert(0, "mentor_id", [f"M{i:03d}" for i in range(n_mentors)])

    Path("data").mkdir(exist_ok=True)
    students.to_csv("data/students.csv", index=False)
    mentors.to_csv("data/mentors.csv", index=False)
    print("Wrote demo files to data/students.csv and data/mentors.csv")


if __name__ == "__main__":
    main()
