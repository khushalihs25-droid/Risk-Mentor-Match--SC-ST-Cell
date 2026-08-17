# Data folder

The project expects an anonymised CSV supplied by the team. Keep the real file
out of GitHub.

The default feature columns are:

- `attendance`
- `assignment_avg`
- `quiz_avg`
- `commute_time_min`
- `prior_backlogs`
- `peer_support_score`

The target column is `at_risk`, encoded as `0` or `1`. `student_id` is used only
to identify rows in output files and is never passed to the model.

Mentor matching uses the same feature columns. The mentor values should be
numeric profile or preference scores on the same scale/meaning as the student
features. Keep `mentor_id` as the identifier column in `data/mentors.csv`.

Before analysis, remove direct identifiers and review whether each feature is
appropriate for student support. The repository intentionally does not include
the institute dataset.
