# Fresher Risk & Mentor Matching

Self project | Deep Learning | SC/ST Cell, IIT Bombay  
Jul 2026 - Present

This repository contains the working code for a student-support pipeline:

1. train a small PyTorch network to flag freshers who may need early support,
2. inspect the predictions with DeepSHAP, and
3. match students to mentors using the Hungarian algorithm on learned embeddings.

The project is designed to be run on an anonymised CSV exported by the team. The
actual student data is not included in this repository.

## Project layout

```text
.
├── configs/
│   └── default.yaml          # training and column settings
├── data/
│   └── README.md             # expected input format
├── scripts/
│   └── make_demo_data.py     # optional local smoke-test data
├── src/
│   ├── data.py               # loading, splitting, and scaling
│   ├── explain.py            # DeepSHAP feature importance
│   ├── losses.py             # weighted focal loss
│   ├── match.py              # embedding extraction + Hungarian matching
│   ├── model.py              # MLP risk model
│   └── train.py              # training entry point
├── tests/
│   └── test_focal_loss.py
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repository-url>
cd fresher-risk-mentor-matching

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The project uses Python 3.10+ and runs on CPU. CUDA works automatically when a
CUDA-enabled PyTorch installation is available.

## Input data

Put an anonymised student file at `data/students.csv`. The default config expects
these columns:

```text
student_id,attendance,assignment_avg,quiz_avg,commute_time_min,
prior_backlogs,peer_support_score,at_risk
```

`at_risk` is a binary label (`0` or `1`). Keep identifiers out of the feature
list. If the real data uses different columns, change
`configs/default.yaml`; no source code change is needed.

Do not commit names, phone numbers, email addresses, roll numbers, or any other
identifying student information. See `data/README.md` for the assumptions used
by the scripts.

## Train

```bash
python -m src.train --config configs/default.yaml
```

The command writes the following local artefacts to `outputs/`:

- `risk_model.pt` - model weights and feature metadata
- `scaler.joblib` - the fitted training scaler
- `metrics.json` - validation metrics and the threshold used

For a quick local check without private data:

```bash
python scripts/make_demo_data.py
python -m src.train --config configs/default.yaml
```

The generated data is only for checking that the pipeline runs. It is not a
benchmark and should not be used to make student-support decisions.

## Explain predictions

DeepSHAP uses a background sample from the training data and reports the mean
absolute contribution of each feature.

```bash
python -m src.explain \
  --config configs/default.yaml \
  --data data/students.csv \
  --output outputs/deepshap_importance.csv
```

The output is an ordered CSV with `feature`, `mean_abs_shap`, and
`mean_shap`. SHAP values describe model behaviour; they are not causal effects.

## Match students to mentors

Mentor profiles should contain the same feature columns used by the risk model.
In practice, these columns are anonymised profile/preference scores mapped to
the student-support dimensions used during training.

```bash
python -m src.match \
  --config configs/default.yaml \
  --students data/students.csv \
  --mentors data/mentors.csv \
  --output outputs/mentor_matches.csv
```

The script embeds both tables with the trained network, builds a cosine-distance
cost matrix, and solves the one-to-one assignment with
`scipy.optimize.linear_sum_assignment`.

The matching output contains the IDs, embedding distance, and a similarity score.
It is intended as a first-pass recommendation for a human coordinator, not an
automatic final assignment.

## Notes

- The model is deliberately small because this is an early-warning tool, not a
  replacement for a student support team.
- A validation split is stratified so the minority risk class is represented in
  both partitions.
- The classification threshold is selected on the validation set using F1 rather
  than assuming `0.5` is appropriate for an imbalanced target.
- Risk predictions should be reviewed with context and never used as a punitive
  label.
