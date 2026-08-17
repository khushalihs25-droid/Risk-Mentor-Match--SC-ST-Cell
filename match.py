"""Match student and mentor profiles in the learned embedding space."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from .data import read_features
from .model import RiskMLP


def match_students(
    checkpoint_path: str,
    scaler_path: str,
    students_path: str,
    mentors_path: str,
    feature_columns: list[str],
    student_id_column: str,
    mentor_id_column: str,
    output_path: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = RiskMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    student_frame = pd.read_csv(students_path)
    mentor_frame = pd.read_csv(mentors_path)
    student_features, _ = read_features(students_path, feature_columns)
    mentor_features, _ = read_features(mentors_path, feature_columns)
    scaler = joblib.load(scaler_path)
    student_x = torch.tensor(scaler.transform(student_features), dtype=torch.float32)
    mentor_x = torch.tensor(scaler.transform(mentor_features), dtype=torch.float32)

    with torch.no_grad():
        student_embeddings = model.embed(student_x).numpy()
        mentor_embeddings = model.embed(mentor_x).numpy()
    costs = pairwise_distances(student_embeddings, mentor_embeddings, metric="cosine")
    student_indices, mentor_indices = linear_sum_assignment(costs)

    result = pd.DataFrame(
        {
            "student_id": student_frame.iloc[student_indices][student_id_column].to_numpy(),
            "mentor_id": mentor_frame.iloc[mentor_indices][mentor_id_column].to_numpy(),
            "embedding_distance": costs[student_indices, mentor_indices],
            "similarity_score": 1 - costs[student_indices, mentor_indices],
        }
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} mentor assignments to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--students", default="data/students.csv")
    parser.add_argument("--mentors", default="data/mentors.csv")
    parser.add_argument("--output", default="outputs/mentor_matches.csv")
    parser.add_argument("--mentor-id-column", default="mentor_id")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    training = config["training"]
    match_students(
        checkpoint_path=f"{training['output_dir']}/risk_model.pt",
        scaler_path=f"{training['output_dir']}/scaler.joblib",
        students_path=args.students,
        mentors_path=args.mentors,
        feature_columns=config["data"]["feature_columns"],
        student_id_column=config["data"]["id_column"],
        mentor_id_column=args.mentor_id_column,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
