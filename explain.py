"""Explain trained predictions with DeepSHAP."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import yaml

from .data import read_features
from .model import RiskMLP


def explain(
    checkpoint_path: str,
    scaler_path: str,
    data_path: str,
    feature_columns: list[str],
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

    frame = pd.read_csv(data_path)
    features, _ = read_features(data_path, feature_columns)
    scaler = joblib.load(scaler_path)
    scaled = torch.tensor(scaler.transform(features), dtype=torch.float32)
    background = scaled[: min(100, len(scaled))]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 0]

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    print(f"Wrote DeepSHAP summary for {len(frame)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/students.csv")
    parser.add_argument("--output", default="outputs/deepshap_importance.csv")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    training = config["training"]
    explain(
        checkpoint_path=f"{training['output_dir']}/risk_model.pt",
        scaler_path=f"{training['output_dir']}/scaler.joblib",
        data_path=args.data,
        feature_columns=config["data"]["feature_columns"],
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
