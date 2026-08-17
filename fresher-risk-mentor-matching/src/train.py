"""Train the risk model.

Run from the repository root:
    python -m src.train --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import make_split, save_scaler
from .losses import WeightedFocalLoss
from .model import RiskMLP


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_threshold(labels: torch.Tensor, probabilities: torch.Tensor) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in torch.arange(0.2, 0.81, 0.05):
        predictions = (probabilities >= threshold).int().numpy()
        score = f1_score(labels.numpy(), predictions, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score
    return best_threshold


def evaluate(
    model: RiskMLP, features: torch.Tensor, labels: torch.Tensor
) -> tuple[float, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits).reshape(-1)
    targets = labels.reshape(-1).int()
    threshold = choose_threshold(targets, probabilities)
    predictions = (probabilities >= threshold).int()
    metrics = {
        "roc_auc": float(roc_auc_score(targets.numpy(), probabilities.numpy())),
        "average_precision": float(
            average_precision_score(targets.numpy(), probabilities.numpy())
        ),
        "f1": float(f1_score(targets.numpy(), predictions.numpy(), zero_division=0)),
        "threshold": threshold,
    }
    return float(nn.functional.binary_cross_entropy(logits.sigmoid(), labels).item()), metrics


def train(config_path: str) -> dict[str, float]:
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    data_config = config["data"]
    train_config = config["training"]
    seed = int(train_config["seed"])
    set_seed(seed)

    split = make_split(
        path=data_config["train_path"],
        feature_columns=data_config["feature_columns"],
        target_column=data_config["target_column"],
        test_size=float(data_config["test_size"]),
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskMLP(
        input_dim=len(split.feature_columns),
        hidden_dim=int(train_config["hidden_dim"]),
        embedding_dim=int(train_config["embedding_dim"]),
        dropout=float(train_config["dropout"]),
    ).to(device)

    positive = float(split.train_y.sum())
    negative = float(split.train_y.numel() - positive)
    loss_fn = WeightedFocalLoss(
        positive_weight=max(negative / max(positive, 1.0), 1.0),
        gamma=float(train_config["focal_gamma"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config["learning_rate"]))
    dataset = TensorDataset(split.train_x, split.train_y)
    loader = DataLoader(dataset, batch_size=int(train_config["batch_size"]), shuffle=True)

    best_loss = float("inf")
    patience = 0
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "risk_model.pt"

    for epoch in range(int(train_config["epochs"])):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(
                model(split.valid_x.to(device)), split.valid_y.to(device)
            ).item()
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": len(split.feature_columns),
                    "hidden_dim": int(train_config["hidden_dim"]),
                    "embedding_dim": int(train_config["embedding_dim"]),
                    "dropout": float(train_config["dropout"]),
                    "feature_columns": split.feature_columns,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= int(train_config["early_stopping_patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    _, metrics = evaluate(model, split.valid_x.to(device), split.valid_y.to(device))
    metrics["validation_loss"] = best_loss
    metrics["epochs_ran"] = epoch + 1
    metrics["device"] = str(device)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    save_scaler(split.scaler, output_dir / "scaler.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    metrics = train(args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
