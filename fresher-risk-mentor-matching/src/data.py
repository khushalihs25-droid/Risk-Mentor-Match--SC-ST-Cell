"""CSV loading and train/validation preparation."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataSplit:
    train_x: torch.Tensor
    train_y: torch.Tensor
    valid_x: torch.Tensor
    valid_y: torch.Tensor
    feature_columns: list[str]
    scaler: StandardScaler


def read_features(
    path: str | Path,
    feature_columns: list[str],
    target_column: str | None = None,
) -> tuple[pd.DataFrame, torch.Tensor | None]:
    frame = pd.read_csv(path)
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise ValueError(f"Missing feature columns in {path}: {', '.join(missing)}")

    features = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    if features.isna().any().any():
        raise ValueError(f"Feature columns in {path} contain missing values")

    if target_column is None:
        return features, None

    if target_column not in frame:
        raise ValueError(f"Missing target column '{target_column}' in {path}")
    target = pd.to_numeric(frame[target_column], errors="raise")
    if not target.isin([0, 1]).all():
        raise ValueError(f"Target column '{target_column}' must contain only 0 and 1")
    return features, torch.tensor(target.to_numpy(), dtype=torch.float32).reshape(-1, 1)


def make_split(
    path: str | Path,
    feature_columns: list[str],
    target_column: str,
    test_size: float,
    seed: int,
) -> DataSplit:
    features, target = read_features(path, feature_columns, target_column)
    assert target is not None
    if target.numel() < 10:
        raise ValueError("At least 10 labelled rows are needed for a train/validation split")
    if target.sum() == 0 or target.sum() == target.numel():
        raise ValueError("The target needs both 0 and 1 examples")

    train_idx, valid_idx = train_test_split(
        range(len(features)),
        test_size=test_size,
        random_state=seed,
        stratify=target.numpy().ravel(),
    )
    scaler = StandardScaler()
    train_values = scaler.fit_transform(features.iloc[train_idx])
    valid_values = scaler.transform(features.iloc[valid_idx])

    return DataSplit(
        train_x=torch.tensor(train_values, dtype=torch.float32),
        train_y=target[list(train_idx)],
        valid_x=torch.tensor(valid_values, dtype=torch.float32),
        valid_y=target[list(valid_idx)],
        feature_columns=feature_columns,
        scaler=scaler,
    )


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
