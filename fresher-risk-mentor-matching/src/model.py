"""Neural network used for risk prediction and embedding extraction."""

import torch
from torch import Tensor, nn


class RiskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        embedding_dim: int = 12,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def embed(self, features: Tensor) -> Tensor:
        return self.encoder(features)

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(self.embed(features))
