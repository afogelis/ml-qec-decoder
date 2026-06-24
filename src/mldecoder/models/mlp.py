"""Neural-network decoder (PyTorch multilayer perceptron).

A small fully-connected network maps the binary syndrome vector to per-observable
flip probabilities. Training uses binary cross-entropy with an Adam optimizer and
early stopping on a held-out validation split. The architecture is deliberately
modest: surface-code syndromes are low-dimensional, and the goal is a fair
comparison against tree models and matching, not a state-of-the-art decoder.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MlDecoder
from ..dataset import Dataset


class _MlpNet(nn.Module):
    def __init__(self, num_features: int, num_labels: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features
        for width in hidden:
            layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = width
        layers.append(nn.Linear(in_dim, num_labels))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MlpDecoder(MlDecoder):
    """A PyTorch MLP that predicts logical observable flips from syndromes."""

    name = "mlp"

    def __init__(
        self,
        *,
        hidden: tuple[int, ...] = (256, 128),
        epochs: int = 60,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        patience: int = 6,
        val_fraction: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: _MlpNet | None = None

    def _fit(self, dataset: Dataset) -> None:
        torch.manual_seed(self.train_seed)
        features = torch.from_numpy(dataset.features).float()
        labels = torch.from_numpy(dataset.labels.astype(np.float32)).float()

        num_val = max(1, int(self.val_fraction * features.shape[0]))
        perm = torch.randperm(features.shape[0])
        val_idx, train_idx = perm[:num_val], perm[num_val:]
        train_loader = DataLoader(
            TensorDataset(features[train_idx], labels[train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_features = features[val_idx].to(self.device)
        val_labels = labels[val_idx].to(self.device)

        net = _MlpNet(dataset.num_features, dataset.num_labels, self.hidden).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        best_state = {key: value.clone() for key, value in net.state_dict().items()}
        epochs_without_improvement = 0

        for _ in range(self.epochs):
            net.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(net(batch_features), batch_labels)
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(val_features), val_labels).item())
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {key: value.clone() for key, value in net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self._net = net

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("network is not trained")
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            logits = self._net(tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return probabilities
