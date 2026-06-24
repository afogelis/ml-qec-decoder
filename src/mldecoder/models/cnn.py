"""Geometry-aware convolutional decoder (PyTorch).

The tabular models (random forest, gradient boosting, MLP) treat the syndrome as
an unordered feature vector and therefore ignore the single most important fact
about a surface code: its detectors live on a 2D lattice, and errors create
*spatially local* clusters of detection events. As the code distance grows the
flat feature space explodes and these tabular models degrade.

This decoder instead reshapes the syndrome back onto the lattice. It reads each
detector's ``(x, y, t)`` coordinate from the Stim circuit, scatters the binary
detection events into a ``(rounds, height, width)`` image, and applies a small
convolutional network whose 3x3 kernels exploit the translation-equivariant,
local structure of surface-code syndromes -- the same inductive bias that makes
CNNs effective on images, applied to the decoding problem.

The architecture is intentionally compact; the point is to demonstrate that a
geometry-aware inductive bias scales better with distance than the tabular
models, not to chase a state-of-the-art decoder.
"""

from __future__ import annotations

import numpy as np
import stim
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import MlDecoder
from ..dataset import Dataset


class _CnnNet(nn.Module):
    """Two conv blocks over the lattice followed by global pooling and a head."""

    def __init__(self, in_channels: int, num_labels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global pooling: robust to any lattice size
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class CnnDecoder(MlDecoder):
    """A convolutional decoder that operates on the syndrome as a lattice image."""

    name = "cnn"

    def __init__(
        self,
        *,
        epochs: int = 40,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        patience: int = 8,
        val_fraction: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: _CnnNet | None = None
        self._grid: tuple[int, int, int] | None = None  # (rounds/time, height, width)
        self._rows: np.ndarray | None = None
        self._cols: np.ndarray | None = None
        self._times: np.ndarray | None = None

    def fit(self, circuit: stim.Circuit) -> None:
        """Recover detector geometry from ``circuit`` before sampling and fitting."""
        self._build_geometry(circuit)
        super().fit(circuit)

    def _build_geometry(self, circuit: stim.Circuit) -> None:
        """Map every detector index to an integer ``(time, row, col)`` lattice cell."""
        coords = circuit.get_detector_coordinates()
        num_detectors = circuit.num_detectors

        xs, ys, ts = [], [], []
        for index in range(num_detectors):
            coord = coords.get(index, [0.0, 0.0, 0.0])
            xs.append(coord[0] if len(coord) > 0 else 0.0)
            ys.append(coord[1] if len(coord) > 1 else 0.0)
            ts.append(coord[2] if len(coord) > 2 else 0.0)

        x_axis = {value: i for i, value in enumerate(sorted(set(xs)))}
        y_axis = {value: i for i, value in enumerate(sorted(set(ys)))}
        t_axis = {value: i for i, value in enumerate(sorted(set(ts)))}

        self._cols = np.array([x_axis[x] for x in xs], dtype=np.int64)
        self._rows = np.array([y_axis[y] for y in ys], dtype=np.int64)
        self._times = np.array([t_axis[t] for t in ts], dtype=np.int64)
        self._grid = (len(t_axis), len(y_axis), len(x_axis))

    def _to_images(self, features: np.ndarray) -> np.ndarray:
        """Scatter ``(shots, num_detectors)`` syndromes into ``(shots, T, H, W)`` images."""
        if self._grid is None:
            raise RuntimeError("geometry not built; call fit() first")
        shots = features.shape[0]
        time_steps, height, width = self._grid
        images = np.zeros((shots, time_steps, height, width), dtype=np.float32)
        images[:, self._times, self._rows, self._cols] = features
        return images

    def _fit(self, dataset: Dataset) -> None:
        torch.manual_seed(self.train_seed)
        images = torch.from_numpy(self._to_images(dataset.features)).float()
        labels = torch.from_numpy(dataset.labels.astype(np.float32)).float()

        num_val = max(1, int(self.val_fraction * images.shape[0]))
        perm = torch.randperm(images.shape[0])
        val_idx, train_idx = perm[:num_val], perm[num_val:]
        train_loader = DataLoader(
            TensorDataset(images[train_idx], labels[train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_images = images[val_idx].to(self.device)
        val_labels = labels[val_idx].to(self.device)

        in_channels = self._grid[0] if self._grid else 1
        net = _CnnNet(in_channels, dataset.num_labels).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        best_state = {key: value.clone() for key, value in net.state_dict().items()}
        epochs_without_improvement = 0

        for _ in range(self.epochs):
            net.train()
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(net(batch_images), batch_labels)
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(val_images), val_labels).item())
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
        images = torch.from_numpy(self._to_images(np.asarray(features, dtype=np.float32)))
        images = images.float().to(self.device)
        with torch.no_grad():
            logits = self._net(images)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return probabilities
