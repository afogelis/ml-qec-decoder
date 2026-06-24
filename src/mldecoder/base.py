"""Shared base class for machine-learning decoders.

The base class handles the parts every ML decoder shares -- generating training
data from the circuit and conforming to the ``decbench`` ``Decoder`` protocol --
so concrete models only implement ``_fit`` and ``_predict_proba``. This keeps
the comparison against classical decoders apples-to-apples: identical circuit,
identical sampling, only the model differs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import stim

from .dataset import Dataset, make_dataset


class MlDecoder(ABC):
    """Abstract ML decoder that predicts logical observable flips from syndromes."""

    name = "ml_base"

    def __init__(self, *, train_shots: int = 50_000, train_seed: int = 1234) -> None:
        self.train_shots = train_shots
        self.train_seed = train_seed
        self._num_labels = 0
        self._fitted = False

    def fit(self, circuit: stim.Circuit) -> None:
        """Sample training data from ``circuit`` and fit the underlying model.

        A distinct training seed keeps the training shots disjoint in
        distribution-controlling randomness from the evaluation shots that the
        benchmark later supplies.
        """
        dataset = make_dataset(circuit, shots=self.train_shots, seed=self.train_seed)
        self._num_labels = dataset.num_labels
        self._fit(dataset)
        self._fitted = True

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("decoder must be fit() before decode_batch()")
        features = np.asarray(detection_events, dtype=np.float32)
        probabilities = self._predict_proba(features)
        return probabilities >= 0.5

    @abstractmethod
    def _fit(self, dataset: Dataset) -> None:
        """Train the model on ``dataset``."""

    @abstractmethod
    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return ``(shots, num_labels)`` probabilities of each observable flipping."""


def constant_label_probability(dataset: Dataset) -> np.ndarray | None:
    """Return the constant prediction to use when a label has a single class.

    When the training set never observed a logical flip for some observable, a
    classifier cannot learn it; we fall back to the majority (always-0) class.
    Returns ``None`` when every label has both classes present.
    """
    per_label_positive = dataset.labels.mean(axis=0)
    if np.all((per_label_positive > 0.0) & (per_label_positive < 1.0)):
        return None
    return (per_label_positive >= 0.5).astype(float)
