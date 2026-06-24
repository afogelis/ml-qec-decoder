"""Generate supervised training data from a surface-code circuit.

An ML decoder is trained as a classifier mapping a syndrome (the detection
events of one shot) to the logical observable flip. We generate labeled pairs
by sampling the same Stim circuit the classical decoders see, so every decoder
is trained and evaluated on the identical noise process.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim
from surfacecode.sampling import sample_syndromes


@dataclass(frozen=True)
class Dataset:
    """Feature matrix and label matrix for a decoding task."""

    features: np.ndarray  # (shots, num_detectors), float32
    labels: np.ndarray  # (shots, num_observables), int8

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_labels(self) -> int:
        return int(self.labels.shape[1])


def make_dataset(circuit: stim.Circuit, *, shots: int, seed: int | None = None) -> Dataset:
    """Sample ``shots`` (syndrome, observable-flip) pairs from ``circuit``."""
    sample = sample_syndromes(circuit, shots=shots, seed=seed)
    features = sample.detection_events.astype(np.float32)
    labels = sample.observable_flips.astype(np.int8)
    return Dataset(features=features, labels=labels)
