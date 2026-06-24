"""Random-forest decoder (scikit-learn backend)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..base import MlDecoder, constant_label_probability
from ..dataset import Dataset


class RandomForestDecoder(MlDecoder):
    """Predicts logical flips with one random forest per observable."""

    name = "rf"

    def __init__(self, *, n_estimators: int = 200, max_depth: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._models: list[RandomForestClassifier | None] = []
        self._constant: np.ndarray | None = None

    def _fit(self, dataset: Dataset) -> None:
        self._constant = constant_label_probability(dataset)
        self._models = []
        for label_index in range(dataset.num_labels):
            target = dataset.labels[:, label_index]
            if len(np.unique(target)) < 2:
                self._models.append(None)
                continue
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=-1,
                random_state=self.train_seed,
            )
            model.fit(dataset.features, target)
            self._models.append(model)

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((features.shape[0], self._num_labels), dtype=float)
        for label_index, model in enumerate(self._models):
            if model is None:
                fallback = 0.0 if self._constant is None else float(self._constant[label_index])
                probabilities[:, label_index] = fallback
                continue
            probabilities[:, label_index] = model.predict_proba(features)[:, 1]
        return probabilities
