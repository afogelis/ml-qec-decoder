"""Gradient-boosted-tree decoder (XGBoost backend)."""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from ..base import MlDecoder, constant_label_probability
from ..dataset import Dataset


class XGBoostDecoder(MlDecoder):
    """Predicts logical flips with one gradient-boosted classifier per observable."""

    name = "xgb"

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._models: list[XGBClassifier | None] = []
        self._constant: np.ndarray | None = None

    def _fit(self, dataset: Dataset) -> None:
        self._constant = constant_label_probability(dataset)
        self._models = []
        for label_index in range(dataset.num_labels):
            target = dataset.labels[:, label_index]
            if len(np.unique(target)) < 2:
                self._models.append(None)
                continue
            model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                tree_method="hist",
                n_jobs=-1,
                eval_metric="logloss",
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
