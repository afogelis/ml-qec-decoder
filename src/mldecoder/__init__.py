"""Machine-learning surface-code decoders that plug into the decbench framework."""

from decbench.registry import register_decoder

from .base import MlDecoder
from .dataset import Dataset, make_dataset
from .models import MlpDecoder, RandomForestDecoder, XGBoostDecoder

__version__ = "0.1.0"


def register_ml_decoders(*, train_shots: int = 50_000, train_seed: int = 1234) -> None:
    """(Re)register the ML decoders with a chosen training-data budget.

    Importing this package already registers ``rf``, ``xgb`` and ``mlp`` with
    default settings. Call this to trade training cost against accuracy, e.g.
    a small ``train_shots`` for quick experiments or CI.
    """
    register_decoder("rf", lambda: RandomForestDecoder(train_shots=train_shots, train_seed=train_seed))
    register_decoder("xgb", lambda: XGBoostDecoder(train_shots=train_shots, train_seed=train_seed))
    register_decoder("mlp", lambda: MlpDecoder(train_shots=train_shots, train_seed=train_seed))


__all__ = [
    "Dataset",
    "MlDecoder",
    "MlpDecoder",
    "RandomForestDecoder",
    "XGBoostDecoder",
    "make_dataset",
    "register_ml_decoders",
]
