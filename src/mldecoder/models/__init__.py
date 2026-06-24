"""Machine-learning decoder models and their registration into ``decbench``."""

from decbench.registry import register_decoder

from .cnn import CnnDecoder
from .mlp import MlpDecoder
from .random_forest import RandomForestDecoder
from .xgboost_decoder import XGBoostDecoder

register_decoder("rf", RandomForestDecoder)
register_decoder("xgb", XGBoostDecoder)
register_decoder("mlp", MlpDecoder)
register_decoder("cnn", CnnDecoder)

__all__ = [
    "CnnDecoder",
    "MlpDecoder",
    "RandomForestDecoder",
    "XGBoostDecoder",
]
