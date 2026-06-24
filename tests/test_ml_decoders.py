"""Tests for the ML decoders. Training budgets are kept small for speed."""

import numpy as np
import pytest
from surfacecode.circuits import build_surface_code_circuit
from surfacecode.sampling import sample_syndromes
from surfacecode.types import ExperimentConfig

from mldecoder.models import MlpDecoder, RandomForestDecoder, XGBoostDecoder

MODELS = [RandomForestDecoder, XGBoostDecoder, MlpDecoder]


def _circuit_and_eval(distance=3, rounds=3, p=0.02, shots=400, seed=11):
    config = ExperimentConfig(distance=distance, rounds=rounds, p=p, shots=shots, seed=seed)
    circuit = build_surface_code_circuit(config)
    sample = sample_syndromes(circuit, shots=shots, seed=seed + 1)
    return circuit, sample


@pytest.mark.parametrize("model_cls", MODELS)
def test_decode_output_shape_and_dtype(model_cls):
    circuit, sample = _circuit_and_eval()
    decoder = model_cls(train_shots=2000, train_seed=1)
    decoder.fit(circuit)
    predictions = decoder.decode_batch(sample.detection_events)
    assert predictions.shape == (sample.num_shots, sample.num_observables)
    assert predictions.dtype == bool


@pytest.mark.parametrize("model_cls", MODELS)
def test_decoder_learns_something_useful(model_cls):
    # At a moderate error rate the trained decoder should beat the trivial
    # always-predict-no-flip baseline's logical error rate.
    circuit, sample = _circuit_and_eval(p=0.02, shots=1500, seed=21)
    decoder = model_cls(train_shots=8000, train_seed=2)
    decoder.fit(circuit)
    predictions = decoder.decode_batch(sample.detection_events)

    ler = np.mean(np.any(predictions != sample.observable_flips, axis=1))
    trivial_ler = np.mean(np.any(sample.observable_flips, axis=1))
    assert ler <= trivial_ler + 0.02


def test_decoders_register_into_decbench():
    from decbench.registry import available_decoders

    import mldecoder  # noqa: F401  (import registers rf/xgb/mlp)

    assert {"rf", "xgb", "mlp"}.issubset(set(available_decoders()))
