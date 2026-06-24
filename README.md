# ML QEC Decoder

Machine-learning decoders for the surface code. Each model learns to predict the logical
observable flip directly from a syndrome, and plugs into the
[`decoder-benchmark`](https://github.com/afogelis/decoder-benchmark) framework so it can be
compared head-to-head with classical decoders (MWPM, union-find, belief propagation) on accuracy,
runtime and memory.

This is repo 4 of a seven-part [QEC research portfolio](../README.md).

## Models

| Name | Backend | Notes |
|------|---------|-------|
| `rf` | scikit-learn `RandomForestClassifier` | One forest per logical observable. |
| `xgb` | XGBoost `XGBClassifier` | Gradient-boosted trees, `hist` method. |
| `mlp` | PyTorch | Fully-connected net, BCE loss, Adam, early stopping on a validation split, GPU when available. |

All three subclass a common `MlDecoder` base that samples training data from the same Stim circuit
the classical decoders see, so comparisons are apples-to-apples.

## What this demonstrates

- **Applied ML:** framing decoding as supervised classification, training tree and neural models, early stopping, fair train/eval separation.
- **Integration:** the ML models implement the same `Decoder` protocol as the classical decoders and register into the shared benchmark.
- **Research judgement:** an honest regime analysis of *when* ML helps and *when* it does not, without overclaiming.

## Research questions and findings

The bundled `examples/ml_vs_classical.py` sweep (distances 3 and 5, p from 0.01 to 0.03,
20k training shots) gives an honest picture:

- **When is ML competitive?** At small distance (d=3) the MLP comes close to MWPM
  (e.g. 0.080 vs 0.064 logical error rate at p=0.01) and does so with very fast inference
  (sub-microsecond per shot, since a forward pass is a couple of matrix multiplies). With more
  training data the gap narrows further.
- **When does it fail?** At larger distance (d=5) every ML decoder degrades sharply: the syndrome
  space grows, logical flips become rarer, and 20k shots no longer cover the input distribution,
  so the classifier underperforms MWPM by a wide margin. MWPM, which uses the *known* error model
  rather than learning it, does not pay this cost.
- **Why MWPM is hard to beat here:** for circuit-level depolarizing noise the matching graph is an
  excellent model, so a learned decoder is competing against a near-optimal baseline. ML decoders
  tend to win in settings the matching graph models poorly (strongly correlated or non-graphlike
  noise) or where training data is abundant relative to distance.
- **How does noise matter?** Tree models need enough positive (flip) examples; the MLP benefits
  most from additional training shots as distance grows.

The takeaway is not "ML beats matching" but a calibrated understanding of the regimes where a
learned decoder is and is not the right tool.

## Install

```bash
pip install -e ".[dev]"
# torch CPU wheels: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For local development with checked-out sibling repos:

```bash
pip install -e ../surface-code-simulator
pip install -e ../decoder-benchmark --no-deps
pip install -e . --no-deps
```

## Quick start

```bash
pytest
python examples/ml_vs_classical.py     # writes outputs/{ml_comparison.json,ml_vs_mwpm.png}
```

```bash
mldecoder compare --distances 3,5 --p 0.01,0.02,0.03 --shots 5000 --train-shots 20000
```

## Library usage

```python
from decbench import run_benchmark, build_leaderboard, format_leaderboard
from decbench.types import BenchmarkConfig
from mldecoder import register_ml_decoders

register_ml_decoders(train_shots=20_000)   # rf, xgb, mlp now in the registry
result = run_benchmark(BenchmarkConfig(
    decoders=["mwpm", "rf", "xgb", "mlp"],
    distances=[3, 5], error_rates=[0.01, 0.02], shots=5_000, seed=2026,
))
print(format_leaderboard(build_leaderboard(result)))
```

## Layout

- `src/mldecoder/models/` — `random_forest`, `xgboost_decoder`, `mlp`
- `src/mldecoder/{base,dataset,analysis,cli}.py`
- `tests/` — small-budget training tests on the real stack
- `examples/ml_vs_classical.py` — regime analysis vs MWPM

## License

MIT — see [LICENSE](LICENSE).
