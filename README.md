# ML QEC Decoder: when do learned decoders fail against MWPM?

A controlled **negative study**. Off-the-shelf machine-learning decoders are a popular idea, so this
repo asks the honest question directly: trained on a realistic data budget, do learned decoders --
tabular *and* a geometry-aware convolutional model -- actually keep up with minimum-weight perfect
matching as the surface code grows? Each model learns to predict the logical observable flip from a
syndrome and plugs into the
[`decoder-benchmark`](https://github.com/afogelis/decoder-benchmark) framework, so the comparison
against MWPM, union-find and belief propagation is apples-to-apples.

The short answer: **no, not at this data budget.** That is the result, and it is worth showing
clearly rather than cherry-picking the one regime where ML looks good.

This is repo 4 of a ten-part [QEC research portfolio](https://github.com/afogelis/qec-portfolio).

## Results at a glance

![Logical error rate versus code distance for MWPM and four ML decoders.](docs/cnn_scaling.png)

*The headline negative result. As the code distance grows from 3 to 7 at a fixed below-threshold physical error rate (p = 0.006), MWPM (green) suppresses the logical error rate, while every learned decoder -- including the geometry-aware CNN -- diverges upward. The CNN's lattice inductive bias buys only a marginal edge over the tabular models and does not prevent the collapse.*

![Logical error rate of MWPM versus the best machine-learning decoder, across code distance and physical error rate.](docs/ml_vs_mwpm.png)

*Best ML decoder vs MWPM per regime. Points above the diagonal are MWPM wins. Learned decoders only approach the diagonal at d=3 and low p; almost everywhere else MWPM is better.*

## Models

| Name | Backend | Notes |
|------|---------|-------|
| `rf` | scikit-learn `RandomForestClassifier` | One forest per logical observable (tabular). |
| `xgb` | XGBoost `XGBClassifier` | Gradient-boosted trees, `hist` method (tabular). |
| `mlp` | PyTorch | Fully-connected net on the flat syndrome vector, BCE loss, Adam, early stopping (tabular). |
| `cnn` | PyTorch | **Geometry-aware syndrome-grid CNN**: scatters detectors back onto their `(t, y, x)` lattice cells and applies small `Conv2d` kernels, giving the same translation-equivariant inductive bias that makes CNNs work on images. |

All four subclass a common `MlDecoder` base that samples training data from the same Stim circuit
the classical decoders see, so comparisons are apples-to-apples. The `cnn` is the most interesting
baseline because it is the model with the *right* inductive bias for a 2D code; the study tests
whether that bias is enough to overcome the data-scaling problem (it is not, here).

## What this demonstrates

- **Applied ML:** framing decoding as supervised classification across tree, fully-connected and convolutional models, with early stopping and a fair train/eval split; recovering detector geometry from a Stim circuit to build the CNN's lattice tensor.
- **Integration:** every ML model implements the same `Decoder` protocol as the classical decoders and registers into the shared benchmark.
- **Research judgement (the point of this repo):** designing and reporting a *negative result* honestly -- showing where ML decoders fail and explaining why, rather than overclaiming a win.

## Research questions and findings

The bundled `examples/cnn_scaling.py` sweep (distances 3, 5, 7 at p = 0.006, 30k training shots)
and `examples/ml_vs_classical.py` (distances 3 and 5, p from 0.01 to 0.03) give a consistent
picture:

- **Where ML is competitive (narrowly).** At small distance (d=3) the learned decoders come close
  to MWPM (logical error rates within roughly a factor of two) and do so with very fast inference,
  since a forward pass is a few matrix multiplies. This is the only regime where ML is in the game.
- **Where it fails (everywhere else).** As distance grows to 5 and 7 the syndrome space expands,
  logical flips become rarer, and a fixed training budget no longer covers the input distribution.
  Every ML decoder's logical error rate climbs steeply while MWPM's falls. MWPM uses the *known*
  error model rather than learning it from data, so it pays none of this cost.
- **Does geometry awareness save ML?** Not here. The CNN tracks slightly below the tabular models
  at d=5 and d=7, confirming its inductive bias helps a little, but it still diverges from MWPM by
  an order of magnitude. The right architecture does not substitute for the missing data.
- **Why MWPM is hard to beat in this setting.** For circuit-level depolarizing noise the matching
  graph is an excellent model, so a learned decoder competes against a near-optimal baseline. ML
  decoders are expected to win only where the matching graph models the noise poorly (strongly
  correlated or non-graphlike noise) or where training data is abundant relative to distance --
  neither of which holds in this controlled comparison.

The takeaway is not "ML beats matching." It is a calibrated, reproducible demonstration of *when a
learned decoder is the wrong tool*, which is the more useful thing to know.

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
python examples/cnn_scaling.py          # writes docs/cnn_scaling.png (the headline figure)
```

```bash
mldecoder compare --distances 3,5 --p 0.01,0.02,0.03 --shots 5000 --train-shots 20000
```

## Library usage

```python
from decbench import run_benchmark, build_leaderboard, format_accuracy_tier
from decbench.types import BenchmarkConfig
from mldecoder import register_ml_decoders

register_ml_decoders(train_shots=20_000)   # rf, xgb, mlp, cnn now in the registry
result = run_benchmark(BenchmarkConfig(
    decoders=["mwpm", "rf", "xgb", "mlp", "cnn"],
    distances=[3, 5], error_rates=[0.01, 0.02], shots=5_000, seed=2026,
))
print(format_accuracy_tier(build_leaderboard(result)))
```

## Layout

- `src/mldecoder/models/` — `random_forest`, `xgboost_decoder`, `mlp`, `cnn` (geometry-aware syndrome-grid CNN)
- `src/mldecoder/{base,dataset,analysis,cli}.py`
- `tests/` — small-budget training tests on the real stack, including the CNN lattice builder
- `examples/ml_vs_classical.py` — regime analysis vs MWPM
- `examples/cnn_scaling.py` — logical-error-vs-distance scaling (the headline negative result)

## License

MIT — see [LICENSE](LICENSE).
