"""Analyse when ML decoders beat classical matching and when they fail.

The research questions this module answers:

* When does an ML decoder outperform MWPM? (Typically small distances and
  higher error rates, where the syndrome-to-flip mapping is learnable from
  modest data.)
* When does it fail? (Larger distances, where the input space grows and the
  flip becomes rare, starving the classifier of positive examples; and any
  shift between the training and evaluation noise distribution.)
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from decbench.runner import run_benchmark
from decbench.types import BenchmarkConfig, BenchmarkResult, RunRecord
from matplotlib.axes import Axes


@dataclass(frozen=True)
class RegimeRow:
    """Comparison of the best ML decoder against MWPM at one (distance, p)."""

    distance: int
    p: float
    mwpm_ler: float
    best_ml_decoder: str
    best_ml_ler: float

    @property
    def ml_wins(self) -> bool:
        return self.best_ml_ler <= self.mwpm_ler


def compare(
    *,
    distances: list[int],
    error_rates: list[float],
    ml_decoders: tuple[str, ...] = ("rf", "xgb", "mlp"),
    baseline: str = "mwpm",
    shots: int = 5_000,
    seed: int = 2026,
) -> BenchmarkResult:
    """Run the baseline and ML decoders over a distance/error-rate grid."""
    config = BenchmarkConfig(
        decoders=[baseline, *ml_decoders],
        distances=distances,
        error_rates=error_rates,
        shots=shots,
        seed=seed,
    )
    return run_benchmark(config)


def regime_table(result: BenchmarkResult, *, baseline: str = "mwpm") -> list[RegimeRow]:
    """Reduce a benchmark sweep to a per-point ML-vs-baseline comparison."""
    points: dict[tuple[int, float], dict[str, RunRecord]] = {}
    for record in result.records:
        points.setdefault((record.distance, record.p), {})[record.decoder] = record

    rows: list[RegimeRow] = []
    for (distance, p), by_decoder in sorted(points.items()):
        if baseline not in by_decoder:
            continue
        ml_records = [(name, rec) for name, rec in by_decoder.items() if name != baseline]
        if not ml_records:
            continue
        best_name, best_record = min(ml_records, key=lambda item: item[1].logical_error_rate)
        rows.append(
            RegimeRow(
                distance=distance,
                p=p,
                mwpm_ler=by_decoder[baseline].logical_error_rate,
                best_ml_decoder=best_name,
                best_ml_ler=best_record.logical_error_rate,
            )
        )
    return rows


def plot_ml_vs_baseline(rows: list[RegimeRow], *, ax: Axes | None = None) -> Axes:
    """Scatter best-ML vs baseline logical error rate; points below the diagonal are ML wins."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    baseline = [row.mwpm_ler for row in rows]
    ml = [row.best_ml_ler for row in rows]
    ax.scatter(baseline, ml, s=60)
    limit = max([*baseline, *ml, 1e-3])
    ax.plot([0, limit], [0, limit], linestyle="--", color="gray", label="parity")

    ax.set_xlabel("MWPM logical error rate")
    ax.set_ylabel("Best ML logical error rate")
    ax.set_title("ML decoders vs MWPM (below diagonal = ML wins)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
