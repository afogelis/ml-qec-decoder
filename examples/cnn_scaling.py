"""Does a geometry-aware CNN scale with code distance better than tabular ML?

Compares the tabular decoders (random forest, gradient boosting, MLP) and the
geometry-aware CNN against the MWPM baseline as the code distance grows. Writes
a logical-error-rate-vs-distance figure to docs/ (committed for the README).

    python examples/cnn_scaling.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mldecoder import register_ml_decoders
from mldecoder.analysis import accuracy_vs_distance, compare, plot_decoder_scaling


def main() -> None:
    os.makedirs("docs", exist_ok=True)
    register_ml_decoders(train_shots=30_000)

    # Below the ~1% threshold, MWPM suppresses the logical error rate as the
    # distance grows; the question is whether the learned decoders can follow.
    p = 0.006
    result = compare(
        distances=[3, 5, 7],
        error_rates=[p],
        shots=20_000,
        seed=2026,
    )

    print(f"Logical error rate by decoder and distance (p = {p}):")
    for name, points in sorted(accuracy_vs_distance(result, p=p).items()):
        trail = "  ".join(f"d={d}:{ler:.3e}" for d, ler in points)
        print(f"  {name:<5} {trail}")

    ax = plot_decoder_scaling(result, p=p)
    ax.figure.tight_layout()
    ax.figure.savefig("docs/cnn_scaling.png", dpi=150)
    plt.close(ax.figure)
    print("saved docs/cnn_scaling.png")


if __name__ == "__main__":
    main()
