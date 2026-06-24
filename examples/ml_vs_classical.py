"""Compare ML decoders against MWPM and save the regime analysis plots.

    python examples/ml_vs_classical.py

Writes outputs/ (gitignored).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decbench.leaderboard import build_leaderboard, format_leaderboard

from mldecoder import register_ml_decoders
from mldecoder.analysis import compare, plot_ml_vs_baseline, regime_table


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    register_ml_decoders(train_shots=20_000)

    result = compare(
        distances=[3, 5],
        error_rates=[0.01, 0.015, 0.02, 0.03],
        shots=5_000,
        seed=2026,
    )

    print(format_leaderboard(build_leaderboard(result)))
    rows = regime_table(result)
    for row in rows:
        verdict = "ML WINS " if row.ml_wins else "mwpm wins"
        print(f"d={row.distance} p={row.p:<7} {verdict} mwpm={row.mwpm_ler:.4e} "
              f"best_ml={row.best_ml_decoder}:{row.best_ml_ler:.4e}")

    with open("outputs/ml_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(json.loads(result.model_dump_json()), handle, indent=2)

    ax = plot_ml_vs_baseline(rows)
    ax.figure.tight_layout()
    ax.figure.savefig("outputs/ml_vs_mwpm.png", dpi=150)
    plt.close(ax.figure)
    print("saved outputs/ml_comparison.json and outputs/ml_vs_mwpm.png")


if __name__ == "__main__":
    main()
