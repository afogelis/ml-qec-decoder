"""Command-line interface for ML-vs-classical decoder comparison.

Examples
--------
    mldecoder compare --distances 3,5 --p 0.01,0.02 --shots 5000 --train-shots 20000
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from decbench.leaderboard import build_leaderboard, format_leaderboard

from . import register_ml_decoders
from .analysis import compare, regime_table


def _ints(raw: str) -> list[int]:
    return [int(token) for token in raw.split(",") if token.strip()]


def _floats(raw: str) -> list[float]:
    return [float(token) for token in raw.split(",") if token.strip()]


def _cmd_compare(args: argparse.Namespace) -> int:
    register_ml_decoders(train_shots=args.train_shots)
    result = compare(
        distances=_ints(args.distances),
        error_rates=_floats(args.p),
        shots=args.shots,
        seed=args.seed,
    )
    print(format_leaderboard(build_leaderboard(result)))
    print()
    for row in regime_table(result):
        verdict = "ML WINS " if row.ml_wins else "mwpm wins"
        print(
            f"d={row.distance} p={row.p:<7} {verdict} "
            f"mwpm={row.mwpm_ler:.4e} best_ml={row.best_ml_decoder}:{row.best_ml_ler:.4e}"
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(json.loads(result.model_dump_json()), handle, indent=2)
        print(f"\nwrote {len(result.records)} records to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mldecoder", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compare_cmd = sub.add_parser("compare", help="Compare ML decoders against MWPM.")
    compare_cmd.add_argument("--distances", type=str, required=True)
    compare_cmd.add_argument("--p", type=str, required=True)
    compare_cmd.add_argument("--shots", type=int, default=5_000)
    compare_cmd.add_argument("--train-shots", type=int, default=20_000)
    compare_cmd.add_argument("--seed", type=int, default=2026)
    compare_cmd.add_argument("--output", type=str, default=None)
    compare_cmd.set_defaults(func=_cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
