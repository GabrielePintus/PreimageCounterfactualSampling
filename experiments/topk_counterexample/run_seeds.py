"""Run the controlled top-k counterexample over multiple random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from end_to_end import DEFAULT_REPOSITORY, HERE, Configuration, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    completed = []
    for position, seed in enumerate(args.seeds, start=1):
        output = args.output / f"seed_{seed}"
        print(f"[SEED {position}/{len(args.seeds)}] seed={seed} -> {output}", flush=True)
        result = run(
            Configuration(seed=seed, n_queries=args.queries),
            args.repository.resolve(),
            output.resolve(),
        )
        completed.append(
            {
                "seed": seed,
                "failure_rate": result["queries"]["failure_rate"],
                "mean_gap": result["queries"]["mean_gap"],
                "mean_ratio": result["queries"]["mean_ratio"],
            }
        )
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
