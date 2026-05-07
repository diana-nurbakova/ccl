"""CLI entry point for cross-experiment synthesis."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.synthesis.cross_experiment import run_synthesis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-experiment synthesis"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results = run_synthesis()
    out_dir = args.output_dir or Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    results["table"].to_csv(out_dir / "synthesis_table.csv", index=False)
    print(f"\nSynthesis table saved to {out_dir}/synthesis_table.csv")


if __name__ == "__main__":
    main()
