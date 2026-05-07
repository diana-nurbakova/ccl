"""CLI entry point for Experiment B."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exp_b_active_passive.analysis import run_experiment_b
from experiments.shared.plotting import save_figure, setup_style


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment B: No-Self-Merge — Active vs Passive AI Use"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    setup_style()

    results = run_experiment_b(force_download=args.force_download)
    out_dir = args.output_dir or Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    results["comparison"].to_csv(out_dir / "exp_b_condition_comparison.csv", index=False)
    if "lmm_table" in results:
        results["lmm_table"].to_csv(
            out_dir / "exp_b_lmm_table.csv", index=False
        )

    # Plot: turn type rates by condition
    metrics_df = results["metrics_df"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (metric, label) in enumerate([
        ("evaluative_rate", "Evaluative Rate"),
        ("active_rate", "Active Rate"),
        ("passive_rate", "Passive Rate"),
    ]):
        for treatment, color in [("vanilla", "#E53935"), ("aug", "#1E88E5")]:
            subset = metrics_df[metrics_df["treatment"] == treatment][metric]
            axes[i].hist(
                subset, bins=20, alpha=0.5, label=treatment, color=color,
            )
        axes[i].set_title(label)
        axes[i].set_xlabel("Rate")
        axes[i].set_ylabel("Count")
        axes[i].legend()

    fig.suptitle("Experiment B: Turn Type Distributions by Condition", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "exp_b_turn_distributions", output_dir=out_dir / "figures")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
