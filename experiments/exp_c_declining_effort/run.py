"""CLI entry point for Experiment C."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exp_c_declining_effort.analysis import run_experiment_c
from experiments.shared.plotting import save_figure, setup_style, trajectory_plot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment C: Declining Effort Across Sessions"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    setup_style()

    results = run_experiment_c(force_download=args.force_download)
    out_dir = args.output_dir or Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    results["session_summary"].to_csv(out_dir / "exp_c_session_summary.csv", index=False)
    results["slopes"].to_csv(out_dir / "exp_c_trajectory_slopes.csv", index=False)

    # Plot: session trajectories by treatment
    summary = results["session_summary"]
    metrics_to_plot = ["passive_rate", "active_rate", "evaluative_rate"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, metric in enumerate(metrics_to_plot):
        for treatment in summary["treatment"].unique():
            t_data = summary[summary["treatment"] == treatment]
            axes[i].plot(
                t_data["session_id"], t_data[metric],
                marker="o", label=treatment,
            )
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_xlabel("Session")
        axes[i].set_ylabel("Rate")
        axes[i].legend()

    fig.suptitle("Experiment C: Engagement Trajectories Across Sessions",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "exp_c_session_trajectories", output_dir=out_dir / "figures")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
