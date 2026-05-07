"""CLI entry point for Experiment D."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exp_d_wildchat_decay.analysis import run_experiment_d
from experiments.shared.plotting import save_figure, setup_style


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment D: Longitudinal Engagement Decay (WildChat)"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    setup_style()

    results = run_experiment_d(force_download=args.force_download)
    out_dir = args.output_dir or Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    results["decay_by_order"].to_csv(out_dir / "exp_d_decay_by_order.csv", index=False)
    results["decay_by_weeks"].to_csv(out_dir / "exp_d_decay_by_weeks.csv", index=False)

    # Plot 1: Engagement metrics over conversation order (binned)
    conv = results["conv_metrics"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("passive_rate", "Passive Rate"),
        ("evaluative_rate", "Evaluative Rate"),
        ("mean_words_per_turn", "Mean Words/Turn"),
        ("mean_ttr", "Lexical Diversity (TTR)"),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics):
        # Bin by conversation order (quintiles)
        conv["order_bin"] = pd.qcut(
            conv["conv_order"], q=10, labels=False, duplicates="drop",
        )
        binned = conv.groupby("order_bin")[metric].agg(["mean", "sem"])
        ax.errorbar(
            binned.index + 1, binned["mean"], yerr=binned["sem"],
            marker="o", capsize=3,
        )
        ax.set_title(title)
        ax.set_xlabel("Conversation Order Decile")
        ax.set_ylabel(title)

    fig.suptitle("Experiment D: Engagement Decay Over Time (WildChat)",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "exp_d_decay_trajectories", output_dir=out_dir / "figures")

    # Plot 2: Slope distributions
    from experiments.shared.stats_utils import per_subject_slopes
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, title) in zip(axes2, [
        ("passive_rate", "Passive Rate Slope"),
        ("evaluative_rate", "Evaluative Rate Slope"),
        ("mean_words_per_turn", "Words/Turn Slope"),
    ]):
        slopes = per_subject_slopes(
            conv, subject_col="hashed_ip",
            time_col="conv_order", metric_col=metric,
        )
        ax.hist(slopes.values, bins=40, edgecolor="white", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.axvline(slopes.mean(), color="blue", linestyle="-", alpha=0.7,
                   label=f"mean={slopes.mean():.4f}")
        ax.set_title(title)
        ax.set_xlabel("Slope")
        ax.legend()

    fig2.suptitle("Experiment D: Per-User Slope Distributions", fontweight="bold")
    fig2.tight_layout()
    save_figure(fig2, "exp_d_slope_distributions", output_dir=out_dir / "figures")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
