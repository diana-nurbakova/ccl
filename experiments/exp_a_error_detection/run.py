"""CLI entry point for Experiment A."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure experiments package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exp_a_error_detection.analysis import run_experiment_a
from experiments.shared.plotting import (
    grouped_bar_chart,
    save_figure,
    setup_style,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment A: Differential Error Detection by Claim Type"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output figures/tables",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download data even if cached",
    )
    args = parser.parse_args()
    setup_style()

    results = run_experiment_a(force_download=args.force_download)
    stats_df = results["stats"]

    # Save results table
    out_dir = args.output_dir or Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_dir / "exp_a_category_stats.csv", index=False)

    # Save GLMM result table (one row, key fields)
    import pandas as pd
    glmm = results.get("glmm", {})
    if glmm and "error" not in glmm:
        glmm_row = {
            k: v for k, v in glmm.items()
            if k != "random_effects_sd"
        }
        for re_name, re_sd in glmm.get("random_effects_sd", {}).items():
            glmm_row[f"sd_{re_name}"] = re_sd
        pd.DataFrame([glmm_row]).to_csv(
            out_dir / "exp_a_glmm.csv", index=False
        )

    # Plot: majority agreement by category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    agreement_data = dict(
        zip(stats_df["ccl_category"], stats_df["majority_agreement"])
    )
    grouped_bar_chart(
        agreement_data,
        title="Majority Agreement Rate by CCL Category",
        ylabel="Proportion (≥2/3 annotators agree)",
        ax=axes[0],
    )

    kappa_data = dict(
        zip(stats_df["ccl_category"], stats_df["fleiss_kappa"])
    )
    grouped_bar_chart(
        kappa_data,
        title="Fleiss' κ by CCL Category",
        ylabel="Fleiss' κ",
        ax=axes[1],
    )

    fig.suptitle("Experiment A: Error Detection Reliability", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "exp_a_agreement_kappa", output_dir=out_dir / "figures")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
