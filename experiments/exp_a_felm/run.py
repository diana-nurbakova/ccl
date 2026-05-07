"""CLI entry point for Experiment A' (FELM)."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exp_a_felm.analysis import (
    compute_error_rates,
    compute_error_type_distribution,
    add_ccl_category,
    run_experiment_a_felm,
)
from experiments.shared.plotting import ccl_palette, save_figure, setup_style


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment A' (FELM): Claim-Type Asymmetry in ChatGPT"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    setup_style()

    results = run_experiment_a_felm(force_download=args.force_download)
    out_dir = args.output_dir or Path(__file__).parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    results["by_domain"].to_csv(out_dir / "exp_a_felm_by_domain.csv", index=False)
    results["by_ccl"].to_csv(out_dir / "exp_a_felm_by_ccl.csv", index=False)
    if "prompt_stats" in results:
        results["prompt_stats"].to_csv(
            out_dir / "exp_a_felm_prompt_stats.csv", index=False
        )
    import pandas as pd
    mw = results.get("mannwhitney_prompts", {})
    kruskal = results.get("kruskal_prompts", {})
    if mw and "error" not in mw:
        row = dict(mw)
        if "error" not in kruskal:
            row.update({"kruskal_H": kruskal["H"],
                        "kruskal_p": kruskal["p_value"]})
        pd.DataFrame([row]).to_csv(
            out_dir / "exp_a_felm_prompt_test.csv", index=False
        )

    palette = ccl_palette()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: error rate by domain (grouped by CCL category)
    by_domain = results["by_domain"].sort_values(["ccl_category", "domain"])
    colors = [palette.get(c, "#888") for c in by_domain["ccl_category"]]
    bars = axes[0].bar(by_domain["domain"], by_domain["error_rate"],
                       color=colors, edgecolor="white")
    axes[0].set_title("Error Rate by Domain")
    axes[0].set_ylabel("Error rate")
    axes[0].set_xlabel("FELM domain")
    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palette.get(cat, "#888"), label=cat)
        for cat in ["FACTUAL", "INTERPRETIVE", "GAP"]
    ]
    axes[0].legend(handles=legend_elements, fontsize=8)

    # Plot 2: error rate by CCL category
    by_ccl = results["by_ccl"]
    cat_colors = [palette.get(c, "#888") for c in by_ccl["ccl_category"]]
    axes[1].bar(by_ccl["ccl_category"], by_ccl["error_rate"],
                color=cat_colors, edgecolor="white")
    axes[1].set_title("Error Rate by CCL Category")
    axes[1].set_ylabel("Error rate")

    # Plot 3: error type heatmap
    type_dist = results["error_type_dist"]
    if len(type_dist) > 0:
        type_cols = [c for c in type_dist.columns if c != "total"]
        if type_cols:
            mat = type_dist[type_cols].values.astype(float)
            # Normalise each row to proportions
            row_sums = mat.sum(axis=1, keepdims=True)
            mat_norm = np.divide(mat, row_sums, where=row_sums > 0)
            im = axes[2].imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
            axes[2].set_xticks(range(len(type_cols)))
            axes[2].set_xticklabels(type_cols, rotation=30, ha="right", fontsize=8)
            axes[2].set_yticks(range(len(type_dist.index)))
            axes[2].set_yticklabels(type_dist.index, fontsize=9)
            axes[2].set_title("Error Type Distribution\n(row-normalised)")
            plt.colorbar(im, ax=axes[2], shrink=0.8)
    else:
        axes[2].text(0.5, 0.5, "No typed errors", ha="center", va="center")
        axes[2].set_title("Error Type Distribution")

    fig.suptitle("Experiment A' (FELM): Claim-Type Asymmetry in ChatGPT",
                 fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "exp_a_felm_error_rates", output_dir=out_dir / "figures")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
