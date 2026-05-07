"""Run all CCL validation experiments and produce complete results.

Usage:
    python -m experiments.run_all [--output-dir DIR] [--force-download]
    python experiments/run_all.py [--output-dir DIR] [--force-download]
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure importability
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all CCL validation experiments"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "output",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["a", "a_felm", "b", "c", "d", "synthesis"],
        help="Skip specific experiments",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()

    # --- Experiment A (FRANK) ---
    if "a" not in args.skip:
        print("\n" + "=" * 70)
        import pandas as pd
        from experiments.exp_a_error_detection.analysis import run_experiment_a
        results["a"] = run_experiment_a(force_download=args.force_download)
        results["a"]["stats"].to_csv(
            args.output_dir / "exp_a_category_stats.csv", index=False
        )
        glmm = results["a"].get("glmm", {})
        if glmm and "error" not in glmm:
            glmm_row = {k: v for k, v in glmm.items() if k != "random_effects_sd"}
            for nm, sd in glmm.get("random_effects_sd", {}).items():
                glmm_row[f"sd_{nm}"] = sd
            pd.DataFrame([glmm_row]).to_csv(
                args.output_dir / "exp_a_glmm.csv", index=False
            )

    # --- Experiment A' (FELM) ---
    if "a_felm" not in args.skip:
        print("\n" + "=" * 70)
        import pandas as pd
        from experiments.exp_a_felm.analysis import run_experiment_a_felm
        results["a_felm"] = run_experiment_a_felm(force_download=args.force_download)
        results["a_felm"]["by_ccl"].to_csv(
            args.output_dir / "exp_a_felm_by_ccl.csv", index=False
        )
        results["a_felm"]["by_domain"].to_csv(
            args.output_dir / "exp_a_felm_by_domain.csv", index=False
        )
        if "prompt_stats" in results["a_felm"]:
            results["a_felm"]["prompt_stats"].to_csv(
                args.output_dir / "exp_a_felm_prompt_stats.csv", index=False
            )
        mw = results["a_felm"].get("mannwhitney_prompts", {})
        kruskal = results["a_felm"].get("kruskal_prompts", {})
        if mw and "error" not in mw:
            row = dict(mw)
            if "error" not in kruskal:
                row.update({"kruskal_H": kruskal["H"],
                            "kruskal_p": kruskal["p_value"]})
            pd.DataFrame([row]).to_csv(
                args.output_dir / "exp_a_felm_prompt_test.csv", index=False
            )

    # --- Experiment B ---
    if "b" not in args.skip:
        print("\n" + "=" * 70)
        from experiments.exp_b_active_passive.analysis import run_experiment_b
        results["b"] = run_experiment_b(force_download=args.force_download)
        results["b"]["comparison"].to_csv(
            args.output_dir / "exp_b_condition_comparison.csv", index=False
        )
        if "lmm_table" in results["b"]:
            results["b"]["lmm_table"].to_csv(
                args.output_dir / "exp_b_lmm_table.csv", index=False
            )

    # --- Experiment C ---
    if "c" not in args.skip:
        print("\n" + "=" * 70)
        from experiments.exp_c_declining_effort.analysis import run_experiment_c
        results["c"] = run_experiment_c(force_download=args.force_download)
        results["c"]["slopes"].to_csv(
            args.output_dir / "exp_c_trajectory_slopes.csv", index=False
        )
        results["c"]["session_summary"].to_csv(
            args.output_dir / "exp_c_session_summary.csv", index=False
        )

    # --- Experiment D ---
    if "d" not in args.skip:
        print("\n" + "=" * 70)
        from experiments.exp_d_wildchat_decay.analysis import run_experiment_d
        results["d"] = run_experiment_d(force_download=args.force_download)
        results["d"]["decay_by_order"].to_csv(
            args.output_dir / "exp_d_decay_by_order.csv", index=False
        )
        results["d"]["decay_by_weeks"].to_csv(
            args.output_dir / "exp_d_decay_by_weeks.csv", index=False
        )

    # --- Synthesis ---
    if "synthesis" not in args.skip:
        print("\n" + "=" * 70)
        from experiments.synthesis.cross_experiment import run_synthesis
        synth = run_synthesis(
            exp_a_results=results.get("a"),
            exp_a_felm_results=results.get("a_felm"),
            exp_b_results=results.get("b"),
            exp_c_results=results.get("c"),
            exp_d_results=results.get("d"),
        )
        synth["table"].to_csv(
            args.output_dir / "synthesis_table.csv", index=False
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All experiments complete in {elapsed:.1f}s")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
