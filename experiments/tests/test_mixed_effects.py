"""Tests for the mixed-effects reanalyses (Statistical Independence Fix).

These tests use synthetic data with a known structure so we can verify
the model fits the expected effect under nested observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Exp A — FRANK GLMM
# ---------------------------------------------------------------------------

class TestFrankGLMM:
    """Synthetic flagged-sentence data with a known FACTUAL > INTERPRETIVE effect."""

    @staticmethod
    def _make_synthetic(n_summaries: int = 60, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        models = [f"m{k}" for k in range(4)]
        rows = []
        for s in range(n_summaries):
            model = rng.choice(models)
            summary_id = f"sum_{s}_{model}"
            # 2 FACTUAL + 2 INTERPRETIVE flagged sentences per summary
            for cat, prob in [("FACTUAL", 0.7), ("INTERPRETIVE", 0.3)]:
                for _ in range(2):
                    rows.append({
                        "sentence_id": f"{summary_id}_{rng.integers(1_000_000)}",
                        "summary_id": summary_id,
                        "model_name": model,
                        "ccl_category": cat,
                        "any_flagged": True,
                        "majority_agree": bool(rng.random() < prob),
                    })
        return pd.DataFrame(rows)

    def test_glmm_recovers_positive_effect(self) -> None:
        from experiments.exp_a_error_detection.analysis import (
            glmm_factual_vs_interpretive,
        )
        df = self._make_synthetic(n_summaries=80, seed=1)
        result = glmm_factual_vs_interpretive(df)

        # Direction: FACTUAL > INTERPRETIVE → positive beta, OR > 1
        assert result["beta_factual"] > 0
        assert result["odds_ratio"] > 1.0
        # Significant at alpha = 0.05 with this synthetic effect size
        assert result["p_value"] < 0.05
        # CI lower bound should be > 1 (consistent with significance)
        assert result["or_ci_low"] > 1.0
        # Sanity-check structure
        assert result["n_observations"] == 4 * 80
        assert result["n_summaries"] == 80
        assert "summary" in result["random_effects_sd"]
        assert "model" in result["random_effects_sd"]


# ---------------------------------------------------------------------------
# Exp A' — FELM prompt-level analysis
# ---------------------------------------------------------------------------

class TestFelmPromptLevel:
    @staticmethod
    def _make_segments(seed: int = 0) -> pd.DataFrame:
        """Synthetic FELM-shaped segments with known per-prompt error rates.

        FACTUAL prompts: ~30 % error rate; INTERPRETIVE prompts: ~10 %.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for p in range(60):
            cat = "FACTUAL"
            error_p = 0.30
            for _ in range(rng.integers(3, 8)):
                rows.append({
                    "record_index": f"f{p}",
                    "domain": "wk",
                    "ccl_category": cat,
                    "is_error": bool(rng.random() < error_p),
                })
        for p in range(60):
            cat = "INTERPRETIVE"
            error_p = 0.10
            for _ in range(rng.integers(3, 8)):
                rows.append({
                    "record_index": f"i{p}",
                    "domain": "reasoning",
                    "ccl_category": cat,
                    "is_error": bool(rng.random() < error_p),
                })
        return pd.DataFrame(rows)

    def test_compute_prompt_level_error_rates(self) -> None:
        from experiments.exp_a_felm.analysis import (
            compute_prompt_level_error_rates,
        )
        df = self._make_segments(seed=2)
        prompt_stats = compute_prompt_level_error_rates(df)
        # One row per prompt
        assert prompt_stats["record_index"].is_unique
        assert len(prompt_stats) == 120
        # Error rate stays in [0, 1]
        assert prompt_stats["error_rate"].between(0, 1).all()

    def test_mannwhitney_recovers_directional_effect(self) -> None:
        from experiments.exp_a_felm.analysis import (
            compute_prompt_level_error_rates,
            mannwhitney_factual_vs_interpretive_prompts,
        )
        df = self._make_segments(seed=3)
        prompt_stats = compute_prompt_level_error_rates(df)
        result = mannwhitney_factual_vs_interpretive_prompts(
            prompt_stats, alternative="greater",
        )
        # FACTUAL prompts should rank higher than INTERPRETIVE
        assert result["mean_factual"] > result["mean_interpretive"]
        # Rank-biserial r > 0 means FACTUAL ranks higher (correct sign)
        assert result["rank_biserial"] > 0
        # Significant under the directional alternative
        assert result["p_value"] < 0.01

    def test_kruskal_runs(self) -> None:
        from experiments.exp_a_felm.analysis import (
            compute_prompt_level_error_rates,
            kruskal_across_ccl_prompts,
        )
        df = self._make_segments(seed=4)
        prompt_stats = compute_prompt_level_error_rates(df)
        result = kruskal_across_ccl_prompts(prompt_stats)
        assert "H" in result
        assert result["p_value"] < 0.05


# ---------------------------------------------------------------------------
# Exp B — LMM with student + class random intercepts
# ---------------------------------------------------------------------------

class TestBastaniLMM:
    @staticmethod
    def _make_metrics(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Synthetic conversation-level metrics with a known treatment effect."""
        rng = np.random.default_rng(seed)
        n_classes = 8
        students_per_class = 6
        convs_per_student = 4

        # half the classes are aug, half are vanilla
        rows_metrics = []
        rows_outcomes = []
        for c in range(n_classes):
            treatment = "aug" if c % 2 == 0 else "vanilla"
            class_intercept = rng.normal(0, 0.3)
            for s in range(students_per_class):
                student_id = f"c{c}_s{s}"
                student_intercept = rng.normal(0, 0.5) + class_intercept
                rows_outcomes.append({
                    "Student ID": student_id,
                    "Class": f"class_{c}",
                })
                for conv in range(convs_per_student):
                    base = 5.0 if treatment == "aug" else 2.5
                    n_turns = max(
                        1, base + student_intercept + rng.normal(0, 0.6)
                    )
                    rows_metrics.append({
                        "username": student_id,
                        "treatment": treatment,
                        "n_turns": n_turns,
                        "mean_words_per_turn": rng.normal(6, 1),
                        "evaluative_rate": rng.uniform(0, 0.05),
                        "active_rate": rng.uniform(0, 0.1),
                        "passive_rate": rng.uniform(0.85, 1.0),
                    })
        return pd.DataFrame(rows_metrics), pd.DataFrame(rows_outcomes)

    def test_attach_class_id_merges_correctly(self) -> None:
        from experiments.exp_b_active_passive.analysis import attach_class_id
        metrics, outcomes = self._make_metrics(seed=5)
        merged = attach_class_id(metrics, outcomes)
        assert "class_id" in merged.columns
        assert merged["class_id"].notna().all()
        assert merged["class_id"].nunique() == 8

    def test_lmm_recovers_treatment_effect_on_n_turns(self) -> None:
        from experiments.exp_b_active_passive.analysis import (
            attach_class_id,
            conversation_level_lmm,
        )
        metrics, outcomes = self._make_metrics(seed=6)
        merged = attach_class_id(metrics, outcomes)
        lmm = conversation_level_lmm(merged, metric_cols=["n_turns"])
        row = lmm.iloc[0]
        # vanilla - aug should be negative (aug has higher n_turns by design)
        assert row["beta_vanilla_minus_aug"] < 0
        # Effect should be detectable with this synthetic effect size
        assert row["p_value"] < 0.05
        assert row["n_obs"] == 8 * 6 * 4
        assert row["n_classes"] == 8
        assert row["n_students"] == 8 * 6
