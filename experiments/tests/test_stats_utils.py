"""Tests for statistical utility functions."""

import numpy as np
import pandas as pd
import pytest

from experiments.shared.stats_utils import (
    cohens_d,
    fisher_exact_2x2,
    fleiss_kappa,
    per_subject_slopes,
    slope_ttest,
    type_token_ratio,
)


class TestFleissKappa:
    def test_perfect_agreement(self) -> None:
        """All raters agree on every subject -> kappa = 1."""
        ratings = np.array([
            [3, 0],  # all 3 raters pick category 0
            [0, 3],  # all 3 raters pick category 1
            [3, 0],
        ])
        assert fleiss_kappa(ratings) == pytest.approx(1.0)

    def test_known_textbook_value(self) -> None:
        """Fleiss (1971) example from Wikipedia / textbooks."""
        # 10 subjects, 6 raters, 5 categories
        ratings = np.array([
            [0, 0, 0, 0, 6],
            [0, 2, 0, 4, 0],
            [0, 0, 4, 2, 0],
            [0, 3, 0, 3, 0],
            [2, 2, 0, 2, 0],
            [0, 0, 6, 0, 0],
            [4, 0, 0, 0, 2],
            [0, 1, 0, 5, 0],
            [0, 0, 4, 2, 0],
            [0, 2, 2, 2, 0],
        ])
        kappa = fleiss_kappa(ratings)
        # Verified against statsmodels.stats.inter_rater.fleiss_kappa
        assert kappa == pytest.approx(0.388, abs=0.01)


class TestCohensD:
    def test_identical_groups(self) -> None:
        g = np.array([1.0, 2.0, 3.0])
        assert cohens_d(g, g) == 0.0

    def test_known_value(self) -> None:
        g1 = np.array([2.0, 4.0, 6.0, 8.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0])
        d = cohens_d(g1, g2)
        # mean diff = 2.5, pooled SD = sqrt((var1+var2)/2) ≈ 2.236
        assert d == pytest.approx(2.5 / np.sqrt(
            (np.var(g1, ddof=1) * 3 + np.var(g2, ddof=1) * 3) / 6
        ), abs=0.01)


class TestFisherExact:
    def test_basic_table(self) -> None:
        table = [[10, 5], [3, 12]]
        result = fisher_exact_2x2(table)
        assert "odds_ratio" in result
        assert "p_value" in result
        assert result["odds_ratio"] > 1.0
        assert result["p_value"] < 0.05


class TestPerSubjectSlopes:
    def test_positive_trend(self) -> None:
        df = pd.DataFrame({
            "subj": ["A", "A", "A", "B", "B", "B"],
            "time": [1, 2, 3, 1, 2, 3],
            "metric": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
        })
        slopes = per_subject_slopes(df, "subj", "time", "metric")
        assert slopes["A"] == pytest.approx(1.0)
        assert slopes["B"] == pytest.approx(2.0)

    def test_slope_ttest(self) -> None:
        slopes = pd.Series([0.1, 0.2, -0.05, 0.15, 0.3])
        result = slope_ttest(slopes)
        assert result["mean_slope"] == pytest.approx(slopes.mean())
        assert result["n"] == 5
        assert 0.0 <= result["pct_declining"] <= 1.0


class TestTypeTTR:
    def test_all_unique(self) -> None:
        assert type_token_ratio("the quick brown fox") == 1.0

    def test_with_repeats(self) -> None:
        assert type_token_ratio("the the the") == pytest.approx(1 / 3)

    def test_empty(self) -> None:
        assert type_token_ratio("") == 0.0
