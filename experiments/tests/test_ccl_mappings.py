"""Tests for CCL category mappings."""

import pytest

from experiments.shared.ccl_mappings import (
    CCL_CATEGORIES,
    FRANK_CODE_TO_CCL,
    FRANK_ERROR_CODES,
    FRANK_NO_ERROR,
    FRANK_TO_CCL,
    classify_frank_errors,
)


class TestFrankToCCL:
    def test_all_error_codes_mapped(self) -> None:
        """Every non-NoE FRANK code should map to exactly one CCL category."""
        all_codes = set()
        for codes in FRANK_TO_CCL.values():
            all_codes.update(codes)
        assert all_codes == FRANK_ERROR_CODES

    def test_no_overlap_between_categories(self) -> None:
        """Each FRANK code maps to exactly one CCL category."""
        seen = set()
        for codes in FRANK_TO_CCL.values():
            for code in codes:
                assert code not in seen, f"{code} mapped to multiple categories"
                seen.add(code)

    def test_factual_codes(self) -> None:
        assert FRANK_TO_CCL["FACTUAL"] == ["EntE", "OutE", "GramE"]

    def test_source_codes(self) -> None:
        assert FRANK_TO_CCL["SOURCE"] == ["CircE"]

    def test_interpretive_codes(self) -> None:
        assert FRANK_TO_CCL["INTERPRETIVE"] == ["RelE", "LinkE", "CorefE"]


class TestClassifyFrankErrors:
    def test_no_error(self) -> None:
        assert classify_frank_errors(["NoE"]) == set()

    def test_single_factual(self) -> None:
        assert classify_frank_errors(["EntE"]) == {"FACTUAL"}

    def test_multiple_same_category(self) -> None:
        assert classify_frank_errors(["EntE", "OutE"]) == {"FACTUAL"}

    def test_cross_category(self) -> None:
        assert classify_frank_errors(["EntE", "RelE"]) == {"FACTUAL", "INTERPRETIVE"}

    def test_all_categories(self) -> None:
        result = classify_frank_errors(["EntE", "CircE", "RelE"])
        assert result == {"FACTUAL", "SOURCE", "INTERPRETIVE"}

    def test_empty_list(self) -> None:
        assert classify_frank_errors([]) == set()

    def test_unknown_code_ignored(self) -> None:
        assert classify_frank_errors(["UnknownCode"]) == set()
