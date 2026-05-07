"""Tests for the regex-based turn classifier."""

import pytest

from experiments.shared.turn_classifier import TurnType, classify_turn, conversation_metrics


class TestClassifyTurn:
    """Test classification against spec-validated examples and edge cases."""

    # Manual validation samples from the spec (all confirmed EVALUATIVE)
    @pytest.mark.parametrize("text", [
        "are you sure that the answer is 7",
        "you are wrong x can not be equal to 5",
        "I found the answer 45, is that correct?",
        "are you sure",
        "that's wrong",
        "that is not correct",
        "you're wrong about that",
        "is that right?",
        "can you double check your answer",
        "check again",
        "I think the answer is 12",
        "I got 42",
        "that doesn't seem right",
        "I don't think that's correct",
        "no, the answer should be 8",
    ])
    def test_evaluative_patterns(self, text: str) -> None:
        assert classify_turn(text) == TurnType.EVALUATIVE

    @pytest.mark.parametrize("text", [
        "let me try solving this",
        "I'll work through it step by step",
        "so if x = 3 then y = 6",
        "give me a hint",
        "what's the first step",
        "how should I approach this problem",
        "step 1: find the derivative",
        "what formula should I use",
        "substituting x into the equation",
        "simplifying the expression",
        "if we multiply both sides by 2",
    ])
    def test_active_patterns(self, text: str) -> None:
        assert classify_turn(text) == TurnType.ACTIVE

    @pytest.mark.parametrize("text", [
        "ok",
        "thanks",
        "yes",
        "got it",
        "alright",
        "",
        "okay",
        "thank you",
    ])
    def test_passive_patterns(self, text: str) -> None:
        assert classify_turn(text) == TurnType.PASSIVE

    def test_evaluative_takes_priority_over_active(self) -> None:
        """If a turn matches both EVALUATIVE and ACTIVE, EVALUATIVE wins."""
        text = "are you sure? let me try calculating it myself"
        assert classify_turn(text) == TurnType.EVALUATIVE


class TestConversationMetrics:
    def test_empty_conversation(self) -> None:
        result = conversation_metrics([])
        assert result["n_turns"] == 0
        assert result["evaluative_rate"] == 0.0

    def test_mixed_conversation(self) -> None:
        turns = [
            "ok",                          # passive
            "are you sure about that?",    # evaluative
            "let me try step 1",           # active
            "thanks",                      # passive
        ]
        result = conversation_metrics(turns)
        assert result["n_turns"] == 4
        assert result["evaluative_rate"] == 0.25
        assert result["active_rate"] == 0.25
        assert result["passive_rate"] == 0.50
