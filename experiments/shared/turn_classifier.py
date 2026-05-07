"""Regex-based turn classifier for human–AI conversation analysis.

Classifies user turns into three categories following the CCL framework:
  - EVALUATIVE: challenges to AI correctness ("are you sure", "that's wrong")
  - ACTIVE: working through the problem (showing work, asking hints)
  - PASSIVE: everything else (acknowledgements, copy requests, short affirmations)
"""

from __future__ import annotations

import re
from enum import Enum


class TurnType(str, Enum):
    EVALUATIVE = "EVALUATIVE"
    ACTIVE = "ACTIVE"
    PASSIVE = "PASSIVE"


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# EVALUATIVE: user challenges the AI's correctness
_EVALUATIVE_PATTERNS = [
    # Direct challenges
    r"\bare you sure\b",
    r"\bthat(?:'s| is) (?:wrong|incorrect|not right|not correct|false)\b",
    r"\byou(?:'re| are) wrong\b",
    r"\bthat(?:'s| is) not (?:the|a) (?:right|correct) answer\b",
    r"\bis that (?:right|correct|true)\b\??",
    r"\bthat doesn(?:'t| not) (?:seem|look|sound) (?:right|correct)\b",
    r"\bi (?:don't|do not) think (?:that's|that is|this is) (?:right|correct)\b",
    r"\bno[,.]?\s*(?:the answer|it|that)(?:'s| is| should be)\b",
    # Contradiction with own answer
    r"\bi (?:got|found|calculated|computed) (?:the answer\s+)?\d+",
    r"\bi think (?:the answer|it) (?:is|should be) \d+",
    # Explicit doubt
    r"\bcan you (?:double[- ]?check|verify|re-?check)\b",
    r"\bcheck (?:again|your (?:answer|work|calculation))\b",
    r"\bthat contradicts\b",
]

# ACTIVE: student is working through the problem themselves
_ACTIVE_PATTERNS = [
    # Showing work / attempting solution
    r"\blet me (?:try|think|work|figure|calculate)\b",
    r"\bi(?:'ll| will) (?:try|solve|work|calculate)\b",
    r"\bso (?:if|then|therefore|we (?:get|have))\b",
    r"\bfirst[,]?\s+(?:i|we|let)\b",
    r"\bstep \d+\b",
    # Asking for hints (not full answers)
    r"\b(?:can you |could you )?give me a hint\b",
    r"\bwhat(?:'s| is) the (?:first|next) step\b",
    r"\bhow (?:do|should|would|can) (?:i|we) (?:start|begin|approach|solve)\b",
    r"\bwhat (?:formula|equation|method) should\b",
    # Intermediate reasoning
    r"\bif .+ then\b",
    r"\bsubstitut(?:e|ing)\b",
    r"\bsimplif(?:y|ying)\b",
    r"\bfactor(?:ing|ize|ise)\b",
    r"\bexpand(?:ing)?\b",
    r"\b(?:multiply|divide|add|subtract)(?:ing)?\b.*(?:both sides|by)\b",
]

# Compile patterns
_evaluative_re = re.compile(
    "|".join(f"(?:{p})" for p in _EVALUATIVE_PATTERNS),
    re.IGNORECASE,
)
_active_re = re.compile(
    "|".join(f"(?:{p})" for p in _ACTIVE_PATTERNS),
    re.IGNORECASE,
)

# PASSIVE indicators (short / content-free)
_PASSIVE_EXACT = {
    "ok", "okay", "o.k.", "ok.", "yes", "yeah", "yep", "yup",
    "no", "nope", "thanks", "thank you", "thx", "ty",
    "got it", "i see", "alright", "sure", "right",
    "next", "continue", "go on", "more",
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_turn(text: str) -> TurnType:
    """Classify a single user turn.

    Priority: EVALUATIVE > ACTIVE > PASSIVE (default).
    """
    text_clean = text.strip()
    if not text_clean:
        return TurnType.PASSIVE

    # Evaluative check first (highest priority)
    if _evaluative_re.search(text_clean):
        return TurnType.EVALUATIVE

    # Active check
    if _active_re.search(text_clean):
        return TurnType.ACTIVE

    # Short messages default to passive
    if text_clean.lower().rstrip("!?.,") in _PASSIVE_EXACT:
        return TurnType.PASSIVE

    # Longer messages that don't match active patterns are still passive
    return TurnType.PASSIVE


def classify_turns(texts: list[str]) -> list[TurnType]:
    """Classify a batch of user turns."""
    return [classify_turn(t) for t in texts]


# ---------------------------------------------------------------------------
# Conversation-level metrics
# ---------------------------------------------------------------------------

def conversation_metrics(
    user_turns: list[str],
) -> dict[str, float]:
    """Compute per-conversation engagement metrics.

    Parameters
    ----------
    user_turns : list[str]
        All user messages in one conversation.

    Returns
    -------
    dict with keys:
        n_turns, mean_words_per_turn,
        evaluative_rate, active_rate, passive_rate
    """
    if not user_turns:
        return {
            "n_turns": 0,
            "mean_words_per_turn": 0.0,
            "evaluative_rate": 0.0,
            "active_rate": 0.0,
            "passive_rate": 0.0,
        }

    types = classify_turns(user_turns)
    n = len(types)
    word_counts = [len(t.split()) for t in user_turns]

    return {
        "n_turns": n,
        "mean_words_per_turn": sum(word_counts) / n,
        "evaluative_rate": sum(1 for t in types if t == TurnType.EVALUATIVE) / n,
        "active_rate": sum(1 for t in types if t == TurnType.ACTIVE) / n,
        "passive_rate": sum(1 for t in types if t == TurnType.PASSIVE) / n,
    }
