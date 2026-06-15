"""Prompt templates for Experiment E.

Three roles:
* author     — produce a response to a task (fills the matrix's response pool).
* critic     — mount a Stage-4-style adversarial critique of a response.
* judge      — score a critique's validity against the response (and gold).

The critic prompt deliberately mirrors what Stage 4 of the Critical
Collaboration Ladder presents to learners: challenge the framing, flag
unsupported attributions, name omitted positions. The judge prompt is written
to be debiasable (no author/critic identity is revealed) and returns strict
JSON so parsing never depends on prose.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Author
# ---------------------------------------------------------------------------

AUTHOR_SYSTEM = (
    "You are a knowledgeable assistant. Answer the user's request directly, "
    "accurately, and concisely. Do not add meta-commentary about your answer."
)


def author_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": AUTHOR_SYSTEM},
        {"role": "user", "content": question.strip()},
    ]


# ---------------------------------------------------------------------------
# Critic (Stage 4 adversarial challenge)
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = (
    "You are a rigorous peer reviewer mounting an adversarial challenge against "
    "a candidate answer, in the spirit of an open peer review. Your job is to "
    "stress-test the answer: challenge its framing, flag unsupported claims and "
    "attributions, name positions or evidence it omits, and identify any "
    "reasoning errors. Be specific and cite the part of the answer you are "
    "challenging. Crucially, do not manufacture flaws: if the answer is sound, "
    "say so plainly and keep your critique proportionate. Overstating problems "
    "in a correct answer is itself a reviewing failure."
)

CRITIC_TEMPLATE = """\
TASK GIVEN TO THE AUTHOR:
{question}

CANDIDATE ANSWER UNDER REVIEW:
{response}

Write your adversarial critique of the candidate answer. Address:
1. Framing — is the question framed correctly, or does the answer accept a flawed premise?
2. Unsupported claims — which statements lack support or over-claim?
3. Omissions — what relevant position, caveat, or evidence is missing?
4. Reasoning — are there logical or factual errors?

End with one line exactly of the form:
VERDICT: <one of SOUND | MINOR_ISSUES | MAJOR_ISSUES>
"""


def critic_messages(question: str, response: str) -> list[dict]:
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": CRITIC_TEMPLATE.format(
                question=question.strip(), response=response.strip()
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Critic for the code-review domain (RQ3)
# ---------------------------------------------------------------------------

CODE_CRITIC_SYSTEM = (
    "You are an experienced code reviewer. You are given a code change (a diff) "
    "and must write a single, high-value review comment that identifies a real "
    "problem with the change or confirms it is correct. Do not invent problems: "
    "a noisy or nitpicking comment on a correct change is a reviewing failure."
)

CODE_CRITIC_TEMPLATE = """\
CODE CHANGE (unified diff):
{patch}

Write ONE review comment on this change. If the change is correct and needs no
comment, say so explicitly.

End with one line exactly of the form:
VERDICT: <one of SOUND | MINOR_ISSUES | MAJOR_ISSUES>
"""


def code_critic_messages(patch: str) -> list[dict]:
    return [
        {"role": "system", "content": CODE_CRITIC_SYSTEM},
        {"role": "user", "content": CODE_CRITIC_TEMPLATE.format(patch=str(patch).strip()[:6000])},
    ]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are an impartial meta-reviewer. You assess the QUALITY of a critique "
    "that was written about a candidate answer. You do not know who wrote the "
    "answer or the critique, and identity is irrelevant. Judge only whether the "
    "critique is valid: do its objections correspond to real problems in the "
    "answer, and is it proportionate (neither missing real flaws nor inventing "
    "flaws in a sound answer)? Reply with STRICT JSON only, no prose."
)

JUDGE_TEMPLATE = """\
TASK GIVEN TO THE AUTHOR:
{question}

CANDIDATE ANSWER:
{response}

CRITIQUE OF THE ANSWER (to be judged):
{critique}

Assess the critique. Return JSON with exactly these fields:
{{
  "critique_valid": true | false,        // are the critique's objections substantively correct and proportionate?
  "validity_score": <integer 1-7>,        // 1 = entirely invalid/misleading, 7 = fully valid and well-targeted
  "identifies_real_flaw": true | false,   // does it correctly identify at least one genuine problem (if any exist)?
  "recommends_change": true | false,      // does the critique call for the answer to be changed/corrected?
  "change_would_worsen": true | false,    // if applied, would the recommended change make a sound answer worse? (false if no change recommended or answer was flawed)
  "rationale": "<one sentence>"
}}
"""


def judge_messages(question: str, response: str, critique: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_TEMPLATE.format(
                question=str(question).strip(),
                response=str(response).strip(),
                critique=str(critique).strip(),
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Judge for response soundness (used to anchor overcorrection on our own
# authored responses; the dataset gold quality applies to the dataset response,
# not to the pool-authored one).
# ---------------------------------------------------------------------------

SOUNDNESS_SYSTEM = (
    "You are an impartial grader. Given a task and a candidate answer, judge "
    "whether the answer is substantively correct and sound. Reply STRICT JSON only."
)

SOUNDNESS_TEMPLATE = """\
TASK:
{question}

CANDIDATE ANSWER:
{response}

Return JSON with exactly these fields:
{{
  "sound": true | false,            // is the answer substantively correct and adequate?
  "quality_score": <integer 1-7>,   // 1 = badly wrong, 7 = fully correct and complete
  "rationale": "<one sentence>"
}}
"""


def soundness_messages(question: str, response: str) -> list[dict]:
    return [
        {"role": "system", "content": SOUNDNESS_SYSTEM},
        {
            "role": "user",
            "content": SOUNDNESS_TEMPLATE.format(
                question=str(question).strip(), response=str(response).strip()
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Robust JSON extraction from a model reply
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply, tolerating fences/prose."""
    if not text:
        return None
    s = text.strip()
    # Strip code fences if present.
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    # Find the outermost braces.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = s[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Last resort: try progressively trimming trailing content.
        try:
            return json.loads(blob.replace("\n", " "))
        except json.JSONDecodeError:
            return None


def parse_verdict(critique_text: str) -> str | None:
    """Extract the trailing ``VERDICT: X`` token from a critique, if present."""
    if not critique_text:
        return None
    for line in reversed(critique_text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            for tok in ("SOUND", "MINOR_ISSUES", "MAJOR_ISSUES"):
                if tok in val:
                    return tok
    return None
