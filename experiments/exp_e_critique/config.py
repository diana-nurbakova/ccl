"""Configuration for Experiment E: Stage 4 self-vs-cross critique validity.

Model pool, judge, generation parameters, and output paths. Model strings are
validated against the live provider at run start (see ``run.py --probe``); edit
``MODEL_POOL`` / ``JUDGE`` here if a provider renames or retires a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
EXP_DIR = OUTPUT_DIR / "exp_e"
LOG_DIR = EXP_DIR / "logs"
STORE_DIR = EXP_DIR / "store"
FIG_DIR = OUTPUT_DIR / "figures"

for _d in (EXP_DIR, LOG_DIR, STORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelSpec:
    key: str            # short stable identifier used in keys/tables
    provider: str       # "openai" | "deepinfra"
    model: str          # exact API model string
    reasoning: bool     # is this a reasoning model?
    reasoning_effort: str | None = None


# ---------------------------------------------------------------------------
# The five-model symmetric pool (spec section 4; decision 2026-06-07).
# Each model both authors and critiques, giving a 5x5 author x critic matrix:
# self = diagonal (model critiques its own output), cross = off-diagonal.
# Model strings reflect DeepInfra naming as of June 2026; --probe verifies them.
# ---------------------------------------------------------------------------

MODEL_POOL: list[ModelSpec] = [
    ModelSpec("gpt41nano", "openai", "gpt-4.1-nano", reasoning=False),
    ModelSpec("deepseekv3", "deepinfra", "deepseek-ai/DeepSeek-V3-0324", reasoning=False),
    ModelSpec("llama70b", "deepinfra", "meta-llama/Llama-3.3-70B-Instruct-Turbo", reasoning=False),
    ModelSpec("gemma27b", "deepinfra", "google/gemma-3-27b-it", reasoning=False),
    ModelSpec("deepseekr1", "deepinfra", "deepseek-ai/DeepSeek-R1-0528", reasoning=True),
]

# Judge: a strong model OUTSIDE the pool so it never scores its own critique.
# NOTE: gpt-oss-120b at reasoning_effort="high" routinely spends the entire
# completion budget on hidden reasoning and returns an EMPTY final answer
# (observed empirically: ~82% empty at max_tokens=900). We use "medium" effort
# with a generous token budget so the structured JSON verdict always fits. This
# is a deliberate deviation from the spec's "high" suggestion, forced by the
# empty-output failure mode; medium reasoning is ample for a rubric judgement.
JUDGE = ModelSpec(
    "judge_gptoss120b", "deepinfra", "openai/gpt-oss-120b",
    reasoning=True, reasoning_effort="medium",
)

# Fallback model strings tried by --probe if the primary is unavailable.
MODEL_FALLBACKS: dict[str, list[str]] = {
    "gemma27b": ["google/gemma-3-27b-it", "google/gemma-2-27b-it"],
    "deepseekv3": ["deepseek-ai/DeepSeek-V3-0324", "deepseek-ai/DeepSeek-V3"],
    "deepseekr1": ["deepseek-ai/DeepSeek-R1-0528", "deepseek-ai/DeepSeek-R1"],
    "llama70b": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-70B-Instruct"],
}
JUDGE_FALLBACKS: list[str] = ["openai/gpt-oss-120b", "Qwen/Qwen3-235B-A22B-Thinking-2507"]

# ---------------------------------------------------------------------------
# Generation / judging parameters (spec section 4 "Provider plumbing")
# ---------------------------------------------------------------------------

GEN_TEMPERATURE = 0.2      # critique + response generation
JUDGE_TEMPERATURE = 0.0    # judging
SEED = 20260615            # set where the API honours it
# Token budgets must cover hidden reasoning tokens for reasoning models
# (DeepSeek-R1 critiques hit a 700 cap; gpt-oss judge returned empty at 900).
# These caps total completion = reasoning + final answer.
AUTHOR_MAX_TOKENS = 1000
CRITIQUE_MAX_TOKENS = 2000
JUDGE_MAX_TOKENS = 3000

# Default stratified sample size from CriticEval, with the high-quality stratum
# oversampled (it carries the overcorrection signal).
DEFAULT_N_ITEMS = 300
HIGH_QUALITY_OVERSAMPLE = 0.5   # fraction of the sample drawn from quality=="high"

# Concurrency for API calls.
MAX_WORKERS = 8

# Judge-human agreement threshold below which the LLM-judge headline is dropped.
KAPPA_THRESHOLD = 0.6
