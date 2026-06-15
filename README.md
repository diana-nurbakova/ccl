# Critical Collaboration Ladder (CCL) — Validation Experiments

Retroactive validation of the CCL framework against four public empirical datasets. The framework makes three structural predictions about how people interact with AI-generated content:

1. **Claim-type asymmetry** — factual errors are more reliably detected than interpretive errors, justifying differential annotation scaffolding.
2. **No-self-merge** — passive acceptance of AI output harms learning; structural enforcement (hint-only AI) is more effective than voluntary evaluation.
3. **Engagement decay** — without explicit process reflection, critical engagement declines over time.

A fourth structural prediction grounds Stage 4 (tested by Experiment E):

- **No-self-critique** — adversarial critique requires architectural separation; a model critiquing its own output is compromised by sycophancy/self-preference, so the Stage 4 adversary should be a *separate* agent from the author (the corollary of no-self-merge).

Six experiments test these predictions:

| Experiment | Dataset | What it tests |
| --- | --- | --- |
| A | FRANK (Pagnoni et al., NAACL 2021) | Inter-annotator agreement on factual vs interpretive error flags |
| A' | FELM (Chen et al., NeurIPS 2023) | ChatGPT factual vs interpretive error rates |
| B | Bastani et al. (PNAS 2025) | Conversation architecture under hint-only vs full-answer AI |
| C | Bastani et al. (PNAS 2025) | Within-student engagement decay across 4 sessions |
| D | WildChat-1M (Allen AI) | Long-term engagement decay across months |
| E | CriticEval + MetaCritique + ManualReviewComment | Self vs cross-model critique validity and overcorrection (Stage 4) |

The full results report (with all tables, figures, and interpretation) lives at [`experiments/output/CCL_Validation_Experiments_Report.md`](experiments/output/CCL_Validation_Experiments_Report.md).

## Key findings

After correcting for the nesting structure of each dataset (see [Statistical methodology](#statistical-methodology)):

- **Claim-type asymmetry — validated (both pre-LLM and LLM-era).** FRANK GLMM odds ratio 5.15 [95 % CI 4.66, 5.70] for majority annotator agreement on FACTUAL vs INTERPRETIVE flags, after summary- and model-level random intercepts. FELM replicates at the prompt level: ChatGPT factual prompts have higher per-prompt error rates than interpretive prompts (mean 0.28 vs 0.16, one-sided Mann-Whitney p ≈ 7.8 × 10⁻⁶, N = 378 + 333 prompts). LLMs make more factual errors, *and* humans detect them more reliably — interpretive claims are the high-stealth category that motivates the differential scaffolding design.
- **No-self-merge — partially validated (length, not disposition).** After class- and student-level random intercepts, GPT Tutor (hint-only) produces ~3 more turns per conversation than GPT Base (β = −2.95, 95 % CI [−4.80, −1.09], p = 0.0018, d_total ≈ −0.27). Differences in evaluative / active / passive turn rates that looked highly significant under Mann-Whitney collapse into between-student variance once nesting is respected and are reported descriptively. The structural finding therefore supports CCL's emphasis on architecture over exhortation, but only in the form of sustained engagement length.
- **Engagement decay — validated within sessions, mixed across conditions.** Per-student slopes (one slope per student) over 4 Bastani sessions: passive rate rises significantly under hint-only scaffolding (`aug` slope +0.0145/session, p < 0.001) and active rate declines in both arms (`aug` p < 0.001; `vanilla` p < 0.001). Evaluative rate *increases* in `vanilla` (p = 0.001), consistent with a learning-curve effect as students notice GPT Base's mistakes. WildChat per-user slopes extend the analysis to months — see `experiments/output/exp_d_decay_by_order.csv` and `experiments/output/exp_d_decay_by_weeks.csv` for the most recent sample.
- **Bastani learning-outcome replication.** Per-session unassisted exam score (Part3Tot) regression returns β = −0.010 (p = 0.84) for GPT Base, diverging from the paper's β = −0.064 (p = 0.016). The published outcome is a single standardised final exam aggregated across sessions; the per-session Part3Tot in the public data file is a closely related but not identical measure.
- **No-self-critique — direction supported, but critique validity is not LLM-judgeable (Experiment E).** Across a symmetric 5-model author×critic matrix, cross-model critique is more valid than self-critique within each critic (mixed-effects logistic OR = 1.37 [1.24, 1.50], p = 8.5×10⁻¹¹ on general critique; OR = 1.29 [1.19, 1.40] in the code-review domain), the self-preference signature the literature predicts, and the effect survives a verbosity control. Overcorrection of sound answers is low and not meaningfully higher under self-critique (~3.4% vs 3.1%). **Critically, the judge failed the pre-registered κ ≥ 0.6 reliability gate even with a frontier model** (gpt-oss-120b κ = 0.15, GPT-4.1 κ = 0.23 against human CriticEval labels; ≈0.14 on code review), so the self/cross effect sizes are reported as *exploratory and consistent-with-prior*, not human-confirmed. That judge failure is itself the point: validity of an adversarial challenge cannot be safely automated, which is why Stage 4 separates the adversary from the author **and** keeps the learner as the evaluator. Full writeup: [`experiments/output/exp_e_stage4_critique_report.md`](experiments/output/exp_e_stage4_critique_report.md).

## Statistical methodology

Each test operates at the unit of analysis appropriate for its nesting structure:

- **Exp A** — binomial GLMM with `(1 | summary_id) + (1 | model_name)` random intercepts (statsmodels `BinomialBayesMixedGLM`).
- **Exp A'** — Mann-Whitney U / Kruskal-Wallis on per-prompt error rates (segments are not independent within a prompt; CCL category is constant within prompt, so a GLMM would be degenerate).
- **Exp B** — linear mixed-effects model with `(1 | class_id) + (1 | student_id)` random intercepts (statsmodels `mixedlm`).
- **Exp C / D** — one-sample t-test on per-student / per-user OLS slopes (one observation per subject).
- **Exp E** — binomial GLMM `critique_valid ~ is_cross + gold_quality + (1 | item) + (1 | critic) + (1 | author)` (statsmodels `BinomialBayesMixedGLM`), so the self/cross effect is estimated *within critic* and is not confounded with critic capability. Overcorrection is a paired within-item comparison (McNemar + item/critic GLMM). The judge is validated against human labels by Cohen's κ; per the pre-registration the LLM-judge headline is dropped because κ < 0.6.

The reasoning behind each replacement is documented in [`specs/Statistical_Independence_Fix_Specs.md`](specs/Statistical_Independence_Fix_Specs.md).

## Repository layout

```
ccl/
├── experiments/
│   ├── exp_a_error_detection/   # FRANK
│   ├── exp_a_felm/              # FELM
│   ├── exp_b_active_passive/    # Bastani conversation analysis
│   ├── exp_c_declining_effort/  # Bastani trajectory slopes
│   ├── exp_d_wildchat_decay/    # WildChat trajectory slopes
│   ├── exp_e_critique/          # Stage 4 self-vs-cross critique (paid API; resumable)
│   ├── shared/                  # Data acquisition, LLM client, JSONL store, stats, plotting
│   ├── synthesis/               # Cross-experiment summary tables
│   ├── tests/                   # pytest suite
│   ├── data/raw/                # Cached datasets (gitignored)
│   ├── output/                  # CSVs, figures, full report
│   └── run_all.py
├── specs/                       # Original specifications + reanalysis spec
├── pyproject.toml
└── README.md
```

## Setup

```bash
# Clone and install
git clone https://github.com/diana-nurbakova/ccl.git
cd ccl

# With uv (recommended)
uv venv
uv pip install -e ".[dev]"

# Or with plain pip
python -m venv .venv
. .venv/Scripts/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

For Experiment D (WildChat) you need a Hugging Face token. Create a `.env` file at the project root:

```
HF_TOKEN=hf_...
```

## Reproducing the experiments

```bash
# Run everything (skip D if you don't have HF_TOKEN)
python -m experiments.run_all --skip d

# Or run individually
python -m experiments.exp_a_error_detection.run
python -m experiments.exp_a_felm.run
python -m experiments.exp_b_active_passive.run
python -m experiments.exp_c_declining_effort.run
python -m experiments.exp_d_wildchat_decay.run

# Force re-download of cached datasets
python -m experiments.run_all --force-download
```

Outputs (CSVs, figures, full report) are written to `experiments/output/`.

### Experiment E (Stage 4 self-vs-cross critique) — makes paid API calls

Experiment E is **excluded from `run_all` by default** because it generates and
judges critiques via paid LLM APIs (OpenAI + DeepInfra). It needs
`OPENAI_API_KEY` and `DEEPINFRA_API_KEY` in `.env`.

```bash
# 1. Check every pool model + judge is live (no generation)
python -m experiments.exp_e_critique.run --probe

# 2. Full run (Path A2: symmetric 5-model author×critic matrix + debiased judge)
python -m experiments.exp_e_critique.run                 # default 300 items
python -m experiments.exp_e_critique.run --n-items 60    # smaller / cheaper

# 3. Include it in run_all (omit 'e' from the skip list)
python -m experiments.run_all --skip d
```

**Logging & recovery.** Every LLM request and response is logged in full to
`experiments/output/exp_e/logs/llm_interactions.jsonl`. Every authored response,
critique, and judge verdict is checkpointed to an append-only store under
`experiments/output/exp_e/store/`, so the run is **fully resumable**: if it
crashes or is interrupted, just run it again (or use the recovery tool) and it
continues from exactly where it stopped without repeating any API call.

```bash
python -m experiments.exp_e_critique.recover --status   # what is done / left
python -m experiments.exp_e_critique.recover --repair   # drop truncated lines after a hard crash
python -m experiments.exp_e_critique.recover --resume   # continue the run
```

**Design (spec `specs/ccl-stage4-critique-experiment-spec.md`).** Five models
(GPT-4.1-nano, DeepSeek-V3, Llama-3.3-70B, Gemma-4-31B, DeepSeek-R1) each author
and critique, giving a 5×5 matrix; *self* = the diagonal (a model critiquing its
own output), *cross* = off-diagonal. The self/cross effect is estimated *within
each critic*, so it is not confounded with critic capability. A judge outside
the pool (gpt-oss-120b) scores critique validity and is validated against human
labels (Cohen's κ; the LLM-judge headline is dropped if κ < 0.6).

> The reported 2026-06-15 run substituted `gemma-3-27b-it` for `gemma-4-31B-it`
> due to an implementation error (`config.py` is now corrected); the conclusions
> are unaffected — see the report's Gemma deviation note.

## Tests

```bash
python -m pytest experiments/tests/ -v
```

The suite covers CCL category mappings, statistical utilities, the regex turn classifier, and the new mixed-effects analyses on synthetic data with known effects.

## Data sources

| Dataset | Source | Cached at |
| --- | --- | --- |
| FRANK | https://github.com/artidoro/frank | `experiments/data/raw/frank/` |
| FELM | https://huggingface.co/datasets/hkust-nlp/felm | `experiments/data/raw/felm/` |
| Bastani | https://github.com/obastani/GenAICanHarmLearning | `experiments/data/raw/bastani/` |
| WildChat | https://huggingface.co/datasets/allenai/WildChat-1M | `experiments/data/raw/wildchat/` |
| CriticEval | https://huggingface.co/datasets/opencompass/CriticBench (Apache-2.0) | `experiments/data/raw/exp_e/criticeval/` |
| MetaCritique | https://github.com/GAIR-NLP/MetaCritique (Apache-2.0) | `experiments/data/raw/exp_e/metacritique/` |
| ManualReviewComment | https://zenodo.org/records/13150598 (CC-BY-4.0) | `experiments/data/raw/exp_e/mrc/` |

All datasets are downloaded on first run by `experiments.shared.data_acquisition` and cached locally. They are excluded from git via `.gitignore`.

## License

MIT.
