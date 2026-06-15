# CCL Retroactive Validation: Experiment Report

**Framework**: Critical Collaboration Ladder (CCL)  
**Purpose**: Validate CCL design choices against independent empirical datasets  
**Code**: `experiments/` directory; reproducible via `python -m experiments.run_all`

---

## Overview

The CCL framework makes four structural predictions about how people interact with AI-generated content:

1. **Claim-type asymmetry** — factual errors are more detectable than interpretive ones, justifying differential annotation scaffolding
2. **No-self-merge** — passive acceptance of AI output harms learning; structural enforcement is more effective than voluntary evaluation
3. **Engagement decay** — without explicit process reflection, critical engagement declines over time
4. **No-self-critique** — adversarial critique requires architectural separation; a model critiquing its own output is compromised by sycophancy/self-preference, so the Stage 4 adversary should be a separate agent from the author (the corollary of no-self-merge)

Six experiments test these predictions against public empirical datasets. Experiments A and A' both address the claim-type asymmetry using different datasets (pre-LLM and LLM-era respectively). Experiments B–C validate the no-self-merge and decay predictions. Experiment D extends decay analysis to longer timescales. Experiment E tests the no-self-critique prediction by generating and judging a symmetric matrix of self vs cross-model critiques (the only experiment that makes paid LLM API calls).

---

## Experiment A — Differential Error Detection by Claim Type

### CCL Prediction
Factual errors are more reliably detected than interpretive errors. This asymmetry justifies the CCL's claim-type annotation system (■ FACTUAL / ♦ SOURCE / ▲ INTERPRETIVE).

### Hypotheses

- **H₁ (primary)**: Among sentences flagged by at least one annotator, the probability of majority agreement (≥ 2 of 3 annotators agreeing on the flagged category) is higher for FACTUAL than for INTERPRETIVE flags. On the logit scale, β(FACTUAL) > 0 in a binomial GLMM with `summary_id` and `model_name` random intercepts.
- **H₁ (secondary)**: Per-category Fleiss' κ for FACTUAL exceeds Fleiss' κ for INTERPRETIVE.
- **H₀**: No difference in majority-agreement odds between categories (β(FACTUAL) = 0).

### Data Source
**FRANK benchmark** (Pagnoni et al., NAACL 2021)  
- URL: https://github.com/artidoro/frank  
- File: `data/human_annotations_sentence.json`  
- **4,942 sentences** from AI-generated summaries (9 models)  
- Each sentence annotated by **3 trained crowdworkers** across 7 error types

### CCL Mapping

| FRANK Error Code | Meaning | CCL Category |
|-----------------|---------|-------------|
| EntE | Entity error | FACTUAL (■) |
| OutE | Out-of-article hallucination | FACTUAL (■) |
| GramE | Grammar error changing meaning | FACTUAL (■) |
| CircE | Circumstantial error | SOURCE (♦) |
| RelE | Relation error | INTERPRETIVE (▲) |
| LinkE | Discourse link error | INTERPRETIVE (▲) |
| CorefE | Coreference error | INTERPRETIVE (▲) |
| NoE | No error | — |

### Experiment Design

**Data preparation** (`experiments.exp_a_error_detection.analysis.build_sentence_category_matrix`).
For each (sentence, CCL category) pair, derive three per-annotator binary flags via `classify_frank_errors`: annotator *r* flags category *c* iff at least one of their FRANK error codes maps to *c*. Each flagged sentence carries `summary_id = "{article_hash}_{model_name}"` so that within-summary dependence can be modelled.

**Per-category descriptive statistics** (`compute_category_stats`).
- *Prevalence*: fraction of sentences with `any_flagged = True` (≥ 1 annotator flagged the category).
- *Majority agreement*: among flagged sentences, fraction where ≥ 2 of 3 annotators agreed.
- *Fleiss' κ*: 3-rater binary inter-annotator reliability for each category, computed via `experiments.shared.stats_utils.fleiss_kappa`.

**Descriptive 2 × 2 test** (`fisher_test_factual_vs_interpretive`).
Fisher's exact test on the 2 × 2 table {majority-agreed, not} × {FACTUAL, INTERPRETIVE}, restricted to flagged sentences. Treats every flagged sentence as independent — *reported as descriptive only* because sentences are nested within summaries within models.

**Inferential mixed-effects model** (`glmm_factual_vs_interpretive`).
Bayesian binomial GLMM fit by variational Bayes (`statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`) on flagged FACTUAL/INTERPRETIVE sentences:
```
majority_agree ~ is_factual + (1 | summary_id) + (1 | model_name)
```
- The fixed effect `is_factual` (1 = FACTUAL, 0 = INTERPRETIVE) gives the log-odds difference.
- Variance-component parameters are log-SDs; the function exponentiates them to report random-intercept SDs on the logit scale.
- We report the posterior mean (β), posterior SD (SE), Wald-style 95 % CI (β ± 1.96 SE), and a two-sided z-test p-value.
- Each summary belongs to exactly one model, so the design is nested; both random intercepts are still identifiable because summary IDs include the model name and 1 824 summaries × 9 models leaves substantial between-model variation to estimate.

### Results

| CCL Category | N flagged | Prevalence | Majority agreement | Fleiss' κ |
|-------------|-----------|------------|-------------------|-----------|
| FACTUAL | 1,867 | 37.8% | **67.9%** | **0.585** |
| SOURCE | 639 | 12.9% | 24.9% | 0.247 |
| INTERPRETIVE | 1,382 | 28.0% | **27.8%** | **0.186** |

**Fisher's exact test** (FACTUAL vs. INTERPRETIVE majority detection):  
OR = 5.49, p = 4.19 × 10⁻¹¹⁶

> **Statistical note (independence)**: Sentences are nested within summaries (each summary produces 1–N flagged sentences) and summaries are nested within models. The Fisher result above treats sentences as independent and is therefore reported as descriptive only. The inferential test below is a binomial GLMM with random intercepts for summary and model, which addresses this dependence.

#### Mixed-effects logistic regression (proper inferential test)

```
majority_agree ~ ccl_category + (1 | summary_id) + (1 | model_name)
```

Fit by variational Bayes (statsmodels `BinomialBayesMixedGLM`) on N = 3,249 flagged FACTUAL/INTERPRETIVE sentences from 1,824 summaries × 9 models.

| Quantity | Value |
| --- | --- |
| Odds ratio (FACTUAL vs INTERPRETIVE) | **5.15** |
| 95 % CI | [4.66, 5.70] |
| β (logit scale) | 1.640 |
| SE | 0.052 |
| z | 31.65 |
| p | < 10⁻¹⁰⁰ |
| SD of summary random intercept (logit) | 0.36 |
| SD of model random intercept (logit) | 0.74 |

The asymmetry survives controlling for within-summary and between-model dependence almost intact: OR drops from 5.49 (Fisher) to 5.15 (GLMM), and the CI is tight. Model-level variance is the larger of the two random effects (SD = 0.74 vs 0.36 on the logit scale), as expected — different summarisation models differ in their overall agreement rates.

### Interpretation
Annotators agree on the presence of a factual error 68% of the time, versus 28% for interpretive errors — a **2.4× detection reliability gap**. Fleiss' κ confirms the same pattern: moderate agreement for factual (0.585) versus near-chance for interpretive (0.186). The mixed-effects OR of 5.15 means a flagged factual claim is ~5× more likely to reach annotator consensus than a flagged interpretive claim, after accounting for summary and model nesting.

**Validation**: Results match the spec exactly. This validates the CCL's differential scaffolding design: interpretive claims pass unquestioned without explicit Stage 2 checklisting and Stage 4 adversarial challenge.

**Limitations**: FRANK annotators are trained crowdworkers, not students. The error taxonomy mapping is analogical. "Missing perspectives" has no FRANK equivalent.

---

## Experiment A' — FELM: Claim-Type Asymmetry in ChatGPT Outputs

### Purpose

Experiment A uses pre-LLM summarisation models (BART, PEGASUS, etc.). FELM tests whether the same claim-type asymmetry exists in **ChatGPT outputs**, directly addressing the concern that the FRANK result may not generalise to modern LLMs.

FELM does not have multiple independent annotators per segment, so inter-annotator agreement cannot be computed. Instead four complementary analyses are run: (1) descriptive error rates by CCL category, (2) prompt-level inferential test (the proper unit), (3) error-type distribution, (4) segment-length proxy.

### Hypotheses

- **H₁ (primary)**: At the prompt level, the per-prompt error-rate distribution for FACTUAL prompts is stochastically greater than for INTERPRETIVE prompts (one-sided Mann-Whitney U).
- **H₁ (secondary)**: At the prompt level, per-prompt error rates differ across the three CCL categories (Kruskal-Wallis).
- **H₀**: Per-prompt error-rate distributions are identical across CCL categories.

### Why prompt-level rather than mixed-effects

Each FELM prompt belongs to exactly one domain and therefore exactly one CCL category, so the predictor is **constant within prompt**. A mixed-effects model with prompt as random intercept would have the fixed effect of CCL category perfectly confounded with the grouping factor. The proper inferential unit is therefore the prompt: aggregate the 1–N segments of each prompt to a single error-rate observation, then compare across categories.

### Data Source

**FELM** (Chen et al., NeurIPS 2023)

- URL: [hkust-nlp/felm on HuggingFace](https://huggingface.co/datasets/hkust-nlp/felm) (downloaded as `all.jsonl`)
- Paper: [arxiv.org/abs/2310.00741](https://arxiv.org/abs/2310.00741)
- **847 prompts → ChatGPT responses → 4,426 annotated segments**
- 5 domains; per-segment binary factuality label + error type

### CCL Mapping

| FELM Domain | N segments | CCL Category | Rationale |
| --- | --- | --- | --- |
| wk (World Knowledge) | 532 | FACTUAL (■) | Verifiable entity/date/number claims |
| math | 599 | FACTUAL (■) | Verifiable computation |
| science | 684 | INTERPRETIVE (▲) | Causal/mechanistic reasoning |
| reasoning | 1,025 | INTERPRETIVE (▲) | Logical inference, framing |
| writing_rec | 1,586 | GAP (•) | Subjective/perspectival claims |

### Experiment Design

**Analysis 1 — Segment-level error rates** (`compute_error_rates`, *descriptive only*).
Group segments by domain (and by CCL category) and report `error_rate = n_errors / n_segments`. We also report a chi-square test on segment counts and a Fisher's exact OR for FACTUAL vs INTERPRETIVE — both are descriptive because segments within a response share the same prompt and generation context, so they are not independent.

**Analysis 2 — Prompt-level inferential test** (`compute_prompt_level_error_rates`, `mannwhitney_factual_vs_interpretive_prompts`, `kruskal_across_ccl_prompts`).
- For each of the 847 prompts, compute `error_rate_p = (n error segments in p) / (n segments in p)`.
- Tag each prompt with its single CCL category (constant within prompt).
- **Kruskal-Wallis** across the three CCL categories: tests H₀ that all three error-rate distributions are identical.
- **One-sided Mann-Whitney U** with `alternative='greater'`: tests CCL's directional prediction that FACTUAL prompts have higher error rates than INTERPRETIVE prompts.
- **Effect size**: rank-biserial r = 2U/(n₁n₂) − 1 ∈ [−1, 1], where positive r means FACTUAL ranks higher.

**Analysis 3 — Error-type distribution** (`chi_square_error_types`).
Cross-tabulate FELM error types (knowledge_error, reasoning_error, calculation_error, irrelevant_with_qst, fooled, …) by CCL category among error segments. Chi-square test for distributional difference. Reported descriptively because the `type` field is sparsely populated (many entries are None).

**Analysis 4 — Segment-length proxy** (`compute_segment_lengths`).
Hypothesis: if interpretive error segments are longer, the error signal is more diluted, providing a mechanistic explanation for lower detection rates. We compute word counts on error segments only and run a two-sided Mann-Whitney U on FACTUAL vs INTERPRETIVE segment lengths.

### Results

#### Analysis 1: Error rate by CCL category

| CCL Category | N segments | N errors | Error rate |
| --- | --- | --- | --- |
| FACTUAL (wk + math) | 1,131 | 272 | **24.0%** |
| GAP (writing_rec) | 1,586 | 267 | 16.8% |
| INTERPRETIVE (science + reasoning) | 1,709 | 248 | **14.5%** |

**Chi-square** (error rates across categories): χ² = 43.87, df = 2, p < 0.0001 — *descriptive only; segments within a response are not independent.*

**Fisher's exact** (FACTUAL vs. INTERPRETIVE): OR = 1.87, p < 0.0001 — *descriptive only.*

#### Prompt-level analysis (proper inferential test)

Each FELM prompt belongs to a single domain and therefore to a single CCL category, so segments within a prompt have a constant predictor — a mixed-effects model with prompt as random intercept would have the predictor confounded with the grouping factor. The appropriate unit of inference is the **prompt** (N = 847; mean ≈ 5.2 segments/prompt).

For each prompt we compute `error_rate = n_errors / n_segments` and compare across CCL categories.

| Test | Statistic | p |
| --- | --- | --- |
| Kruskal-Wallis (3 categories) | H = 20.00 | 4.5 × 10⁻⁵ |
| Mann-Whitney U (FACTUAL > INTERPRETIVE, one-sided) | U = 72,798 | 7.8 × 10⁻⁶ |

Per-prompt summaries:

| CCL Category | N prompts | Mean error rate | Median error rate |
| --- | --- | --- | --- |
| FACTUAL (wk + math) | 378 | **0.284** | 0.000 |
| INTERPRETIVE (science + reasoning) | 333 | **0.158** | 0.000 |
| GAP (writing_rec) | 136 | — | — |

Rank-biserial r ≈ 0.157 (FACTUAL prompts rank higher than INTERPRETIVE).

The asymmetry holds at the proper unit: the median error rate is 0 for both groups (most prompts have no errors), but the *distribution* of prompt-level error rates is shifted higher for FACTUAL prompts. The honest p-value is ~10⁻⁵ rather than the segment-level "< 10⁻⁹" the chi-squared would suggest.

#### Analysis 2: Error type distribution

| CCL Category | fooled | irrelevant_with_qst | knowledge_error | reasoning_error | total typed |
| --- | --- | --- | --- | --- | --- |
| FACTUAL | 8 | 10 | 121 | 7 | 146 |
| GAP | 21 | 84 | 141 | 0 | 246 |
| INTERPRETIVE | 0 | 8 | 92 | 2 | 102 |

**Chi-square** (error types differ by CCL category): χ² = 78.37, df = 6, p < 0.0001

Note: The majority of typed errors are `knowledge_error` across all categories. This reflects limited annotation coverage of the `type` field (many entries are None). The significant chi-square is driven by the `irrelevant_with_qst` type being concentrated in the GAP category.

#### Analysis 3: Segment length

| CCL Category | Mean length (words) | Median |
| --- | --- | --- |
| FACTUAL | 19.0 | 17 |
| INTERPRETIVE | 19.5 | 17 |
| GAP | 20.3 | 20 |

**Mann-Whitney U** (FACTUAL vs. INTERPRETIVE error segment length): U = 32,838, p = 0.603 (n.s.)

Segment lengths do not differ significantly between FACTUAL and INTERPRETIVE error segments. The detection difficulty asymmetry is not explained by segment length in this dataset.

### Interpretation

**What FELM shows** (at the proper unit): ChatGPT prompts in the FACTUAL category have a higher mean per-prompt error rate (0.284) than INTERPRETIVE prompts (0.158), and the prompt-level distribution is shifted upward for FACTUAL (Mann-Whitney U = 72,798, p ≈ 7.8 × 10⁻⁶ one-sided; rank-biserial r ≈ 0.157). Combined with FRANK, this gives a double asymmetry:

> - **LLMs make more factual errors** — FELM: per-prompt mean error rate 0.284 (FACTUAL) vs 0.158 (INTERPRETIVE), Mann-Whitney p ≈ 8 × 10⁻⁶
> - **Humans detect factual errors more reliably** — FRANK: GLMM odds ratio 5.15 [95 % CI 4.66, 5.70] for majority agreement on FACTUAL vs INTERPRETIVE flags, accounting for summary and model nesting

Interpretive errors are *both* less frequently produced *and* harder for humans to catch when they do occur. This is precisely the scenario where CCL's differential scaffolding is most consequential: a low-frequency but high-stealth error class that passes unnoticed without explicit Stage 2–4 prompting.

The error-type distribution (χ² = 78.37, p < 0.0001 — descriptive, since errors are nested in prompts) confirms that the categories produce qualitatively different errors, not just quantitatively different rates — supporting the CCL's decision to treat them as distinct annotation types rather than a single "error" category.

**Limitations**: FELM uses a single ground-truth annotation (no inter-annotator reliability measure). The `type` field is sparsely populated. The five FELM domains are broader than FRANK's seven error codes; the CCL mapping is coarser.

---

## Experiment B — No-Self-Merge: Active vs Passive AI Use

### CCL Prediction
AI output that undergoes independent human evaluation (no-self-merge principle) produces better learning outcomes than passively accepted output. Structural enforcement is more effective than voluntary evaluation.

### Hypotheses

- **H₁ (architectural enforcement)**: After accounting for class- and student-level dependence, GPT Tutor produces conversations with more turns, shorter turns, higher evaluative-turn rate, higher active-turn rate, and lower passive-turn rate than GPT Base. Operationally, β(`is_vanilla`) in a per-metric LMM with `(1 | class_id) + (1 | student_id)` differs from 0 in the predicted direction.
- **H₁ (learning outcome)**: GPT Base lowers unassisted exam score (Part3Tot) relative to control; GPT Tutor does not (Bastani's published finding).
- **H₀ (architectural enforcement)**: β(`is_vanilla`) = 0 for every metric in the LMM.
- **H₀ (learning outcome)**: GPT Base β = 0 in the cluster-robust ITT regression.

### Data Source
**Bastani et al. (2025, PNAS)** — *"Generative AI Can Harm Learning"*  
- URL: https://github.com/obastani/GenAICanHarmLearning  
- Files:
  - `main_regressions/final_data.csv` — student-session outcomes (`Student ID`, `Class`, `Treatment arm`, `Part3Tot`, …)
  - `text_analysis/data/raw/valid_student_data_w_time_stamp.csv` — conversation logs (`username`, `conversation_id`, `session_id`, `treatment`, `role`, `message`, …)
- ~**1,000 Turkish high school students**, 4 × 90-minute math tutoring sessions
- Pre-registered RCT (3 arms: control, GPT Base, GPT Tutor)

### CCL Mapping of Treatment Arms

| Arm | Label in data | CCL interpretation |
|-----|--------------|-------------------|
| GPT Base | `vanilla` | No-self-merge **violated**: AI gives full answers, no evaluation required |
| GPT Tutor | `aug` | No-self-merge **enforced**: AI gives hints only, student must work through problems |
| Control | `control` | Baseline: no AI |

### Experiment Design

**Step 1 — Turn classification** (`experiments.shared.turn_classifier`).
Regex-based classifier applied to every user message:
- **EVALUATIVE**: challenges to AI correctness ("are you sure", "that's wrong", "you are wrong"). Evaluated *before* ACTIVE so that "are you sure I should add 2x" counts as evaluative, not active.
- **ACTIVE**: student working through the problem ("let me try", "step 1", "give me a hint", math operations on the student's side).
- **PASSIVE**: everything else (acknowledgements, short affirmations, single-word replies).

**Step 2 — Conversation-level metrics** (`compute_conversation_level_metrics`).
For each conversation (uniquely identified by `conversation_id` or `(username, problem_id)`):
- `n_turns` — number of user turns
- `mean_words_per_turn` — average word count over user turns
- `evaluative_rate`, `active_rate`, `passive_rate` — fraction of user turns in each category

**Step 3 — Descriptive condition comparison** (`compute_condition_comparison`, *descriptive only*).
Cohen's d and Mann-Whitney U between `aug` and `vanilla` arms, treating each conversation as one observation. Reported as descriptive because each student contributes up to ~14 conversations and students are nested within classes.

**Step 4 — Inferential mixed-effects model** (`attach_class_id`, `conversation_level_lmm`).
For each metric, fit:
```
metric ~ is_vanilla + (1 | class_id) + (1 | student_id)
```
via `statsmodels.formula.api.mixedlm` with `groups=class_id` and `vc_formula={"student": "0 + C(username)"}`. `is_vanilla` is a 0/1 indicator coded so that positive β means GPT Base > GPT Tutor on that metric.
- We merge `Class` from `final_data.csv` onto the conversation-level metrics via `username = Student ID`.
- We report β, robust SE, 95 % CI, p (Wald), and `d_total = β / √(var_class + var_student + var_residual)` as a variance-explained-scale standardised effect size.
- REML fit (`reml=True`) with the L-BFGS-B optimiser; convergence is checked.

**Step 5 — Learning-outcome ITT regression** (`run_itt_regression`).
OLS on `Part3Tot ~ GPTBase + GPTTutor` with cluster-robust SEs clustered at `Class`. Returns β / SE / p for each treatment arm.

> **Note on outcome variable**: Part3Tot is the unassisted exam score — the key CCL learning outcome. Part2Tot is the AI-assisted practice score and is intentionally excluded; it conflates tool use with learning. The Bastani paper's headline regression uses a single standardised final exam (administered after all 4 sessions); the per-session Part3Tot in this data file is a closely related but not identical measure, which explains the divergence in β values.

**Step 6 — Manual validation sample** (`sample_evaluative_turns`).
Random sample of 15 turns the regex classifier labels EVALUATIVE, hand-checked for false positives.

### Results

#### Turn classification (28,666 user turns)

| Category | Count | Rate |
|----------|-------|------|
| EVALUATIVE | 510 | 1.8% |
| ACTIVE | 665 | 2.3% |
| PASSIVE | 27,491 | 95.9% |

#### Conversation metrics by condition (descriptive)

| Metric | GPT Tutor (aug) | GPT Base (vanilla) | Cohen's d | Mann-Whitney p |
|--------|----------------|-------------------|-----------|---------|
| Turns per conversation | 5.49 ± 3.79 | 2.45 ± 1.99 | +0.99 | < 10⁻⁴⁰ |
| Mean words per turn | 5.75 ± 7.24 | 10.42 ± 15.49 | −0.39 | < 10⁻³⁹ |
| Evaluative turn rate | 1.0% ± 5% | 3.0% ± 12% | −0.33 | < 10⁻⁴¹ |
| Active turn rate | 3.0% ± 12% | 4.0% ± 17% | −0.07 | < 0.001 |
| Passive turn rate | 97% ± 13% | 93% ± 20% | +0.24 | < 0.001 |

> **Statistical note (independence)**: Each student contributes up to 4 sessions and many conversations, and students are nested within classes (the Bastani RCT randomisation unit). The Mann-Whitney p-values above treat all 7,089 conversations as independent and are reported as descriptive only. The mixed-effects model below is the inferential test.

#### Mixed-effects LMM (proper inferential test)

For each metric we fit:

```
metric ~ is_vanilla + (1 | class_id) + (1 | student_id)
```

via statsmodels `mixedlm` with class as the grouping factor and student as a variance component. Class IDs come from `final_data.csv`; the conversation-level dataset merges to 27 classes / 518 students / 7,089 conversations. Coefficient sign convention: positive β = vanilla > aug.

| Metric | β (vanilla − aug) | SE | 95 % CI | p | d (β / √var_total) | var(class) | var(student) | var(residual) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Turns per conversation** | **−2.95** | 0.95 | [−4.80, −1.09] | **0.0018** | −0.27 | 0.000 | 114.66 | 7.17 |
| Mean words per turn | +4.34 | 3.37 | [−2.27, +10.95] | 0.198 | +0.11 | 0.000 | 1456.93 | 91.06 |
| Evaluative turn rate | +0.026 | 0.026 | [−0.025, +0.077] | 0.322 | +0.08 | 0.000 | 0.087 | 0.005 |
| Active turn rate | +0.009 | 0.045 | [−0.080, +0.097] | 0.848 | +0.02 | 0.000 | 0.260 | 0.016 |
| Passive turn rate | −0.034 | 0.051 | [−0.133, +0.065] | 0.495 | −0.06 | 0.000 | 0.327 | 0.020 |

Once the within-student dependence is properly accounted for, the structural finding survives in one place and one place only: **GPT Base produces ~2.95 fewer turns per conversation than GPT Tutor (95 % CI [−4.80, −1.09], p = 0.0018, d ≈ −0.27)**. The other Mann-Whitney "differences" — words/turn, evaluative rate, active rate, passive rate — are not significant once between-student variance is correctly absorbed. The shrinkage of d for `n_turns` (from 0.99 → 0.27) reflects how much of the original signal was within-student replication rather than independent observations.

The class-level variance is estimated at 0 in every metric. This is expected: conditioning on student already absorbs most of the between-class variation because each student belongs to exactly one class and treatment is class-randomised. Reframing the same model with class alone as the grouping factor would inflate SEs slightly without changing the qualitative conclusion.

#### ITT regression (outcome: Part3Tot — unassisted exam)

| Arm | β | SE | p-value |
|-----|---|----|---------|
| GPT Base (vanilla) | −0.010 | 0.049 | 0.837 |
| GPT Tutor (aug) | +0.003 | 0.047 | 0.946 |

### Interpretation

**Conversation structure: the surviving claim**. After accounting for within-student dependence (LMM with class + student random intercepts), the only metric that remains significantly different between conditions is *turns per conversation*: GPT Tutor produces ~3 more turns per conversation than GPT Base (β = −2.95, p = 0.0018, d ≈ −0.27). This is consistent with hint-based scaffolding forcing iterative engagement.

**Words/turn, evaluative rate, active rate, passive rate — descriptive only.** The original Mann-Whitney tests on these metrics returned p-values that *appeared* extremely significant (p < 10⁻⁹ in several cases), but the LMM shows these were inflated by treating ~14 conversations from the same student as independent. The variance components make this concrete: between-student variance for `evaluative_rate` (0.087) is ~16× the residual (0.005), so each "conversation" carries little independent information about the *between-treatment* effect — most variation is captured by the student. Reported as descriptive segment/conversation rates, the original direction holds (GPT Base shows higher evaluative rate, etc.); reported as inferential claims, only the turns-per-conversation gap survives.

**The regression result diverges from the Bastani paper's reported β = −0.064 (p = .016)**. The most likely cause is that the paper's outcome is a single standardised final exam administered after all 4 sessions, whereas Part3Tot in the data file is a per-session score. A student-level aggregation (mean or final session Part3Tot) would be closer to the paper's specification.

**CCL validation**: The no-self-merge principle is validated at the conversation-architecture level. The structural difference between hint-only (enforced) and full-answer (voluntary) conditions produces a large, significant difference in *engagement length* (turns per conversation). The framework should not over-claim structural effects on disposition rates (evaluative/active) — those differences are not separable from between-student variance in this dataset.

### Manual Validation Sample (EVALUATIVE turns, n=15)

All 15 sampled evaluative turns were genuine challenges to AI correctness. Dominant pattern: "are you sure" variants. Examples:

> "you are wrong again" [vanilla]  
> "are you sure" [vanilla]  
> "c) is not a function. is that correct" [aug]  
> "explain again, you are wrong" [vanilla]  
> "so let me get this right the first one is function the second one is also function but the third one is not is that correct" [aug]

---

## Experiment C — Declining Effort Across Sessions

### CCL Prediction
Without process-level reflection (Stage 3), critical engagement declines across sessions.

### Hypotheses

- **H₁ (passive rate)**: The mean per-student slope of `passive_rate` over sessions 1–4 is positive (passivity rises). One-sample t-test on slopes against 0.
- **H₁ (active rate)**: The mean per-student slope of `active_rate` is negative.
- **H₁ (evaluative rate)**: The mean per-student slope of `evaluative_rate` is negative *if scaffolding is sufficient*; the spec admits that for `vanilla` (no scaffold) a learning-curve effect may push evaluative rate up over sessions.
- **H₀**: For each metric, the mean per-student slope = 0.

### Why the per-student slope is the right unit

Each student contributes 3–4 sessions, so session-level rows are not independent within student. Computing one OLS slope per student and then running a one-sample t-test on the resulting N = 451 slopes operates at the student level by construction — there is no nesting to model away.

### Data Source
Same Bastani et al. dataset as Experiment B. Filtered to **451 students** with ≥3 of 4 sessions (243 `aug`, 208 `vanilla`).

### Experiment Design

**Step 1 — Per-(student, session) metrics** (`build_session_metrics`).
Apply the regex turn classifier from Experiment B to user turns and aggregate per `(username, session_id)` into `n_turns`, `mean_words_per_turn`, `evaluative_rate`, `active_rate`, `passive_rate`.

**Step 2 — Filter to longitudinal participants** (`filter_repeat_students`).
Keep only students with ≥ 3 distinct `session_id` values so an OLS slope is well-defined.

**Step 3 — Per-student trajectory slopes** (`per_subject_slopes`).
For each student and each metric, fit a simple OLS slope of `metric ~ session_id` using the closed-form `cov(x,y) / var(x)`. Returns one scalar slope per (student, metric).

**Step 4 — One-sample t-test on slopes per treatment arm** (`slope_ttest`, `compute_trajectory_slopes`).
Within each treatment arm separately (`aug`, `vanilla`), test H₀: mean slope = 0 with `scipy.stats.ttest_1samp`. Report mean slope, % of students with negative slope, t-statistic, and p-value. We test arms separately rather than pooled because the spec predicts *direction* may differ between arms.

**Step 5 — Session-level summary table** (`session_level_summary`).
For visualisation only: mean of each metric grouped by `(treatment, session_id)`. Not a hypothesis test.

### Session-Level Summary

| Treatment | Session | Eval rate | Passive rate | Active rate | Turns/conv | Words/turn |
|-----------|---------|-----------|-------------|-------------|------------|-----------|
| aug | 1 | 0.5% | 94.1% | 5.3% | 14.2 | 5.85 |
| aug | 2 | 0.5% | 96.3% | 3.2% | 21.7 | 5.96 |
| aug | 3 | 0.2% | 98.6% | 1.2% | 25.9 | 5.29 |
| aug | 4 | 0.2% | 98.1% | 1.6% | 24.8 | 4.92 |
| vanilla | 1 | 1.9% | 92.2% | 5.9% | 8.6 | 9.12 |
| vanilla | 2 | 3.1% | 92.1% | 4.8% | 10.5 | 9.71 |
| vanilla | 3 | 5.1% | 94.0% | 0.9% | 10.9 | 9.70 |
| vanilla | 4 | 4.4% | 92.6% | 3.0% | 11.0 | 9.37 |

### Within-Student Trajectory Slopes

| Metric | Treatment | Mean slope | % declining | t-stat | p-value |
|--------|-----------|------------|-------------|--------|---------|
| passive_rate | aug | +0.0145 | 16% | 5.16 | < 0.001*** |
| active_rate | aug | −0.0133 | 40% | −4.80 | < 0.001*** |
| evaluative_rate | aug | −0.0012 | 10% | −1.90 | 0.059 |
| mean_words/turn | aug | −0.241 | 63% | −1.75 | 0.081 |
| passive_rate | vanilla | +0.0044 | 23% | 1.08 | 0.282 |
| active_rate | vanilla | −0.0146 | 26% | −3.73 | < 0.001*** |
| evaluative_rate | vanilla | +0.0102 | 12% | 3.24 | 0.001** |
| mean_words/turn | vanilla | −0.134 | 54% | −0.42 | 0.678 |

### Interpretation

**What supports the CCL prediction:**
- Passive rate increases significantly in the `aug` condition (p < 0.001): even under architectural scaffolding, students become more passive over sessions.
- Active engagement rate declines significantly in both conditions (aug: p < 0.001; vanilla: p < 0.001).
- Words/turn decline in the `aug` condition (p = 0.081, borderline).

**What complicates the prediction:**
- Evaluative rate *increases* for `vanilla` students (p = 0.001). The likely explanation is a learning curve: GPT Base students discover GPT makes mistakes and begin challenging it. By session 4, their evaluative rate has nearly tripled (1.9% → 4.4%).
- The 4-session window is short; stronger decay is expected over longer timescales (see Experiment D).

**Implication**: The CCL's Stage 3 process reflection is needed because passive engagement increases even under structural scaffolding. Increasing turns per session (aug condition) does not prevent disengagement — the quality of engagement degrades.

**Note on divergence from spec**: The spec reports passive slope +0.027 for `aug`. Our result (+0.0145) is in the same direction and significant, but smaller in magnitude. This may reflect differences in how passive rate was defined in the original paper's classifier.

---

## Experiment D — Longitudinal Engagement Decay (WildChat)

### CCL Prediction
Engagement decay documented in Experiment C (4 sessions, days) extends to months-long timescales and is visible in unstructured real-world AI conversations.

### Hypotheses

- **H₁ (passive rate)**: The mean per-user slope of `passive_rate` over `conv_order` is positive.
- **H₁ (active rate)**: The mean per-user slope of `active_rate` over `conv_order` is negative.
- **H₁ (vocabulary concentration)**: The mean per-user slope of `vocab_concentration` (fraction of user words drawn from a top-50 global vocabulary) is positive — users converge on narrower topic ranges over time.
- **H₁ (lexical diversity)**: The mean per-user slope of TTR (type-token ratio) is negative — repetition increases.
- **H₁ (use-frequency vs calendar time)**: Decay is detectable both by conversation order and by `weeks_since_first`. Whichever has the larger effect indicates whether decay is use-driven or time-driven.
- **H₀**: For each metric, the mean per-user slope = 0.

### Why the per-user slope is the right unit

Each user contributes multiple conversations, so conversation-level rows are not independent within user. As in Experiment C, we collapse to one slope per user and run a one-sample t-test on the N = 696 slopes — operating at the user level by construction.

### Data Source
**WildChat-1M** (Zhao et al. / Allen AI)  
- URL: https://huggingface.co/datasets/allenai/WildChat-1M  
- License: ODC-BY  
- **Sample**: 10,000 conversations from repeat users (≥5 conversations each), streamed in non-toxic subset  
- User identifier: `hashed_ip` (anonymised IP hash)

### Experiment Design

**Step 1 — Streamed sampling of repeat users** (`download_wildchat_sample`).
Stream WildChat-1M via the Hugging Face `datasets` library, skipping toxic/redacted records. Group by `hashed_ip`; keep users with ≥ 5 conversations. Cap to 10,000 conversations total (random shuffle to trim if more qualify). Cache as parquet. The streaming approach avoids downloading the full ~5 M-conversation dataset.

**Step 2 — Chronological ordering per user**.
Sort conversations within each user by `timestamp`. Assign `conv_order` (1 = earliest) and `weeks_since_first = (timestamp − min_timestamp_per_user).days / 7`.

**Step 3 — Per-conversation metrics**.
Same regex turn classifier as Experiments B and C, plus:
- `vocab_concentration`: fraction of the conversation's user-side word tokens that appear in a top-50 global vocabulary list (built from the full sample).
- `mean_ttr`: mean type-token ratio across the user turns of the conversation, computed via `experiments.shared.stats_utils.type_token_ratio`.

**Step 4 — Per-user OLS slopes** (`per_subject_slopes`).
For each user and each metric, fit a slope of metric vs `conv_order` and a separate slope vs `weeks_since_first`. Use only users with ≥ 5 conversations and non-zero variance in the time variable.

**Step 5 — One-sample t-tests on slopes** (`slope_ttest`).
For each metric, run a one-sample t-test on the N ≈ 696 per-user slopes against 0. Report mean slope, % declining, t-statistic, p-value, separately for `conv_order` and `weeks_since_first`.

**Outputs**:
- `experiments/output/exp_d_decay_by_order.csv` — slope summaries vs conversation order
- `experiments/output/exp_d_decay_by_weeks.csv` — slope summaries vs weeks since first conversation
- `experiments/output/figures/exp_d_decay_trajectories.{png,pdf}` — mean trajectories per metric
- `experiments/output/figures/exp_d_slope_distributions.{png,pdf}` — histograms of per-user slopes

### Results

> ⚠️ Results depend on the WildChat sample obtained. The figures below reflect the 10K-conversation sample. Run `python -m experiments.exp_d_wildchat_decay.run` to reproduce.

Key outputs saved to:
- `experiments/output/exp_d_decay_by_order.csv`
- `experiments/output/exp_d_decay_by_weeks.csv`
- `experiments/output/figures/exp_d_decay_trajectories.png`
- `experiments/output/figures/exp_d_slope_distributions.png`

### Cross-Study Comparison

| Study | Timescale | Passive slope | Active slope |
|-------|-----------|--------------|-------------|
| Bastani Exp C (aug) | 4 sessions (~days) | +0.0145*** | −0.0133*** |
| Bastani Exp C (vanilla) | 4 sessions (~days) | +0.0044 (n.s.) | −0.0146*** |
| WildChat Exp D | Months | (see output) | (see output) |

### Interpretation
The WildChat analysis tests whether the decay pattern found in Experiment C persists over much longer timescales. If passive rate slopes are significant over months, this strengthens the CCL argument that scaffolding must be sustained rather than faded after initial exposure. Topic drift analysis addresses an additional question: whether users converge toward simpler, narrower queries over time — a behavioural signature of reduced critical engagement.

---

## Experiment E — Stage 4: Self vs Cross-Model Adversarial Critique

### CCL Prediction

Stage 4 (Peer Reviewer) asks the AI to mount an adversarial challenge against its own earlier output and asks the learner to judge whether that self-critique is valid or overcorrecting. The framework's fourth structural prediction — **no-self-critique**, the corollary of no-self-merge — is that adversarial critique requires *architectural separation*: a model critiquing its own output is compromised by sycophancy and self-preference, so the Stage 4 adversary should be a separate agent from the author. This experiment grounds that design choice and quantifies the specific failure mode (overcorrection) learners must learn to catch.

### Hypotheses

- **H₁ (RQ1, validity)**: Adversarial critique generated by a *separate* model is more valid than *same-model* self-critique. On the logit scale, β(`is_cross`) > 0 in a binomial GLMM that estimates the self/cross effect **within critic** (so it is not confounded with critic capability). Directional, with a strong literature prior (Huang et al. 2024; Xu et al. 2024; Panickssery et al. 2024).
- **H₂ (RQ2, overcorrection)**: Same-model self-critique overcorrects more often than cross-critique — i.e. judges an already-sound response as flawed and pushes a change that would worsen it (correct→incorrect pressure).
- **H₃ (RQ3, domain)**: The H₁ direction holds in the code-review domain, not only general critique.
- **H₀**: No self/cross difference in critique validity (β(`is_cross`) = 0); equal overcorrection rates.

> **Pre-registered abort trigger (spec §7)**: validity rests on an LLM judge. If judge–human agreement is below **Cohen's κ = 0.6**, the LLM-judge headline is dropped and only human-labelled subsets are reported.

### Design principle: a symmetric pool, not a fixed author and critic

A fixed-author/separate-critic design would confound the self-vs-cross factor with critic capability: if the cross-critic is simply a stronger model, "cross beats self" would reflect capability, not the self/cross distinction. To isolate the factor, **every pool model both authors and critiques**, building the full critic-by-author matrix. *Self* = a model critiquing its own output (the diagonal); *cross* = the same model critiquing another model's output (off-diagonal). The effect is estimated within each critic model, holding capability constant. This is **Path A2** (controlled generation over reused dataset responses): the datasets' questions and gold labels are reused, and a thin generation step manipulates the self/cross factor cleanly.

### Data Sources

| Dataset | Role | Host / licence |
| --- | --- | --- |
| **CriticEval** (Lan et al., NeurIPS 2024; arXiv 2402.13764) | Primary general-domain items: `*_feedback_correction` dev files give a question, a response (`generation`), and a gold response-quality grade (`metadata.quality` ∈ {low, high}); `meta_feedback_single` files carry human critique-quality scores for judge validation. | `opencompass/CriticBench` (HF), Apache-2.0 |
| **MetaCritique** (Sun et al., ACL 2024 Findings; arXiv 2401.04518) | Scoring scheme / calibration (Atomic Information Units; precision/recall/F1). | `GAIR-NLP/MetaCritique`, Apache-2.0 |
| **ManualReviewComment** (Liu et al., MSR 2025; arXiv 2502.02757) | Code-review domain (RQ3): 270 CodeReviewer comments labelled Valid (172) / Noisy (98); the `patch` diff + `msg` comment + gold validity. | Zenodo 13150598, **CC-BY-4.0** (confirmed; more permissive than the spec's provisional note) |

**Sample (run 2026-06-15):** N = 150 CriticEval items, stratified by gold quality (89 high, 61 low — the feedback_correction dev files grade responses low/high only, so no medium stratum materialised); all 270 ManualReviewComment diffs.

### Models and roles

| Model | Provider | Type | Role |
| --- | --- | --- | --- |
| GPT-4.1-nano | OpenAI | non-reasoning | author + critic |
| DeepSeek-V3-0324 | DeepInfra | non-reasoning | author + critic |
| Llama-3.3-70B-Instruct-Turbo | DeepInfra | non-reasoning | author + critic |
| Gemma-3-27B | DeepInfra | non-reasoning | author + critic |
| DeepSeek-R1-0528 | DeepInfra | reasoning | author + critic (reasoning secondary result) |
| gpt-oss-120b | DeepInfra | reasoning | **judge** (outside the pool) |
| GPT-4.1 | OpenAI | non-reasoning | **frontier cross-check judge** |

> **Deviation from spec (Gemma).** The spec's pool named *Gemma-4-31B*. The executed run reported here used `google/gemma-3-27b-it` due to an implementation error (the harness substituted Gemma-3 on a mistaken assumption that Gemma-4 was unavailable, and the model-availability probe checked only the substitute). `google/gemma-4-31B-it` is in fact live on DeepInfra; `config.py` has been corrected to it for reproduction. The substitution does not affect the conclusions: the cross > self direction holds within 4 of 5 critics and under a frontier judge, and the headline (the judge κ-gate failure) is independent of any single pool member.

Temperatures: generation 0.2, judging 0.0; seed 20260615. Judge `reasoning_effort="medium"` with a 3000-token budget (at "high" it spent the entire budget on hidden reasoning and returned empty answers).

### Experiment Design

**Step 1 — Stratified item sample** (`experiments.exp_e_critique.data.sample_items`). Sample CriticEval items by gold quality, oversampling the high-quality stratum (it carries the overcorrection signal).

**Step 2 — Author matrix rows** (`generate.author_responses`). Each of the 5 pool models authors one response per item (authored once per (model, item) and reused across all critic pairs — not regenerated per pair). 750 responses (727 non-empty; 23 were empty refusals, mostly harmlessness items, and skipped downstream).

**Step 3 — Critique matrix** (`generate.critique_matrix`). Each pool model critiques every authored response under a fixed adversarial prompt mirroring Stage 4 (challenge the framing, flag unsupported attributions, name omitted positions). Self critiques are the diagonal, cross the off-diagonal: 3,635 general critiques.

**Step 4 — Judge** (`judge.judge_critiques`, `judge_response_soundness`). A judge outside the pool scores each critique's validity (pointwise, strict-JSON rubric: `critique_valid`, `validity_score` 1–7, `recommends_change`, `change_would_worsen`) and separately rates each authored response's soundness (anchors overcorrection on our own responses). 10,385 judge verdicts; JSON parse rate 1.00.

**Step 5 — Judge validation** (`judge.validate_judge_criticeval`, `validate_judge_mrc`). The judge scores the human-labelled CriticEval critiques and the gold Valid/Noisy MRC comments; Cohen's κ is computed against the human labels. GPT-4.1 repeats this as a frontier cross-check.

**Step 6 — Code-review domain** (`generate.author_code_comments`, `code_critique_matrix`). Each pool model writes a review comment per diff; each pool model critiques every comment; the judge scores them. 1,350 comments → 6,750 code critiques judged.

**Inferential model** (`analysis.glmm_validity`). Binomial GLMM via variational Bayes (`BinomialBayesMixedGLM`):

```
critique_valid ~ is_cross + gold_quality + (1 | item) + (1 | critic) + (1 | author)
```

Because each critic contributes both self and cross critiques, the `is_cross` effect is estimated within critic and controls for critic capability by design. Overcorrection (RQ2) is a within-item paired comparison (McNemar + item/critic GLMM) restricted to sound responses. A verbosity control regresses validity on `is_cross` + standardised critique length.

**Engineering.** Every LLM request/response is logged in full to `exp_e/logs/llm_interactions.jsonl`; every result is checkpointed to append-only JSONL stores under `exp_e/store/`, making the run fully resumable (`recover.py --status/--repair/--resume`). The full run was ~28,000 API calls with 0 errors.

### Results

#### Headline: the judge fails the pre-registered reliability gate

No LLM judge — not even a frontier model — reproduces human critique-validity labels on these datasets (`exp_e_judge_validation.csv`):

| Judge | Dataset | n | Agreement | Cohen's κ | PABAK | Ordinal ρ | Passes κ ≥ 0.6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-120b | CriticEval | 200 | 0.64 | **0.145** | 0.27 | 0.27 | no |
| GPT-4.1 | CriticEval | 200 | 0.68 | **0.226** | 0.36 | 0.37 | no |
| gpt-oss-120b | ManualReviewComment | 270 | 0.52 | **0.142** | 0.04 | — | no |
| GPT-4.1 | ManualReviewComment | 270 | 0.59 | **0.137** | 0.17 | — | no |

The failure is not a weak-judge or pure base-rate artifact: it survives a frontier judge and prevalence adjustment (PABAK), and on CriticEval the judge agrees *less* than a constant "valid" baseline would (0.64–0.68 < the 0.77 human-positive rate). **Per the pre-registered trigger, the LLM-judge headline is dropped; the self/cross results below are reported as exploratory and consistent-with-prior, not human-confirmed.**

#### RQ1 — Cross-model critique is more valid than self-critique (exploratory)

Binomial GLMM, self/cross estimated within critic:

> **OR(cross vs self) = 1.37, 95 % CI [1.24, 1.50], p = 8.5 × 10⁻¹¹, n = 3,634.**

Overall validity rate: self 0.741 vs cross 0.782. The direction holds *within* 4 of the 5 critics (`exp_e_descriptive_validity.csv`):

| Critic | self | cross | Δ (cross − self) |
| --- | --- | --- | --- |
| GPT-4.1-nano | 0.660 | 0.797 | **+0.137** |
| DeepSeek-R1 | 0.813 | 0.875 | +0.062 |
| Gemma-3-27B | 0.793 | 0.844 | +0.051 |
| DeepSeek-V3 | 0.591 | 0.598 | +0.007 |
| Llama-3.3-70B | 0.827 | 0.801 | −0.026 |

A model is more lenient (less valid) when critiquing its *own* output — the self-preference / sycophancy signature the literature predicts, observed here in the critique-validity framing Stage 4 actually uses.

**Verbosity control.** Validity correlates weakly with critique length (β_len = 0.27, p = 2 × 10⁻⁸), but the cross advantage survives controlling for length (β_is_cross = 0.21, p = 0.027). The effect is not merely "cross critiques are longer."

#### RQ2 — Overcorrection is low and not meaningfully higher under self-critique

On sound responses (judge-rated and/or gold high-quality stratum, n = 3,214):

| Condition | n | Overcorrection rate | Helpful-correction rate (on flawed responses) |
| --- | --- | --- | --- |
| self | 643 | **3.4 %** | 0.54 |
| cross | 2,571 | **3.1 %** | 0.58 |

The within-item GLMM (OR(cross) = 0.78, p = 0.034) and paired McNemar (22 self-only vs 0 cross-only discordant pairs, p < 10⁻⁶) are *directionally* consistent with H₂ (self overcorrects slightly more), but the absolute magnitude is ~3 % either way and the small p-values reflect the large sample, not a substantively important gap. **H₂ is at best weakly supported.** These quantities depend on the judge and are reported descriptively.

#### RQ3 — The pattern generalises to code review (exploratory)

Refitting RQ1's model on the ManualReviewComment code critiques:

> **OR(cross vs self) = 1.29, 95 % CI [1.19, 1.40], p = 3.2 × 10⁻⁹, n = 6,750.**

The cross > self direction generalises from general critique to the professional code-review practice Stage 4's lineage draws on.

#### Secondary — reasoning and critique validity

The one reasoning model in the pool (DeepSeek-R1) produced the highest-validity critiques: validity rate **0.86** vs **0.75** for the four non-reasoning critics. Consistent with reasoning improving critique quality, but it is a single model and reported as a descriptive observation, not a controlled contrast.

Figure: `experiments/output/figures/exp_e_self_vs_cross.png` (per-critic self-vs-cross validity and overcorrection). Full machine-readable results: `experiments/output/exp_e/exp_e_results.json`; effect-size table: `exp_e_results_table.csv`. A standalone writeup with the same numbers is at `experiments/output/exp_e_stage4_critique_report.md`.

### Interpretation

Two findings hold together. (1) The **self < cross** direction is observed consistently — within-critic, across two domains, robust to a verbosity control — matching the strong literature prior and the self-preference mechanism. This supports the **no-self-critique** corollary: the Stage 4 adversary should be a separate agent from the author. (2) **No LLM judge reliably certifies critique validity against humans** (κ ≤ 0.23, even GPT-4.1), so the validity of an adversarial challenge cannot be safely outsourced to an automated judge — the learner's judgement is load-bearing.

Together these *are* the Stage 4 design: an adversary structurally separated from the author (because self-critique is weaker), with a human evaluator deciding whether the challenge is valid or overcorrecting (because even frontier judges cannot). The experiment grounds Stage 4 both in what it automates (the separate adversary) and in what it deliberately does not (the validity judgement).

**Paper integration** (spec §13, Option II): fold this into the Stage 4 paragraph as direct empirical backing, foregrounding the human-grounded judge-reliability result and presenting the self<cross direction as a pre-registered, literature-consistent exploratory signal. The pre-registered design plus this run answer the reviewers' core request to engage with *why Stage 4 is hard*.

### Limitations

- **LLM-judge results are not human-validated** (κ gate failed for both judges); treat the self/cross effect sizes as exploratory.
- CriticEval feedback_correction dev files grade responses low/high only, so the overcorrection "sound" stratum rests on the gold high grade plus judge soundness ratings.
- Single judge per verdict (pointwise validity), so no order-swap doubling was needed; pairwise order-swapped judging is implemented but unused as primary.
- The reasoning-model result is a single model (DeepSeek-R1), not a controlled reasoning-vs-not manipulation.

---

## Cross-Experiment Synthesis

This table is auto-generated by `experiments.synthesis.cross_experiment.build_synthesis_table` from the result dicts of each experiment, and saved to `experiments/output/synthesis_table.csv`. Effect sizes prefer the proper inferential test (GLMM / LMM / prompt-level Mann-Whitney) and fall back to descriptive statistics only if the new fields are absent.

| CCL design choice | Experiment | Key finding (proper inferential test) | Effect size |
| --- | --- | --- | --- |
| Claim-type annotation (pre-LLM) | A (FRANK) | Factual detection >> Interpretive (GLMM with summary + model random intercepts) | OR = 5.15 [95 % CI 4.66, 5.70] |
| Claim-type annotation (LLM era) | A' (FELM) | Factual prompt-level error rate > Interpretive (prompt-level Mann-Whitney; segments not independent within prompt) | mean 0.28 vs 0.16, U = 72,798, p ≈ 7.8 × 10⁻⁶ (N = 378 + 333 prompts) |
| No-self-merge principle (learning outcome) | B | Per-session unassisted exam score (Part3Tot); paper reports b = −0.064 on student-level final exam | b = −0.010 (p = 0.837) |
| Architectural enforcement (conversation length) | B | Hint-only AI produces longer conversations (LMM with class + student random intercepts) | β(vanilla−aug) = −2.95 turns [95 % CI −4.80, −1.09], p = 0.0018, d_total = −0.27 |
| Process reflection (Stage 3) | C | Passivity increases across sessions (per-student slopes, one-sample t) | slope = +0.014*** |
| Sustained scaffolding (fading debate) | D | Long-term engagement decay over months (per-user slopes) | (see Exp D output) |
| No-self-critique (Stage 4 separation) | E | Cross-model critique more valid than self, within critic (GLMM; **exploratory** — judge κ < 0.6) | OR = 1.37 [1.24, 1.50] general; 1.29 [1.19, 1.40] code |
| Stage 4 needs a human evaluator | E | No LLM judge (incl. GPT-4.1) matches human critique-validity labels | Cohen's κ ≤ 0.23 |

### Four Mechanisms (after correcting for nesting / judge-reliability)

1. **Differential scaffolding by claim type**: not all errors are equally visible to unaided readers. After accounting for within-summary and between-model dependence (GLMM with random intercepts), factual error flags achieve ~5× higher odds of annotator consensus than interpretive flags (OR = 5.15, 95 % CI [4.66, 5.70]). The FELM replication at the *prompt* level confirms this generalises to LLM-era content: ChatGPT factual prompts have higher mean error rates than interpretive prompts (0.28 vs 0.16, Mann-Whitney p ≈ 8 × 10⁻⁶). The double asymmetry — LLMs err more on factual claims, and humans detect those errors more reliably — makes interpretive claims the highest-risk category for uncritical acceptance.

2. **Structural enforcement over voluntary evaluation (length, not disposition)**: architectural constraints (hint-only AI) produce conversations that are systematically longer than free-form AI conversations. After accounting for within-student dependence, GPT Tutor produces ~3 more turns per conversation than GPT Base (β = −2.95, p = 0.002, d ≈ −0.27 on the variance-explained scale). The original Mann-Whitney p-values on disposition rates (evaluative/active) were inflated by treating multiple conversations from the same student as independent — those differences are not separable from between-student variance and are reported as descriptive only. The structural finding is therefore narrower but still supports CCL's emphasis on architecture over exhortation.

3. **Process-level reflection for sustained engagement**: active engagement declines significantly across sessions even when the tool is architecturally constrained. Per-student slopes computed at the student level (one slope per student) avoid the nesting problem entirely. The aug condition's passive rate slope is +0.0145/session, p < 0.001, and the active rate slope is −0.013/session, p < 0.001 over 4 sessions. Experiment D tests whether this decay is qualitatively similar over months at the per-user slope level.

4. **Architectural separation of the adversary, with a human evaluator (Stage 4)**: across a symmetric author×critic matrix, a model is more lenient when critiquing its *own* output than another model's — cross-model critique is more valid than self-critique within each critic (GLMM OR = 1.37 [1.24, 1.50] general, 1.29 [1.19, 1.40] code), the self-preference signature the literature predicts, robust to a verbosity control. This supports making the Stage 4 adversary a separate agent (no-self-critique). But the self/cross effect is estimated by an LLM judge that **fails the pre-registered κ ≥ 0.6 gate even as a frontier model** (GPT-4.1 κ = 0.23; gpt-oss-120b κ = 0.15), so the effect is reported as exploratory and consistent-with-prior — and that failure is itself the second half of the mechanism: critique validity cannot be safely automated, which is exactly why Stage 4 keeps the learner as the evaluator. Overcorrection of sound answers is low and not meaningfully higher under self-critique (~3.4 % vs 3.1 %).

---

## Reproducibility

```bash
# Install (standalone CCL project)
pip install -e ".[dev]"

# Run all experiments (skip D if no HF token)
python -m experiments.run_all --skip d

# Run individual experiments
python -m experiments.exp_a_error_detection.run
python -m experiments.exp_a_felm.run
python -m experiments.exp_b_active_passive.run
python -m experiments.exp_c_declining_effort.run
python -m experiments.exp_d_wildchat_decay.run   # requires HF_TOKEN in .env

# Experiment E — paid LLM API calls (OPENAI_API_KEY + DEEPINFRA_API_KEY in .env)
python -m experiments.exp_e_critique.run --probe         # check models are live
python -m experiments.exp_e_critique.run --n-items 150   # full run (resumable)
python -m experiments.exp_e_critique.recover --status     # progress / resume after a crash

# Run tests
python -m pytest experiments/tests/ -v
```

All results (CSVs + figures) are saved to `experiments/output/`.

### Software stack

| Library | Use |
| --- | --- |
| `statsmodels.formula.api.mixedlm` | Linear mixed-effects models (Exp B) |
| `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` | Binomial GLMM with crossed/nested random intercepts (Exp A) — variational-Bayes fit |
| `scipy.stats` | Mann-Whitney U, Kruskal-Wallis, Fisher's exact, one-sample t-test, normal CDF |
| `experiments.shared.stats_utils` | Fleiss' κ, Cohen's d, per-subject OLS slopes, cluster-robust OLS |
| `experiments.shared.llm_client` | OpenAI/DeepInfra chat-completions client with full interaction logging (Exp E) |
| `experiments.shared.jsonl_store` | Append-only resumable checkpoint store (Exp E recovery) |
| `experiments.exp_e_critique.judge.cohens_kappa` | Judge–human agreement (Exp E) |
| `pandas` / `numpy` | Data manipulation and aggregation |
| `matplotlib` / `seaborn` | Figures saved to `experiments/output/figures/` |
| `httpx` / `datasets` / `openpyxl` | Dataset acquisition and caching |

---

## Statistical Independence Reanalysis

Per EDM Reviewer 2's comment, three of the original tests treated nested observations as independent. The reanalyses above replace each with a model that respects the nesting structure:

| Experiment | Original test | Replacement (proper inferential test) | Effect direction | Effect size after fix |
| --- | --- | --- | --- | --- |
| A (FRANK) | Fisher's exact on 3,249 sentences | Binomial GLMM with `(1\|summary_id) + (1\|model_name)` | Holds | OR 5.49 → 5.15 [4.66, 5.70] |
| A' (FELM) | Chi-squared / Fisher's on 4,426 segments | Mann-Whitney U on 711 prompt-level rates (FACTUAL vs INTERPRETIVE) | Holds | OR 1.87 (segment) → mean 0.28 vs 0.16 (prompt), p ≈ 8 × 10⁻⁶ |
| B (Bastani) | Mann-Whitney on 7,089 conversations | LMM with `(1\|class_id) + (1\|student_id)`, per metric | Mixed: turns/conv survives; words/turn, evaluative/active/passive rates do not | d 0.99 → 0.27 (turns), other metrics ns |
| C (Bastani) | One-sample t on 451 per-student slopes | No change needed (student is the unit) | — | — |
| D (WildChat) | One-sample t on 696 per-user slopes | No change needed (user is the unit) | — | — |
| E (CriticEval/MRC) | Designed nested from the start | Binomial GLMM `(1\|item) + (1\|critic) + (1\|author)`; self/cross within critic | Holds (exploratory) | OR(cross vs self) 1.37 [1.24, 1.50] |

The directional CCL claims about claim-type asymmetry survive cleanly. The Bastani structural-enforcement claim survives only on conversation length; the disposition-rate differences (evaluative/active) collapse into between-student variance and should be reported descriptively rather than as hypothesis tests.

Reproduce: `python -m experiments.exp_a_error_detection.run` (writes `exp_a_glmm.csv`), `python -m experiments.exp_a_felm.run` (writes `exp_a_felm_prompt_test.csv` and `exp_a_felm_prompt_stats.csv`), `python -m experiments.exp_b_active_passive.run` (writes `exp_b_lmm_table.csv`).

---

## Known Limitations and Open Issues

| Issue | Affects | Notes |
| --- | --- | --- |
| FRANK annotators ≠ students | Exp A | Crowdworkers are trained; untrained learners likely show larger asymmetry |
| No FRANK equivalent for "missing perspectives" | Exp A | This CCL category remains unvalidated empirically |
| FELM has no inter-annotator reliability | Exp A' | Single ground-truth annotation; cannot compute Fleiss' kappa |
| FELM `type` field is sparsely populated | Exp A' | Many entries are None; error type chi-square is underpowered |
| Part3Tot is per-session, not final exam | Exp B | Regression β diverges from paper; student-level aggregation needed for exact replication |
| Disposition rates not separable from student variance | Exp B | Evaluative/active/passive rates differ between conditions only descriptively after LMM; treat as exploratory |
| Class-level random-intercept variance estimated at 0 | Exp B | Expected because treatment is class-randomised and student is nested in class; student RI absorbs class signal |
| Turn classifier is regex-based | B, C, D | Conservative (95.9% PASSIVE); rates differ from spec's classifier |
| WildChat uses IP hash, not user account | Exp D | Same IP ≠ same user; shared IPs inflate apparent repeat users |
| 4-session window may be too short | Exp C | Exp D tests longer-timescale prediction |
| LLM judge fails κ ≥ 0.6 gate (incl. GPT-4.1) | Exp E | Self/cross validity effect sizes are exploratory, not human-confirmed; the judge-reliability failure is itself a reported result |
| CriticEval dev files grade responses low/high only | Exp E | No medium stratum; overcorrection "sound" set rests on gold-high + judge soundness |
| Reasoning result is a single model | Exp E | DeepSeek-R1 only; not a controlled reasoning-vs-not manipulation |
| Self/cross factor is model-generated, not native | Exp E | Path A2 manipulates the factor over reused responses; not observed in the wild |
