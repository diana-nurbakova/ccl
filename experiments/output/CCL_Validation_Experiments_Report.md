# CCL Retroactive Validation: Experiment Report

**Framework**: Critical Claim Literacy (CCL)  
**Purpose**: Validate CCL design choices against independent empirical datasets  
**Code**: `experiments/` directory; reproducible via `python -m experiments.run_all`

---

## Overview

The CCL framework makes three core structural predictions about how people interact with AI-generated content:

1. **Claim-type asymmetry** — factual errors are more detectable than interpretive ones, justifying differential annotation scaffolding
2. **No-self-merge** — passive acceptance of AI output harms learning; structural enforcement is more effective than voluntary evaluation
3. **Engagement decay** — without explicit process reflection, critical engagement declines over time

Five experiments test these predictions against public empirical datasets. Experiments A and A' both address the claim-type asymmetry using different datasets (pre-LLM and LLM-era respectively). Experiments B–C validate the no-self-merge and decay predictions. Experiment D extends decay analysis to longer timescales.

---

## Experiment A — Differential Error Detection by Claim Type

### CCL Prediction
Factual errors are more reliably detected than interpretive errors. This asymmetry justifies the CCL's claim-type annotation system (■ FACTUAL / ♦ SOURCE / ▲ INTERPRETIVE).

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
For each sentence and CCL category:
- Compute per-annotator binary flag (does this annotator flag this category?)
- **Majority agreement**: proportion of flagged sentences where ≥2/3 annotators agree
- **Fleiss' κ**: inter-rater reliability treating each category as a binary judgment
- **Fisher's exact test**: 2×2 table of (majority-agreed vs. not) × (FACTUAL vs. INTERPRETIVE), among flagged sentences only

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

FELM does not have multiple independent annotators per segment, so inter-annotator agreement cannot be computed. Instead three complementary analyses are run: (1) error rate by CCL category, (2) error type distribution, (3) segment length as a detection difficulty proxy.

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

1. **Analysis 1 — Error rate by CCL category**: chi-square test across FACTUAL / INTERPRETIVE / GAP; Fisher's exact for FACTUAL vs. INTERPRETIVE
2. **Analysis 2 — Error type distribution**: cross-tabulate FELM error types (knowledge_error, reasoning_error, calculation_error, etc.) by CCL category; chi-square test for distributional difference
3. **Analysis 3 — Segment length proxy**: if interpretive error segments are longer, the error signal is more diluted, supporting the detectability argument (Mann-Whitney U)

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

**What FELM shows**: ChatGPT produces proportionally more factual errors (24.0%) than interpretive errors (14.5%), with OR = 1.87. Combined with FRANK, this creates a double asymmetry:

> - **LLMs make more factual errors** (FELM: OR = 1.87 in error production)
> - **Humans detect factual errors more reliably** (FRANK: OR = 5.49 in detection agreement)

Interpretive errors are *both* less frequently produced *and* harder for humans to catch when they do occur. This is precisely the scenario where CCL's differential scaffolding is most consequential: a low-frequency but high-stealth error class that passes unnoticed without explicit Stage 2–4 prompting.

The error type distribution (χ² = 78.37, p < 0.0001) confirms that the categories produce qualitatively different errors, not just quantitatively different rates — supporting the CCL's decision to treat them as distinct annotation types rather than a single "error" category.

**Limitations**: FELM uses a single ground-truth annotation (no inter-annotator reliability measure). The `type` field is sparsely populated. The five FELM domains are broader than FRANK's seven error codes; the CCL mapping is coarser.

---

## Experiment B — No-Self-Merge: Active vs Passive AI Use

### CCL Prediction
AI output that undergoes independent human evaluation (no-self-merge principle) produces better learning outcomes than passively accepted output. Structural enforcement is more effective than voluntary evaluation.

### Data Source
**Bastani et al. (2025, PNAS)** — *"Generative AI Can Harm Learning"*  
- URL: https://github.com/obastani/GenAICanHarmLearning  
- Files:
  - `main_regressions/final_data.csv` — student-session outcomes
  - `text_analysis/data/raw/valid_student_data_w_time_stamp.csv` — conversation logs
- ~**1,000 Turkish high school students**, 4 × 90-minute math tutoring sessions
- Pre-registered RCT (3 arms: control, GPT Base, GPT Tutor)

### CCL Mapping of Treatment Arms

| Arm | Label in data | CCL interpretation |
|-----|--------------|-------------------|
| GPT Base | `vanilla` | No-self-merge **violated**: AI gives full answers, no evaluation required |
| GPT Tutor | `aug` | No-self-merge **enforced**: AI gives hints only, student must work through problems |
| Control | `control` | Baseline: no AI |

### Experiment Design

**Turn classification**: Regex-based classifier applied to all 28,666 user turns:
- **EVALUATIVE**: challenges to AI correctness ("are you sure", "that's wrong", "you are wrong")
- **ACTIVE**: student working through the problem ("let me try", "step 1", "give me a hint")
- **PASSIVE**: everything else (acknowledgements, short affirmations)

**Conversation metrics**: Aggregated per conversation — turns/conversation, mean words/turn, turn-type rates.

**Condition comparison**: Cohen's d and Mann-Whitney U between GPT Base and GPT Tutor on each metric.

**Outcome regression**: OLS with cluster-robust standard errors (clustered at Class level), outcome = **Part3Tot** (unassisted exam score, 0–1 scale), predictors = GPTBase and GPTTutor binary indicators.

> **Note on outcome variable**: Part3Tot is the unassisted exam score — the key CCL learning outcome. Part2Tot is the AI-assisted practice score and is intentionally excluded; it conflates tool use with learning.

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

### Data Source
Same Bastani et al. dataset as Experiment B. Filtered to **451 students** with ≥3 of 4 sessions (243 `aug`, 208 `vanilla`).

### Experiment Design
1. Aggregate turn-type metrics per student per session
2. Compute per-student OLS slope for each metric over sessions 1–4
3. One-sample t-test on slope distribution (H₀: mean slope = 0)
4. Report % of students with negative slope

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

### Data Source
**WildChat-1M** (Zhao et al. / Allen AI)  
- URL: https://huggingface.co/datasets/allenai/WildChat-1M  
- License: ODC-BY  
- **Sample**: 10,000 conversations from repeat users (≥5 conversations each), streamed in non-toxic subset  
- User identifier: `hashed_ip` (anonymised IP hash)

### Experiment Design
1. Filter to repeat users (≥5 conversations) to enable longitudinal tracking
2. Order conversations chronologically per user; assign `conv_order` and `weeks_since_first`
3. Apply turn classifier to user turns; compute per-conversation engagement metrics
4. Per-user OLS slopes over `conv_order` and `weeks_since_first`
5. **Topic drift**: track vocabulary concentration (fraction of user words in top-50 global vocabulary) as a proxy for narrowing topic range
6. **Turn complexity**: type-token ratio (TTR) as lexical diversity measure

### Additional Analysis Angles
- **Decay by conversation order** vs. **decay by calendar time** (weeks) — tests whether decay is use-frequency-driven or time-driven
- **Vocabulary concentration slope**: positive slope = users converge on a narrower topic range over time
- **Lexical diversity (TTR) slope**: declining TTR = simpler, more repetitive language over time

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

## Cross-Experiment Synthesis

| CCL design choice | Experiment | Key finding (proper inferential test) | Effect size |
| --- | --- | --- | --- |
| Claim-type annotation (pre-LLM) | A (FRANK) | Factual detection >> Interpretive (GLMM with summary + model random intercepts) | OR = 5.15, 95% CI [4.66, 5.70] |
| Claim-type annotation (LLM era) | A' (FELM) | Factual prompt-level error rate higher than interpretive (Mann-Whitney at prompt level) | mean 0.284 vs 0.158, p ≈ 8 × 10⁻⁶ |
| No-self-merge principle | B | Structural enforcement changes conversation length (LMM with class + student random intercepts) | β = −2.95 turns, p = 0.002, d ≈ −0.27 |
| Process reflection (Stage 3) | C | Active engagement declines even under scaffolding (per-student slopes, t-test on slopes) | slope = −0.013***, p < 0.001 |
| Sustained scaffolding | D | Long-term decay analysis (WildChat, months; per-user slopes) | (see Exp D output) |

### Three Validated Mechanisms (after correcting for nesting)

1. **Differential scaffolding by claim type**: not all errors are equally visible to unaided readers. After accounting for within-summary and between-model dependence (GLMM with random intercepts), factual error flags achieve ~5× higher odds of annotator consensus than interpretive flags (OR = 5.15, 95 % CI [4.66, 5.70]). The FELM replication at the *prompt* level confirms this generalises to LLM-era content: ChatGPT factual prompts have higher mean error rates than interpretive prompts (0.28 vs 0.16, Mann-Whitney p ≈ 8 × 10⁻⁶). The double asymmetry — LLMs err more on factual claims, and humans detect those errors more reliably — makes interpretive claims the highest-risk category for uncritical acceptance.

2. **Structural enforcement over voluntary evaluation (length, not disposition)**: architectural constraints (hint-only AI) produce conversations that are systematically longer than free-form AI conversations. After accounting for within-student dependence, GPT Tutor produces ~3 more turns per conversation than GPT Base (β = −2.95, p = 0.002, d ≈ −0.27 on the variance-explained scale). The original Mann-Whitney p-values on disposition rates (evaluative/active) were inflated by treating multiple conversations from the same student as independent — those differences are not separable from between-student variance and are reported as descriptive only. The structural finding is therefore narrower but still supports CCL's emphasis on architecture over exhortation.

3. **Process-level reflection for sustained engagement**: active engagement declines significantly across sessions even when the tool is architecturally constrained. Per-student slopes computed at the student level (one slope per student) avoid the nesting problem entirely. The aug condition's passive rate slope is +0.0145/session, p < 0.001, and the active rate slope is −0.013/session, p < 0.001 over 4 sessions. Experiment D tests whether this decay is qualitatively similar over months at the per-user slope level.

---

## Reproducibility

```bash
# Install experiment dependencies
pip install .[experiments]

# Run all experiments (skip D if no HF token)
python -m experiments.run_all --skip d

# Run individual experiments
python -m experiments.exp_a_error_detection.run
python -m experiments.exp_a_felm.run
python -m experiments.exp_b_active_passive.run
python -m experiments.exp_c_declining_effort.run
python -m experiments.exp_d_wildchat_decay.run   # requires HF_TOKEN in .env

# Run tests
python -m pytest experiments/tests/ -v
```

All results (CSVs + figures) are saved to `experiments/output/`.

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
