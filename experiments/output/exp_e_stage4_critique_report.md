# Experiment E — Stage 4: Self vs Cross-Model Adversarial Critique

Secondary analysis grounding Stage 4 (Peer Reviewer) of the Critical
Collaboration Ladder. Path A2 (controlled generation over reused dataset
responses) with a symmetric five-model author×critic matrix and a debiased
judge. Full specification: [`specs/ccl-stage4-critique-experiment-spec.md`](../../specs/ccl-stage4-critique-experiment-spec.md).

**Run:** 2026-06-15. N = 150 CriticEval items (stratified by gold quality:
89 high, 61 low), all 270 ManualReviewComment code diffs. Pool = GPT-4.1-nano,
DeepSeek-V3-0324, Llama-3.3-70B-Instruct-Turbo, Gemma-3-27B, DeepSeek-R1-0528
(each authors and critiques). Judge = gpt-oss-120b (outside the pool), with
GPT-4.1 as a frontier cross-check. Temperatures: generation 0.2, judging 0.0,
seed 20260615. 0 API errors across ~28,000 calls. Every interaction logged to
`exp_e/logs/llm_interactions.jsonl`; every result checkpointed under
`exp_e/store/` (fully resumable).

---

## Headline (read this first): the judge fails the pre-registered reliability gate

The spec (section 7) pre-registered a **κ ≥ 0.6 abort trigger**: if judge–human
agreement falls below it, *drop the LLM-judge headline and report only the
human-labelled subsets*. **The trigger fired.** No LLM judge — not even a
frontier model — reproduces human critique-validity labels on these datasets:

| Judge | Dataset | n | Agreement | Cohen's κ | PABAK | Ordinal ρ | Passes κ≥0.6 |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | CriticEval | 200 | 0.64 | **0.145** | 0.27 | 0.27 | no |
| GPT-4.1 | CriticEval | 200 | 0.68 | **0.226** | 0.36 | 0.37 | no |
| gpt-oss-120b | ManualReviewComment | 270 | 0.52 | **0.142** | 0.04 | — | no |
| GPT-4.1 | ManualReviewComment | 270 | 0.59 | **0.137** | 0.17 | — | no |

(`exp_e_judge_validation.csv`.) The failure is not a weak-judge or pure
base-rate artifact: it survives a frontier judge (GPT-4.1) and prevalence
adjustment (PABAK), and on CriticEval the judge agrees *less* than a constant
"valid" baseline would (0.64–0.68 < the 0.77 human-positive rate). This is
itself consistent with the LLM-as-judge bias literature (Zheng et al. 2023;
Panickssery et al. 2024) and **directly reinforces Stage 4's premise**: judging
whether an adversarial critique is *valid* is genuinely hard, which is exactly
why Stage 4 keeps a human (the learner) in the loop and why the adversary must
be architecturally separated from the author.

**Consequence.** Because the self-vs-cross contrast is intrinsically
judge-based (the human labels carry no self/cross factor), the results below are
reported as **exploratory and consistent-with-prior, not as human-validated
confirmatory findings.**

---

## RQ1 — Does cross-model critique achieve higher validity than self-critique?

**Direction confirmed (exploratory).** Mixed-effects logistic regression
`critique_valid ~ is_cross + gold_quality + (1|item) + (1|critic) + (1|author)`,
estimating the self/cross effect *within critic* (so it is not confounded with
critic capability):

> **OR(cross vs self) = 1.37, 95% CI [1.24, 1.50], p = 8.5×10⁻¹¹, n = 3,634.**

Overall validity rate: self 0.741 vs cross 0.782. The effect is consistent
*within* 4 of the 5 critic models (`exp_e_descriptive_validity.csv`):

| Critic | self | cross | Δ (cross − self) |
|---|---|---|---|
| GPT-4.1-nano | 0.660 | 0.797 | +0.137 |
| Gemma-3-27B | 0.793 | 0.844 | +0.051 |
| DeepSeek-R1 | 0.813 | 0.875 | +0.062 |
| DeepSeek-V3 | 0.591 | 0.598 | +0.007 |
| Llama-3.3-70B | 0.827 | 0.801 | −0.026 |

The within-critic pattern (a model is more lenient/less valid when critiquing
its *own* output) is the **self-preference / sycophancy signature** the
literature predicts (Xu et al. 2024; Panickssery et al. 2024), observed here in
the critique-validity framing Stage 4 actually presents.

**Verbosity control.** Validity correlates weakly with critique length
(β_len = 0.27, p = 2×10⁻⁸), but the cross advantage **survives** controlling for
length (β_is_cross = 0.21, p = 0.027). The effect is not merely "cross critiques
are longer."

## RQ2 — Does self-critique overcorrect more (push a sound answer toward wrong)?

**Not meaningfully.** On responses that are sound (judge-rated and/or gold
high-quality stratum, n = 3,214), the overcorrection rate (critique recommends a
change to a sound answer that would worsen it) is:

> **self 3.4% vs cross 3.1%; difference +0.3 points.**

The within-item GLMM gives OR(cross) = 0.78 (p = 0.034) and the paired McNemar is
"significant" (22 self-only vs 0 cross-only discordant pairs, p < 10⁻⁶), both
*directionally* consistent with H2 (self overcorrects slightly more) — but the
absolute magnitude is tiny (~3% either way) and the p-values reflect the large
sample, not a substantively important gap. **H2 is at best weakly supported.**
For context, on flawed responses the helpful-correction rate (I→C) is 0.54
(self) vs 0.58 (cross). All of these depend on the judge, which is not
human-validated, so they are reported descriptively only.

## RQ3 — Does the pattern hold in the code-review domain?

**Yes (exploratory).** Refitting RQ1's model on the ManualReviewComment code
critiques:

> **OR(cross vs self) = 1.29, 95% CI [1.19, 1.40], p = 3.2×10⁻⁹, n = 6,750.**

The cross > self direction generalises from general critique to code review,
the professional practice Stage 4's lineage draws on.

## Secondary — does reasoning improve critique validity?

The one reasoning model in the pool (DeepSeek-R1) produced the highest-validity
critiques: validity rate 0.86 vs 0.75 for the four non-reasoning critics. This
is consistent with reasoning improving critique quality, but it is a single
model and is reported as a descriptive observation, not a controlled contrast.

---

## Interpretation for the framework

Two things hold together. (1) The **self < cross** direction is observed
consistently — within-critic, across two domains, and robust to a verbosity
control — matching the strong literature prior and the self-preference
mechanism. This supports the **no-self-critique** corollary of no-self-merge:
the Stage 4 adversary should be a *separate* agent from the author. (2) **No
LLM judge reliably certifies critique validity against humans** (κ ≤ 0.23, even
GPT-4.1), so the validity of an adversarial challenge cannot be safely
outsourced to an automated judge — the learner's judgement is load-bearing.

Together these are precisely the Stage 4 design: an adversary that is
*structurally separated* from the author (because self-critique is weaker), with
a *human evaluator* deciding whether the challenge is valid or overcorrecting
(because even frontier judges cannot). The experiment therefore grounds Stage 4
both in what it automates (the separate adversary) and in what it deliberately
does not (the validity judgement).

**Paper integration.** Per spec section 13, Option II: fold this into the Stage 4
paragraph as direct empirical backing, foregrounding the human-grounded
judge-reliability result and presenting the self<cross direction as a
pre-registered, literature-consistent exploratory signal (the LLM-judge
confirmatory headline is dropped per the κ gate). The pre-registered design plus
this run satisfy the reviewers' core request to engage with *why Stage 4 is hard*.

## Outputs

| File | Contents |
|---|---|
| `exp_e/exp_e_results_table.csv` | RQ1/RQ2/RQ3 effect sizes + CIs |
| `exp_e/exp_e_judge_validation.csv` | judge–human κ (both judges, both datasets) |
| `exp_e/exp_e_descriptive_validity.csv` | self/cross validity per critic (general) |
| `exp_e/exp_e_descriptive_validity_code.csv` | same, code-review domain |
| `exp_e/exp_e_results.json` | full machine-readable results |
| `figures/exp_e_self_vs_cross.png` | self-vs-cross validity + overcorrection figure |
| `exp_e/store/*.jsonl` | all responses, critiques, judge verdicts (resumable) |
| `exp_e/logs/llm_interactions.jsonl` | full prompt+response audit log |

## Caveats

- LLM-judge results are **not human-validated** (κ gate failed); treat the
  self/cross effect sizes as exploratory.
- The CriticEval feedback_correction dev files graded responses low/high only
  (no medium stratum materialised in the sample), so the overcorrection "sound"
  stratum rests on the high grade plus judge soundness ratings.
- ManualReviewComment is CC-BY-4.0 on Zenodo (more permissive than the spec's
  provisional CC-BY-NC-SA note); derivatives and release are fine with attribution.
- Single judge per verdict (pointwise validity), so no order-swap was needed;
  pairwise order-swapped judging remains available but unused.
