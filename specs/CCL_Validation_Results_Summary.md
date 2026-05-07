# CCL Retroactive Validation: Results Summary

## Experiment A — Differential Error Detection by Claim Type

**CCL prediction**: Factual errors are more reliably detected than interpretive errors, which are more reliably detected than missing perspectives. This justifies the claim-type annotation system.

**Data**: FRANK benchmark (Pagnoni et al., NAACL 2021). 4,942 sentences from AI-generated summaries, each annotated by 3 trained annotators across 7 error types.

**CCL mapping**:

- FACTUAL (■): EntE + OutE + GramE
- SOURCE (♦): CircE
- INTERPRETIVE (▲): RelE + LinkE + CorefE
- Missing perspectives (•): no FRANK equivalent

### Key results

| CCL Category | N flagged | Prevalence | Majority agreement* | Fleiss' κ |
| --- | --- | --- | --- | --- |
| FACTUAL | 1,867 | 37.8% | **67.9%** | **0.585** |
| SOURCE | 639 | 12.9% | 24.9% | 0.247 |
| INTERPRETIVE | 1,382 | 28.0% | **27.8%** | **0.186** |

*Majority agreement = proportion of flagged sentences where ≥2 of 3 annotators agree.

**Fisher's exact test** (Factual vs. Interpretive majority detection): OR = 5.49, p < 10⁻¹¹⁶

**Interpretation**: When a factual error exists in AI-generated text, annotators agree on its presence 68% of the time. For interpretive errors, agreement drops to 28%. Even trained NLP annotators show a 2.4× detection reliability gap between factual and interpretive errors. Untrained learners would show this asymmetry more strongly. This validates the CCL's claim-type annotation system: without explicit scaffolding, interpretive claims pass unquestioned (exactly what the CCL's Stage 2 checklist and Stage 4 adversarial challenge are designed to address).

**Limitation**: FRANK annotators are trained crowdworkers, not students. The mapping from FRANK's summarisation error taxonomy to CCL's claim-type categories is analogical, not exact. "Missing perspectives" has no FRANK equivalent.

### Possible MQM extension

The WMT MQM dataset offers a complementary analysis with professional translators. The CCL mapping would be: Accuracy/Mistranslation → FACTUAL, Accuracy/Omission → Missing perspective, Style → INTERPRETIVE. This connects directly to the ToM-PE paper's error taxonomy and uses the existing analysis pipeline.

---

## Experiment A' — FELM: Claim-Type Asymmetry in ChatGPT Outputs

**CCL prediction**: The same claim-type detection asymmetry found in pre-LLM models (Experiment A) generalises to modern LLM-generated content. Factual claims should be more error-prone and structurally distinct from interpretive claims in ChatGPT outputs.

**Data**: FELM benchmark (Chen et al., NeurIPS 2023). 847 prompts → ChatGPT responses → 4,426 annotated segments across 5 domains. Per-segment binary factuality label + error type.

**CCL mapping**:

- FACTUAL (■): `wk` (World Knowledge) + `math` — verifiable entity/number/computation claims
- INTERPRETIVE (▲): `science` + `reasoning` — causal/mechanistic and logical inference claims
- GAP (•): `writing_rec` — subjective and perspectival claims

### FELM Results

#### Analysis 1: Error rate by CCL category

| CCL Category | N segments | N errors | Error rate |
| --- | --- | --- | --- |
| FACTUAL (wk + math) | 1,131 | 272 | **24.0%** |
| GAP (writing_rec) | 1,586 | 267 | 16.8% |
| INTERPRETIVE (science + reasoning) | 1,709 | 248 | **14.5%** |

**Chi-square** (error rates across categories): χ² = 43.87, df = 2, p < 0.0001

**Fisher's exact** (FACTUAL vs. INTERPRETIVE): OR = 1.87, p < 0.0001

#### Analysis 2: Error type distribution

| CCL Category | fooled | irrelevant_with_qst | knowledge_error | reasoning_error | total typed |
| --- | --- | --- | --- | --- | --- |
| FACTUAL | 8 | 10 | 121 | 7 | 146 |
| GAP | 21 | 84 | 141 | 0 | 246 |
| INTERPRETIVE | 0 | 8 | 92 | 2 | 102 |

**Chi-square** (error types differ by CCL category): χ² = 78.37, df = 6, p < 0.0001

The `irrelevant_with_qst` type is concentrated in the GAP category; `knowledge_error` dominates FACTUAL and INTERPRETIVE. This confirms that the claim categories produce qualitatively different error profiles, not just different rates.

#### Analysis 3: Segment length proxy

**Mann-Whitney U** (FACTUAL vs. INTERPRETIVE error segments): U = 32,838, p = 0.603 (n.s.)

Segment lengths do not differ significantly. Detection difficulty is not explained by segment length.

**Interpretation**: FELM shows a **double asymmetry** when read alongside FRANK:

1. **LLMs err more on factual claims** (FELM OR = 1.87): ChatGPT is 87% more likely to produce a factual error than an interpretive one.
2. **Humans detect factual errors more reliably** (FRANK OR = 5.49): annotators agree on factual errors 2.4× more often.

The highest-risk scenario for uncritical acceptance is therefore *interpretive* claims: they are produced with fewer errors but those errors are systematically harder for humans to catch. This is precisely the failure mode CCL's Stage 2 checklist and Stage 4 adversarial challenge are designed to address.

**Limitation**: FELM uses a single ground-truth annotation (no inter-annotator reliability measure). The `type` field is sparsely populated. The CCL mapping (5 FELM domains → 3 CCL categories) is coarser than FRANK's 7-code mapping.

---

## Experiment B — No-Self-Merge: Active vs. Passive AI Use

**CCL prediction**: AI output that undergoes independent human evaluation (no-self-merge) produces better learning outcomes than passively accepted output.

**Data**: Bastani et al. (2025, PNAS). ~1,000 Turkish high school students, 4 × 90-minute math tutoring sessions, pre-registered RCT with three arms.

### CCL mapping of experimental conditions

| Arm | CCL interpretation |
| --- | --- |
| GPT Base | No-self-merge **violated**: student receives full answers, no evaluation required |
| GPT Tutor | No-self-merge **enforced architecturally**: AI gives hints only, student must work through problems |
| Control | Baseline: no AI |

### Outcome results (ITT regression, cluster-robust SEs)

| Outcome | GPT Base (β) | GPT Tutor (β) |
| --- | --- | --- |
| AI-assisted practice score | +0.102*** (0.032) | +0.359*** (0.032) |
| Unassisted exam score | **-0.064** (0.027, p=0.016) | +0.001 (0.018, p=0.94) |

**The no-self-merge cost**: GPT Base students score 6.4 percentage points lower than controls on unassisted exams. GPT Tutor students show no harm. The gap between conditions (≈6.5pp) is the cost of violating the no-self-merge principle.

### Conversation analysis (novel contribution)

Automated turn classification of 28,666 user turns using regex-based classifier.

| Metric | GPT Tutor | GPT Base | Cohen's d | p |
| --- | --- | --- | --- | --- |
| Turns per conversation | 5.49 ± 3.79 | 2.45 ± 1.99 | +1.01 | < 10⁻⁴⁰ |
| Mean words per turn | 5.74 ± 7.24 | 10.41 ± 15.50 | -0.39 | < 10⁻⁴⁰ |
| Evaluative turn rate | 0.7% | 5.3% | -0.39 | < 10⁻⁵⁴ |
| Passive turn rate | 68.6% | 59.0% | +0.27 | < 10⁻¹⁰ |
| Active turn rate | 7.1% | 19.0% | -0.46 | < 10⁻³⁸ |

### The surprise that tells the story

GPT Base students show **higher** evaluative and active rates than GPT Tutor students. This seems paradoxical. The explanation reveals how no-self-merge actually works:

1. **GPT Tutor works through architectural constraint**, not through voluntary evaluation. It withholds direct answers, forcing students to work through problems (many short, procedural turns). The interaction structure *prevents* self-merge by giving nothing to merge.

2. **GPT Base allows self-merge**, and most students accept passively. But the *minority* who do push back ("are you sure?", "that's wrong") produce the higher evaluative rate. These voluntary challenges are insufficient: 95% of GPT Base turns are non-evaluative, and the condition still harms learning.

3. **Implication for the CCL**: The no-self-merge principle is not "evaluate more." It is "structure the interaction so that passive acceptance is impossible." This supports the CCL's emphasis on *structural* features (checklists, retrospectives, adversarial exchanges) over exhortations to "think critically."

### Manual validation sample

15 randomly sampled EVALUATIVE turns were inspected. All were correctly classified (genuine challenges to AI correctness). Dominant pattern: "are you sure" variants (vanilla condition). Sample:

> "are you sure that the answer is 7" [vanilla, s4]
> "you are wrong x can not be equal to 5" [vanilla, s3]
> "I found the answer 45, is that correct?" [aug, s3]

---

## Experiment C — Declining Effort Across Sessions (Bastani data)

**CCL prediction**: Without process-level reflection (Stage 3), critical engagement declines across sessions.

**Data**: Same Bastani et al. dataset. 451 students with ≥3 of 4 sessions. Within-student trajectory analysis.

### Session-level trends

| Treatment | Session | Eval rate | Passive rate | Active rate | Turns/conv |
| --- | --- | --- | --- | --- | --- |
| vanilla | 1 | 2.5% | 57.6% | 16.8% | 2.57 |
| vanilla | 2 | 5.0% | 55.9% | 19.6% | 2.49 |
| vanilla | 3 | 5.7% | 59.0% | 20.7% | 2.46 |
| vanilla | 4 | 6.4% | 60.3% | 19.4% | 2.34 |
| aug | 1 | 0.8% | 64.3% | 9.6% | 4.25 |
| aug | 2 | 0.6% | 64.2% | 8.4% | 5.40 |
| aug | 3 | 0.6% | 68.4% | 5.7% | 6.01 |
| aug | 4 | 1.0% | 71.1% | 5.9% | 5.84 |

### Within-student trajectory slopes

| Metric | Treatment | Mean slope | % declining | t | p |
| --- | --- | --- | --- | --- | --- |
| Passive rate | vanilla | +0.026 | 42% | 2.65 | **0.009** |
| Passive rate | aug | +0.027 | 44% | 3.79 | **<0.001** |
| Active rate | aug | -0.012 | 52% | -2.72 | **0.007** |
| Active rate | vanilla | +0.000 | 38% | 0.02 | 0.987 |
| Eval rate | vanilla | +0.012 | 20% | 3.34 | **0.001** |
| Eval rate | aug | +0.000 | 12% | 0.31 | 0.760 |

### Interpretation

The picture is nuanced, not a simple "effort declines":

**What supports the CCL prediction:**

- Passive rates increase significantly in **both** conditions (p<0.01). Students become more passive over time regardless of condition.
- Active engagement rate declines significantly for GPT Tutor students (p=0.007). Even structured scaffolding does not fully prevent disengagement over time.
- Mean words per turn show a declining trend in the GPT Tutor condition (p=0.06), suggesting decreasing elaboration.

**What complicates the prediction:**

- Evaluative rate *increases* for GPT Base students across sessions. This likely reflects a learning curve: students in the unrestricted condition gradually discover that GPT makes mistakes and begin challenging it. By session 4, their eval rate has tripled from session 1 (2.5% → 6.4%).
- The 4-session window (each 90 minutes) is likely too short for the dramatic decay documented in longer studies (Fan et al.'s multi-week study, WildChat's month-long trajectories).

**Implication**: The CCL's Stage 3 (retrospective/process reflection) is needed not because evaluation rate drops to zero, but because **passivity increases even under scaffolding**. Without explicit process-level reflection, the proportion of non-evaluative turns grows. The WildChat analysis (longer timescales) should show a clearer decay pattern.

---

## Cross-experiment synthesis

| CCL design choice | Experiment | Key finding | Effect size |
| --- | --- | --- | --- |
| Claim-type annotation (pre-LLM) | A (FRANK) | Factual detection >> Interpretive | OR = 5.49 |
| Claim-type annotation (LLM era) | A' (FELM) | Factual error rate 24% vs. interpretive 14.5% | OR = 1.87 |
| No-self-merge principle | B | Passive AI use harms learning | b = -0.064 (p=.02) |
| Architectural enforcement | B | Structure > voluntary evaluation | d = 1.01 (turns) |
| Process reflection (Stage 3) | C | Passivity increases across sessions | slope = +0.027*** |

The four experiments validate four independent mechanisms. Together they support the CCL's claim that critical engagement with AI requires:

1. **Differential scaffolding by claim type** (not all errors are equally visible — true for both pre-LLM and LLM-era models)
2. **Structural enforcement of evaluation** (not voluntary critical thinking)
3. **Process-level reflection over time** (engagement decays without it)

---

## Remaining work

1. **WildChat sample analysis** (Exp D): longitudinal decay over months, not just 4 sessions
2. **WMT MQM reanalysis** (Exp A extension): inter-rater agreement by error type using the existing Wasserstein pipeline, connecting to the ToM-PE claim taxonomy
3. **Discussion section on fading**: CCL fades scaffolding for a metacognitive skill (critical evaluation capacity), not for task performance. The Belland et al. meta-analytic counter-evidence applies to task scaffolding fading; CCL's target is different.
