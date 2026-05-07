# Experiment A' — FELM Extension: Claim-Type Asymmetry in ChatGPT Outputs

## Purpose

FRANK (Exp A) uses pre-LLM summarisation models (BART, PEGASUS, etc.). A reviewer could ask whether the detection asymmetry transfers to LLM-generated content. FELM directly addresses this: its annotations are on **ChatGPT responses**, with segment-level factuality labels across five domains that map onto CCL claim types.

FELM does not provide multiple independent annotators per segment (unlike FRANK), so we cannot compute inter-annotator agreement. Instead, we analyse two complementary signals:
1. **Error rate by domain**: Do different claim-type domains produce different error rates in ChatGPT? (Shows the *problem* exists for LLMs.)
2. **Error type distribution**: Do the types of errors differ systematically across domains? (Shows the *character* of errors differs.)

---

## Data Source

**FELM** (Chen et al., NeurIPS 2023)
- URL: https://huggingface.co/datasets/hkust-nlp/felm
- Paper: https://arxiv.org/abs/2310.00741
- 847 prompts → ChatGPT responses → 4,427 annotated segments
- 5 domains: World Knowledge (wk), Science/Technology (st), Writing/Recommendation (wr), Reasoning (re), Math (math)
- Per-segment: binary factuality label (true/false), error type, error reason, reference links

## Download

```python
# Method 1: Direct download
wget https://huggingface.co/datasets/hkust-nlp/felm/resolve/main/all.jsonl

# Method 2: HuggingFace datasets
from datasets import load_dataset
# Load by domain:
ds_wk = load_dataset("hkust-nlp/felm", "wk")
ds_st = load_dataset("hkust-nlp/felm", "st")
ds_wr = load_dataset("hkust-nlp/felm", "wr")
ds_re = load_dataset("hkust-nlp/felm", "re")
ds_math = load_dataset("hkust-nlp/felm", "math")
```

---

## CCL Mapping

| FELM Domain | N segments | N errors | CCL Category | Rationale |
|-------------|-----------|----------|-------------|-----------|
| World Knowledge | 532 | 147 | **FACTUAL** (■) | Verifiable entity/date/number claims |
| Math | 599 | 122 | **FACTUAL** (■) | Verifiable computational claims |
| Science/Technology | 683 | 101 | **INTERPRETIVE** (▲) | Causal/mechanistic reasoning |
| Reasoning | 1,025 | 148 | **INTERPRETIVE** (▲) | Logical inference, framing |
| Writing/Recommendation | 1,588 | 267 | **GAP/PERSPECTIVAL** (•) | Subjective, perspectival completeness |

Note: This mapping is coarser than FRANK's. FELM domains don't distinguish "source" (♦) as a category. The Writing/Recommendation domain contains perspectival claims that are closest to "missing perspectives" but also include stylistic judgments.

---

## Analysis Plan

### Analysis 1: Error Rate by CCL Category

```python
import json
import pandas as pd
from scipy import stats

# Load all.jsonl
records = []
with open('felm_all.jsonl') as f:
    for line in f:
        item = json.loads(line)
        # Determine domain from source or load separately
        for i, (seg, label) in enumerate(zip(item['segmented_response'], item['labels'])):
            records.append({
                'index': item['index'],
                'segment': seg,
                'label': label,  # True = factual, False = error
                'error_type': item['type'][i] if item['type'][i] else None,
                'domain': item.get('domain', 'unknown')
            })

df = pd.DataFrame(records)

# If domain field is not present, load by domain separately:
# for domain_name in ['wk', 'st', 'wr', 're', 'math']:
#     ds = load_dataset("hkust-nlp/felm", domain_name)
#     for item in ds['test']:
#         ... (same structure, add domain=domain_name)

# CCL mapping
ccl_map = {
    'wk': 'FACTUAL', 'math': 'FACTUAL',
    'st': 'INTERPRETIVE', 're': 'INTERPRETIVE',
    'wr': 'GAP'
}
df['ccl_category'] = df['domain'].map(ccl_map)

# Error rate by CCL category
error_rates = df.groupby('ccl_category').agg(
    n_segments=('label', 'count'),
    n_errors=('label', lambda x: (~x).sum()),  # False = error
    error_rate=('label', lambda x: (~x).mean())
).reset_index()

print(error_rates)
```

**Expected output (from README statistics):**

| CCL Category | N segments | N errors | Error rate |
|-------------|-----------|----------|-----------|
| FACTUAL (wk+math) | 1,131 | 269 | 23.8% |
| INTERPRETIVE (st+re) | 1,708 | 249 | 14.6% |
| GAP (wr) | 1,588 | 267 | 16.8% |

**Statistical test**: Chi-square or Fisher's exact on error counts across categories.

**CCL prediction**: If the asymmetry is about human *detection* difficulty (not AI error production), we'd expect FACTUAL errors to be more frequent but more obviously wrong, while INTERPRETIVE errors, though less frequent, are harder to identify. The error rate alone doesn't test detectability — it tests AI reliability. But if combined with the FRANK agreement data, it shows that the claim types where LLMs are most error-prone (factual) are also the ones humans detect most easily, while the types humans miss (interpretive) do produce errors that slip through unnoticed.

### Analysis 2: Error Type Distribution

```python
# Error types by domain
error_types = df[df['label'] == False].groupby(['domain', 'error_type']).size().unstack(fill_value=0)
print(error_types)

# Do error types cluster differently by CCL category?
error_types_ccl = df[df['label'] == False].groupby(['ccl_category', 'error_type']).size().unstack(fill_value=0)
print(error_types_ccl)
```

FELM error types include: `knowledge_error`, `reasoning_error`, `calculation_error`, `output_format_error`, and others. If FACTUAL domains are dominated by `knowledge_error` (verifiable, catchable) while INTERPRETIVE domains show `reasoning_error` (subtle, harder to detect), this directly supports the CCL taxonomy.

### Analysis 3: Segment-Level Difficulty Proxy

```python
# Are erroneous segments in interpretive domains longer?
# (Longer = error is embedded in more context = harder to isolate)
df['seg_len'] = df['segment'].apply(lambda x: len(x.split()))

error_segs = df[df['label'] == False]
print(error_segs.groupby('ccl_category')['seg_len'].describe())

# Mann-Whitney: are interpretive error segments longer than factual?
fact_lens = error_segs[error_segs['ccl_category']=='FACTUAL']['seg_len']
interp_lens = error_segs[error_segs['ccl_category']=='INTERPRETIVE']['seg_len']
u, p = stats.mannwhitneyu(fact_lens, interp_lens)
print(f"Factual error segments: {fact_lens.mean():.1f} words")
print(f"Interpretive error segments: {interp_lens.mean():.1f} words")
print(f"U={u:.0f}, p={p:.4f}")
```

If interpretive error segments are longer, the error signal is more diluted, supporting the claim that these errors are inherently harder to spot.

---

## Integration into the Paper

### Where it fits

In Section 2.1 (Observation 1), after the FRANK results:

> To test whether this asymmetry extends to LLM-generated content, we analyse FELM \cite{chen_felm_2023}, which annotates 4,427 segments from ChatGPT responses across five domains. Mapping these to CCL categories (World Knowledge + Math → Factual; Reasoning + Science/Tech → Interpretive; Writing/Recommendation → Gap), we find [RESULT]. The error-type distribution confirms [RESULT]: factual domains are dominated by verifiable knowledge errors, while interpretive domains show reasoning errors that are harder to isolate.

### What it adds

1. **Directly addresses the pre-LLM concern**: FELM uses ChatGPT, not BART/PEGASUS.
2. **Shows error types differ qualitatively**: Not just detection rates, but the *character* of errors varies by CCL category.
3. **Strengthens the design argument**: ChatGPT produces errors across all claim types, but the types humans are worst at catching (interpretive) are exactly where the errors are most subtle.

### One row in synthesis table

| CCL design choice | Experiment | Key finding | Effect size |
|---|---|---|---|
| Claim-type annotation (LLM era) | A' (FELM) | Error types differ across CCL categories in ChatGPT | [chi-square result] |

---

## Bib Entry

```bibtex
@inproceedings{chen_felm_2023,
  author = {Chen, Shiqi and Zhao, Yiran and Zhang, Jinghan and Chern, I-Chun and Gao, Siyang and Liu, Pengfei and He, Junxian},
  title = {{FELM}: Benchmarking Factuality Evaluation of Large Language Models},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {36},
  year = {2023}
}
```

---

## Effort Estimate

- Data download: 5 minutes
- Analysis 1 (error rates): 30 minutes
- Analysis 2 (error types): 30 minutes  
- Analysis 3 (segment length): 15 minutes
- Paper integration: 30 minutes
- **Total: ~2 hours**

## Expected Outcome

FELM will likely show that ChatGPT's error rates differ by domain and that error types cluster differently across CCL categories. This doesn't directly replicate the FRANK inter-annotator agreement result (FELM has single ground-truth labels, not multiple annotators), but it provides a complementary line of evidence: the phenomenon that motivates the CCL's claim-type system is present in LLM outputs specifically, not just in pre-LLM summarisers.

The strongest possible finding would be: FACTUAL domains show the highest error rate but the simplest error types (knowledge_error), while INTERPRETIVE domains show lower error rates but more complex error types (reasoning_error). This would mean LLMs make fewer interpretive errors but the ones they make are harder to catch — exactly the scenario where CCL's differential scaffolding is most needed.
