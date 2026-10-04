# Licence

This repository and its archived Zenodo snapshot contain three kinds of material, licensed differently.

## 1. Code

All source code (`experiments/`, including `run_all.py`, `shared/`, and the per-experiment packages) is licensed under the **MIT Licence**.

```
Copyright (c) 2026 Diana Nurbakova and Liana Ermakova

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Documentation, specifications and result tables

The written materials and derived statistics authored by us are licensed under
**Creative Commons Attribution 4.0 International (CC-BY-4.0)**:
<https://creativecommons.org/licenses/by/4.0/>

This covers `README.md`, `specs/`, the supplementary materials, and the aggregate
result files in `experiments/output/` (the `exp_*.csv` tables, `exp_e_results.json`,
and the validation report).

This is the licence recorded on the Zenodo deposit, because the deposit is
predominantly data and documentation. The code within it remains MIT-licensed
under section 1.

## 3. Third-party content embedded in generated artefacts

The files under `experiments/output/exp_e/store/` contain, alongside our generated
model responses and critiques, **verbatim content from third-party datasets**:

| File(s) | Embedded third-party content | Upstream licence |
|---|---|---|
| `responses.jsonl`, `critiques_general.jsonl`, `judgments.jsonl`, `soundness.jsonl`, `validate_criticeval*.jsonl` | CriticEval questions and reference responses (`question`, `response` fields) | Apache-2.0 |
| `code_comments.jsonl`, `critiques_code.jsonl`, `validate_mrc*.jsonl` | ManualReviewComment code patches and gold review comments (`patch`, `comment` fields) | CC-BY-4.0 |

That content remains under its original licence and is **not** relicensed by this
repository. Our additions to those files (generated responses, critiques, judge
verdicts and derived fields) are CC-BY-4.0 under section 2.

### Attribution and notices

**CriticEval** (Lan et al., NeurIPS 2024), Apache-2.0. Licensed under the Apache
Licence, Version 2.0; see <http://www.apache.org/licenses/LICENSE-2.0>. *Changes
made:* items were sampled and stratified, and new model responses, critiques and
judge verdicts were generated over them.

**MetaCritique** (Sun et al., ACL Findings 2024), Apache-2.0. Licensed under the
Apache Licence, Version 2.0. *Changes made:* the critique-quality scheme informed
our validity rubric.

**ManualReviewComment** (Liu et al., MSR 2025), CC-BY-4.0,
<https://zenodo.org/records/13150598>. *Changes made:* comments were paired with
newly generated model critiques and judge verdicts.

## 4. Datasets not redistributed

The following are downloaded on first run and excluded from version control via
`.gitignore`. They are **not** redistributed here, so only citation applies:

- FRANK (Pagnoni et al., NAACL 2021)
- FELM (Chen et al., NeurIPS 2023)
- Conversation logs from Bastani et al. (PNAS 2025)
- WildChat-1M (Allen AI), <https://huggingface.co/datasets/allenai/WildChat-1M>

Users who re-run the pipeline obtain these directly from their sources and are
bound by those sources' own terms.

## How to cite

If you use this work, please cite the paper and the archived snapshot. BibTeX
entries are given in the [Citation section of the README](README.md#citation).
