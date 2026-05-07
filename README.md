# Critical Claim Literacy (CCL) — Validation Experiments

Retroactive validation of the CCL framework against four public empirical datasets. The framework makes three structural predictions about how people interact with AI-generated content:

1. **Claim-type asymmetry** — factual errors are more reliably detected than interpretive errors, justifying differential annotation scaffolding.
2. **No-self-merge** — passive acceptance of AI output harms learning; structural enforcement (hint-only AI) is more effective than voluntary evaluation.
3. **Engagement decay** — without explicit process reflection, critical engagement declines over time.

Five experiments test these predictions:

| Experiment | Dataset | What it tests |
| --- | --- | --- |
| A | FRANK (Pagnoni et al., NAACL 2021) | Inter-annotator agreement on factual vs interpretive error flags |
| A' | FELM (Chen et al., NeurIPS 2023) | ChatGPT factual vs interpretive error rates |
| B | Bastani et al. (PNAS 2025) | Conversation architecture under hint-only vs full-answer AI |
| C | Bastani et al. (PNAS 2025) | Within-student engagement decay across 4 sessions |
| D | WildChat-1M (Allen AI) | Long-term engagement decay across months |

The full results report (with all tables, figures, and interpretation) lives at [`experiments/output/CCL_Validation_Experiments_Report.md`](experiments/output/CCL_Validation_Experiments_Report.md).

## Statistical methodology

Each test operates at the unit of analysis appropriate for its nesting structure:

- **Exp A** — binomial GLMM with `(1 | summary_id) + (1 | model_name)` random intercepts (statsmodels `BinomialBayesMixedGLM`).
- **Exp A'** — Mann-Whitney U / Kruskal-Wallis on per-prompt error rates (segments are not independent within a prompt; CCL category is constant within prompt, so a GLMM would be degenerate).
- **Exp B** — linear mixed-effects model with `(1 | class_id) + (1 | student_id)` random intercepts (statsmodels `mixedlm`).
- **Exp C / D** — one-sample t-test on per-student / per-user OLS slopes (one observation per subject).

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
│   ├── shared/                  # Data acquisition, statistical utilities, plotting
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

All datasets are downloaded on first run by `experiments.shared.data_acquisition` and cached locally. They are excluded from git via `.gitignore`.

## License

MIT.
