"""Experiment A: Differential Error Detection by Claim Type.

CCL prediction: Factual errors are more reliably detected than interpretive
errors, which are more reliably detected than missing perspectives.

Data: FRANK benchmark (Pagnoni et al., NAACL 2021).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.shared.ccl_mappings import (
    CCL_CATEGORIES,
    FRANK_CODE_TO_CCL,
    FRANK_NO_ERROR,
    classify_frank_errors,
)
from experiments.shared.data_acquisition import frank_to_sentence_df, load_frank
from experiments.shared.stats_utils import fisher_exact_2x2, fleiss_kappa


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def build_sentence_category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a matrix: sentence × (category, annotator_flags, majority).

    For each sentence and each CCL category, determine:
    - Whether each annotator flagged that category
    - Whether majority (≥2/3) agree it is present

    Includes ``summary_id`` (article hash + model) and ``model_name`` so that
    downstream nested random-effects models can account for within-summary
    and within-model dependence.
    """
    rows = []
    for _, row in df.iterrows():
        ann_errors = [
            row["ann_0_errors"],
            row["ann_1_errors"],
            row["ann_2_errors"],
        ]
        summary_id = f"{row['hash']}_{row['model_name']}"

        for cat in CCL_CATEGORIES:
            # Per-annotator: does this annotator flag this category?
            flags = []
            for errors in ann_errors:
                if not isinstance(errors, list):
                    errors = []
                ccl_cats = classify_frank_errors(errors)
                flags.append(1 if cat in ccl_cats else 0)

            any_flagged = sum(flags) > 0
            majority = sum(flags) >= 2

            rows.append({
                "sentence_id": f"{summary_id}_{row['sentence_idx']}",
                "summary_id": summary_id,
                "model_name": row["model_name"],
                "ccl_category": cat,
                "ann_0": flags[0],
                "ann_1": flags[1],
                "ann_2": flags[2],
                "any_flagged": any_flagged,
                "majority_agree": majority,
            })

    return pd.DataFrame(rows)


def compute_category_stats(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute per-category statistics matching the spec table.

    Returns DataFrame with columns:
      ccl_category, n_flagged, prevalence, majority_agreement, fleiss_kappa
    """
    total_sentences = matrix.groupby("ccl_category").size().iloc[0]  # same for all
    results = []

    for cat in CCL_CATEGORIES:
        cat_data = matrix[matrix["ccl_category"] == cat]
        n_flagged = cat_data["any_flagged"].sum()
        prevalence = n_flagged / len(cat_data)

        # Majority agreement: among flagged sentences, proportion with ≥2/3 agreement
        flagged_data = cat_data[cat_data["any_flagged"]]
        if len(flagged_data) > 0:
            majority_pct = flagged_data["majority_agree"].mean()
        else:
            majority_pct = 0.0

        # Fleiss' kappa for this category (binary: flagged or not, 3 raters)
        # Ratings matrix: n_sentences × 2 (not-flagged, flagged)
        ratings = np.column_stack([
            3 - (cat_data["ann_0"] + cat_data["ann_1"] + cat_data["ann_2"]).values,
            (cat_data["ann_0"] + cat_data["ann_1"] + cat_data["ann_2"]).values,
        ])
        kappa = fleiss_kappa(ratings)

        results.append({
            "ccl_category": cat,
            "n_flagged": int(n_flagged),
            "prevalence": prevalence,
            "majority_agreement": majority_pct,
            "fleiss_kappa": kappa,
        })

    return pd.DataFrame(results)


def fisher_test_factual_vs_interpretive(
    matrix: pd.DataFrame,
) -> dict[str, float]:
    """Fisher's exact test comparing majority detection rates.

    2×2 table: (majority-agreed, not-agreed) × (FACTUAL, INTERPRETIVE)
    among flagged sentences.
    """
    fact = matrix[(matrix["ccl_category"] == "FACTUAL") & matrix["any_flagged"]]
    interp = matrix[(matrix["ccl_category"] == "INTERPRETIVE") & matrix["any_flagged"]]

    table = np.array([
        [fact["majority_agree"].sum(), (~fact["majority_agree"].astype(bool)).sum()],
        [interp["majority_agree"].sum(), (~interp["majority_agree"].astype(bool)).sum()],
    ])
    return fisher_exact_2x2(table)


# ---------------------------------------------------------------------------
# Mixed-effects logistic regression (addresses within-summary / within-model
# dependence flagged by EDM Reviewer 2)
# ---------------------------------------------------------------------------

def glmm_factual_vs_interpretive(
    matrix: pd.DataFrame,
    fit_method: str = "vb",
) -> dict:
    """Bayesian binomial GLMM with summary + model random intercepts.

    Model:
        majority_agree ~ ccl_category + (1 | summary_id) + (1 | model_name)

    Each FRANK summary belongs to exactly one model, so the design is nested
    (model_name/summary_id). We include both random intercepts so the
    fixed-effect estimate for ccl_category accounts for within-summary
    correlation *and* model-level variation.

    Parameters
    ----------
    matrix : DataFrame
        Output of ``build_sentence_category_matrix``.
    fit_method : {"vb", "map"}
        ``"vb"`` runs variational-Bayes inference (returns posterior mean
        and SD; we summarise with a Wald-style 95% interval and z-test).
        ``"map"`` runs maximum-a-posteriori with Laplace SEs (faster, but
        the SE underestimates uncertainty in the random effects).
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    flagged = matrix[
        matrix["any_flagged"]
        & matrix["ccl_category"].isin(["FACTUAL", "INTERPRETIVE"])
    ].copy()
    flagged["majority_int"] = flagged["majority_agree"].astype(int)
    flagged["is_factual"] = (flagged["ccl_category"] == "FACTUAL").astype(int)

    formula = "majority_int ~ is_factual"
    vc_formulas = {
        "summary": "0 + C(summary_id)",
        "model": "0 + C(model_name)",
    }

    md = BinomialBayesMixedGLM.from_formula(
        formula, vc_formulas, flagged
    )
    if fit_method == "map":
        result = md.fit_map()
    else:
        result = md.fit_vb()

    # Locate the fixed-effect coefficient for is_factual
    fe_names = list(result.model.exog_names)
    idx = fe_names.index("is_factual")
    beta = float(result.fe_mean[idx])
    se = float(result.fe_sd[idx])
    z = beta / se if se > 0 else float("nan")
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    # Two-sided Wald p-value
    from scipy.stats import norm
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))

    # Variance-component *parameters* are log-SDs (statsmodels parameterisation)
    vcp_names = list(result.model.vcp_names)
    vcp_logsd = np.asarray(result.vcp_mean)
    vc_summary = {
        name: float(np.exp(vcp_logsd[i]))
        for i, name in enumerate(vcp_names)
    }

    return {
        "method": (
            "Binomial GLMM (variational Bayes), random intercepts for "
            "summary_id and model_name"
        ),
        "n_observations": int(len(flagged)),
        "n_summaries": int(flagged["summary_id"].nunique()),
        "n_models": int(flagged["model_name"].nunique()),
        "beta_factual": beta,
        "se": se,
        "z": float(z),
        "p_value": p_value,
        "odds_ratio": float(np.exp(beta)),
        "or_ci_low": float(np.exp(ci_low)),
        "or_ci_high": float(np.exp(ci_high)),
        "random_effects_sd": vc_summary,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_experiment_a(force_download: bool = False) -> dict:
    """Run the full Experiment A analysis.

    Returns
    -------
    dict with keys: 'sentence_df', 'category_matrix', 'stats', 'fisher_test'
    """
    print("=" * 60)
    print("EXPERIMENT A: Differential Error Detection by Claim Type")
    print("=" * 60)

    # Load and flatten FRANK data
    print("\n1. Loading FRANK benchmark data...")
    raw = load_frank(force_download=force_download)
    sentence_df = frank_to_sentence_df(raw)
    print(f"   {len(sentence_df)} sentences from {len(raw)} summaries")

    # Build category matrix
    print("\n2. Building CCL category matrix...")
    matrix = build_sentence_category_matrix(sentence_df)

    # Compute per-category statistics
    print("\n3. Computing per-category statistics...")
    stats_df = compute_category_stats(matrix)
    print("\n   Results:")
    print(stats_df.to_string(index=False))

    # Fisher's exact test
    print("\n4. Fisher's exact test (Factual vs. Interpretive)...")
    fisher = fisher_test_factual_vs_interpretive(matrix)
    print(f"   Odds ratio = {fisher['odds_ratio']:.2f}")
    print(f"   p-value = {fisher['p_value']:.2e}")
    print("   (Treats sentences as independent — does not account for nesting)")

    # Mixed-effects logistic regression: addresses within-summary / model
    # dependence flagged by EDM Reviewer 2
    print("\n5. Mixed-effects logistic regression (Factual vs. Interpretive)...")
    print("   Model: majority_agree ~ is_factual + (1|summary_id) + (1|model_name)")
    try:
        glmm = glmm_factual_vs_interpretive(matrix)
        print(f"   Method: {glmm['method']}")
        print(f"   N obs = {glmm['n_observations']}, "
              f"summaries = {glmm['n_summaries']}, "
              f"models = {glmm['n_models']}")
        print(f"   Odds ratio (FACTUAL vs INTERPRETIVE) = {glmm['odds_ratio']:.2f} "
              f"[95% CI: {glmm['or_ci_low']:.2f}, {glmm['or_ci_high']:.2f}]")
        print(f"   beta = {glmm['beta_factual']:.3f}, SE = {glmm['se']:.3f}, "
              f"z = {glmm['z']:.2f}, p = {glmm['p_value']:.2e}")
        print(f"   Random-effect SDs (logit scale): "
              f"summary = {glmm['random_effects_sd'].get('summary', 0):.3f}, "
              f"model = {glmm['random_effects_sd'].get('model', 0):.3f}")
    except Exception as e:
        print(f"   GLMM fit failed: {e}")
        glmm = {"error": str(e)}

    # Interpretation
    print("\n6. Interpretation:")
    fact_row = stats_df[stats_df["ccl_category"] == "FACTUAL"].iloc[0]
    interp_row = stats_df[stats_df["ccl_category"] == "INTERPRETIVE"].iloc[0]
    ratio = fact_row["majority_agreement"] / interp_row["majority_agreement"]
    print(f"   Detection reliability gap: {ratio:.1f}x "
          f"(FACTUAL {fact_row['majority_agreement']:.1%} vs "
          f"INTERPRETIVE {interp_row['majority_agreement']:.1%})")
    print(f"   Fleiss' kappa gap: {fact_row['fleiss_kappa']:.3f} vs "
          f"{interp_row['fleiss_kappa']:.3f}")

    return {
        "sentence_df": sentence_df,
        "category_matrix": matrix,
        "stats": stats_df,
        "fisher_test": fisher,
        "glmm": glmm,
    }
