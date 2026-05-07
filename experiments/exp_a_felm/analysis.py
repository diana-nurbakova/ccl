"""Experiment A' (FELM): Claim-Type Asymmetry in ChatGPT Outputs.

Complements Experiment A (FRANK) by validating the CCL claim-type asymmetry
on LLM-generated content (ChatGPT), not pre-LLM summarisers.

Three analyses:
  1. Error rate by CCL category (chi-square test)
  2. Error type distribution by CCL category
  3. Segment-length proxy for detection difficulty (Mann-Whitney U)

Data: FELM (Chen et al., NeurIPS 2023) — 847 prompts, 4,427 segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from experiments.shared.ccl_mappings import FELM_DOMAIN_TO_CCL
from experiments.shared.data_acquisition import load_felm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_ccl_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add ccl_category column using FELM_DOMAIN_TO_CCL mapping."""
    df = df.copy()
    df["ccl_category"] = df["domain"].map(FELM_DOMAIN_TO_CCL)
    return df


# ---------------------------------------------------------------------------
# Analysis 1: Error rate by CCL category
# ---------------------------------------------------------------------------

def compute_error_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute error rate per domain and per CCL category.

    Returns two DataFrames: by-domain and by-CCL-category.
    """
    df = df.dropna(subset=["is_error"])

    by_domain = (
        df.groupby("domain")
        .agg(
            n_segments=("is_error", "count"),
            n_errors=("is_error", "sum"),
        )
        .assign(error_rate=lambda x: x["n_errors"] / x["n_segments"])
        .reset_index()
    )
    by_domain["ccl_category"] = by_domain["domain"].map(FELM_DOMAIN_TO_CCL)
    by_domain = by_domain.sort_values("ccl_category")

    by_ccl = (
        df.groupby("ccl_category")
        .agg(
            n_segments=("is_error", "count"),
            n_errors=("is_error", "sum"),
        )
        .assign(error_rate=lambda x: x["n_errors"] / x["n_segments"])
        .reset_index()
    )
    return by_domain, by_ccl


def chi_square_error_rates(by_ccl: pd.DataFrame) -> dict:
    """Chi-square test across CCL categories on error counts.

    Tests H0: error rates are equal across FACTUAL, INTERPRETIVE, GAP.
    """
    # contingency table: rows = categories, cols = [errors, non-errors]
    table = np.array([
        [row["n_errors"], row["n_segments"] - row["n_errors"]]
        for _, row in by_ccl.iterrows()
    ])
    chi2, p, dof, expected = sp_stats.chi2_contingency(table)
    return {"chi2": chi2, "p_value": p, "dof": dof}


def fisher_factual_vs_interpretive(by_ccl: pd.DataFrame) -> dict:
    """Fisher's exact test: FACTUAL vs INTERPRETIVE error rates."""
    from experiments.shared.stats_utils import fisher_exact_2x2
    fact = by_ccl[by_ccl["ccl_category"] == "FACTUAL"].iloc[0]
    interp = by_ccl[by_ccl["ccl_category"] == "INTERPRETIVE"].iloc[0]
    table = [
        [int(fact["n_errors"]), int(fact["n_segments"] - fact["n_errors"])],
        [int(interp["n_errors"]), int(interp["n_segments"] - interp["n_errors"])],
    ]
    return fisher_exact_2x2(table)


# ---------------------------------------------------------------------------
# Prompt-level analysis (addresses within-response segment dependence flagged
# by EDM Reviewer 2). FELM domains are constant within prompt, so each prompt
# belongs to a single CCL category — the appropriate unit of inference.
# ---------------------------------------------------------------------------

def compute_prompt_level_error_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-segment labels to one error rate per prompt.

    Returns DataFrame with columns:
      record_index, domain, ccl_category, n_segments, n_errors, error_rate
    """
    df = df.dropna(subset=["is_error", "ccl_category"]).copy()
    df["is_error_int"] = df["is_error"].astype(int)
    prompt_stats = (
        df.groupby(["record_index", "domain", "ccl_category"])
        .agg(
            n_segments=("is_error_int", "count"),
            n_errors=("is_error_int", "sum"),
        )
        .reset_index()
    )
    prompt_stats["error_rate"] = (
        prompt_stats["n_errors"] / prompt_stats["n_segments"]
    )
    return prompt_stats


def kruskal_across_ccl_prompts(prompt_stats: pd.DataFrame) -> dict:
    """Kruskal-Wallis on per-prompt error rates across CCL categories."""
    groups = [
        g["error_rate"].values
        for _, g in prompt_stats.groupby("ccl_category")
    ]
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        return {"error": "Insufficient groups"}
    H, p = sp_stats.kruskal(*groups)
    return {
        "H": float(H),
        "p_value": float(p),
        "n_groups": len(groups),
        "n_prompts_total": int(sum(len(g) for g in groups)),
    }


def mannwhitney_factual_vs_interpretive_prompts(
    prompt_stats: pd.DataFrame,
    alternative: str = "greater",
) -> dict:
    """Mann-Whitney U on prompt-level error rates: FACTUAL vs INTERPRETIVE.

    Default ``alternative='greater'`` tests CCL's directional prediction:
    factual error rates exceed interpretive error rates.
    """
    fact = prompt_stats.loc[
        prompt_stats["ccl_category"] == "FACTUAL", "error_rate"
    ].values
    interp = prompt_stats.loc[
        prompt_stats["ccl_category"] == "INTERPRETIVE", "error_rate"
    ].values
    if len(fact) == 0 or len(interp) == 0:
        return {"error": "Missing FACTUAL or INTERPRETIVE prompts"}

    U, p = sp_stats.mannwhitneyu(fact, interp, alternative=alternative)

    # Rank-biserial correlation as effect size: r = 2*U/(n1*n2) - 1
    # scipy returns U1 = #(x_i > y_j) (+ 0.5 ties), so r > 0 ↔ FACTUAL ranks higher.
    n1, n2 = len(fact), len(interp)
    rank_biserial = (2.0 * U) / (n1 * n2) - 1.0

    return {
        "U": float(U),
        "p_value": float(p),
        "alternative": alternative,
        "n_factual_prompts": int(n1),
        "n_interpretive_prompts": int(n2),
        "median_factual": float(np.median(fact)),
        "median_interpretive": float(np.median(interp)),
        "mean_factual": float(np.mean(fact)),
        "mean_interpretive": float(np.mean(interp)),
        "rank_biserial": float(rank_biserial),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Error type distribution
# ---------------------------------------------------------------------------

def compute_error_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate error types by CCL category (errors only).

    Returns a DataFrame: rows = CCL categories, columns = error types.
    """
    errors = df[df["is_error"] == True].dropna(subset=["error_type"])
    if len(errors) == 0:
        return pd.DataFrame()

    crosstab = (
        errors.groupby(["ccl_category", "error_type"])
        .size()
        .unstack(fill_value=0)
    )
    # Add row totals and proportions
    crosstab["total"] = crosstab.sum(axis=1)
    return crosstab


def chi_square_error_types(df: pd.DataFrame) -> dict:
    """Chi-square test: does error type distribution differ by CCL category?"""
    errors = df[df["is_error"] == True].dropna(subset=["error_type", "ccl_category"])
    if len(errors) < 10:
        return {"error": "Too few typed errors for chi-square"}

    crosstab = pd.crosstab(errors["ccl_category"], errors["error_type"])
    # Drop columns with all zeros
    crosstab = crosstab.loc[:, (crosstab != 0).any(axis=0)]
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return {"error": "Insufficient categories for chi-square"}

    chi2, p, dof, _ = sp_stats.chi2_contingency(crosstab.values)
    return {"chi2": chi2, "p_value": p, "dof": dof, "crosstab": crosstab}


# ---------------------------------------------------------------------------
# Analysis 3: Segment length as detection difficulty proxy
# ---------------------------------------------------------------------------

def compute_segment_lengths(df: pd.DataFrame) -> dict:
    """Compare error segment lengths across CCL categories.

    Longer segments = error signal more diluted = harder to detect.
    Returns descriptive stats + Mann-Whitney U (FACTUAL vs INTERPRETIVE).
    """
    df = df.copy()
    df["seg_len"] = df["segment"].apply(lambda x: len(str(x).split()))

    errors = df[df["is_error"] == True].dropna(subset=["ccl_category"])

    desc = errors.groupby("ccl_category")["seg_len"].describe()

    fact_lens = errors[errors["ccl_category"] == "FACTUAL"]["seg_len"].values
    interp_lens = errors[errors["ccl_category"] == "INTERPRETIVE"]["seg_len"].values

    if len(fact_lens) > 0 and len(interp_lens) > 0:
        u_stat, p_value = sp_stats.mannwhitneyu(
            fact_lens, interp_lens, alternative="two-sided"
        )
    else:
        u_stat, p_value = None, None

    return {
        "descriptive": desc,
        "factual_mean": float(fact_lens.mean()) if len(fact_lens) > 0 else None,
        "interpretive_mean": float(interp_lens.mean()) if len(interp_lens) > 0 else None,
        "mannwhitney_u": u_stat,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_experiment_a_felm(force_download: bool = False) -> dict:
    """Run the full FELM analysis (Experiment A')."""
    print("=" * 60)
    print("EXPERIMENT A' (FELM): Claim-Type Asymmetry in ChatGPT")
    print("=" * 60)

    # Load
    print("\n1. Loading FELM data...")
    df = load_felm(force_download=force_download)
    df = add_ccl_category(df)
    n_total = len(df)
    n_errors = int(df["is_error"].sum())
    print(f"   {n_total} segments from {df['record_index'].nunique()} prompts")
    print(f"   {n_errors} errors ({n_errors/n_total:.1%} overall error rate)")
    print(f"   Domain distribution: {df['domain'].value_counts().to_dict()}")

    # Analysis 1: error rates
    print("\n2. Error rates by domain and CCL category...")
    by_domain, by_ccl = compute_error_rates(df)
    print("\n   By domain:")
    print(by_domain[["domain", "ccl_category", "n_segments", "n_errors", "error_rate"]]
          .to_string(index=False))
    print("\n   By CCL category:")
    print(by_ccl.to_string(index=False))

    chi2_rates = chi_square_error_rates(by_ccl)
    print(f"\n   Chi-square on segment counts (across categories): "
          f"chi2={chi2_rates['chi2']:.2f}, df={chi2_rates['dof']}, "
          f"p={chi2_rates['p_value']:.4f}  "
          "[descriptive only — segments not independent]")

    fisher = fisher_factual_vs_interpretive(by_ccl)
    print(f"   Fisher's exact on segment counts (FACTUAL vs INTERPRETIVE): "
          f"OR={fisher['odds_ratio']:.2f}, p={fisher['p_value']:.4f}  "
          "[descriptive only]")

    # Prompt-level inferential test (the appropriate unit: CCL category is
    # constant within prompt, so segment-level tests inflate significance).
    print("\n   Prompt-level analysis (one observation per prompt):")
    prompt_stats = compute_prompt_level_error_rates(df)
    print(f"     N prompts = {len(prompt_stats)} "
          f"(mean {prompt_stats['n_segments'].mean():.1f} segments/prompt)")

    kruskal = kruskal_across_ccl_prompts(prompt_stats)
    if "error" not in kruskal:
        print(f"     Kruskal-Wallis (3 categories): "
              f"H={kruskal['H']:.2f}, p={kruskal['p_value']:.4f}")

    mw = mannwhitney_factual_vs_interpretive_prompts(
        prompt_stats, alternative="greater"
    )
    if "error" not in mw:
        print(f"     Mann-Whitney U (FACTUAL > INTERPRETIVE, one-sided): "
              f"U={mw['U']:.0f}, p={mw['p_value']:.4f}")
        print(f"     Median error rate per prompt: "
              f"FACTUAL = {mw['median_factual']:.3f}  "
              f"INTERPRETIVE = {mw['median_interpretive']:.3f}")
        print(f"     Mean error rate per prompt: "
              f"FACTUAL = {mw['mean_factual']:.3f}  "
              f"INTERPRETIVE = {mw['mean_interpretive']:.3f}")
        print(f"     Rank-biserial r = {mw['rank_biserial']:.3f}  "
              f"(N_factual = {mw['n_factual_prompts']}, "
              f"N_interpretive = {mw['n_interpretive_prompts']})")

    # Analysis 2: error types
    print("\n3. Error type distribution by CCL category...")
    type_dist = compute_error_type_distribution(df)
    if len(type_dist) > 0:
        print(type_dist.to_string())
    else:
        print("   No typed errors found.")

    chi2_types = chi_square_error_types(df)
    if "error" not in chi2_types:
        print(f"\n   Chi-square (error types differ by category): "
              f"chi2={chi2_types['chi2']:.2f}, df={chi2_types['dof']}, "
              f"p={chi2_types['p_value']:.4f}")
    else:
        print(f"   {chi2_types['error']}")

    # Analysis 3: segment lengths
    print("\n4. Segment length (detection difficulty proxy)...")
    seg_lengths = compute_segment_lengths(df)
    print(seg_lengths["descriptive"].to_string())
    if seg_lengths["p_value"] is not None:
        print(f"\n   Factual error segments: {seg_lengths['factual_mean']:.1f} words")
        print(f"   Interpretive error segments: {seg_lengths['interpretive_mean']:.1f} words")
        print(f"   Mann-Whitney U={seg_lengths['mannwhitney_u']:.0f}, "
              f"p={seg_lengths['p_value']:.4f}")

    # Summary
    print("\n5. CCL interpretation:")
    fact_row = by_ccl[by_ccl["ccl_category"] == "FACTUAL"]
    interp_row = by_ccl[by_ccl["ccl_category"] == "INTERPRETIVE"]
    if len(fact_row) > 0 and len(interp_row) > 0:
        fact_rate = fact_row.iloc[0]["error_rate"]
        interp_rate = interp_row.iloc[0]["error_rate"]
        print(f"   FACTUAL error rate:      {fact_rate:.1%}")
        print(f"   INTERPRETIVE error rate: {interp_rate:.1%}")
        if fact_rate > interp_rate:
            print("   ChatGPT produces more factual errors than interpretive errors.")
            print("   Combined with FRANK: factual errors are both more frequent AND")
            print("   more detectable — interpretive errors are rarer but harder to catch.")
        else:
            print("   ChatGPT produces more interpretive errors than factual errors.")
            print("   Combined with FRANK: interpretive errors slip past both AI and human.")

    return {
        "df": df,
        "by_domain": by_domain,
        "by_ccl": by_ccl,
        "chi2_rates": chi2_rates,
        "fisher": fisher,
        "prompt_stats": prompt_stats,
        "kruskal_prompts": kruskal,
        "mannwhitney_prompts": mw,
        "error_type_dist": type_dist,
        "chi2_types": chi2_types,
        "segment_lengths": seg_lengths,
    }
