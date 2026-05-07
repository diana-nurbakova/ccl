"""Statistical utility functions for CCL validation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Fleiss' kappa
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings: np.ndarray) -> float:
    """Compute Fleiss' kappa for inter-rater agreement.

    Parameters
    ----------
    ratings : np.ndarray, shape (n_subjects, n_categories)
        Each row is one subject.  Each cell is the number of raters
        who assigned that subject to that category.
        All rows must sum to the same value (number of raters).

    Returns
    -------
    float
        Fleiss' kappa coefficient.
    """
    n_subjects, n_categories = ratings.shape
    n_raters = ratings[0].sum()

    # Proportion of all assignments to each category
    p_j = ratings.sum(axis=0) / (n_subjects * n_raters)

    # Per-subject agreement
    P_i = (np.sum(ratings ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()

    # Expected agreement by chance
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (standardised mean difference).

    Uses pooled standard deviation.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / pooled_std


# ---------------------------------------------------------------------------
# Fisher's exact test (2×2)
# ---------------------------------------------------------------------------

def fisher_exact_2x2(
    table: np.ndarray | list[list[int]],
) -> dict[str, float]:
    """Run Fisher's exact test on a 2×2 contingency table.

    Returns dict with 'odds_ratio' and 'p_value'.
    """
    table = np.asarray(table)
    odds_ratio, p_value = stats.fisher_exact(table)
    return {"odds_ratio": odds_ratio, "p_value": p_value}


# ---------------------------------------------------------------------------
# Per-student trajectory slopes
# ---------------------------------------------------------------------------

def per_subject_slopes(
    df: pd.DataFrame,
    subject_col: str,
    time_col: str,
    metric_col: str,
) -> pd.Series:
    """Fit a simple OLS slope per subject over time.

    Parameters
    ----------
    df : DataFrame
        Long-format data with one row per subject-timepoint.
    subject_col : str
        Column identifying subjects.
    time_col : str
        Column for the time variable (e.g. session number).
    metric_col : str
        Column for the dependent variable.

    Returns
    -------
    pd.Series
        Index = subject IDs, values = OLS slopes.
    """
    slopes = {}
    for subj, grp in df.groupby(subject_col):
        x = grp[time_col].values.astype(float)
        y = grp[metric_col].values.astype(float)
        if len(x) < 2:
            continue
        # Simple OLS slope: cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        var_x = ((x - x_mean) ** 2).sum()
        if var_x == 0:
            slopes[subj] = 0.0
        else:
            slopes[subj] = ((x - x_mean) * (y - y_mean)).sum() / var_x
    return pd.Series(slopes, name=f"{metric_col}_slope")


def slope_ttest(slopes: pd.Series) -> dict[str, float]:
    """One-sample t-test: H0: mean slope = 0.

    Returns dict with 't_stat', 'p_value', 'mean_slope', 'pct_declining'.
    """
    arr = slopes.dropna().values
    t_stat, p_value = stats.ttest_1samp(arr, 0.0)
    return {
        "mean_slope": float(arr.mean()),
        "pct_declining": float((arr < 0).mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n": len(arr),
    }


# ---------------------------------------------------------------------------
# Cluster-robust OLS regression
# ---------------------------------------------------------------------------

def ols_cluster_robust(
    y: np.ndarray,
    X: np.ndarray,
    cluster_ids: np.ndarray,
    add_constant: bool = True,
) -> dict:
    """OLS regression with cluster-robust standard errors.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, k)
    cluster_ids : array-like, shape (n,)
    add_constant : bool
        Whether to add a constant column to X.

    Returns
    -------
    dict with 'params', 'bse' (robust SEs), 'pvalues', 'summary'.
    """
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if add_constant:
        X = sm.add_constant(X)

    model = sm.OLS(y, X)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": np.asarray(cluster_ids)},
    )
    return {
        "params": result.params,
        "bse": result.bse,
        "pvalues": result.pvalues,
        "conf_int": result.conf_int(),
        "summary": str(result.summary()),
    }


# ---------------------------------------------------------------------------
# Lexical diversity (type-token ratio)
# ---------------------------------------------------------------------------

def type_token_ratio(text: str) -> float:
    """Compute type-token ratio as a simple lexical diversity measure."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
