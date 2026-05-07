"""Experiment B: No-Self-Merge — Active vs Passive AI Use.

CCL prediction: AI output that undergoes independent human evaluation
(no-self-merge) produces better learning outcomes than passively accepted output.

Data: Bastani et al. (2025, PNAS). ~1,000 Turkish high-school students.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from experiments.shared.data_acquisition import (
    load_bastani_conversations,
    load_bastani_outcomes,
)
from experiments.shared.stats_utils import cohens_d, ols_cluster_robust
from experiments.shared.turn_classifier import (
    TurnType,
    classify_turn,
    conversation_metrics,
)


# ---------------------------------------------------------------------------
# Treatment arm mapping
# ---------------------------------------------------------------------------

# Bastani uses "vanilla" = GPT Base (no-self-merge violated)
#                "augmented" = GPT Tutor (no-self-merge enforced)
TREATMENT_MAP = {
    "vanilla": "GPT Base",
    "augmented": "GPT Tutor",
}


# ---------------------------------------------------------------------------
# Conversation analysis
# ---------------------------------------------------------------------------

def classify_all_turns(conv_df: pd.DataFrame) -> pd.DataFrame:
    """Add turn classification to conversation dataframe.

    Expects columns: role, message (or content), treatment, username,
    conversation_id (or problem_id + session_id).
    """
    df = conv_df.copy()

    # Normalize column names
    if "content" in df.columns and "message" not in df.columns:
        df["message"] = df["content"]

    # Only classify user turns
    user_mask = df["role"] == "user"
    df["turn_type"] = None
    df.loc[user_mask, "turn_type"] = df.loc[user_mask, "message"].apply(
        lambda t: classify_turn(str(t)).value if pd.notna(t) else TurnType.PASSIVE.value
    )
    return df


def compute_conversation_level_metrics(
    conv_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate metrics per conversation.

    Groups by (username, problem_id) or conversation_id.
    """
    df = conv_df[conv_df["role"] == "user"].copy()

    # Determine grouping columns
    if "conversation_id" in df.columns:
        group_cols = ["conversation_id"]
    else:
        group_cols = ["username", "problem_id"]

    # Carry treatment and session info
    if "treatment" in df.columns:
        group_cols_with_meta = group_cols + ["treatment"]
    else:
        group_cols_with_meta = group_cols

    if "session_id" in df.columns:
        group_cols_with_meta = list(set(group_cols_with_meta + ["session_id"]))

    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        turns = grp["message"].dropna().tolist()
        metrics = conversation_metrics([str(t) for t in turns])

        meta = dict(zip(group_cols, keys))
        # Pick treatment and session from first row
        meta["treatment"] = grp["treatment"].iloc[0] if "treatment" in grp.columns else None
        meta["session_id"] = grp["session_id"].iloc[0] if "session_id" in grp.columns else None
        meta["username"] = grp["username"].iloc[0] if "username" in grp.columns else None
        meta.update(metrics)
        rows.append(meta)

    return pd.DataFrame(rows)


def compute_condition_comparison(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare GPT Base vs GPT Tutor on conversation metrics.

    Returns a summary table with means, SDs, Cohen's d, and p-values.
    """
    metric_cols = [
        "n_turns", "mean_words_per_turn",
        "evaluative_rate", "active_rate", "passive_rate",
    ]
    display_names = {
        "n_turns": "Turns per conversation",
        "mean_words_per_turn": "Mean words per turn",
        "evaluative_rate": "Evaluative turn rate",
        "active_rate": "Active turn rate",
        "passive_rate": "Passive turn rate",
    }

    base = metrics_df[metrics_df["treatment"] == "vanilla"]
    tutor = metrics_df[metrics_df["treatment"] == "aug"]

    rows = []
    for col in metric_cols:
        b = base[col].dropna().values
        t = tutor[col].dropna().values
        d = cohens_d(t, b)  # GPT Tutor - GPT Base
        _, p = sp_stats.mannwhitneyu(t, b, alternative="two-sided")
        rows.append({
            "Metric": display_names.get(col, col),
            "GPT Tutor (mean ± sd)": f"{t.mean():.2f} ± {t.std():.2f}",
            "GPT Base (mean ± sd)": f"{b.mean():.2f} ± {b.std():.2f}",
            "Cohen's d": round(d, 2),
            "p-value": p,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mixed-effects models (address within-student / within-class dependence
# flagged by EDM Reviewer 2)
# ---------------------------------------------------------------------------

def attach_class_id(
    metrics_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add a `class_id` column to per-conversation metrics.

    Each Bastani student belongs to exactly one Class. We left-join on
    Student ID == username to attach the randomisation unit.
    """
    student_to_class = (
        outcomes_df[["Student ID", "Class"]]
        .drop_duplicates("Student ID")
        .rename(columns={"Student ID": "username", "Class": "class_id"})
    )
    # Bastani Student IDs are integers, conv usernames are also integers
    student_to_class["username"] = student_to_class["username"].astype(str)
    out = metrics_df.copy()
    out["username"] = out["username"].astype(str)
    out = out.merge(student_to_class, on="username", how="left")
    return out


def conversation_level_lmm(
    metrics_df: pd.DataFrame,
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Mixed-effects LMM per metric: outcome ~ treatment + (1|student) + (1|class).

    Uses statsmodels.mixedlm with class as the top-level grouping factor and
    student as a variance component, which is equivalent to crossed/nested
    random intercepts in lme4. Conversations are nested in students nested
    in classes.
    """
    import statsmodels.formula.api as smf

    if metric_cols is None:
        metric_cols = [
            "n_turns", "mean_words_per_turn",
            "evaluative_rate", "active_rate", "passive_rate",
        ]

    df = metrics_df.copy()
    df = df[df["treatment"].isin(["vanilla", "aug"])].copy()
    if "class_id" not in df.columns:
        raise ValueError(
            "metrics_df must have class_id column "
            "(call attach_class_id first)."
        )
    df = df.dropna(subset=["username", "class_id"])
    # GPT Tutor (aug) is the reference: positive coefficient on is_vanilla
    # means GPT Base is higher on the metric.
    df["is_vanilla"] = (df["treatment"] == "vanilla").astype(int)

    rows = []
    for col in metric_cols:
        sub = df.dropna(subset=[col]).copy()
        if len(sub) < 30 or sub["class_id"].nunique() < 3:
            rows.append({"metric": col, "error": "insufficient data"})
            continue
        try:
            md = smf.mixedlm(
                f"{col} ~ is_vanilla",
                data=sub,
                groups=sub["class_id"],
                vc_formula={"student": "0 + C(username)"},
            )
            res = md.fit(method="lbfgs", reml=True, maxiter=200)
            beta = float(res.params.get("is_vanilla", float("nan")))
            se = float(res.bse.get("is_vanilla", float("nan")))
            p = float(res.pvalues.get("is_vanilla", float("nan")))
            ci = res.conf_int().loc["is_vanilla"].tolist()

            # Variance components: class-level + student-level + residual
            var_class = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] else 0.0
            var_student = float(res.vcomp[0]) if len(res.vcomp) else 0.0
            var_resid = float(res.scale)
            total_var = var_class + var_student + var_resid
            d_marginal = (
                beta / np.sqrt(total_var) if total_var > 0 else float("nan")
            )

            rows.append({
                "metric": col,
                "beta_vanilla_minus_aug": beta,
                "se": se,
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "p_value": p,
                "var_class": var_class,
                "var_student": var_student,
                "var_residual": var_resid,
                "d_total_var": d_marginal,
                "n_obs": int(res.nobs),
                "n_classes": int(sub["class_id"].nunique()),
                "n_students": int(sub["username"].nunique()),
                "converged": bool(res.converged) if hasattr(res, "converged")
                              else None,
            })
        except Exception as e:
            rows.append({"metric": col, "error": str(e)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outcome regression (ITT)
# ---------------------------------------------------------------------------

def run_itt_regression(
    outcomes_df: pd.DataFrame,
) -> dict:
    """Intent-to-treat regression: unassisted exam score ~ treatment.

    Uses Part3Tot (unassisted exam) as the outcome — the key CCL measure.
    Part2Tot is the AI-assisted practice score (not the learning outcome).
    Clusters at Class level with robust SEs.
    """
    df = outcomes_df.copy()

    if "Part3Tot" not in df.columns:
        return {"error": "Part3Tot (unassisted exam score) column not found"}

    df["exam_score"] = df["Part3Tot"]
    df = df.dropna(subset=["exam_score", "Class"])

    # Treatment indicators — outcomes df uses "augmented", conversations use "aug"
    if "GPTBase" not in df.columns:
        df["GPTBase"] = (df["Treatment arm"] == "vanilla").astype(int)
    if "GPTTutor" not in df.columns:
        df["GPTTutor"] = (df["Treatment arm"] == "augmented").astype(int)

    y = df["exam_score"].values
    X = df[["GPTBase", "GPTTutor"]].values
    clusters = df["Class"].values

    result = ols_cluster_robust(y, X, clusters)

    return {
        "gpt_base_beta": result["params"][1],
        "gpt_base_se": result["bse"][1],
        "gpt_base_p": result["pvalues"][1],
        "gpt_tutor_beta": result["params"][2],
        "gpt_tutor_se": result["bse"][2],
        "gpt_tutor_p": result["pvalues"][2],
        "summary": result["summary"],
    }


# ---------------------------------------------------------------------------
# Manual validation sample
# ---------------------------------------------------------------------------

def sample_evaluative_turns(
    conv_df: pd.DataFrame,
    n: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Random sample of EVALUATIVE turns for manual validation."""
    evals = conv_df[
        (conv_df["role"] == "user") & (conv_df["turn_type"] == TurnType.EVALUATIVE.value)
    ]
    return evals.sample(n=min(n, len(evals)), random_state=seed)[
        ["message", "treatment", "session_id"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_experiment_b(force_download: bool = False) -> dict:
    """Run the full Experiment B analysis."""
    print("=" * 60)
    print("EXPERIMENT B: No-Self-Merge — Active vs Passive AI Use")
    print("=" * 60)

    # Load data
    print("\n1. Loading Bastani conversation data...")
    conv_df = load_bastani_conversations(force_download=force_download)
    print(f"   {len(conv_df)} message turns loaded")

    print("   Loading outcome data...")
    outcomes_df = load_bastani_outcomes(force_download=force_download)
    print(f"   {len(outcomes_df)} student-session rows loaded")

    # Classify turns
    print("\n2. Classifying user turns...")
    conv_df = classify_all_turns(conv_df)
    user_turns = conv_df[conv_df["role"] == "user"]
    n_user = len(user_turns)
    print(f"   {n_user} user turns classified")

    # Turn type distribution
    type_counts = user_turns["turn_type"].value_counts()
    for tt in TurnType:
        cnt = type_counts.get(tt.value, 0)
        print(f"   {tt.value}: {cnt} ({cnt/n_user:.1%})")

    # Conversation-level metrics
    print("\n3. Computing conversation-level metrics...")
    metrics_df = compute_conversation_level_metrics(conv_df)
    print(f"   {len(metrics_df)} conversations analysed")

    # Condition comparison (descriptive — treats conversations as independent)
    print("\n4. Comparing conditions (GPT Base vs GPT Tutor)...")
    comparison = compute_condition_comparison(metrics_df)
    print("\n" + comparison.to_string(index=False))
    print("\n   Note: the table above treats conversations as independent.")
    print("   The mixed-effects model below is the inferential test.")

    # Mixed-effects LMM: addresses within-student / within-class dependence
    print("\n5. Mixed-effects LMM per metric "
          "(treatment + (1|student) + (1|class))...")
    metrics_with_class = attach_class_id(metrics_df, outcomes_df)
    n_missing_class = metrics_with_class["class_id"].isna().sum()
    if n_missing_class:
        print(f"   {n_missing_class} conversations dropped — no class match")
    lmm_table = conversation_level_lmm(metrics_with_class)
    for _, row in lmm_table.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            print(f"   {row['metric']:25s} | {row['error']}")
            continue
        sig = ("***" if row["p_value"] < 0.001
               else "**" if row["p_value"] < 0.01
               else "*" if row["p_value"] < 0.05
               else "")
        print(f"   {row['metric']:25s} | "
              f"beta(vanilla-aug)={row['beta_vanilla_minus_aug']:+.3f}, "
              f"SE={row['se']:.3f}, p={row['p_value']:.4g}{sig}, "
              f"d_total={row['d_total_var']:+.2f}")
        print(f"     variance: class={row['var_class']:.4f}, "
              f"student={row['var_student']:.4f}, "
              f"residual={row['var_residual']:.4f}, "
              f"N={row['n_obs']} convs / {row['n_students']} students "
              f"/ {row['n_classes']} classes")

    # ITT regression
    print("\n6. Running ITT regression (exam score ~ treatment)...")
    regression = run_itt_regression(outcomes_df)
    if "error" not in regression:
        print(f"   GPT Base b = {regression['gpt_base_beta']:.3f} "
              f"(SE = {regression['gpt_base_se']:.3f}, "
              f"p = {regression['gpt_base_p']:.4f})")
        print(f"   GPT Tutor b = {regression['gpt_tutor_beta']:.3f} "
              f"(SE = {regression['gpt_tutor_se']:.3f}, "
              f"p = {regression['gpt_tutor_p']:.4f})")
    else:
        print(f"   Regression skipped: {regression['error']}")

    # Validation sample
    print("\n7. Manual validation sample (EVALUATIVE turns):")
    sample = sample_evaluative_turns(conv_df)
    if len(sample) > 0:
        for _, row in sample.iterrows():
            print(f"   [{row['treatment']}] \"{row['message']}\"")
    else:
        print("   No evaluative turns found in sample")

    return {
        "conv_df": conv_df,
        "outcomes_df": outcomes_df,
        "metrics_df": metrics_df,
        "metrics_with_class": metrics_with_class,
        "comparison": comparison,
        "lmm_table": lmm_table,
        "regression": regression,
        "validation_sample": sample,
    }
