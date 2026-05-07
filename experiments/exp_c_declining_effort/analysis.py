"""Experiment C: Declining Effort Across Sessions.

CCL prediction: Without process-level reflection (Stage 3), critical
engagement declines across sessions.

Data: Bastani et al. — 451 students with ≥3 of 4 sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.shared.data_acquisition import load_bastani_conversations
from experiments.shared.stats_utils import per_subject_slopes, slope_ttest
from experiments.shared.turn_classifier import TurnType, classify_turn


# ---------------------------------------------------------------------------
# Session-level aggregation
# ---------------------------------------------------------------------------

def build_session_metrics(conv_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-student, per-session engagement metrics.

    Returns DataFrame with columns:
      username, session_id, treatment, n_turns, mean_words_per_turn,
      evaluative_rate, active_rate, passive_rate
    """
    df = conv_df[conv_df["role"] == "user"].copy()
    df["message_str"] = df["message"].fillna("").astype(str)

    # Classify turns
    df["turn_type"] = df["message_str"].apply(lambda t: classify_turn(t).value)
    df["word_count"] = df["message_str"].apply(lambda t: len(t.split()))

    # Group by student + session
    group_cols = ["username", "session_id", "treatment"]
    available = [c for c in group_cols if c in df.columns]

    rows = []
    for keys, grp in df.groupby(available):
        meta = dict(zip(available, keys if isinstance(keys, tuple) else (keys,)))
        n = len(grp)
        meta["n_turns"] = n
        meta["mean_words_per_turn"] = grp["word_count"].mean()
        meta["evaluative_rate"] = (grp["turn_type"] == TurnType.EVALUATIVE.value).mean()
        meta["active_rate"] = (grp["turn_type"] == TurnType.ACTIVE.value).mean()
        meta["passive_rate"] = (grp["turn_type"] == TurnType.PASSIVE.value).mean()
        rows.append(meta)

    return pd.DataFrame(rows)


def filter_repeat_students(
    session_df: pd.DataFrame,
    min_sessions: int = 3,
) -> pd.DataFrame:
    """Keep only students with at least *min_sessions* sessions."""
    counts = session_df.groupby("username")["session_id"].nunique()
    keep = counts[counts >= min_sessions].index
    filtered = session_df[session_df["username"].isin(keep)].copy()
    return filtered


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def compute_trajectory_slopes(
    session_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-student slopes and t-tests for each metric × treatment.

    Returns a summary DataFrame with one row per (metric, treatment).
    """
    if metrics is None:
        metrics = [
            "passive_rate", "active_rate", "evaluative_rate",
            "mean_words_per_turn",
        ]

    rows = []
    for treatment, grp in session_df.groupby("treatment"):
        for metric in metrics:
            slopes = per_subject_slopes(
                grp, subject_col="username", time_col="session_id",
                metric_col=metric,
            )
            tt = slope_ttest(slopes)
            rows.append({
                "treatment": treatment,
                "metric": metric,
                "mean_slope": tt["mean_slope"],
                "pct_declining": tt["pct_declining"],
                "t_stat": tt["t_stat"],
                "p_value": tt["p_value"],
                "n_students": tt["n"],
            })

    return pd.DataFrame(rows)


def session_level_summary(session_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean metrics per treatment × session (for display table)."""
    group_cols = ["treatment", "session_id"]
    metric_cols = [
        "evaluative_rate", "passive_rate", "active_rate",
        "n_turns", "mean_words_per_turn",
    ]
    available = [c for c in metric_cols if c in session_df.columns]
    return session_df.groupby(group_cols)[available].mean().reset_index()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_experiment_c(force_download: bool = False) -> dict:
    """Run the full Experiment C analysis."""
    print("=" * 60)
    print("EXPERIMENT C: Declining Effort Across Sessions")
    print("=" * 60)

    # Load conversations
    print("\n1. Loading Bastani conversation data...")
    conv_df = load_bastani_conversations(force_download=force_download)
    print(f"   {len(conv_df)} turns loaded")

    # Build session metrics
    print("\n2. Building per-student, per-session metrics...")
    session_df = build_session_metrics(conv_df)
    print(f"   {len(session_df)} student-session rows")

    # Filter repeat students
    print("\n3. Filtering students with >=3 sessions...")
    filtered = filter_repeat_students(session_df, min_sessions=3)
    n_students = filtered["username"].nunique()
    print(f"   {n_students} students retained")

    # Session-level summary table
    print("\n4. Session-level summary:")
    summary = session_level_summary(filtered)
    print(summary.to_string(index=False))

    # Trajectory slopes
    print("\n5. Within-student trajectory slopes:")
    slopes_df = compute_trajectory_slopes(filtered)
    for _, row in slopes_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"   {row['treatment']:10s} | {row['metric']:25s} | "
              f"slope={row['mean_slope']:+.4f} | "
              f"{row['pct_declining']:.0%} declining | "
              f"t={row['t_stat']:.2f} | p={row['p_value']:.4f}{sig}")

    return {
        "session_df": session_df,
        "filtered_df": filtered,
        "session_summary": summary,
        "slopes": slopes_df,
    }
