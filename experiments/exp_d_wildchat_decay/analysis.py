"""Experiment D: Longitudinal Engagement Decay (WildChat).

Extends Experiment C from 4 sessions (days) to months-long timescales.
Additional angles: turn complexity decay, topic drift, evaluative behaviour decay.

Data: 10K-conversation sample from WildChat-1M (repeat users with ≥5 conversations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter

from experiments.shared.data_acquisition import load_wildchat_sample
from experiments.shared.stats_utils import (
    per_subject_slopes,
    slope_ttest,
    type_token_ratio,
)
from experiments.shared.turn_classifier import TurnType, classify_turn


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_wildchat(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to the raw WildChat sample."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["content_str"] = df["content"].fillna("").astype(str)
    df["word_count"] = df["content_str"].apply(lambda t: len(t.split()))
    return df


def build_user_conversation_index(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-user conversation index ordered by time.

    Returns one row per conversation with: hashed_ip, conversation_hash,
    conv_order (1-based rank within user), timestamp_first, n_user_turns.
    """
    user_turns = df[df["role"] == "user"]

    # Per-conversation aggregation
    conv_agg = user_turns.groupby(["hashed_ip", "conversation_hash"]).agg(
        timestamp_first=("timestamp", "min"),
        n_user_turns=("content_str", "count"),
    ).reset_index()

    # Rank conversations within each user by time
    conv_agg = conv_agg.sort_values(["hashed_ip", "timestamp_first"])
    conv_agg["conv_order"] = conv_agg.groupby("hashed_ip").cumcount() + 1

    # Compute weeks since user's first conversation
    first_ts = conv_agg.groupby("hashed_ip")["timestamp_first"].transform("min")
    conv_agg["days_since_first"] = (
        conv_agg["timestamp_first"] - first_ts
    ).dt.total_seconds() / 86400
    conv_agg["weeks_since_first"] = conv_agg["days_since_first"] / 7

    return conv_agg


# ---------------------------------------------------------------------------
# Per-conversation engagement metrics
# ---------------------------------------------------------------------------

def compute_conversation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engagement metrics per conversation.

    Returns one row per conversation with: hashed_ip, conversation_hash,
    conv_order, weeks_since_first, plus metrics.
    """
    user_df = df[df["role"] == "user"].copy()
    user_df["turn_type"] = user_df["content_str"].apply(
        lambda t: classify_turn(t).value
    )
    user_df["ttr"] = user_df["content_str"].apply(type_token_ratio)

    conv_metrics = user_df.groupby(["hashed_ip", "conversation_hash"]).agg(
        n_user_turns=("content_str", "count"),
        mean_words_per_turn=("word_count", "mean"),
        evaluative_rate=("turn_type", lambda x: (x == TurnType.EVALUATIVE.value).mean()),
        active_rate=("turn_type", lambda x: (x == TurnType.ACTIVE.value).mean()),
        passive_rate=("turn_type", lambda x: (x == TurnType.PASSIVE.value).mean()),
        mean_ttr=("ttr", "mean"),
    ).reset_index()

    # Merge in temporal ordering
    conv_index = build_user_conversation_index(df)
    conv_metrics = conv_metrics.merge(
        conv_index[["hashed_ip", "conversation_hash", "conv_order",
                     "days_since_first", "weeks_since_first"]],
        on=["hashed_ip", "conversation_hash"],
        how="left",
    )
    return conv_metrics


# ---------------------------------------------------------------------------
# Longitudinal decay analysis (Exp C extension)
# ---------------------------------------------------------------------------

def compute_decay_slopes(
    conv_metrics: pd.DataFrame,
    metrics: list[str] | None = None,
    time_col: str = "conv_order",
) -> pd.DataFrame:
    """Per-user slopes over time for each metric.

    Same methodology as Experiment C, but over longer timescales.
    """
    if metrics is None:
        metrics = [
            "passive_rate", "active_rate", "evaluative_rate",
            "mean_words_per_turn", "mean_ttr",
        ]

    rows = []
    for metric in metrics:
        slopes = per_subject_slopes(
            conv_metrics,
            subject_col="hashed_ip",
            time_col=time_col,
            metric_col=metric,
        )
        tt = slope_ttest(slopes)
        rows.append({
            "metric": metric,
            "time_variable": time_col,
            "mean_slope": tt["mean_slope"],
            "pct_declining": tt["pct_declining"],
            "t_stat": tt["t_stat"],
            "p_value": tt["p_value"],
            "n_users": tt["n"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Topic drift analysis
# ---------------------------------------------------------------------------

def compute_topic_diversity(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Measure topic diversity per conversation using keyword concentration.

    Uses a simple approach: track how concentrated the vocabulary is
    in later conversations vs earlier ones (TF-based, no heavy models).
    """
    user_df = df[df["role"] == "user"].copy()

    # Build global vocabulary (top-N most common words, excluding stopwords)
    all_words = " ".join(user_df["content_str"]).lower().split()
    # Minimal stopword list
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "it", "this", "that", "i", "you",
        "he", "she", "we", "they", "my", "your", "his", "her", "our", "their",
        "me", "him", "us", "them", "and", "or", "but", "not", "no", "if",
        "so", "just", "about", "what", "how", "when", "where", "who", "which",
    }
    filtered = [w for w in all_words if w not in stopwords and len(w) > 2]
    vocab = [w for w, _ in Counter(filtered).most_common(top_n)]
    vocab_set = set(vocab)

    # Per conversation: fraction of user words that are in top-N vocab
    rows = []
    for (hashed_ip, conv_hash), grp in user_df.groupby(["hashed_ip", "conversation_hash"]):
        words = " ".join(grp["content_str"]).lower().split()
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        if not content_words:
            concentration = 0.0
            unique_topics = 0
        else:
            in_vocab = sum(1 for w in content_words if w in vocab_set)
            concentration = in_vocab / len(content_words)
            unique_topics = len(set(content_words) - vocab_set)

        rows.append({
            "hashed_ip": hashed_ip,
            "conversation_hash": conv_hash,
            "vocab_concentration": concentration,
            "unique_topic_words": unique_topics,
            "n_content_words": len(content_words),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_experiment_d(force_download: bool = False) -> dict:
    """Run the full Experiment D analysis."""
    print("=" * 60)
    print("EXPERIMENT D: Longitudinal Engagement Decay (WildChat)")
    print("=" * 60)

    # Load data
    print("\n1. Loading WildChat sample...")
    raw_df = load_wildchat_sample(force_download=force_download)
    df = prepare_wildchat(raw_df)
    n_users = df["hashed_ip"].nunique()
    n_convos = df["conversation_hash"].nunique()
    print(f"   {len(df)} turns, {n_convos} conversations, {n_users} users")

    # Build conversation metrics
    print("\n2. Computing per-conversation engagement metrics...")
    conv_metrics = compute_conversation_metrics(df)
    print(f"   {len(conv_metrics)} conversation records")

    # Decay slopes (by conversation order)
    print("\n3. Longitudinal decay analysis (by conversation order)...")
    decay_order = compute_decay_slopes(conv_metrics, time_col="conv_order")
    print("\n   By conversation order:")
    for _, row in decay_order.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"   {row['metric']:25s} | slope={row['mean_slope']:+.5f} | "
              f"{row['pct_declining']:.0%} declining | "
              f"t={row['t_stat']:.2f} | p={row['p_value']:.4f}{sig}")

    # Decay slopes (by weeks)
    print("\n   By weeks since first conversation:")
    decay_weeks = compute_decay_slopes(conv_metrics, time_col="weeks_since_first")
    for _, row in decay_weeks.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"   {row['metric']:25s} | slope={row['mean_slope']:+.5f} | "
              f"{row['pct_declining']:.0%} declining | "
              f"t={row['t_stat']:.2f} | p={row['p_value']:.4f}{sig}")

    # Topic drift
    print("\n4. Topic drift analysis...")
    topic_df = compute_topic_diversity(df)
    topic_merged = topic_df.merge(
        conv_metrics[["hashed_ip", "conversation_hash", "conv_order"]],
        on=["hashed_ip", "conversation_hash"],
        how="left",
    )
    topic_slopes = per_subject_slopes(
        topic_merged, subject_col="hashed_ip",
        time_col="conv_order", metric_col="vocab_concentration",
    )
    topic_tt = slope_ttest(topic_slopes)
    print(f"   Vocab concentration slope: {topic_tt['mean_slope']:+.5f} "
          f"(t={topic_tt['t_stat']:.2f}, p={topic_tt['p_value']:.4f})")
    print(f"   {topic_tt['pct_declining']:.0%} of users show decreasing concentration")

    # Comparison with Bastani (Exp C)
    print("\n5. Cross-study comparison:")
    print("   Bastani (4 sessions, days): passive slope ~ +0.027")
    bastani_passive = decay_order[decay_order["metric"] == "passive_rate"]
    if len(bastani_passive) > 0:
        wc_slope = bastani_passive.iloc[0]["mean_slope"]
        print(f"   WildChat (months): passive slope = {wc_slope:+.5f}")

    return {
        "df": df,
        "conv_metrics": conv_metrics,
        "decay_by_order": decay_order,
        "decay_by_weeks": decay_weeks,
        "topic_diversity": topic_df,
        "topic_slope_test": topic_tt,
    }
