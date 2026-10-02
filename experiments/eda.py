"""Exploratory data analysis: descriptive statistics of every dataset used.

Writes ``output/dataset_stats.json`` with the main counts of each dataset as
loaded by the experiments (same loaders, same filters), so that descriptive
figures quoted in the paper and supplement trace to a committed file.

Usage:
    python -m experiments.eda [--output PATH] [--force-download]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments.exp_a_felm.analysis import add_ccl_category
from experiments.exp_c_declining_effort.analysis import (
    build_session_metrics,
    filter_repeat_students,
)
from experiments.exp_d_wildchat_decay.analysis import prepare_wildchat
from experiments.exp_e_critique import data as exp_e_data
from experiments.shared.data_acquisition import (
    frank_to_sentence_df,
    load_bastani_conversations,
    load_bastani_outcomes,
    load_felm,
    load_frank,
    load_wildchat_sample,
)
from experiments.shared.turn_classifier import classify_turn

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _counts(s: pd.Series) -> dict:
    """Value counts as a plain dict with string keys, sorted by frequency."""
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}


def _describe(s: pd.Series) -> dict:
    """Min / median / mean / max of a numeric series."""
    return {
        "min": float(s.min()),
        "median": float(s.median()),
        "mean": round(float(s.mean()), 3),
        "max": float(s.max()),
    }


# ---------------------------------------------------------------------------
# Per-dataset statistics
# ---------------------------------------------------------------------------

def frank_stats(force_download: bool = False) -> dict:
    """FRANK (Experiment A): sentences, summaries, models, annotations."""
    df = frank_to_sentence_df(load_frank(force_download=force_download))
    ann_cols = ["ann_0_errors", "ann_1_errors", "ann_2_errors"]
    n_flagging = df[ann_cols].apply(
        lambda r: sum(bool([e for e in errs if e != "NoE"]) for errs in r), axis=1
    )
    return {
        "used_in": ["A"],
        "n_sentences": len(df),
        "n_summaries": int(df["hash"].astype(str).str.cat(df["model_name"], "_").nunique()),
        "n_articles": int(df["hash"].nunique()),
        "n_models": int(df["model_name"].nunique()),
        "sentences_per_model": _counts(df["model_name"]),
        "annotators_per_sentence": len(ann_cols),
        "sentences_by_n_annotators_flagging_any_error": _counts(n_flagging.sort_values()),
    }


def felm_stats(force_download: bool = False) -> dict:
    """FELM (Experiment A'): prompts, segments, error rates by domain."""
    df = add_ccl_category(load_felm(force_download=force_download))
    by_domain = df.groupby("domain").agg(
        n_prompts=("record_index", "nunique"),
        n_segments=("segment", "size"),
        n_error_segments=("is_error", "sum"),
    )
    by_domain["segment_error_rate"] = (
        by_domain["n_error_segments"] / by_domain["n_segments"]
    ).round(4)
    by_domain["ccl_category"] = df.groupby("domain")["ccl_category"].first()
    seg_per_prompt = df.groupby("record_index").size()
    return {
        "used_in": ["A'"],
        "n_prompts": int(df["record_index"].nunique()),
        "n_segments": len(df),
        "n_error_segments": int(df["is_error"].sum()),
        "segment_error_rate": round(float(df["is_error"].mean()), 4),
        "segments_per_prompt": _describe(seg_per_prompt),
        "by_domain": json.loads(by_domain.to_json(orient="index")),
    }


def bastani_stats(force_download: bool = False) -> dict:
    """Bastani et al. RCT (Experiments B, C): outcomes and conversation logs."""
    out = load_bastani_outcomes(force_download=force_download)
    conv = load_bastani_conversations(force_download=force_download)
    user = conv[conv["role"] == "user"]
    turn_types = user["message"].fillna("").astype(str).map(lambda t: classify_turn(t).value)

    per_arm = {}
    for arm, g in conv.groupby("treatment"):
        u = g[g["role"] == "user"]
        per_arm[str(arm)] = {
            "n_messages": len(g),
            "n_user_turns": len(u),
            "n_conversations": int(g["conversation_id"].nunique()),
            "n_students": int(g["username"].nunique()),
        }

    sessions = build_session_metrics(conv)
    repeat = filter_repeat_students(sessions, min_sessions=3)
    return {
        "used_in": ["B", "C"],
        "outcomes": {
            "n_student_session_rows": len(out),
            "n_students": int(out["Student ID"].nunique()),
            "n_classes": int(out["Class"].nunique()),
            "sessions": sorted(int(s) for s in out["Session"].unique()),
            "students_by_arm": {
                str(k): int(v)
                for k, v in out.groupby("Treatment arm")["Student ID"].nunique().items()
            },
        },
        "conversations": {
            "n_messages": len(conv),
            "messages_by_role": _counts(conv["role"]),
            "n_user_turns": len(user),
            "n_conversations": int(conv["conversation_id"].nunique()),
            "n_students": int(conv["username"].nunique()),
            "grades": sorted(int(g) for g in conv["grade"].dropna().unique()),
            "by_arm": per_arm,
            "user_turn_types": _counts(turn_types),
            "user_turn_types_pct": {
                k: round(v * 100, 1) for k, v in turn_types.value_counts(normalize=True).items()
            },
        },
        "experiment_c_repeat_students": {
            "min_sessions": 3,
            "n_students": int(repeat["username"].nunique()),
            "by_arm": {
                str(k): int(v) for k, v in repeat.groupby("treatment")["username"].nunique().items()
            },
        },
        "arm_labels": {"vanilla": "GPT Base", "aug / augmented": "GPT Tutor"},
    }


def wildchat_stats(force_download: bool = False) -> dict:
    """WildChat sample (Experiment D): conversations, users, time span."""
    df = prepare_wildchat(load_wildchat_sample(force_download=force_download))
    conv_per_user = df.groupby("hashed_ip")["conversation_hash"].nunique()
    ts = df["timestamp"].dropna()
    return {
        "used_in": ["D"],
        "n_messages": len(df),
        "messages_by_role": _counts(df["role"]),
        "n_user_turns": int((df["role"] == "user").sum()),
        "n_conversations": int(df["conversation_hash"].nunique()),
        "n_users": int(df["hashed_ip"].nunique()),
        "conversations_per_user": _describe(conv_per_user),
        "time_span": {
            "first": ts.min().isoformat(),
            "last": ts.max().isoformat(),
            "days": round((ts.max() - ts.min()).total_seconds() / 86400, 2),
        },
        "messages_by_model": _counts(df["model"]),
        "top_languages_by_message": dict(list(_counts(df["language"]).items())[:10]),
    }


def critique_stats(force_download: bool = False) -> dict:
    """CriticEval and ManualReviewComment (Experiment E)."""
    items = exp_e_data.load_criticeval_items(force_download=force_download)
    human = exp_e_data.load_criticeval_human_critiques(force_download=force_download)
    mrc = exp_e_data.load_manual_review_comment(force_download=force_download)
    sampled_path = OUTPUT_DIR / "exp_e" / "sampled_items.csv"
    sampled = pd.read_csv(sampled_path) if sampled_path.exists() else None
    return {
        "criticeval": {
            "used_in": ["E"],
            "n_candidate_items": len(items),
            "items_by_gold_quality": _counts(items["gold_quality"]),
            "items_by_domain": _counts(items["domain"]),
            "n_human_scored_critiques": len(human),
            "sampled_items": None if sampled is None else {
                "n": len(sampled),
                "by_gold_quality": _counts(sampled["gold_quality"]),
                "by_domain": _counts(sampled["domain"]),
            },
        },
        "manual_review_comment": {
            "used_in": ["E"],
            "n_comments": len(mrc),
            "by_label": _counts(mrc["quality_label"]),
        },
        "metacritique": {
            "used_in": ["E"],
            "note": "Used for the critique-quality scoring scheme only; no records are loaded.",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eda(force_download: bool = False) -> dict:
    """Compute descriptive statistics for all datasets."""
    stats = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frank": frank_stats(force_download),
        "felm": felm_stats(force_download),
        "bastani": bastani_stats(force_download),
        "wildchat": wildchat_stats(force_download),
    }
    stats.update(critique_stats(force_download))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Descriptive statistics of all datasets")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "dataset_stats.json")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    stats = run_eda(force_download=args.force_download)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Dataset statistics written to {args.output}")


if __name__ == "__main__":
    main()
