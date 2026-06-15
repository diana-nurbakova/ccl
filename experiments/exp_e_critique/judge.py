"""Judge harness for Experiment E (resumable).

A single strong judge OUTSIDE the pool scores every critique for validity, and
scores each authored response for soundness (needed to anchor overcorrection on
our own responses). The judge is also validated against human labels:
* CriticEval meta_feedback human critique scores -> Cohen's kappa;
* ManualReviewComment valid/noisy gold -> Cohen's kappa.

Bias mitigations (spec section 7): the judge never sees author/critic identity;
critique length is recorded for the verbosity control; if judge-human kappa is
below the threshold the LLM-judge headline is dropped downstream.

All judging is pointwise (one critique scored against its response + question),
so no order-swap doubling is required here. Order-swap is only needed for the
optional pairwise self-vs-cross comparison, which we do not run as primary.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from experiments.shared.jsonl_store import JsonlStore
from experiments.shared.llm_client import InteractionLogger, make_client

from . import config, prompts


def build_judge(logger: InteractionLogger, spec=None):
    spec = spec or config.JUDGE
    return make_client(
        spec.provider, spec.model,
        temperature=config.JUDGE_TEMPERATURE,
        max_tokens=config.JUDGE_MAX_TOKENS,
        reasoning_effort=spec.reasoning_effort,
        logger=logger,
    )


def _run(tasks, fn, max_workers, desc):
    total = len(tasks)
    if total == 0:
        print(f"  [{desc}] nothing to do (all cached)")
        return
    print(f"  [{desc}] {total} judge calls ...")
    done = errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fn, t) for t in tasks]
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"    ! error: {type(e).__name__}: {str(e)[:160]}")
            if done % 25 == 0 or done == total:
                print(f"    [{desc}] {done}/{total} (errors={errors})")


# ---------------------------------------------------------------------------
# Judge every critique for validity
# ---------------------------------------------------------------------------


def judge_critiques(crit_store: JsonlStore, judge_client, judge_store: JsonlStore,
                    max_workers=None):
    max_workers = max_workers or config.MAX_WORKERS
    crits = [c for c in crit_store.read_all() if c.get("critique", "").strip()]
    tasks = []
    for c in crits:
        key = f"judge::{c['key']}"
        if judge_store.has(key):
            continue
        tasks.append((c, key))

    def do(task):
        c, key = task
        text = judge_client.complete(
            prompts.judge_messages(c["question"], c["response"], c["critique"]),
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
            seed=config.SEED,
            task_key=key,
            task_kind="judge_validity",
        )
        v = prompts.extract_json(text) or {}
        judge_store.append(key, {
            "crit_key": c["key"],
            "domain_kind": c.get("domain_kind"),
            "critic": c["critic"],
            "author": c["author"],
            "condition": c["condition"],
            "item_id": c.get("item_id"),
            "msg_id": c.get("msg_id"),
            "domain": c.get("domain"),
            "gold_quality": c.get("gold_quality"),
            "gold_is_valid": c.get("gold_is_valid"),
            "critique_len": c.get("critique_len"),
            "critique_valid": _as_bool(v.get("critique_valid")),
            "validity_score": _as_int(v.get("validity_score")),
            "identifies_real_flaw": _as_bool(v.get("identifies_real_flaw")),
            "recommends_change": _as_bool(v.get("recommends_change")),
            "change_would_worsen": _as_bool(v.get("change_would_worsen")),
            "parse_ok": bool(v),
            "raw": text if not v else None,
        })

    _run(tasks, do, max_workers, "judge-validity")


# ---------------------------------------------------------------------------
# Judge response soundness (anchors overcorrection)
# ---------------------------------------------------------------------------


def judge_response_soundness(resp_store: JsonlStore, judge_client,
                             sound_store: JsonlStore, max_workers=None):
    max_workers = max_workers or config.MAX_WORKERS
    responses = [r for r in resp_store.read_all() if r.get("response", "").strip()]
    tasks = []
    for r in responses:
        key = f"sound::{r['author']}::{r['item_id']}"
        if sound_store.has(key):
            continue
        tasks.append((r, key))

    def do(task):
        r, key = task
        text = judge_client.complete(
            prompts.soundness_messages(r["question"], r["response"]),
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
            seed=config.SEED,
            task_key=key,
            task_kind="judge_soundness",
        )
        v = prompts.extract_json(text) or {}
        sound_store.append(key, {
            "author": r["author"],
            "item_id": r["item_id"],
            "domain": r.get("domain"),
            "gold_quality": r.get("gold_quality"),
            "sound": _as_bool(v.get("sound")),
            "quality_score": _as_int(v.get("quality_score")),
            "parse_ok": bool(v),
        })

    _run(tasks, do, max_workers, "judge-soundness")


# ---------------------------------------------------------------------------
# Judge-human validation (Cohen's kappa)
# ---------------------------------------------------------------------------


def validate_judge_criticeval(human_df, judge_client, store: JsonlStore,
                              human_valid_cut: float = 5.0, max_workers=None):
    """Judge the human-scored CriticEval critiques; return judge vs human."""
    max_workers = max_workers or config.MAX_WORKERS
    df = human_df[human_df["critique"].str.len() > 0].copy()
    tasks = []
    for _, row in df.iterrows():
        key = f"val_ce::{row['item_id']}"
        if store.has(key):
            continue
        tasks.append((row, key))

    def do(task):
        row, key = task
        text = judge_client.complete(
            prompts.judge_messages(row["question"], row["response"], row["critique"]),
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
            seed=config.SEED, task_key=key, task_kind="judge_validate_ce",
        )
        v = prompts.extract_json(text) or {}
        store.append(key, {
            "item_id": row["item_id"],
            "human_score": float(row["human_score"]),
            "human_valid": bool(row["human_score"] >= human_valid_cut),
            "judge_valid": _as_bool(v.get("critique_valid")),
            "judge_score": _as_int(v.get("validity_score")),
            "parse_ok": bool(v),
        })

    _run(tasks, do, max_workers, "validate-criticeval")
    return store


def validate_judge_mrc(mrc_df, judge_client, store: JsonlStore, max_workers=None):
    """Judge the gold human review comments valid/noisy; return judge vs gold."""
    max_workers = max_workers or config.MAX_WORKERS
    tasks = []
    for _, row in mrc_df.iterrows():
        mid = str(row["msg_id"])
        key = f"val_mrc::{mid}"
        if store.has(key):
            continue
        tasks.append((row, mid, key))

    def do(task):
        row, mid, key = task
        question = f"Review this code change:\n{str(row['patch'])[:6000]}"
        # The human review comment IS a critique of the code; judge its validity.
        text = judge_client.complete(
            prompts.judge_messages(question, str(row["patch"])[:6000], str(row["msg"])),
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
            seed=config.SEED, task_key=key, task_kind="judge_validate_mrc",
        )
        v = prompts.extract_json(text) or {}
        store.append(key, {
            "msg_id": mid,
            "gold_is_valid": bool(row["is_valid"]),
            "judge_valid": _as_bool(v.get("critique_valid")),
            "judge_score": _as_int(v.get("validity_score")),
            "parse_ok": bool(v),
        })

    _run(tasks, do, max_workers, "validate-mrc")
    return store


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def cohens_kappa(a, b) -> float:
    """Cohen's kappa for two binary label sequences."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    # Expected agreement
    pe = 0.0
    for cls in (0, 1):
        pa = float((a == cls).mean())
        pb = float((b == cls).mean())
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# small coercion helpers
# ---------------------------------------------------------------------------


def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "yes", "1")
    if isinstance(x, (int, float)):
        return bool(x)
    return None


def _as_int(x):
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return None
