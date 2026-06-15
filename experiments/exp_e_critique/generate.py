"""Path A2 generation harness for Experiment E (resumable).

Builds the symmetric author x critic matrix:
  1. each pool model AUTHORS a response to each sampled item;
  2. each pool model CRITIQUES every authored response (self = diagonal).

Every authored response and every critique is appended to a JsonlStore keyed by
a deterministic id, so an interrupted run resumes from exactly where it stopped
(no API call is ever repeated). All raw requests/responses are also written to
the interaction log by the LLM client.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.shared.jsonl_store import JsonlStore
from experiments.shared.llm_client import InteractionLogger, make_client

from . import config, prompts


# ---------------------------------------------------------------------------
# Client pool
# ---------------------------------------------------------------------------


def build_clients(logger: InteractionLogger, pool=None):
    """Construct one LLMClient per pool model, keyed by ModelSpec.key."""
    pool = pool or config.MODEL_POOL
    clients = {}
    for spec in pool:
        clients[spec.key] = make_client(
            spec.provider, spec.model,
            temperature=config.GEN_TEMPERATURE,
            max_tokens=config.CRITIQUE_MAX_TOKENS,
            reasoning_effort=("low" if spec.reasoning else None),
            logger=logger,
        )
    return clients


def _run_tasks(tasks, fn, max_workers, desc):
    """Run ``fn(task)`` over ``tasks`` with a thread pool; report progress."""
    done = 0
    total = len(tasks)
    if total == 0:
        print(f"  [{desc}] nothing to do (all cached)")
        return
    print(f"  [{desc}] {total} calls to make ...")
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fn, t) for t in tasks]
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - log & continue; resume later
                errors += 1
                print(f"    ! error: {type(e).__name__}: {str(e)[:160]}")
            if done % 25 == 0 or done == total:
                print(f"    [{desc}] {done}/{total} (errors={errors})")


# ---------------------------------------------------------------------------
# CriticEval (general domain)
# ---------------------------------------------------------------------------


def author_responses(items, clients, store: JsonlStore, max_workers=None):
    """Each pool model authors a response to each sampled item."""
    max_workers = max_workers or config.MAX_WORKERS
    tasks = []
    for spec in config.MODEL_POOL:
        for _, item in items.iterrows():
            key = f"resp::{spec.key}::{item['item_id']}"
            if store.has(key):
                continue
            tasks.append((spec, item, key))

    def do(task):
        spec, item, key = task
        client = clients[spec.key]
        text = client.complete(
            prompts.author_messages(item["question"]),
            temperature=config.GEN_TEMPERATURE,
            max_tokens=config.AUTHOR_MAX_TOKENS,
            seed=config.SEED,
            task_key=key,
            task_kind="author",
        )
        store.append(key, {
            "kind": "response",
            "author": spec.key,
            "item_id": item["item_id"],
            "domain": item["domain"],
            "gold_quality": item["gold_quality"],
            "question": item["question"],
            "response": text,
            "response_len": len(text.split()),
        })

    _run_tasks(tasks, do, max_workers, "author")


def critique_matrix(resp_store: JsonlStore, clients, crit_store: JsonlStore,
                    max_workers=None):
    """Every pool model critiques every authored response (fills the matrix)."""
    max_workers = max_workers or config.MAX_WORKERS
    responses = [r for r in resp_store.read_all() if r.get("response", "").strip()]

    tasks = []
    for spec in config.MODEL_POOL:
        for r in responses:
            author = r["author"]
            key = f"crit::{spec.key}::{author}::{r['item_id']}"
            if crit_store.has(key):
                continue
            tasks.append((spec, author, r, key))

    def do(task):
        spec, author, r, key = task
        client = clients[spec.key]
        text = client.complete(
            prompts.critic_messages(r["question"], r["response"]),
            temperature=config.GEN_TEMPERATURE,
            max_tokens=config.CRITIQUE_MAX_TOKENS,
            seed=config.SEED,
            task_key=key,
            task_kind="critique",
        )
        crit_store.append(key, {
            "kind": "critique",
            "domain_kind": "general",
            "critic": spec.key,
            "author": author,
            "condition": "self" if spec.key == author else "cross",
            "item_id": r["item_id"],
            "domain": r["domain"],
            "gold_quality": r["gold_quality"],
            "question": r["question"],
            "response": r["response"],
            "critique": text,
            "critique_len": len(text.split()),
            "self_verdict": prompts.parse_verdict(text),
        })

    _run_tasks(tasks, do, max_workers, "critique")


# ---------------------------------------------------------------------------
# ManualReviewComment (code-review domain, RQ3)
# ---------------------------------------------------------------------------


def author_code_comments(mrc, clients, store: JsonlStore, max_workers=None):
    """Each pool model writes a review comment on each code diff."""
    max_workers = max_workers or config.MAX_WORKERS
    tasks = []
    for spec in config.MODEL_POOL:
        for _, row in mrc.iterrows():
            mid = str(row["msg_id"])
            key = f"coderesp::{spec.key}::{mid}"
            if store.has(key):
                continue
            tasks.append((spec, row, mid, key))

    def do(task):
        spec, row, mid, key = task
        client = clients[spec.key]
        text = client.complete(
            prompts.code_critic_messages(row["patch"]),
            temperature=config.GEN_TEMPERATURE,
            max_tokens=config.CRITIQUE_MAX_TOKENS,
            seed=config.SEED,
            task_key=key,
            task_kind="code_author",
        )
        store.append(key, {
            "kind": "code_comment",
            "author": spec.key,
            "msg_id": mid,
            "patch": str(row["patch"])[:6000],
            "gold_is_valid": bool(row["is_valid"]),
            "comment": text,
            "comment_len": len(text.split()),
        })

    _run_tasks(tasks, do, max_workers, "code-author")


def code_critique_matrix(code_store: JsonlStore, clients, crit_store: JsonlStore,
                         max_workers=None):
    """Every pool model critiques every authored review comment."""
    max_workers = max_workers or config.MAX_WORKERS
    comments = [c for c in code_store.read_all() if c.get("comment", "").strip()]
    tasks = []
    for spec in config.MODEL_POOL:
        for c in comments:
            author = c["author"]
            key = f"codecrit::{spec.key}::{author}::{c['msg_id']}"
            if crit_store.has(key):
                continue
            tasks.append((spec, author, c, key))

    def do(task):
        spec, author, c, key = task
        client = clients[spec.key]
        # Treat the diff as the "task" and the review comment as the "response".
        question = f"Review this code change:\n{c['patch']}"
        text = client.complete(
            prompts.critic_messages(question, c["comment"]),
            temperature=config.GEN_TEMPERATURE,
            max_tokens=config.CRITIQUE_MAX_TOKENS,
            seed=config.SEED,
            task_key=key,
            task_kind="code_critique",
        )
        crit_store.append(key, {
            "kind": "critique",
            "domain_kind": "code",
            "critic": spec.key,
            "author": author,
            "condition": "self" if spec.key == author else "cross",
            "msg_id": c["msg_id"],
            "gold_is_valid": c["gold_is_valid"],
            "question": question,
            "response": c["comment"],
            "critique": text,
            "critique_len": len(text.split()),
            "self_verdict": prompts.parse_verdict(text),
        })

    _run_tasks(tasks, do, max_workers, "code-critique")
