"""CLI orchestration for Experiment E (Stage 4 self-vs-cross critique).

Fully resumable: every API result is checkpointed to a JsonlStore, so re-running
this command (or the dedicated ``recover.py``) continues from exactly where a
crash or interruption left off without repeating any call. All raw LLM
interactions are written to ``output/exp_e/logs/llm_interactions.jsonl``.

Usage:
    python -m experiments.exp_e_critique.run --probe            # check models live
    python -m experiments.exp_e_critique.run                    # full run (resumes)
    python -m experiments.exp_e_critique.run --n-items 60       # smaller sample
    python -m experiments.exp_e_critique.run --status           # progress only
    python -m experiments.exp_e_critique.run --analyze-only     # skip generation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.shared.jsonl_store import JsonlStore
from experiments.shared.llm_client import InteractionLogger, make_client

from experiments.exp_e_critique import analysis, config, data, generate, judge


# ---------------------------------------------------------------------------
# Store handles
# ---------------------------------------------------------------------------


def _stores() -> dict[str, JsonlStore]:
    sd = config.STORE_DIR
    return {
        "responses": JsonlStore(sd / "responses.jsonl"),
        "critiques": JsonlStore(sd / "critiques_general.jsonl"),
        "code_comments": JsonlStore(sd / "code_comments.jsonl"),
        "code_critiques": JsonlStore(sd / "critiques_code.jsonl"),
        "judgments": JsonlStore(sd / "judgments.jsonl"),
        "soundness": JsonlStore(sd / "soundness.jsonl"),
        "val_ce": JsonlStore(sd / "validate_criticeval.jsonl"),
        "val_mrc": JsonlStore(sd / "validate_mrc.jsonl"),
    }


def _logger() -> InteractionLogger:
    return InteractionLogger(config.LOG_DIR / "llm_interactions.jsonl")


# ---------------------------------------------------------------------------
# Model availability probe
# ---------------------------------------------------------------------------


def probe_models() -> dict:
    """Send a one-token call to each pool model + judge; swap in fallbacks.

    Returns the resolved pool/judge and prints a report. Edits are in-memory
    only (the run uses the returned specs); persistent edits go in config.py.
    """
    log = InteractionLogger(None)
    resolved_pool = []
    print("Probing model availability ...")
    for spec in config.MODEL_POOL:
        candidates = [spec.model] + [
            m for m in config.MODEL_FALLBACKS.get(spec.key, []) if m != spec.model
        ]
        chosen = _first_live(spec.provider, candidates, spec, log)
        status = chosen or "UNAVAILABLE"
        print(f"  {spec.key:12s} {spec.provider:9s} -> {status}")
        if chosen:
            resolved_pool.append(dataclasses.replace(spec, model=chosen))
    # Judge
    jcands = [config.JUDGE.model] + [m for m in config.JUDGE_FALLBACKS if m != config.JUDGE.model]
    jchosen = _first_live(config.JUDGE.provider, jcands, config.JUDGE, log)
    print(f"  {'judge':12s} {config.JUDGE.provider:9s} -> {jchosen or 'UNAVAILABLE'}")
    return {
        "pool": resolved_pool,
        "judge": dataclasses.replace(config.JUDGE, model=jchosen) if jchosen else None,
    }


def _first_live(provider, candidates, spec, log) -> str | None:
    for model in candidates:
        try:
            client = make_client(
                provider, model, temperature=0.0, max_tokens=8,
                reasoning_effort=(spec.reasoning_effort if spec.reasoning else None),
                logger=log,
            )
            client.complete(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=8, task_kind="probe", task_key=f"probe::{model}",
            )
            client.close()
            return model
        except Exception as e:  # noqa: BLE001
            print(f"      x {model}: {type(e).__name__}: {str(e)[:100]}")
    return None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def print_status(n_items: int) -> None:
    st = _stores()
    n_models = len(config.MODEL_POOL)
    exp_resp = n_items * n_models
    exp_crit = exp_resp * n_models
    print("Experiment E — progress")
    print(f"  responses        : {len(st['responses']):>6} / ~{exp_resp}")
    print(f"  general critiques: {len(st['critiques']):>6} / ~{exp_crit}")
    print(f"  response soundness: {len(st['soundness']):>6} / ~{exp_resp}")
    print(f"  code comments    : {len(st['code_comments']):>6}")
    print(f"  code critiques   : {len(st['code_critiques']):>6}")
    print(f"  judge verdicts   : {len(st['judgments']):>6}")
    print(f"  judge val (CE)   : {len(st['val_ce']):>6}")
    print(f"  judge val (MRC)  : {len(st['val_mrc']):>6}")
    print(f"  interaction log  : {config.LOG_DIR / 'llm_interactions.jsonl'}")


# ---------------------------------------------------------------------------
# Generation + judging pipeline
# ---------------------------------------------------------------------------


def run_generation(n_items, max_workers, force_download, skip_code,
                   pool=None, judge_spec=None):
    if pool:
        config.MODEL_POOL[:] = pool
    if judge_spec:
        # replace module-level JUDGE used by build_judge default
        config.JUDGE = judge_spec  # type: ignore[assignment]

    log = _logger()
    st = _stores()

    # --- sample items (cached to disk for reproducibility) ---
    print("\n[1/6] Sampling CriticEval items ...")
    items_all = data.load_criticeval_items(force_download=force_download)
    items = data.sample_items(items_all, n_items,
                              high_oversample=config.HIGH_QUALITY_OVERSAMPLE,
                              seed=config.SEED)
    items.to_csv(config.EXP_DIR / "sampled_items.csv", index=False)
    print(f"   sampled {len(items)} items "
          f"(quality dist: {items['gold_quality'].value_counts().to_dict()})")

    clients = generate.build_clients(log)
    judge_client = judge.build_judge(log)

    print("\n[2/6] Authoring responses (matrix rows) ...")
    generate.author_responses(items, clients, st["responses"], max_workers)

    print("\n[3/6] Critique matrix (every model critiques every response) ...")
    generate.critique_matrix(st["responses"], clients, st["critiques"], max_workers)

    print("\n[4/6] Judging response soundness + critique validity ...")
    judge.judge_response_soundness(st["responses"], judge_client, st["soundness"], max_workers)
    judge.judge_critiques(st["critiques"], judge_client, st["judgments"], max_workers)

    print("\n[5/6] Validating judge against human labels ...")
    human_ce = data.load_criticeval_human_critiques(force_download=force_download)
    # cap calibration set for cost; stratified-ish by taking a stable sample
    if len(human_ce) > 200:
        human_ce = human_ce.sample(n=200, random_state=config.SEED)
    judge.validate_judge_criticeval(human_ce, judge_client, st["val_ce"], max_workers=max_workers)
    mrc = data.load_manual_review_comment(force_download=force_download)
    judge.validate_judge_mrc(mrc, judge_client, st["val_mrc"], max_workers=max_workers)

    if not skip_code:
        print("\n[6/6] Code-review domain (RQ3) ...")
        generate.author_code_comments(mrc, clients, st["code_comments"], max_workers)
        generate.code_critique_matrix(st["code_comments"], clients, st["code_critiques"], max_workers)
        judge.judge_critiques(st["code_critiques"], judge_client, st["judgments"], max_workers)
    else:
        print("\n[6/6] Skipping code-review domain (--skip-code)")

    for c in clients.values():
        c.close()
    judge_client.close()
    return st


# ---------------------------------------------------------------------------
# Analysis + outputs
# ---------------------------------------------------------------------------


def run_analysis(st: dict[str, JsonlStore]) -> dict:
    print("\n=== Analysis ===")
    out = config.EXP_DIR
    results: dict = {}

    judgments = st["judgments"].read_all()
    if not judgments:
        print("  No judge verdicts yet — run generation first.")
        return results
    jdf = analysis.judgments_to_df(judgments)
    jdf = analysis.attach_soundness(jdf, st["soundness"].read_all())

    gen = jdf[jdf["domain_kind"] == "general"].copy()
    code = jdf[jdf["domain_kind"] == "code"].copy()

    # --- judge validation (kappa) ---
    k_ce = analysis.judge_kappa(st["val_ce"].read_all(), "human_valid")
    k_mrc = analysis.judge_kappa(st["val_mrc"].read_all(), "gold_is_valid")
    results["kappa_criticeval"] = k_ce
    results["kappa_mrc"] = k_mrc
    print(f"  Judge-human kappa (CriticEval): {k_ce.get('cohens_kappa')!r} (n={k_ce.get('n')})")
    print(f"  Judge-human kappa (MRC)       : {k_mrc.get('cohens_kappa')!r} (n={k_mrc.get('n')})")
    judge_trustworthy = (k_ce.get("cohens_kappa") or 0) >= config.KAPPA_THRESHOLD

    # --- descriptives ---
    desc = analysis.descriptive_validity(gen)
    desc.to_csv(out / "exp_e_descriptive_validity.csv", index=False)
    results["descriptive"] = desc

    # --- RQ1 ---
    print("\n  RQ1: self vs cross validity (general) ...")
    try:
        rq1 = analysis.glmm_validity(gen)
    except Exception as e:  # noqa: BLE001
        rq1 = {"error": str(e)}
    results["rq1"] = rq1
    _print_glmm(rq1)

    # --- RQ2 overcorrection ---
    print("\n  RQ2: overcorrection on sound responses ...")
    rq2 = analysis.overcorrection_analysis(gen)
    results["rq2"] = rq2
    print(f"    self  overcorrection: {rq2['rates']['self']['overcorrection_rate']:.3f} "
          f"(n={rq2['rates']['self']['n']})")
    print(f"    cross overcorrection: {rq2['rates']['cross']['overcorrection_rate']:.3f} "
          f"(n={rq2['rates']['cross']['n']})")
    print(f"    self - cross = {rq2['self_minus_cross']:.3f}; McNemar p = {rq2['mcnemar'].get('p_value')}")
    results["helpful_correction"] = analysis.helpful_correction_rate(gen)

    # --- RQ3 code domain ---
    if len(code) > 0:
        print("\n  RQ3: self vs cross validity (code review) ...")
        try:
            rq3 = analysis.glmm_validity(code, item_col="msg_id")
        except Exception as e:  # noqa: BLE001
            rq3 = {"error": str(e)}
        results["rq3"] = rq3
        _print_glmm(rq3)
        analysis.descriptive_validity(code).to_csv(
            out / "exp_e_descriptive_validity_code.csv", index=False)

    # --- length control ---
    results["length_control"] = analysis.length_control(gen)

    # --- save consolidated results table + JSON ---
    _save_results_table(results, out)
    (out / "exp_e_results.json").write_text(
        json.dumps(_jsonable(results), indent=2, default=str), encoding="utf-8")
    results["judge_trustworthy"] = judge_trustworthy

    _make_figure(gen, code, results, config.FIG_DIR)
    print(f"\n  Outputs written to {out}/")
    return results


def _print_glmm(r: dict) -> None:
    if "error" in r:
        print(f"    GLMM unavailable: {r['error']}")
        return
    print(f"    OR(cross vs self) = {r['odds_ratio']:.2f} "
          f"[95% CI {r['or_ci_low']:.2f}, {r['or_ci_high']:.2f}], "
          f"beta={r['beta_is_cross']:.3f}, p={r['p_value']:.2e}, n={r['n_obs']}")


def _save_results_table(results: dict, out: Path) -> None:
    import pandas as pd

    rows = []
    for rq, key in [("RQ1 validity (general)", "rq1"), ("RQ3 validity (code)", "rq3")]:
        r = results.get(key, {})
        if r and "error" not in r:
            rows.append({
                "result": rq, "metric": "OR(cross vs self)",
                "estimate": round(r["odds_ratio"], 3),
                "ci_low": round(r["or_ci_low"], 3),
                "ci_high": round(r["or_ci_high"], 3),
                "p_value": r["p_value"], "n": r["n_obs"],
            })
    rq2 = results.get("rq2", {})
    if rq2:
        rows.append({
            "result": "RQ2 overcorrection", "metric": "self - cross rate diff",
            "estimate": round(rq2["self_minus_cross"], 3),
            "ci_low": None, "ci_high": None,
            "p_value": rq2["mcnemar"].get("p_value"), "n": rq2["n_sound"],
        })
    if rows:
        pd.DataFrame(rows).to_csv(out / "exp_e_results_table.csv", index=False)


def _make_figure(gen, code, results, fig_dir) -> None:
    try:
        import matplotlib.pyplot as plt
        from experiments.shared.plotting import save_figure, setup_style
    except Exception:  # noqa: BLE001
        return
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: validity rate self vs cross, overall + per critic (general)
    d = gen.dropna(subset=["valid_int"])
    if len(d):
        critics = sorted(d["critic"].unique())
        import numpy as np
        x = np.arange(len(critics))
        self_rates = [d[(d.critic == c) & (d.condition == "self")]["valid_int"].mean() for c in critics]
        cross_rates = [d[(d.critic == c) & (d.condition == "cross")]["valid_int"].mean() for c in critics]
        axes[0].bar(x - 0.2, self_rates, 0.4, label="self", color="#E57373")
        axes[0].bar(x + 0.2, cross_rates, 0.4, label="cross", color="#64B5F6")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(critics, rotation=30, ha="right", fontsize=8)
        axes[0].set_ylabel("Critique validity rate")
        axes[0].set_title("Self vs cross critique validity (per critic)")
        axes[0].legend()

    # Panel 2: overcorrection rates
    rq2 = results.get("rq2", {})
    if rq2:
        r = rq2["rates"]
        axes[1].bar(["self", "cross"],
                    [r["self"]["overcorrection_rate"], r["cross"]["overcorrection_rate"]],
                    color=["#E57373", "#64B5F6"])
        axes[1].set_ylabel("Overcorrection rate (sound responses)")
        axes[1].set_title("Overcorrection: self vs cross")

    fig.suptitle("Experiment E: Stage 4 self-vs-cross critique", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "exp_e_self_vs_cross", output_dir=fig_dir)


def _jsonable(obj):
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment E: Stage 4 self-vs-cross critique")
    p.add_argument("--n-items", type=int, default=config.DEFAULT_N_ITEMS)
    p.add_argument("--max-workers", type=int, default=config.MAX_WORKERS)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-code", action="store_true")
    p.add_argument("--probe", action="store_true", help="check model availability and exit")
    p.add_argument("--status", action="store_true", help="show progress and exit")
    p.add_argument("--analyze-only", action="store_true", help="skip generation, just analyse stores")
    args = p.parse_args()

    if args.status:
        print_status(args.n_items)
        return
    if args.probe:
        probe_models()
        return

    if args.analyze_only:
        st = _stores()
    else:
        probe = probe_models()
        if probe["judge"] is None or len(probe["pool"]) < 2:
            print("\nAborting: judge or too few pool models unavailable. "
                  "Fix credentials/model strings in config.py.")
            return
        st = run_generation(args.n_items, args.max_workers, args.force_download,
                            args.skip_code, pool=probe["pool"], judge_spec=probe["judge"])

    run_analysis(st)


if __name__ == "__main__":
    main()
