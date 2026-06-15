"""Recovery utility for Experiment E.

Because every API result is checkpointed to an append-only JsonlStore, a crashed
or interrupted run is recovered simply by running generation again — completed
work is skipped. This script makes that explicit and adds safety/inspection:

    python -m experiments.exp_e_critique.recover --status     # what is done / left
    python -m experiments.exp_e_critique.recover --repair     # drop truncated lines
    python -m experiments.exp_e_critique.recover --resume     # continue the run
    python -m experiments.exp_e_critique.recover --resume --n-items 60

"--repair" rewrites each store dropping any malformed/truncated trailing line a
hard crash may have left (data already loaded fine, but this keeps files clean).
"--resume" re-enters the same pipeline as run.py; all finished calls are reused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exp_e_critique import config, run


def repair_store(path: Path) -> tuple[int, int]:
    """Rewrite a JSONL file keeping only valid, unique-keyed records.

    Returns (kept, dropped).
    """
    if not path.exists():
        return (0, 0)
    kept_rows = []
    seen = set()
    dropped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            k = rec.get("key")
            if k is None or k in seen:
                dropped += 1
                continue
            seen.add(k)
            kept_rows.append(line)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(kept_rows) + ("\n" if kept_rows else ""), encoding="utf-8")
    tmp.replace(path)
    return (len(kept_rows), dropped)


def repair_all() -> None:
    print("Repairing stores (dropping any truncated/duplicate lines) ...")
    for name in [
        "responses.jsonl", "critiques_general.jsonl", "code_comments.jsonl",
        "critiques_code.jsonl", "judgments.jsonl", "soundness.jsonl",
        "validate_criticeval.jsonl", "validate_mrc.jsonl",
    ]:
        path = config.STORE_DIR / name
        kept, dropped = repair_store(path)
        if path.exists():
            print(f"  {name:28s} kept={kept:>6} dropped={dropped}")


def main() -> None:
    p = argparse.ArgumentParser(description="Recover/resume Experiment E from checkpoints")
    p.add_argument("--status", action="store_true")
    p.add_argument("--repair", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-items", type=int, default=config.DEFAULT_N_ITEMS)
    p.add_argument("--max-workers", type=int, default=config.MAX_WORKERS)
    p.add_argument("--skip-code", action="store_true")
    args = p.parse_args()

    if args.repair:
        repair_all()
    if args.status or not (args.resume or args.repair):
        run.print_status(args.n_items)
    if args.resume:
        print("\nResuming run (completed calls will be skipped) ...")
        probe = run.probe_models()
        if probe["judge"] is None or len(probe["pool"]) < 2:
            print("Aborting: judge or too few pool models available.")
            return
        st = run.run_generation(
            args.n_items, args.max_workers, force_download=False,
            skip_code=args.skip_code, pool=probe["pool"], judge_spec=probe["judge"],
        )
        run.run_analysis(st)


if __name__ == "__main__":
    main()
