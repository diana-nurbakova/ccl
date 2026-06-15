"""Dataset acquisition and sampling for Experiment E.

Three sources (all licences confirmed against their hosts on 2026-06-15):
* CriticEval  -- opencompass/CriticBench on HuggingFace (Apache-2.0). The
  ``*_feedback_correction.json`` dev files give question + response +
  gold quality grade (low/medium/high); ``meta_feedback_single`` files carry
  human critique-quality scores used to validate the judge.
* MetaCritique -- GAIR-NLP/MetaCritique (Apache-2.0). benchmark_data.json
  supplies the precision/recall scoring scheme and a calibration set.
* ManualReviewComment -- Zenodo 13150598 (CC-BY-4.0, confirmed on the record).
  270 CodeReviewer comments labelled Valid (172) / Noisy (98) for RQ3.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "exp_e"

# ---------------------------------------------------------------------------
# CriticEval (CriticBench HF)
# ---------------------------------------------------------------------------

_HF_BASE = "https://huggingface.co/datasets/opencompass/CriticBench/resolve/main/"

# 9 domains x {obj, sub} feedback_correction dev files (question + response +
# gold quality grade). These are the items we sample for Path A2.
CRITICEVAL_DOMAINS = [
    "qa", "chat", "summary", "translate", "harmlessness",
    "math_cot", "math_pot", "code_exec", "code_not_exec",
]
CRITICEVAL_FC_FILES = (
    [f"obj_dev_data/{d}_feedback_correction.json" for d in CRITICEVAL_DOMAINS]
    + [f"sub_dev_data/{d}_feedback_correction.json" for d in CRITICEVAL_DOMAINS]
)
# Human-scored critiques for judge validation (Cohen's kappa).
CRITICEVAL_META_FILES = [
    f"obj_dev_data/meta_feedback_single/meta_feedback_singlewise_{d}.json"
    for d in CRITICEVAL_DOMAINS
]

METACRITIQUE_URL = (
    "https://raw.githubusercontent.com/GAIR-NLP/MetaCritique/master/"
    "data/benchmark_data.json"
)
MRC_URL = (
    "https://zenodo.org/api/records/13150598/files/"
    "valid_noisy_manual_labeling_270.xlsx/content"
)

_TIMEOUT = 180.0


def _download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


# ---------------------------------------------------------------------------
# CriticEval loaders
# ---------------------------------------------------------------------------


def load_criticeval_items(force_download: bool = False) -> pd.DataFrame:
    """Load CriticEval feedback_correction records as candidate items.

    Returns one row per (question, dataset response, gold quality) with columns:
      item_id, domain, split, question, dataset_response, dataset_feedback,
      dataset_feedback_score, gold_quality, source_model, data_source
    """
    rows = []
    for rel in CRITICEVAL_FC_FILES:
        dest = DATA_DIR / "criticeval" / rel
        path = _download(_HF_BASE + rel, dest, force=force_download)
        records = json.loads(path.read_text(encoding="utf-8"))
        split = "obj" if rel.startswith("obj_") else "sub"
        domain = Path(rel).stem.replace("_feedback_correction", "")
        for i, rec in enumerate(records):
            meta = rec.get("metadata", {}) or {}
            rows.append({
                "item_id": f"{split}_{domain}_{i}",
                "domain": domain,
                "split": split,
                "question": rec.get("question", ""),
                "dataset_response": rec.get("generation", ""),
                "dataset_feedback": rec.get("feedback", ""),
                "dataset_feedback_score": rec.get("feedback_score", None),
                "gold_quality": (meta.get("quality") or "").lower() or None,
                "source_model": meta.get("llm_name", ""),
                "data_source": rec.get("data_source", ""),
            })
    df = pd.DataFrame(rows)
    df = df[df["question"].str.len() > 0].reset_index(drop=True)
    return df


def load_criticeval_human_critiques(force_download: bool = False) -> pd.DataFrame:
    """Load human-scored critiques (meta_feedback) for judge validation.

    Columns: item_id, domain, question, response, critique, human_score (1-7),
    gold_quality. ``human_score`` is the human annotation of the *critique's*
    quality (``annotated_scores``), used to compute judge-human Cohen's kappa.
    """
    rows = []
    for rel in CRITICEVAL_META_FILES:
        dest = DATA_DIR / "criticeval" / rel
        path = _download(_HF_BASE + rel, dest, force=force_download)
        records = json.loads(path.read_text(encoding="utf-8"))
        domain = Path(rel).stem.replace("meta_feedback_singlewise_", "")
        for i, rec in enumerate(records):
            score = rec.get("annotated_scores") or rec.get("raw_scores")
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            critique = rec.get("evaluated_feedback") or rec.get("feedback") or ""
            if not critique:
                continue
            rows.append({
                "item_id": f"meta_{domain}_{i}",
                "domain": domain,
                "question": rec.get("question", ""),
                "response": rec.get("generation", ""),
                "critique": critique,
                "human_score": score,
                "gold_quality": (rec.get("raw_quality") or "").lower() or None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MetaCritique (calibration / scoring scheme)
# ---------------------------------------------------------------------------


def load_metacritique(force_download: bool = False) -> pd.DataFrame:
    """Load MetaCritique benchmark items (question/response/reference critique)."""
    dest = DATA_DIR / "metacritique" / "benchmark_data.json"
    path = _download(METACRITIQUE_URL, dest, force=force_download)
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for rec in records:
        crit = rec.get("gpt4_critique", {})
        ref_critique = crit.get("critique", "") if isinstance(crit, dict) else str(crit)
        rows.append({
            "item_id": f"mc_{rec.get('index', rec.get('shepherd_id', ''))}",
            "dataset": rec.get("dataset", ""),
            "question": rec.get("question", ""),
            "response": rec.get("response", ""),
            "reference_answer": rec.get("gpt4_answer", ""),
            "reference_critique": ref_critique,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ManualReviewComment (RQ3, code-review domain)
# ---------------------------------------------------------------------------


def load_manual_review_comment(force_download: bool = False) -> pd.DataFrame:
    """Load the 270 manually labelled CodeReviewer comments.

    Columns: msg_id, oldf, patch, msg (review comment), quality_label
    (Valid/Noisy), is_valid (bool).
    """
    dest = DATA_DIR / "mrc" / "valid_noisy_manual_labeling_270.xlsx"
    path = _download(MRC_URL, dest, force=force_download)
    df = pd.read_excel(path)
    df = df.rename(columns={"quality_label": "quality_label"})
    df["is_valid"] = df["quality_label"].astype(str).str.strip().str.lower().eq("valid")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stratified item sampling for Path A2
# ---------------------------------------------------------------------------


def sample_items(
    items: pd.DataFrame,
    n_items: int,
    high_oversample: float = 0.5,
    seed: int = 20260615,
) -> pd.DataFrame:
    """Stratified sample of CriticEval items by gold quality.

    The ``high`` quality stratum is oversampled to ``high_oversample`` of the
    sample because it carries the overcorrection signal (a sound answer that a
    self-critique may wrongly push toward worse). The remainder is split between
    ``medium`` and ``low``.
    """
    rng_items = items[items["gold_quality"].isin(["low", "medium", "high"])].copy()
    if len(rng_items) == 0:
        raise ValueError("No CriticEval items with a low/medium/high gold_quality")

    n_high = int(round(n_items * high_oversample))
    n_rest = n_items - n_high
    n_med = n_rest // 2
    n_low = n_rest - n_med

    def take(df, n):
        if len(df) == 0 or n <= 0:
            return df.head(0)
        return df.sample(n=min(n, len(df)), random_state=seed)

    parts = [
        take(rng_items[rng_items["gold_quality"] == "high"], n_high),
        take(rng_items[rng_items["gold_quality"] == "medium"], n_med),
        take(rng_items[rng_items["gold_quality"] == "low"], n_low),
    ]
    out = pd.concat(parts, ignore_index=True)
    # If a stratum was short, top up from the remaining pool to hit n_items.
    if len(out) < n_items:
        remaining = rng_items[~rng_items["item_id"].isin(out["item_id"])]
        out = pd.concat(
            [out, take(remaining, n_items - len(out))], ignore_index=True
        )
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def download_all(force_download: bool = False) -> None:
    """Pre-download every dataset (used by run_all / --force-download)."""
    print("Experiment E: downloading datasets ...")
    load_criticeval_items(force_download=force_download)
    load_criticeval_human_critiques(force_download=force_download)
    load_metacritique(force_download=force_download)
    load_manual_review_comment(force_download=force_download)
    print("  done.")
