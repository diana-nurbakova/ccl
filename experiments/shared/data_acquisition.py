"""Data acquisition: download and cache datasets for CCL validation experiments."""

from __future__ import annotations

import json
import csv
import io
import os
from pathlib import Path

import httpx
import pandas as pd

# Load .env so HF_TOKEN (and others) are available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass  # python-dotenv is already in main project deps; safe to ignore if missing

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

FRANK_URL = (
    "https://raw.githubusercontent.com/artidoro/frank/main/"
    "data/human_annotations_sentence.json"
)

FELM_URL = (
    "https://huggingface.co/datasets/hkust-nlp/felm/resolve/main/all.jsonl"
)

BASTANI_BASE = (
    "https://raw.githubusercontent.com/obastani/GenAICanHarmLearning/main/"
)
BASTANI_OUTCOMES_URL = BASTANI_BASE + "main_regressions/final_data.csv"
BASTANI_CONVERSATIONS_URL = (
    BASTANI_BASE + "text_analysis/data/raw/valid_student_data.csv"
)
BASTANI_CONVERSATIONS_TS_URL = (
    BASTANI_BASE + "text_analysis/data/raw/valid_student_data_w_time_stamp.csv"
)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_CLIENT_TIMEOUT = 120.0  # seconds


def _download(url: str, dest: Path, force: bool = False) -> Path:
    """Download *url* to *dest* if not already cached."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    with httpx.Client(timeout=_CLIENT_TIMEOUT, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  -> saved to {dest}")
    return dest


# ---------------------------------------------------------------------------
# FRANK benchmark
# ---------------------------------------------------------------------------

def download_frank(force: bool = False) -> Path:
    """Download FRANK human_annotations_sentence.json and return its path."""
    dest = DATA_DIR / "frank" / "human_annotations_sentence.json"
    return _download(FRANK_URL, dest, force=force)


def load_frank(force_download: bool = False) -> list[dict]:
    """Load FRANK annotations as a list of dicts.

    Each dict represents one model-generated summary and contains:
      - hash, model_name, article, summary, reference
      - summary_sentences: list[str]
      - summary_sentences_annotations: dict mapping "annotator_0/1/2"
        to list-of-lists (one inner list of error codes per sentence)
    """
    path = download_frank(force=force_download)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def frank_to_sentence_df(data: list[dict]) -> pd.DataFrame:
    """Flatten FRANK data to one row per sentence with annotator labels.

    Returns DataFrame with columns:
      hash, model_name, sentence_idx, sentence,
      ann_0_errors, ann_1_errors, ann_2_errors
    where each ann_*_errors is a list of FRANK error codes.

    The actual FRANK structure for summary_sentences_annotations is a list
    of per-sentence dicts:
      [{"annotator_0": [...], "annotator_1": [...], "annotator_2": [...]}, ...]
    """
    rows = []
    for record in data:
        annotations = record["summary_sentences_annotations"]
        sentences = record.get("summary_sentences", [])

        for i, sent_ann in enumerate(annotations):
            rows.append({
                "hash": record.get("hash", ""),
                "model_name": record.get("model_name", ""),
                "sentence_idx": i,
                "sentence": sentences[i] if i < len(sentences) else "",
                "ann_0_errors": sent_ann.get("annotator_0", []),
                "ann_1_errors": sent_ann.get("annotator_1", []),
                "ann_2_errors": sent_ann.get("annotator_2", []),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bastani et al. (GenAICanHarmLearning)
# ---------------------------------------------------------------------------

def download_bastani(force: bool = False) -> dict[str, Path]:
    """Download Bastani outcome + conversation data. Returns dict of paths."""
    paths = {}
    paths["outcomes"] = _download(
        BASTANI_OUTCOMES_URL,
        DATA_DIR / "bastani" / "final_data.csv",
        force=force,
    )
    paths["conversations"] = _download(
        BASTANI_CONVERSATIONS_URL,
        DATA_DIR / "bastani" / "valid_student_data.csv",
        force=force,
    )
    paths["conversations_ts"] = _download(
        BASTANI_CONVERSATIONS_TS_URL,
        DATA_DIR / "bastani" / "valid_student_data_w_time_stamp.csv",
        force=force,
    )
    return paths


# ---------------------------------------------------------------------------
# FELM benchmark
# ---------------------------------------------------------------------------

def download_felm(force: bool = False) -> Path:
    """Download FELM all.jsonl and return its path (~300 KB)."""
    dest = DATA_DIR / "felm" / "all.jsonl"
    return _download(FELM_URL, dest, force=force)


def load_felm(force_download: bool = False) -> pd.DataFrame:
    """Load FELM as a flat DataFrame with one row per segment.

    Real schema (confirmed from dataset):
      - domain: 'wk' | 'math' | 'science' | 'reasoning' | 'writing_rec'
      - segmented_response: list[str]  — one entry per segment
      - labels: list[bool]             — False = error present
      - type: list[str | None]         — error type per segment (often None/empty)

    Returns DataFrame with columns:
      record_index, domain, segment_idx, segment, is_error,
      error_type, prompt
    """
    path = download_felm(force=force_download)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            segments = rec.get("segmented_response", [])
            labels = rec.get("labels", [])
            types = rec.get("type", [])

            for i, seg in enumerate(segments):
                label = labels[i] if i < len(labels) else None
                etype = types[i] if i < len(types) else None
                rows.append({
                    "record_index": rec.get("index", ""),
                    "domain": rec.get("domain", ""),
                    "segment_idx": i,
                    "segment": seg,
                    "is_error": (label is False) if label is not None else None,
                    "error_type": etype if etype else None,
                    "prompt": rec.get("prompt", ""),
                })
    return pd.DataFrame(rows)


def load_bastani_outcomes(force_download: bool = False) -> pd.DataFrame:
    """Load student-session outcome data.

    Key columns: Student ID, Session, Treatment arm, GPTBase, GPTTutor,
    Part2Tot, Part3Tot, Class, gpa_prev, female, ...
    """
    paths = download_bastani(force=force_download)
    return pd.read_csv(paths["outcomes"])


def load_bastani_conversations(
    force_download: bool = False,
    with_timestamps: bool = True,
) -> pd.DataFrame:
    """Load conversation turn data.

    Each row = one message turn. Key columns:
      role, username, problem_id, session_id, grade, treatment, message,
      conversation_id, [timestamp]
    """
    paths = download_bastani(force=force_download)
    key = "conversations_ts" if with_timestamps else "conversations"
    return pd.read_csv(paths[key])


# ---------------------------------------------------------------------------
# WildChat (HuggingFace streaming sample)
# ---------------------------------------------------------------------------

def download_wildchat_sample(
    n_conversations: int = 10_000,
    min_user_conversations: int = 5,
    force: bool = False,
    seed: int = 42,
) -> Path:
    """Download a sample of WildChat conversations from repeat users.

    Uses the HuggingFace ``datasets`` library in streaming mode to avoid
    downloading the full 4.8M-conversation dataset. Filters for users
    with at least *min_user_conversations* conversations so that
    longitudinal analysis is possible.

    WildChat schema (one record = one full conversation):
      - hashed_ip     : user identifier (top-level)
      - conversation_hash : unique conversation ID
      - timestamp     : datetime of the conversation (top-level)
      - conversation  : list of turn dicts, each with 'role' and 'content'
      - model, language, toxic, redacted, ...

    Parameters
    ----------
    n_conversations : int
        Target number of conversations to keep (from qualifying users).
    min_user_conversations : int
        Minimum conversations per user to include them.
    force : bool
        Re-download even if cached file exists.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    Path to the cached parquet file.
    """
    dest = DATA_DIR / "wildchat" / "wildchat_sample.parquet"
    if dest.exists() and not force:
        print(f"WildChat sample already cached at {dest}")
        return dest

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the `datasets` package: pip install datasets>=2.14"
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN not found. Add it to your .env file: HF_TOKEN=hf_..."
        )

    print(f"Streaming WildChat to find repeat users "
          f"(min {min_user_conversations} conversations)...")

    ds = load_dataset(
        "allenai/WildChat-1M",
        split="train",
        streaming=True,
        token=hf_token,
    )

    # Each record = one full conversation. Group by hashed_ip (user identifier).
    user_convos: dict[str, list[dict]] = {}
    total_scanned = 0
    scan_limit = n_conversations * 20

    for example in ds:
        total_scanned += 1
        if total_scanned > scan_limit:
            break

        # Skip toxic / redacted conversations
        if example.get("toxic", False) or example.get("redacted", False):
            continue

        user_id = example.get("hashed_ip", "")
        if not user_id:
            continue

        if user_id not in user_convos:
            user_convos[user_id] = []

        user_convos[user_id].append({
            "conversation_hash": example.get("conversation_hash", ""),
            "hashed_ip": user_id,
            "model": example.get("model", ""),
            "timestamp": example.get("timestamp", None),
            "language": example.get("language", ""),
            "turns": example.get("conversation", []),
        })

        # Early exit once we have enough qualifying conversations
        qualifying = sum(
            len(v) for v in user_convos.values()
            if len(v) >= min_user_conversations
        )
        if qualifying >= n_conversations:
            break

    print(f"  Scanned {total_scanned} conversations, "
          f"found {len(user_convos)} unique users")

    # Filter to repeat users only
    repeat_users = {
        uid: convos
        for uid, convos in user_convos.items()
        if len(convos) >= min_user_conversations
    }
    print(f"  {len(repeat_users)} users with >= {min_user_conversations} conversations")

    if not repeat_users:
        raise RuntimeError(
            f"No users found with >= {min_user_conversations} conversations "
            f"after scanning {total_scanned} records. "
            "Try increasing scan_limit or reducing min_user_conversations."
        )

    # Collect and trim to target size
    import random
    random.seed(seed)
    all_convos = [c for convos in repeat_users.values() for c in convos]
    if len(all_convos) > n_conversations:
        random.shuffle(all_convos)
        all_convos = all_convos[:n_conversations]

    # Flatten to one row per turn
    rows = []
    for convo in all_convos:
        ts = convo["timestamp"]
        # Normalise timestamp to string so parquet handles it uniformly
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts) if ts else ""
        for turn_idx, turn in enumerate(convo["turns"]):
            rows.append({
                "conversation_hash": convo["conversation_hash"],
                "hashed_ip": convo["hashed_ip"],
                "model": convo["model"],
                "timestamp": ts_str,
                "language": convo["language"],
                "turn_idx": turn_idx,
                "role": turn.get("role", ""),
                "content": turn.get("content", ""),
            })

    df = pd.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"  Saved {len(df)} turns from {len(all_convos)} conversations to {dest}")
    return dest


def load_wildchat_sample(
    force_download: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load the cached WildChat sample as a DataFrame."""
    path = download_wildchat_sample(force=force_download, **kwargs)
    return pd.read_parquet(path)
