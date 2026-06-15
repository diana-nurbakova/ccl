"""Append-only, resumable JSONL result store.

Every expensive unit of work in Experiment E (an authored response, a critique
matrix cell, a judge verdict) is keyed by a deterministic string and appended
here as one JSON line. On restart the harness loads the set of completed keys
and skips them, so a crashed or interrupted run continues from exactly where it
stopped without repeating any API call. This is the recovery mechanism: the
store *is* the checkpoint.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator


class JsonlStore:
    """A resumable append-only store of records, each carrying a unique ``key``.

    Records are flushed to disk immediately on ``append`` (line-buffered +
    fsync-free flush) so an abrupt termination loses at most the in-flight
    record. Reading back is by full scan, which is fine at this scale
    (tens of thousands of rows).
    """

    KEY_FIELD = "key"

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._keys: set[str] = set()
        self._load_keys()

    def _load_keys(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a truncated final line from a hard crash.
                    continue
                k = rec.get(self.KEY_FIELD)
                if k is not None:
                    self._keys.add(k)

    # -- queries -----------------------------------------------------------

    def has(self, key: str) -> bool:
        return key in self._keys

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> set[str]:
        return set(self._keys)

    def read_all(self) -> list[dict]:
        return list(self.iter_records())

    def iter_records(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    # -- writes ------------------------------------------------------------

    def append(self, key: str, record: dict) -> None:
        """Append a record under ``key``. No-op if the key already exists."""
        with self._lock:
            if key in self._keys:
                return
            row = {self.KEY_FIELD: key, **record}
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            self._keys.add(key)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.read_all())
