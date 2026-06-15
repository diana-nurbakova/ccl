"""Tests for Experiment E (Stage 4 self-vs-cross critique).

Cover the offline machinery: the resumable store (recovery), prompt parsing,
Cohen's kappa, stratified sampling, and the analyses on synthetic data with a
known self<cross validity gap and a known self-overcorrection effect. No API
calls are made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Resumable JsonlStore (the recovery mechanism)
# ---------------------------------------------------------------------------


class TestJsonlStore:
    def test_append_dedup_and_resume(self, tmp_path):
        from experiments.shared.jsonl_store import JsonlStore

        path = tmp_path / "s.jsonl"
        s = JsonlStore(path)
        s.append("a", {"x": 1})
        s.append("b", {"x": 2})
        s.append("a", {"x": 99})  # duplicate key -> ignored
        assert len(s) == 2
        assert "a" in s and "b" in s

        # Re-open: completed keys are remembered (resume from where we stopped).
        s2 = JsonlStore(path)
        assert s2.has("a") and s2.has("b")
        assert len(s2) == 2
        recs = {r["key"]: r["x"] for r in s2.read_all()}
        assert recs == {"a": 1, "b": 2}

    def test_tolerates_truncated_final_line(self, tmp_path):
        from experiments.shared.jsonl_store import JsonlStore

        path = tmp_path / "s.jsonl"
        s = JsonlStore(path)
        s.append("a", {"x": 1})
        # Simulate a hard crash mid-write: append a partial JSON line.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"key": "b", "x": 2')  # no closing brace / newline
        s2 = JsonlStore(path)
        assert s2.has("a")
        assert not s2.has("b")  # partial record is ignored, not crashed on


class TestRepair:
    def test_repair_drops_bad_lines(self, tmp_path, monkeypatch):
        from experiments.exp_e_critique import config, recover

        path = tmp_path / "responses.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"key":"a","x":1}\n')
            fh.write('{"key":"a","x":2}\n')   # duplicate
            fh.write('{"key":"b","x":3\n')     # malformed
        kept, dropped = recover.repair_store(path)
        assert kept == 1
        assert dropped == 2
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_extract_json_plain(self):
        from experiments.exp_e_critique.prompts import extract_json

        v = extract_json('{"critique_valid": true, "validity_score": 6}')
        assert v["critique_valid"] is True
        assert v["validity_score"] == 6

    def test_extract_json_fenced_with_prose(self):
        from experiments.exp_e_critique.prompts import extract_json

        text = 'Here is my answer:\n```json\n{"sound": false, "quality_score": 2}\n```\nDone.'
        v = extract_json(text)
        assert v["sound"] is False
        assert v["quality_score"] == 2

    def test_extract_json_garbage(self):
        from experiments.exp_e_critique.prompts import extract_json

        assert extract_json("no json here") is None
        assert extract_json("") is None

    def test_parse_verdict(self):
        from experiments.exp_e_critique.prompts import parse_verdict

        assert parse_verdict("blah\nVERDICT: MAJOR_ISSUES") == "MAJOR_ISSUES"
        assert parse_verdict("text\nVERDICT: sound") == "SOUND"
        assert parse_verdict("no verdict line") is None


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


class TestKappa:
    def test_perfect_agreement(self):
        from experiments.exp_e_critique.judge import cohens_kappa

        a = [1, 0, 1, 1, 0]
        assert cohens_kappa(a, a) == pytest.approx(1.0)

    def test_chance_agreement_near_zero(self):
        from experiments.exp_e_critique.judge import cohens_kappa

        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 400)
        b = rng.integers(0, 2, 400)
        assert abs(cohens_kappa(a, b)) < 0.2


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


class TestSampling:
    @staticmethod
    def _items(n=200, seed=0):
        rng = np.random.default_rng(seed)
        q = rng.choice(["low", "medium", "high"], size=n)
        return pd.DataFrame({
            "item_id": [f"it{i}" for i in range(n)],
            "domain": "qa",
            "question": [f"q{i}" for i in range(n)],
            "gold_quality": q,
        })

    def test_oversamples_high(self):
        from experiments.exp_e_critique.data import sample_items

        items = self._items(300, seed=1)
        s = sample_items(items, n_items=60, high_oversample=0.5, seed=1)
        assert len(s) == 60
        frac_high = (s["gold_quality"] == "high").mean()
        # High should be ~0.5 of the sample (oversampled), clearly above 1/3.
        assert frac_high > 0.4
        assert s["item_id"].is_unique


# ---------------------------------------------------------------------------
# Analysis: synthetic self < cross validity, and self overcorrection
# ---------------------------------------------------------------------------


class TestAnalysis:
    @staticmethod
    def _synthetic_judgments(seed=0, n_items=80):
        """Each of 5 critics critiques each of 5 authored responses per item.

        Self (critic==author) is less valid and overcorrects sound responses
        more than cross, by construction.
        """
        rng = np.random.default_rng(seed)
        models = [f"m{k}" for k in range(5)]
        rows = []
        for it in range(n_items):
            quality = rng.choice(["low", "medium", "high"])
            for author in models:
                resp_sound = quality == "high"
                for critic in models:
                    condition = "self" if critic == author else "cross"
                    # Cross more valid than self.
                    p_valid = 0.85 if condition == "cross" else 0.6
                    valid = bool(rng.random() < p_valid)
                    # On sound responses, self overcorrects more.
                    if resp_sound:
                        p_oc = 0.4 if condition == "self" else 0.1
                        oc = rng.random() < p_oc
                    else:
                        oc = False
                    rows.append({
                        "domain_kind": "general",
                        "critic": critic, "author": author,
                        "condition": condition,
                        "item_id": f"it{it}", "domain": "qa",
                        "gold_quality": quality,
                        "critique_len": int(rng.integers(20, 120)),
                        "critique_valid": valid,
                        "validity_score": 6 if valid else 2,
                        "identifies_real_flaw": (not resp_sound),
                        "recommends_change": bool(oc or (not resp_sound)),
                        "change_would_worsen": bool(oc),
                        "resp_sound": resp_sound,
                    })
        return rows

    def test_glmm_recovers_cross_advantage(self):
        from experiments.exp_e_critique import analysis

        df = analysis.judgments_to_df(self._synthetic_judgments(seed=2))
        res = analysis.glmm_validity(df)
        assert "error" not in res
        # Cross more valid than self -> positive beta, OR > 1, significant.
        assert res["beta_is_cross"] > 0
        assert res["odds_ratio"] > 1.0
        assert res["or_ci_low"] > 1.0
        assert res["p_value"] < 0.05

    def test_overcorrection_self_higher(self):
        from experiments.exp_e_critique import analysis

        df = analysis.judgments_to_df(self._synthetic_judgments(seed=3))
        res = analysis.overcorrection_analysis(df)
        assert res["rates"]["self"]["overcorrection_rate"] > \
            res["rates"]["cross"]["overcorrection_rate"]
        assert res["self_minus_cross"] > 0
        # McNemar should detect the paired difference.
        assert res["mcnemar"]["p_value"] < 0.05

    def test_descriptive_validity_shape(self):
        from experiments.exp_e_critique import analysis

        df = analysis.judgments_to_df(self._synthetic_judgments(seed=4))
        desc = analysis.descriptive_validity(df)
        assert {"self", "cross"} <= set(desc["condition"])
        overall = desc[desc["scope"] == "overall"]
        assert len(overall) == 2

    def test_judge_kappa_helper(self):
        from experiments.exp_e_critique import analysis

        recs = [
            {"human_valid": True, "judge_valid": True},
            {"human_valid": False, "judge_valid": False},
            {"human_valid": True, "judge_valid": True},
            {"human_valid": False, "judge_valid": True},
        ]
        res = analysis.judge_kappa(recs, "human_valid")
        assert res["n"] == 4
        assert "cohens_kappa" in res
