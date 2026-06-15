"""Statistical analysis for Experiment E (self vs cross critique validity).

Respects the dependence structure (spec section 8): self and cross critiques
target the same response, so comparisons are within-item and paired, and the
self/cross effect is estimated within critic (holding capability constant).

* RQ1 -- mixed-effects logistic: critique_valid ~ is_cross + gold_quality
         + (1|item) + (1|critic) + (1|author). Single judge, so no judge term.
* RQ2 -- overcorrection (correct->incorrect pressure) on sound responses:
         self vs cross rate difference + paired McNemar + mixed logistic.
* RQ3 -- the RQ1 model refit on the code-review domain, reported separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def judgments_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "critique_valid" in df.columns:
        df["valid_int"] = df["critique_valid"].map({True: 1, False: 0})
    df["is_cross"] = (df["condition"] == "cross").astype(int)
    return df


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------


def descriptive_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition validity rate, overall and within each critic model."""
    d = df.dropna(subset=["valid_int"])
    rows = []
    for cond in ["self", "cross"]:
        sub = d[d["condition"] == cond]
        rows.append({
            "scope": "overall", "critic": "ALL", "condition": cond,
            "n": len(sub), "validity_rate": sub["valid_int"].mean(),
            "mean_validity_score": sub["validity_score"].mean(),
        })
    for critic, g in d.groupby("critic"):
        for cond in ["self", "cross"]:
            sub = g[g["condition"] == cond]
            rows.append({
                "scope": "within_critic", "critic": critic, "condition": cond,
                "n": len(sub),
                "validity_rate": sub["valid_int"].mean() if len(sub) else np.nan,
                "mean_validity_score": sub["validity_score"].mean() if len(sub) else np.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ1 / RQ3 mixed-effects logistic
# ---------------------------------------------------------------------------


def glmm_validity(df: pd.DataFrame, item_col: str = "item_id",
                  include_quality: bool = True) -> dict:
    """Binomial GLMM: valid_int ~ is_cross [+ C(gold_quality)] + REs.

    Random intercepts for item, critic and author. Positive beta on ``is_cross``
    means cross-model critique is more valid than self (H1 direction).
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from scipy.stats import norm

    d = df.dropna(subset=["valid_int"]).copy()
    d = d[d[item_col].notna()]
    # Need both conditions present and variation in the outcome.
    if d["condition"].nunique() < 2 or d["valid_int"].nunique() < 2:
        return {"error": "insufficient variation for GLMM"}

    use_quality = (
        include_quality
        and "gold_quality" in d.columns
        and d["gold_quality"].nunique() > 1
    )
    formula = "valid_int ~ is_cross" + (" + C(gold_quality)" if use_quality else "")
    vc = {
        "item": f"0 + C({item_col})",
        "critic": "0 + C(critic)",
        "author": "0 + C(author)",
    }
    md = BinomialBayesMixedGLM.from_formula(formula, vc, d)
    res = md.fit_vb()

    names = list(res.model.exog_names)
    idx = names.index("is_cross")
    beta = float(res.fe_mean[idx])
    se = float(res.fe_sd[idx])
    z = beta / se if se > 0 else float("nan")
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))

    vcp_names = list(res.model.vcp_names)
    vcp = np.asarray(res.vcp_mean)
    re_sd = {nm: float(np.exp(vcp[i])) for i, nm in enumerate(vcp_names)}

    return {
        "method": "Binomial GLMM (variational Bayes); REs: item, critic, author",
        "formula": formula,
        "n_obs": int(len(d)),
        "n_items": int(d[item_col].nunique()),
        "n_critics": int(d["critic"].nunique()),
        "beta_is_cross": beta,
        "se": se,
        "z": float(z),
        "p_value": p,
        "odds_ratio": float(np.exp(beta)),
        "or_ci_low": float(np.exp(beta - 1.96 * se)),
        "or_ci_high": float(np.exp(beta + 1.96 * se)),
        "random_effects_sd": re_sd,
    }


# ---------------------------------------------------------------------------
# RQ2 overcorrection
# ---------------------------------------------------------------------------


def attach_soundness(judge_df: pd.DataFrame, sound_records: list[dict]) -> pd.DataFrame:
    """Merge per-(author,item) judge soundness onto the critique judgments."""
    s = pd.DataFrame(sound_records)
    if s.empty:
        judge_df = judge_df.copy()
        judge_df["resp_sound"] = np.nan
        return judge_df
    s = s[["author", "item_id", "sound", "quality_score"]].rename(
        columns={"sound": "resp_sound", "quality_score": "resp_quality_score"}
    )
    return judge_df.merge(s, on=["author", "item_id"], how="left")


def overcorrection_analysis(df: pd.DataFrame, use_gold_high: bool = True) -> dict:
    """Overcorrection = on a SOUND response, the critique recommends a change
    that would worsen it (correct->incorrect pressure).

    A response is treated as sound if the judge rated it sound, OR (fallback)
    if the dataset gold_quality is 'high'. Compares self vs cross.
    """
    d = df.copy()
    sound_mask = pd.Series(False, index=d.index)
    if "resp_sound" in d.columns:
        sound_mask = sound_mask | (d["resp_sound"] == True)  # noqa: E712
    if use_gold_high and "gold_quality" in d.columns:
        sound_mask = sound_mask | (d["gold_quality"] == "high")
    sound = d[sound_mask].copy()

    def _oc(row):
        rc = row.get("recommends_change")
        cw = row.get("change_would_worsen")
        if rc is None and cw is None:
            return np.nan
        return 1 if (rc and cw) else 0

    sound["overcorrect"] = sound.apply(_oc, axis=1)
    sound = sound.dropna(subset=["overcorrect"])

    rates = {}
    for cond in ["self", "cross"]:
        sub = sound[sound["condition"] == cond]
        rates[cond] = {
            "n": int(len(sub)),
            "overcorrection_rate": float(sub["overcorrect"].mean()) if len(sub) else float("nan"),
        }

    # Paired McNemar: per sound response (author,item) compare self vs majority cross.
    mcnemar = _paired_mcnemar(sound, "overcorrect")

    # Mixed logistic (item + critic REs) when there is variation.
    glmm = {"error": "skipped"}
    try:
        if sound["overcorrect"].nunique() > 1 and sound["condition"].nunique() > 1:
            glmm = _glmm_binary(sound, "overcorrect")
    except Exception as e:  # noqa: BLE001
        glmm = {"error": str(e)}

    diff = rates["self"]["overcorrection_rate"] - rates["cross"]["overcorrection_rate"]
    return {
        "n_sound": int(len(sound)),
        "rates": rates,
        "self_minus_cross": diff,
        "mcnemar": mcnemar,
        "glmm": glmm,
    }


def helpful_correction_rate(df: pd.DataFrame) -> dict:
    """I->C helpful-correction rate on NON-sound responses, for context."""
    d = df.copy()
    flawed_mask = pd.Series(False, index=d.index)
    if "resp_sound" in d.columns:
        flawed_mask = flawed_mask | (d["resp_sound"] == False)  # noqa: E712
    if "gold_quality" in d.columns:
        flawed_mask = flawed_mask | (d["gold_quality"].isin(["low", "medium"]))
    flawed = d[flawed_mask].copy()

    def _helpful(row):
        rc = row.get("recommends_change")
        cw = row.get("change_would_worsen")
        rf = row.get("identifies_real_flaw")
        if rc is None:
            return np.nan
        return 1 if (rc and rf and not cw) else 0

    flawed["helpful"] = flawed.apply(_helpful, axis=1)
    flawed = flawed.dropna(subset=["helpful"])
    rates = {}
    for cond in ["self", "cross"]:
        sub = flawed[flawed["condition"] == cond]
        rates[cond] = {
            "n": int(len(sub)),
            "helpful_rate": float(sub["helpful"].mean()) if len(sub) else float("nan"),
        }
    return {"n_flawed": int(len(flawed)), "rates": rates}


# ---------------------------------------------------------------------------
# Length control (verbosity bias check)
# ---------------------------------------------------------------------------


def length_control(df: pd.DataFrame) -> dict:
    """Check whether validity is explained by critique verbosity, and whether
    the self/cross effect survives controlling for length."""
    import statsmodels.formula.api as smf

    d = df.dropna(subset=["valid_int", "critique_len"]).copy()
    if len(d) < 30 or d["valid_int"].nunique() < 2:
        return {"error": "insufficient data"}
    d["len_z"] = (d["critique_len"] - d["critique_len"].mean()) / (
        d["critique_len"].std(ddof=0) or 1.0
    )
    corr = float(np.corrcoef(d["critique_len"], d["valid_int"])[0, 1])
    try:
        m = smf.logit("valid_int ~ is_cross + len_z", data=d).fit(disp=False)
        return {
            "corr_len_validity": corr,
            "beta_is_cross_controlling_len": float(m.params.get("is_cross", np.nan)),
            "p_is_cross_controlling_len": float(m.pvalues.get("is_cross", np.nan)),
            "beta_len_z": float(m.params.get("len_z", np.nan)),
            "p_len_z": float(m.pvalues.get("len_z", np.nan)),
        }
    except Exception as e:  # noqa: BLE001
        return {"corr_len_validity": corr, "error": str(e)}


# ---------------------------------------------------------------------------
# Judge-human kappa
# ---------------------------------------------------------------------------


def judge_kappa(val_records: list[dict], gold_col: str, judge_col: str = "judge_valid") -> dict:
    from .judge import cohens_kappa

    d = pd.DataFrame(val_records)
    d = d.dropna(subset=[gold_col, judge_col])
    if len(d) == 0:
        return {"error": "no validation records", "n": 0}
    a = d[gold_col].astype(int).values
    b = d[judge_col].astype(int).values
    k = cohens_kappa(a, b)
    return {
        "n": int(len(d)),
        "cohens_kappa": float(k),
        "judge_human_agreement": float((a == b).mean()),
        "gold_positive_rate": float(a.mean()),
        "judge_positive_rate": float(b.mean()),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _paired_mcnemar(sound: pd.DataFrame, outcome: str) -> dict:
    """Pair self vs (majority) cross per sound response; McNemar exact test."""
    pairs = []
    # Use only key columns that are actually populated in this subset, otherwise
    # pandas drops all-NaN groups (e.g. msg_id is null for the general domain).
    key_cols = [
        c for c in ["author", "item_id", "msg_id"]
        if c in sound.columns and sound[c].notna().any()
    ]
    for _, g in sound.groupby(key_cols, dropna=False):
        self_rows = g[g["condition"] == "self"]
        cross_rows = g[g["condition"] == "cross"]
        if len(self_rows) == 0 or len(cross_rows) == 0:
            continue
        s = int(self_rows[outcome].iloc[0])
        c = int(round(cross_rows[outcome].mean()))  # majority of cross critics
        pairs.append((s, c))
    if not pairs:
        return {"error": "no pairs", "n_pairs": 0}
    b = sum(1 for s, c in pairs if s == 1 and c == 0)  # self yes, cross no
    cc = sum(1 for s, c in pairs if s == 0 and c == 1)  # self no, cross yes
    n_disc = b + cc
    if n_disc == 0:
        return {"n_pairs": len(pairs), "self_only": b, "cross_only": cc,
                "p_value": 1.0, "note": "no discordant pairs"}
    # Exact binomial McNemar
    p = float(stats.binomtest(b, n_disc, 0.5).pvalue)
    return {
        "n_pairs": len(pairs),
        "self_only": b,         # overcorrects under self but not cross
        "cross_only": cc,
        "p_value": p,
    }


def _glmm_binary(d: pd.DataFrame, outcome: str) -> dict:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from scipy.stats import norm

    item_col = "item_id" if "item_id" in d.columns and d["item_id"].notna().any() else "msg_id"
    dd = d.dropna(subset=[outcome, item_col]).copy()
    dd[outcome] = dd[outcome].astype(int)
    vc = {"item": f"0 + C({item_col})", "critic": "0 + C(critic)"}
    md = BinomialBayesMixedGLM.from_formula(f"{outcome} ~ is_cross", vc, dd)
    res = md.fit_vb()
    names = list(res.model.exog_names)
    idx = names.index("is_cross")
    beta = float(res.fe_mean[idx])
    se = float(res.fe_sd[idx])
    z = beta / se if se > 0 else float("nan")
    return {
        "n_obs": int(len(dd)),
        "beta_is_cross": beta, "se": se, "z": float(z),
        "p_value": float(2.0 * (1.0 - norm.cdf(abs(z)))),
        "odds_ratio": float(np.exp(beta)),
    }
