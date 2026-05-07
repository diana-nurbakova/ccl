"""CCL category mappings from external error taxonomies."""

# ---------------------------------------------------------------------------
# FRANK benchmark  →  CCL claim-type categories
# ---------------------------------------------------------------------------
# FRANK error codes (Pagnoni et al., NAACL 2021):
#   NoE   = No Error
#   EntE  = Entity Error        (wrong entity)
#   OutE  = Out-of-article Error (hallucinated fact)
#   GramE = Grammatical Error    (syntactic distortion changing meaning)
#   CircE = Circumstantial Error (wrong circumstance / context)
#   RelE  = Relation Error       (wrong relation between entities)
#   LinkE = Link Error           (discourse link error)
#   CorefE= Coreference Error    (wrong coreference resolution)

FRANK_TO_CCL: dict[str, list[str]] = {
    "FACTUAL":      ["EntE", "OutE", "GramE"],
    "SOURCE":       ["CircE"],
    "INTERPRETIVE": ["RelE", "LinkE", "CorefE"],
}

# Reverse lookup: FRANK code → CCL category
FRANK_CODE_TO_CCL: dict[str, str] = {
    code: cat
    for cat, codes in FRANK_TO_CCL.items()
    for code in codes
}

# No-error sentinel (excluded from CCL mapping)
FRANK_NO_ERROR = "NoE"

# All valid FRANK error codes (excluding NoE)
FRANK_ERROR_CODES = set(FRANK_CODE_TO_CCL.keys())

# ---------------------------------------------------------------------------
# WMT MQM  →  CCL (for potential future extension)
# ---------------------------------------------------------------------------
MQM_TO_CCL: dict[str, list[str]] = {
    "FACTUAL":              ["Accuracy/Mistranslation"],
    "MISSING_PERSPECTIVE":  ["Accuracy/Omission"],
    "INTERPRETIVE":         ["Style"],
}

# ---------------------------------------------------------------------------
# CCL display metadata
# ---------------------------------------------------------------------------
CCL_CATEGORIES = ["FACTUAL", "SOURCE", "INTERPRETIVE"]

CCL_SYMBOLS: dict[str, str] = {
    "FACTUAL":      "■",
    "SOURCE":       "♦",
    "INTERPRETIVE": "▲",
    "MISSING":      "•",
}

CCL_COLORS: dict[str, str] = {
    "FACTUAL":      "#2196F3",   # blue
    "SOURCE":       "#FF9800",   # orange
    "INTERPRETIVE": "#4CAF50",   # green
    "MISSING":      "#9E9E9E",   # grey
}


# ---------------------------------------------------------------------------
# FELM benchmark  →  CCL claim-type categories
# ---------------------------------------------------------------------------
# FELM domains (actual names in dataset):
#   wk           = World Knowledge  → FACTUAL  (verifiable entity/date/number)
#   math         = Math             → FACTUAL  (verifiable computation)
#   science      = Science/Tech     → INTERPRETIVE (causal/mechanistic reasoning)
#   reasoning    = Reasoning        → INTERPRETIVE (logical inference, framing)
#   writing_rec  = Writing/Rec      → GAP/PERSPECTIVAL (subjective, missing perspective)

FELM_DOMAIN_TO_CCL: dict[str, str] = {
    "wk":          "FACTUAL",
    "math":        "FACTUAL",
    "science":     "INTERPRETIVE",
    "reasoning":   "INTERPRETIVE",
    "writing_rec": "GAP",
}

# FELM error types that appear in the `type` field
FELM_ERROR_TYPES = [
    "knowledge_error",
    "reasoning_error",
    "calculation_error",
    "output_format_error",
    "irrelevant_with_qst",
    "commonsense_error",
]

# For colour consistency in plots, map GAP to the MISSING colour
CCL_COLORS["GAP"] = CCL_COLORS["MISSING"]


def classify_frank_errors(annotator_errors: list[str]) -> set[str]:
    """Map a list of FRANK error codes to CCL categories.

    Parameters
    ----------
    annotator_errors : list[str]
        Error codes from a single annotator for one sentence,
        e.g. ["EntE", "RelE"] or ["NoE"].

    Returns
    -------
    set[str]
        CCL categories triggered, e.g. {"FACTUAL", "INTERPRETIVE"}.
        Empty set if only NoE is present.
    """
    categories = set()
    for code in annotator_errors:
        if code in FRANK_CODE_TO_CCL:
            categories.add(FRANK_CODE_TO_CCL[code])
    return categories
