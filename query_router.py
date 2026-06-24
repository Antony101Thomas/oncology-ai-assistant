import re

# Keywords that signal the user wants the latest/recent evidence
RECENCY_SIGNALS = [
    "latest", "recent", "new", "updated", "2023", "2024", "2025",
    "current", "emerging", "just approved", "newly", "this year",
]

# Keywords that signal a precision / exact-match lookup
PRECISION_SIGNALS = [
    "nct", "trial id", "pmid", "doi", "find trial", "find study",
    "specific trial", "study number", "protocol",
]

# Patterns that look like NCT IDs or PubMed IDs
PRECISION_PATTERNS = [
    r"\bNCT\d{6,}\b",     # ClinicalTrials ID e.g. NCT02296125
    r"\bPMID\s*\d+\b",    # PubMed ID
    r"\bdoi:\S+\b",       # DOI
]


def classify_query(question: str) -> str:
    """
    Classify an oncology question into one of three routing categories:

    - "precision"   : exact lookup by trial ID, PMID, DOI, or specific code
    - "recency"     : wants the latest / most recent evidence
    - "conceptual"  : general medical / scientific question

    Returns one of: "precision", "recency", "conceptual"
    """
    q = question.lower()

    # Check for precision patterns first (highest priority)
    for pattern in PRECISION_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            print(f"[Router] precision  ← matched pattern '{pattern}'")
            return "precision"

    for signal in PRECISION_SIGNALS:
        if signal in q:
            print(f"[Router] precision  ← matched keyword '{signal}'")
            return "precision"

    # Check for recency signals
    for signal in RECENCY_SIGNALS:
        if signal in q:
            print(f"[Router] recency    ← matched keyword '{signal}'")
            return "recency"

    # Default: treat as a conceptual/general question
    print("[Router] conceptual ← no precision or recency signals found")
    return "conceptual"


def should_call_live_apis(route: str, fused_results: list) -> bool:
    """
    Decide whether to call PubMed / ClinicalTrials.gov based on:
    - route == 'recency' → always call live APIs
    - route == 'precision' → always call live APIs (exact lookup needed)
    - route == 'conceptual' with weak local evidence → call live APIs as backup
    """
    if route in ("recency", "precision"):
        return True
    # Conceptual: call live APIs only if local evidence is thin
    return len(fused_results) < 2


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("What is HER2-low metastatic breast cancer?",          "conceptual"),
        ("Find trial NCT02296125",                              "precision"),
        ("What are the latest KRAS G12C results in 2024?",      "recency"),
        ("PMID 35665782",                                       "precision"),
        ("What is immunotherapy?",                              "conceptual"),
        ("Recent PubMed articles about lung cancer treatment",  "recency"),
        ("doi:10.1056/NEJMoa2206307",                          "precision"),
    ]

    print("── Query routing tests ──")
    all_pass = True
    for question, expected in tests:
        result = classify_query(question)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"  {status}  [{result:12s}]  {question}")

    print()
    print("All tests passed!" if all_pass else "Some tests FAILED — check logic above.")
