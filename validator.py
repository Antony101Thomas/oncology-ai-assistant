import re

# Matches citation markers like [Source 1], [Source 2][Source 3], [source 12]
_CITATION_MARKER_RE = re.compile(r"\[\s*source\s*\d+\s*\]", re.IGNORECASE)


def _strip_citation_markers(text: str) -> str:
    """Remove [Source N] markers so they aren't mistaken for numeric/term claims."""
    return _CITATION_MARKER_RE.sub(" ", text)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric values from text (integers, decimals, percentages)."""
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", _strip_citation_markers(text))


def _source_texts(sources: list[dict]) -> str:
    """Combine all source texts into one searchable string."""
    parts = []
    for s in sources:
        # Support both dict shapes used in the project
        text = s.get("text") or s.get("abstract") or s.get("description") or ""
        parts.append(text)
    return " ".join(parts).lower()


# ── Numeric guard ─────────────────────────────────────────────────────────────

def numeric_guard(answer: str, sources: list[dict]) -> dict:
    """
    Check every number in the answer against the retrieved source texts.

    Returns:
        {
            "passed": bool,
            "checked": int,          # total numbers found in answer
            "supported": int,        # numbers found in sources
            "unsupported": list[str] # numbers NOT found in any source
        }
    """
    numbers = _extract_numbers(answer)

    if not numbers:
        return {
            "passed": True,
            "checked": 0,
            "supported": 0,
            "unsupported": [],
            "note": "No numeric claims to verify.",
        }

    combined = _source_texts(sources)
    unsupported = []

    for num in numbers:
        # Strip trailing % for a bare-number search too
        bare = num.rstrip("%")
        if num not in combined and bare not in combined:
            unsupported.append(num)

    passed = len(unsupported) == 0

    return {
        "passed": passed,
        "checked": len(numbers),
        "supported": len(numbers) - len(unsupported),
        "unsupported": unsupported,
    }


# ── Answer validation ─────────────────────────────────────────────────────────

# Phrases that signal the LLM admitted it couldn't answer
REFUSAL_PHRASES = [
    "i don't have enough evidence",
    "i cannot answer",
    "insufficient evidence",
    "not enough information",
    "no relevant evidence",
    "outside the scope",
    "i do not have",
    "cannot provide",
]

# Minimum word count for a valid answer (very short = likely a refusal or error)
MIN_ANSWER_WORDS = 10


def validate_answer(answer: str, sources: list[dict]) -> dict:
    """
    Validate whether the generated answer is actually supported by evidence.

    Checks:
    1. Answer is not a refusal / too short
    2. At least one key noun from the answer appears in the sources
    3. Numeric guard passes

    Returns:
        {
            "valid": bool,
            "status": "supported" | "partially_supported" | "unsupported" | "refused",
            "reason": str,
            "numeric_guard": dict,
        }
    """
    answer_lower = _strip_citation_markers(answer).lower().strip()

    # ── Check 1: Refusal detection ──
    for phrase in REFUSAL_PHRASES:
        if phrase in answer_lower:
            return {
                "valid": False,
                "status": "refused",
                "reason": f"Answer contains refusal phrase: '{phrase}'",
                "numeric_guard": numeric_guard(answer, sources),
            }

    # ── Check 2: Minimum length ──
    word_count = len(answer.split())
    if word_count < MIN_ANSWER_WORDS:
        return {
            "valid": False,
            "status": "unsupported",
            "reason": f"Answer too short ({word_count} words) — likely not substantive.",
            "numeric_guard": numeric_guard(answer, sources),
        }

    # ── Check 3: Key term overlap ──
    # Extract meaningful words from the answer (ignore stopwords)
    STOPWORDS = {
        "the","a","an","is","in","of","to","and","or","for","with",
        "on","at","by","from","as","are","was","were","be","been",
        "has","have","had","it","its","this","that","these","those",
        "which","who","what","how","when","where","can","may","also",
        "not","no","but","so","if","than","then","there","their","they",
        "we","our","you","your","about","will","would","could","should",
    }
    answer_words = {
        w for w in re.findall(r"[a-z]{4,}", answer_lower)
        if w not in STOPWORDS
    }

    combined = _source_texts(sources)
    matched = {w for w in answer_words if w in combined}

    if not answer_words:
        overlap_ratio = 0.0
    else:
        overlap_ratio = len(matched) / len(answer_words)

    # ── Check 4: Numeric guard ──
    ng = numeric_guard(answer, sources)

    # ── Decide status ──
    if overlap_ratio >= 0.4 and ng["passed"]:
        status = "supported"
        valid = True
        reason = (
            f"{len(matched)}/{len(answer_words)} key terms found in sources "
            f"({overlap_ratio:.0%} overlap). All numeric claims verified."
        )
    elif overlap_ratio >= 0.2 or (overlap_ratio >= 0.4 and not ng["passed"]):
        status = "partially_supported"
        valid = True
        reason = (
            f"{len(matched)}/{len(answer_words)} key terms found in sources "
            f"({overlap_ratio:.0%} overlap)."
        )
        if not ng["passed"]:
            reason += f" Unverified numbers: {ng['unsupported']}"
    else:
        status = "unsupported"
        valid = False
        reason = (
            f"Only {len(matched)}/{len(answer_words)} key terms found in sources "
            f"({overlap_ratio:.0%} overlap) — answer may not be grounded."
        )

    return {
        "valid": valid,
        "status": status,
        "reason": reason,
        "numeric_guard": ng,
    }


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sources = [
        {
            "text": (
                "HER2-low metastatic breast cancer is defined by IHC scores of 1+ or 2+ "
                "with negative ISH. It represents 55-60% of all metastatic breast cancers. "
                "Trastuzumab deruxtecan showed significant benefit in the DESTINY-Breast04 trial."
            )
        }
    ]

    print("── Numeric guard tests ──")
    ans1 = "HER2-low breast cancer represents 55-60% of metastatic cases."
    print("Answer:", ans1)
    print("Result:", numeric_guard(ans1, sources))

    ans2 = "HER2-low breast cancer affects 90% of all patients worldwide."
    print("\nAnswer:", ans2)
    print("Result:", numeric_guard(ans2, sources))

    print("\n── Validate answer tests ──")
    good = (
        "HER2-low metastatic breast cancer is defined by IHC scores of 1+ or 2+ "
        "with negative ISH, representing 55-60% of all metastatic breast cancers."
    )
    print("GOOD answer:", validate_answer(good, sources))

    refused = "I don't have enough evidence to answer this question."
    print("\nREFUSAL:    ", validate_answer(refused, sources))

    hallucinated = "The sky is purple and dogs can fly at 200 miles per hour."
    print("\nHALLUCINATED:", validate_answer(hallucinated, sources))
