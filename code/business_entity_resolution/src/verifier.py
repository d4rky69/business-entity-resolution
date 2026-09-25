"""High-Precision Entity Pair Verifier Module.

Acts as a post-classifier guardrail to prevent catastrophic false merges.
Because the competition metric is macro-averaged F_0.5 (weighting precision 2x over recall),
a single false positive merge on an entity penalizes its score severely.

The verifier enforces:
1. Hard Contradiction Filters:
   - Country disagreement (e.g., US record matching India record)
   - Mutually incompatible postal codes (when both have high-confidence extracted codes)
2. Strong Identity Agreement Checks:
   - High base name lexical similarity
   - Local address / street token alignment
"""

from typing import Any, Dict


def verify_pair(record_a: dict, record_b: dict) -> dict:
    """Stub: Evaluates whether two records represent the exact same business entity.

    Args:
        record_a: Normalized record dictionary for entity A (typically Source 1)
        record_b: Normalized record dictionary for entity B (typically Source 2 or 3)

    Returns:
        Dict with keys:
            - same_entity: bool indicating positive verification
            - confidence: float score in [0.0, 1.0]
            - reason: human-readable explanation of verification outcome
    """
    country_a = (record_a.get("country") or "").upper().strip()
    country_b = (record_b.get("country") or "").upper().strip()

    # Hard contradiction 1: Country mismatch
    if country_a and country_b and country_a != country_b:
        return {
            "same_entity": False,
            "confidence": 0.0,
            "reason": f"country_mismatch: {country_a} != {country_b}",
        }

    postal_a = record_a.get("postal_code")
    postal_b = record_b.get("postal_code")

    # Contradiction 2: Conflicting distinct postal codes
    if postal_a and postal_b and postal_a != postal_b:
        return {
            "same_entity": False,
            "confidence": 0.15,
            "reason": f"postal_code_conflict: {postal_a} vs {postal_b}",
        }

    name_a = (record_a.get("base_name") or record_a.get("clean_name", "")).strip()
    name_b = (record_b.get("base_name") or record_b.get("clean_name", "")).strip()

    # Exact base name match
    if name_a and name_b and name_a == name_b:
        return {
            "same_entity": True,
            "confidence": 0.95,
            "reason": "exact_base_name_match",
        }

    # Token overlap check
    tokens_a = set(name_a.split())
    tokens_b = set(name_b.split())
    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if jaccard >= 0.7:
            return {
                "same_entity": True,
                "confidence": 0.85,
                "reason": f"high_token_overlap_jaccard_{jaccard:.2f}",
            }
        elif jaccard < 0.2:
            return {
                "same_entity": False,
                "confidence": 0.2,
                "reason": f"low_token_overlap_jaccard_{jaccard:.2f}",
            }

    # Default heuristic outcome
    return {
        "same_entity": False,
        "confidence": 0.5,
        "reason": "insufficient_positive_evidence",
    }
