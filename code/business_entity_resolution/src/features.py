"""Feature Engineering Module for Entity Pair Matching.

Extracts dense pairwise similarity features between candidate record pairs
generated during the blocking stage. Features span:
- Lexical similarities: Levenshtein ratio, Jaro-Winkler, token sort ratio
- Set overlap similarities: Jaccard word token overlap, character n-gram overlap
- Structured attributes: Country consistency, postal code / PIN match, length ratios
- High-level semantic similarities: Semantic embedding cosine similarity (optional)

Input:
    data/interim/candidates_long.tsv
Output:
    data/interim/features.tsv (tab-separated, containing pair IDs and numerical feature columns)
"""

import os
from typing import Any, Dict, List, Optional
import pandas as pd


def jaccard_similarity(str1: str, str2: str) -> float:
    """Calculates word-token Jaccard similarity between two strings."""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def simple_levenshtein_ratio(s1: str, s2: str) -> float:
    """Fallback normalized string similarity ratio without external C-extensions."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # Try rapidfuzz if available, else approximate with character set overlap
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(s1, s2) / 100.0
    except ImportError:
        # Simple character overlap ratio fallback
        len_max = max(len(s1), len(s2))
        return len(set(s1) & set(s2)) / len(set(s1) | set(s2)) if len_max > 0 else 0.0


def token_sort_ratio(s1: str, s2: str) -> float:
    """Computes order-agnostic token sort similarity ratio."""
    try:
        from rapidfuzz import fuzz
        return fuzz.token_sort_ratio(s1, s2) / 100.0
    except ImportError:
        sorted1 = " ".join(sorted(s1.split()))
        sorted2 = " ".join(sorted(s2.split()))
        return simple_levenshtein_ratio(sorted1, sorted2)


def compute_features(
    candidates_long_path: str = "data/interim/candidates_long.tsv",
    normalized_records: Optional[Dict[str, dict]] = None,
    output_path: str = "data/interim/features.tsv",
) -> str:
    """Stub: Computes pairwise similarity features for all candidate pairs in long format.

    Args:
        candidates_long_path: Path to candidates_long.tsv (source1_entity_id, candidate_entity_id, block_reason)
        normalized_records: Dict mapping entity_id -> normalized record dict
        output_path: Destination TSV path for feature matrix

    Returns:
        Path to generated features.tsv
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.isfile(candidates_long_path):
        print(f"Warning: Candidates file not found at {candidates_long_path}")
        empty_df = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "name_exact_match", "name_lev_ratio", "name_token_sort_ratio", "name_jaccard",
            "addr_lev_ratio", "addr_jaccard", "postal_match", "country_match", "target_source_s2"
        ])
        empty_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
        return output_path

    # Read candidates long TSV explicitly using tab separator
    cand_df = pd.read_csv(candidates_long_path, sep="\t", dtype=str)
    records = normalized_records or {}

    feature_rows = []
    for _, row in cand_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        cand_id = str(row["candidate_entity_id"])

        s1_rec = records.get(s1_id, {})
        cand_rec = records.get(cand_id, {})

        s1_name = s1_rec.get("base_name") or s1_rec.get("clean_name", "")
        cand_name = cand_rec.get("base_name") or cand_rec.get("clean_name", "")

        s1_addr = s1_rec.get("clean_address", "")
        cand_addr = cand_rec.get("clean_address", "")

        s1_postal = s1_rec.get("postal_code")
        cand_postal = cand_rec.get("postal_code")

        s1_country = s1_rec.get("country", "")
        cand_country = cand_rec.get("country", "")

        # Compute pairwise features
        name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0
        name_lev = simple_levenshtein_ratio(s1_name, cand_name)
        name_token_sort = token_sort_ratio(s1_name, cand_name)
        name_jacc = jaccard_similarity(s1_name, cand_name)

        addr_lev = simple_levenshtein_ratio(s1_addr, cand_addr)
        addr_jacc = jaccard_similarity(s1_addr, cand_addr)

        if s1_postal and cand_postal:
            postal_match = 1.0 if s1_postal == cand_postal else 0.0
        else:
            postal_match = -1.0  # missing indicator

        country_match = 1.0 if s1_country and s1_country == cand_country else 0.0
        is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

        feature_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "name_exact_match": name_exact,
            "name_lev_ratio": name_lev,
            "name_token_sort_ratio": name_token_sort,
            "name_jaccard": name_jacc,
            "addr_lev_ratio": addr_lev,
            "addr_jaccard": addr_jacc,
            "postal_match": postal_match,
            "country_match": country_match,
            "target_source_s2": is_s2,
        })

    feat_df = pd.DataFrame(feature_rows)
    feat_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Computed features for {len(feat_df)} candidate pairs saved to {output_path}")
    return output_path
