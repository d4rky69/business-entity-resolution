"""Candidate Generation (Blocking) Module.

Scalable multi-pass candidate generation to reduce comparison space from O(N_S1 * (N_S2 + N_S3))
to manageable, high-recall candidate pairs for the downstream matching classifier.

Blocking Strategies:
1. Standard Key Blocking: Exact postal code / PIN match + first letter of business name
2. Token Inverted Index Blocking: Shared rare name/address n-grams or tokens
3. Phonetic Blocking: Soundex / Metaphone on primary name token within same country
4. Embedding / ANN Blocking: Approximate nearest neighbors via vector index (FAISS)

Writes internal working format:
    data/interim/candidates_long.tsv
    Columns: source1_entity_id, candidate_entity_id, block_reason
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import pandas as pd


def generate_candidates(
    s1_records: Iterable[dict],
    s2_records: Iterable[dict],
    s3_records: Iterable[dict],
    output_path: str = "data/interim/candidates_long.tsv",
    max_candidates_per_s1: int = 50,
) -> str:
    """Stub: Multi-pass candidate generation across Source 1 and Target Sources (S2, S3).

    Generates candidate match pairs using indexing and multi-pass blocking keys.
    Deduplicates (S1, target) pairs while concatenating the triggering block reasons.

    Args:
        s1_records: Iterable of normalized Source 1 records (deduplicated anchor entities)
        s2_records: Iterable of normalized Source 2 records
        s3_records: Iterable of normalized Source 3 records
        output_path: Destination TSV path for internal candidate pair records
        max_candidates_per_s1: Guardrail cap on maximum candidates retained per anchor

    Returns:
        output_path where candidate pairs were written
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # In-memory candidate storage: (s1_id, cand_id) -> list of block_reasons
    candidates: Dict[Tuple[str, str], List[str]] = {}

    # Combine S2 and S3 target pools
    target_pool: List[dict] = list(s2_records) + list(s3_records)

    # Build lightweight inverted indexes over target pool
    # Index 1: (country, postal_code) -> [target_id]
    geo_index: Dict[Tuple[str, str], List[str]] = {}
    # Index 2: (country, first_token) -> [target_id]
    name_token_index: Dict[Tuple[str, str], List[str]] = {}

    for target in target_pool:
        t_id = target.get("entity_id", "")
        country = target.get("country", "")
        postal = target.get("postal_code")
        base_name = target.get("base_name") or target.get("clean_name", "")
        tokens = [t for t in base_name.split() if len(t) > 2]

        if postal:
            geo_index.setdefault((country, postal), []).append(t_id)

        if tokens:
            name_token_index.setdefault((country, tokens[0]), []).append(t_id)

    # Pass 1 & 2: Query candidate matches for each S1 entity
    for s1 in s1_records:
        s1_id = s1.get("entity_id", "")
        country = s1.get("country", "")
        postal = s1.get("postal_code")
        base_name = s1.get("base_name") or s1.get("clean_name", "")
        tokens = [t for t in base_name.split() if len(t) > 2]

        # Postal match
        if postal and (country, postal) in geo_index:
            for t_id in geo_index[(country, postal)]:
                candidates.setdefault((s1_id, t_id), []).append("postal_match")

        # Name token match
        if tokens and (country, tokens[0]) in name_token_index:
            for t_id in name_token_index[(country, tokens[0])]:
                candidates.setdefault((s1_id, t_id), []).append("name_token_match")

    # Format into tabular rows
    rows = []
    for (s1_id, cand_id), reasons in candidates.items():
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "block_reason": "|".join(sorted(set(reasons))),
        })

    candidates_df = pd.DataFrame(rows)
    if candidates_df.empty:
        candidates_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "block_reason"])

    # Explicitly write tab-separated TSV with no index
    candidates_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Generated {len(candidates_df)} candidate pairs saved to {output_path}")
    return output_path
