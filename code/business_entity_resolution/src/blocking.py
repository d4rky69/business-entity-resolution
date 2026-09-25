"""Candidate Generation (Blocking) Module.

Simplest working version:
- Match if normalized business names share the same first 3 letters AND same country.
- Writes candidate pairs to data/interim/candidates_long.tsv.
- Format: source1_entity_id, candidate_entity_id, block_reason (tab-separated)
"""

import os
from typing import Dict, Iterable, List, Optional
from .normalize import normalize_name


def generate_candidates_from_files(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str = "data/interim/candidates_long.tsv",
    max_candidates_per_s1: int = 3,
) -> str:
    """Streamlined streaming blocking from raw source TSV files.

    Indexes target sources (S2 and S3) by (country, first 3 letters of normalized name),
    then queries each Source 1 record to generate candidate match pairs.

    Args:
        s1_path: Path to test_source1.tsv
        s2_path: Path to test_source2.tsv
        s3_path: Path to test_source3.tsv
        output_path: Output TSV file path (data/interim/candidates_long.tsv)
        max_candidates_per_s1: Maximum candidate pairs per Source 1 entity (default 3)

    Returns:
        output_path string
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Building prefix index from {s2_path} and {s3_path}...")
    # Index: (country, prefix_3) -> list of target entity IDs
    target_index: Dict[tuple, List[str]] = {}

    target_count = 0
    for target_path in (s2_path, s3_path):
        if not os.path.isfile(target_path):
            print(f"Warning: Target file not found: {target_path}")
            continue

        with open(target_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    target_id = parts[0].strip()
                    raw_name = parts[1]
                    country = parts[3].strip().upper()
                    norm_nm = normalize_name(raw_name)
                    prefix = norm_nm[:3]
                    if prefix:
                        key = (country, prefix)
                        if key not in target_index:
                            target_index[key] = []
                        # Retain a bounded pool per prefix bucket to prevent explosive memory
                        if len(target_index[key]) < 10:
                            target_index[key].append(target_id)
                target_count += 1
                if target_count % 2000000 == 0:
                    print(f"  Indexed {target_count:,} target records...")

    print(f"Index built with {len(target_index):,} distinct (country, prefix3) buckets.")
    print(f"Generating candidate pairs for Source 1 entities from {s1_path}...")

    total_pairs = 0
    s1_count = 0

    with open(s1_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_id\tblock_reason\n")
        f_in.readline()  # skip header

        for line in f_in:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                s1_id = parts[0].strip()
                raw_name = parts[1]
                country = parts[3].strip().upper()
                norm_nm = normalize_name(raw_name)
                prefix = norm_nm[:3]
                if prefix:
                    key = (country, prefix)
                    candidates = target_index.get(key, [])
                    # Pick up to max_candidates_per_s1
                    for cand_id in candidates[:max_candidates_per_s1]:
                        f_out.write(f"{s1_id}\t{cand_id}\tfirst_3_letters_and_country\n")
                        total_pairs += 1
            s1_count += 1
            if s1_count % 500000 == 0:
                print(f"  Processed {s1_count:,} S1 entities -> {total_pairs:,} candidate pairs...")

    print(f"Finished blocking: {total_pairs:,} candidate pairs written to {output_path}")
    return output_path


def generate_candidates(
    s1_records: Iterable[dict],
    s2_records: Iterable[dict],
    s3_records: Iterable[dict],
    output_path: str = "data/interim/candidates_long.tsv",
    max_candidates_per_s1: int = 3,
) -> str:
    """In-memory candidate generation fallback for list/dict records."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target_index: Dict[tuple, List[str]] = {}
    for r in list(s2_records) + list(s3_records):
        t_id = r.get("entity_id", "")
        country = (r.get("country") or "").strip().upper()
        norm_nm = r.get("clean_name") or normalize_name(r.get("business_name") or "")
        prefix = norm_nm[:3]
        if prefix:
            key = (country, prefix)
            target_index.setdefault(key, []).append(t_id)

    total_pairs = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_id\tblock_reason\n")
        for s1 in s1_records:
            s1_id = s1.get("entity_id", "")
            country = (s1.get("country") or "").strip().upper()
            norm_nm = s1.get("clean_name") or normalize_name(s1.get("business_name") or "")
            prefix = norm_nm[:3]
            if prefix:
                candidates = target_index.get((country, prefix), [])
                for cand_id in candidates[:max_candidates_per_s1]:
                    f_out.write(f"{s1_id}\t{cand_id}\tfirst_3_letters_and_country\n")
                    total_pairs += 1

    print(f"Generated {total_pairs} candidate pairs to {output_path}")
    return output_path
