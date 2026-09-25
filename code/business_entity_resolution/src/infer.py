"""End-to-End Inference and Submission Aggregation Module.

Simplest working pipeline:
1. Generate candidates using blocking.py (same first 3 letters of normalized name + same country)
2. Read all candidate pairs from data/interim/candidates_long.tsv
3. Treat every candidate as an accepted match (baseline run, no classifier)
4. Stream-aggregate into the official submission TSVs:
   - output/matching_results.tsv (columns: source1_entity_id, matched_entity_ids)
   - output/candidate_pairs.tsv  (columns: source1_entity_id, candidate_entity_ids)
"""

import argparse
import os
import sys
import time
from typing import Dict, Iterable, List, Set

from .blocking import generate_candidates_from_files


def aggregate_to_submission_format(
    candidates_long_path: str,
    s1_source_path: str,
    matching_output_path: str = "output/matching_results.tsv",
    candidate_output_path: str = "output/candidate_pairs.tsv",
) -> None:
    """Stream-aggregates candidates into official one-row-per-source1_entity TSVs.

    Guarantees strict compliance with validator rules:
    - Every S1 entity in test_source1.tsv appears on exactly one row.
    - Headers are exactly:
        matching_results.tsv -> source1_entity_id\\tmatched_entity_ids
        candidate_pairs.tsv  -> source1_entity_id\\tcandidate_entity_ids
    - Separated by TAB ('\\t').
    - Empty string for S1 entities with no matches/candidates.
    - Comma-separated list with no internal duplicates and no S1 self-matches.

    Args:
        candidates_long_path: Path to data/interim/candidates_long.tsv
        s1_source_path: Path to test_source1.tsv (to ensure all S1 entities appear)
        matching_output_path: Destination for matching_results.tsv
        candidate_output_path: Destination for candidate_pairs.tsv
    """
    os.makedirs(os.path.dirname(matching_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(candidate_output_path), exist_ok=True)

    print(f"Reading candidates from {candidates_long_path}...")
    # Map: source1_entity_id -> list of unique candidate_ids
    s1_candidates: Dict[str, List[str]] = {}

    if os.path.isfile(candidates_long_path):
        with open(candidates_long_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    s1 = parts[0].strip()
                    cand = parts[1].strip()
                    # Filter out self matches and duplicates
                    if cand and not cand.startswith("S1-"):
                        if s1 not in s1_candidates:
                            s1_candidates[s1] = [cand]
                        elif cand not in s1_candidates[s1]:
                            s1_candidates[s1].append(cand)

    print(f"Aggregated candidates for {len(s1_candidates):,} distinct Source 1 entities.")
    print(f"Streaming final TSVs for all Source 1 entities from {s1_source_path}...")

    total_rows = 0
    non_empty_rows = 0

    with open(s1_source_path, "r", encoding="utf-8") as f_s1, \
         open(matching_output_path, "w", encoding="utf-8") as f_match, \
         open(candidate_output_path, "w", encoding="utf-8") as f_cand:

        # Write exact required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        f_s1.readline()  # skip header in test_source1.tsv

        for line in f_s1:
            parts = line.split("\t", 1)
            s1_id = parts[0].strip()
            if not s1_id:
                continue

            cands = s1_candidates.get(s1_id, [])
            cand_str = ",".join(cands)

            # Write one row per S1 entity
            f_match.write(f"{s1_id}\t{cand_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")

            total_rows += 1
            if cand_str:
                non_empty_rows += 1

    print(f"Generated {total_rows:,} rows ({non_empty_rows:,} non-empty, {total_rows - non_empty_rows:,} empty)")
    print(f"  -> Matching results saved: {matching_output_path}")
    print(f"  -> Candidate pairs saved:  {candidate_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline end-to-end entity resolution pipeline.")
    parser.add_argument("--test-dir", default="data/raw/dataset/test", help="Path to test datasets directory")
    parser.add_argument("--interim-dir", default="data/interim", help="Path to interim directory")
    parser.add_argument("--output-dir", default="output", help="Path to final submission directory")
    parser.add_argument("--max-candidates-per-s1", type=int, default=3, help="Max candidates per S1 entity")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("Starting Baseline Business Entity Resolution Pipeline")
    print(f"Test Directory:     {args.test_dir}")
    print(f"Interim Directory:  {args.interim_dir}")
    print(f"Output Directory:   {args.output_dir}")
    print("=" * 60)

    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")
    cand_path = os.path.join(args.interim_dir, "candidates_long.tsv")

    # Step 1: Candidate Generation (Blocking on first 3 letters of normalized name + country)
    print("\n[Step 1/2] Candidate Generation (Blocking)...")
    generate_candidates_from_files(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_path=cand_path,
        max_candidates_per_s1=args.max_candidates_per_s1,
    )

    # Step 2: Aggregation to official submission TSV formats
    print("\n[Step 2/2] Aggregation to Official Submission Format...")
    matching_out = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")

    aggregate_to_submission_format(
        candidates_long_path=cand_path,
        s1_source_path=s1_path,
        matching_output_path=matching_out,
        candidate_output_path=candidate_out,
    )

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Pipeline Completed Successfully in {elapsed:.2f}s!")
    print(f"Official Submission Files:")
    print(f"  1. {matching_out} (Leaderboard matching results)")
    print(f"  2. {candidate_out} (Blocking candidate pairs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
