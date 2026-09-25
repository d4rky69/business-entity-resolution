"""End-to-End Inference and Submission Aggregation Module.

Chains all pipeline stages:
1. Load raw test datasets (test_source1.tsv, test_source2.tsv, test_source3.tsv)
2. Normalize records using country-specific and generic legal suffixes
3. Generate candidate pairs via multi-pass blocking
4. Compute dense pairwise similarity features
5. Run trained matching model / verifier scoring
6. Aggregate results into official submission TSVs:
   - output/matching_results.tsv  (source1_entity_id, matched_entity_ids)
   - output/candidate_pairs.tsv   (source1_entity_id, candidate_entity_ids)
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Set
import pandas as pd

from .normalize import get_source, normalize_record
from .blocking import generate_candidates
from .features import compute_features
from .verifier import verify_pair


def load_raw_tsv(file_path: str) -> List[dict]:
    """Reads raw source TSV file with explicit tab delimiter into a list of record dicts.

    Columns expected: entity_id, business_name, business_address, country
    Note: Raw files do not contain a source column.
    """
    if not os.path.isfile(file_path):
        print(f"Warning: File not found: {file_path}")
        return []
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    return df.to_dict(orient="records")


def aggregate_to_submission_format(
    long_df: pd.DataFrame,
    id_col_name: str,
    all_s1_ids: Iterable[str],
    output_path: str,
) -> None:
    """Writes the official one-row-per-source1_entity TSV file.

    Guarantees full compliance with competition validator rules:
    - Exactly one row for every Source 1 entity in all_s1_ids
    - Header is exactly: source1_entity_id\t{id_col_name}
    - Tab-separated (.tsv) explicitly (sep='\\t')
    - Comma-separated target IDs with no internal duplicate IDs
    - No self-matches (no S1 IDs in target list)
    - Empty string for entities with zero matches/candidates

    Args:
        long_df: DataFrame containing at least ['source1_entity_id', 'target_entity_id']
        id_col_name: Column name for list ('matched_entity_ids' or 'candidate_entity_ids')
        all_s1_ids: Full universe of Source 1 entity IDs that must be included
        output_path: Destination file path (e.g. output/matching_results.tsv)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Aggregate target IDs grouped by source1_entity_id
    grouped_map: Dict[str, List[str]] = {}
    if not long_df.empty and "source1_entity_id" in long_df.columns:
        # Determine candidate/target ID column
        target_col = "candidate_entity_id" if "candidate_entity_id" in long_df.columns else "target_entity_id"
        if target_col in long_df.columns:
            for s1_id, group in long_df.groupby("source1_entity_id"):
                s1_id_str = str(s1_id).strip()
                seen_ids: Set[str] = set()
                clean_target_ids: List[str] = []
                for tid in group[target_col].dropna():
                    tid_str = str(tid).strip()
                    # Filter out self matches and duplicates
                    if tid_str and not tid_str.startswith("S1-") and tid_str not in seen_ids:
                        seen_ids.add(tid_str)
                        clean_target_ids.append(tid_str)
                grouped_map[s1_id_str] = clean_target_ids

    # Build full output for EVERY required Source 1 entity in stable order
    rows = []
    for s1_id in sorted(all_s1_ids):
        s1_clean = str(s1_id).strip()
        matched = grouped_map.get(s1_clean, [])
        rows.append({
            "source1_entity_id": s1_clean,
            id_col_name: ",".join(matched),
        })

    out_df = pd.DataFrame(rows)
    # Write tab-separated TSV with no row index
    out_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Saved {len(out_df)} rows to {output_path} ({id_col_name})")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end business entity resolution inference.")
    parser.add_argument("--test-dir", default="data/raw/dataset/test", help="Directory with test_source1/2/3.tsv")
    parser.add_argument("--config", default="config/suffixes.json", help="Path to suffixes.json")
    parser.add_argument("--interim-dir", default="data/interim", help="Path to store interim working files")
    parser.add_argument("--output-dir", default="output", help="Directory to store submission TSV files")
    parser.add_argument("--model-path", default="models/matching_model.pkl", help="Trained model pickle")
    args = parser.parse_args()

    print(f"=== Starting Entity Resolution Pipeline ===")
    print(f"Test Directory: {args.test_dir}")
    print(f"Output Directory: {args.output_dir}")

    # Step 1: Load Legal Suffixes Configuration
    suffix_dict = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            suffix_dict = json.load(f)

    # Step 2: Load Raw Datasets
    s1_raw = load_raw_tsv(os.path.join(args.test_dir, "test_source1.tsv"))
    s2_raw = load_raw_tsv(os.path.join(args.test_dir, "test_source2.tsv"))
    s3_raw = load_raw_tsv(os.path.join(args.test_dir, "test_source3.tsv"))

    all_s1_ids = {r["entity_id"] for r in s1_raw if "entity_id" in r}
    print(f"Loaded {len(s1_raw)} S1, {len(s2_raw)} S2, {len(s3_raw)} S3 records.")

    # Step 3: Normalization
    print("Normalizing records...")
    norm_s1 = [normalize_record(r, suffix_dict) for r in s1_raw]
    norm_s2 = [normalize_record(r, suffix_dict) for r in s2_raw]
    norm_s3 = [normalize_record(r, suffix_dict) for r in s3_raw]

    norm_lookup = {r["entity_id"]: r for r in norm_s1 + norm_s2 + norm_s3}

    # Step 4: Candidate Generation (Blocking)
    print("Generating candidate pairs (blocking)...")
    cand_long_path = os.path.join(args.interim_dir, "candidates_long.tsv")
    generate_candidates(norm_s1, norm_s2, norm_s3, output_path=cand_long_path)

    # Step 5: Feature Extraction
    print("Computing pairwise features...")
    feat_path = os.path.join(args.interim_dir, "features.tsv")
    compute_features(candidates_long_path=cand_long_path, normalized_records=norm_lookup, output_path=feat_path)

    # Step 6: Scoring & High-Precision Verification
    # candidate_pairs.tsv must reflect the LAST-stage candidate set scored by the classifier
    cand_df = pd.read_csv(cand_long_path, sep="\t", dtype=str) if os.path.isfile(cand_long_path) else pd.DataFrame()

    matched_pairs = []
    if not cand_df.empty:
        for _, row in cand_df.iterrows():
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]
            v_res = verify_pair(norm_lookup.get(s1_id, {}), norm_lookup.get(cand_id, {}))
            if v_res.get("same_entity", False) and v_res.get("confidence", 0.0) >= 0.7:
                matched_pairs.append({
                    "source1_entity_id": s1_id,
                    "target_entity_id": cand_id,
                })

    matched_df = pd.DataFrame(matched_pairs)

    # Step 7: Write OFFICIAL submission outputs
    matching_out = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")

    print("Formatting submission outputs...")
    aggregate_to_submission_format(matched_df, "matched_entity_ids", all_s1_ids, matching_out)
    aggregate_to_submission_format(cand_df, "candidate_entity_ids", all_s1_ids, candidate_out)

    print("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
