"""ML Challenge 2026 - Official F_0.5 Evaluation Metric.

Computes the macro-averaged F_0.5 score per Source 1 entity across the evaluation set:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Precision is weighted 2x over recall (beta = 0.5) to heavily penalize false merges.
Singletons (Source 1 entities with no true matches) score:
  - 1.0 if correctly predicted with an empty match list
  - 0.0 if any match was predicted
"""

from typing import Dict, List
import argparse
import sys


def f_half_score(pred: dict[str, list[str]], truth: dict[str, list[str]]) -> float:
    scores = []
    for s1_id, true_ids in truth.items():
        true_set = set(true_ids)
        pred_set = set(pred.get(s1_id, []))
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(true_set & pred_set)
        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        denom = 0.25 * precision + recall
        scores.append((1.25 * precision * recall / denom) if denom > 0 else 0.0)
    return sum(scores) / len(scores)


def load_tsv_mapping(path: str, id_col: str, list_col: str) -> Dict[str, List[str]]:
    """Reads a TSV file and returns {source1_id: [matched_ids]}."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return mapping
        headers = [c.strip() for c in header_line.rstrip("\n").split("\t")]
        id_idx = headers.index(id_col)
        list_idx = headers.index(list_col)

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= id_idx:
                continue
            s1_id = parts[id_idx].strip()
            if len(parts) > list_idx and parts[list_idx].strip():
                matched_ids = [m.strip() for m in parts[list_idx].split(",") if m.strip()]
            else:
                matched_ids = []
            mapping[s1_id] = matched_ids
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions using Macro F_0.5 score.")
    parser.add_argument("--pred", "-p", required=True, help="Path to predicted matching_results.tsv")
    parser.add_argument("--truth", "-t", required=True, help="Path to ground truth TSV (e.g. train_ground_truth.tsv)")
    parser.add_argument("--pred-id-col", default="source1_entity_id", help="Source 1 ID column in pred file")
    parser.add_argument("--pred-list-col", default="matched_entity_ids", help="Match list column in pred file")
    parser.add_argument("--truth-id-col", default="source1_entity_id", help="Source 1 ID column in truth file")
    parser.add_argument("--truth-list-col", default="matched_entity_ids", help="Match list column in truth file")
    args = parser.parse_args()

    pred = load_tsv_mapping(args.pred, args.pred_id_col, args.pred_list_col)
    truth = load_tsv_mapping(args.truth, args.truth_id_col, args.truth_list_col)

    score = f_half_score(pred, truth)
    print(f"Macro F_0.5 Score: {score:.6f}")


if __name__ == "__main__":
    main()
