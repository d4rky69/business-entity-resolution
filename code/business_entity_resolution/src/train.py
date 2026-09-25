"""Model Training and F_0.5 Threshold Optimization Module.

Trains a pairwise ranking / classification model (e.g. LightGBM) to predict whether
a (Source 1, Candidate) pair represents the same real-world business entity.

Key Aspects:
- Grouped Train/Validation split: Grouped by source1_entity_id to prevent data leakage
- Target definition: Label 1 if candidate_entity_id is in ground truth matched_entity_ids, else 0
- Threshold optimization: Grid search to maximize macro-averaged F_0.5 score
  (weighting precision 2x over recall, handling singletons)
"""

import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import official metric
from ..utils.score_f_half import f_half_score


def load_ground_truth_sets(ground_truth_path: str) -> Dict[str, set]:
    """Loads ground truth into {source1_id: set(matched_ids)}."""
    gt_map = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            s1_id = parts[0].strip()
            matched = [m.strip() for m in parts[1].split(",") if m.strip()] if len(parts) > 1 else []
            gt_map[s1_id] = set(matched)
    return gt_map


def optimize_threshold(
    val_df: pd.DataFrame,
    val_probs: np.ndarray,
    val_ground_truth: Dict[str, list],
    threshold_range: Iterable = np.arange(0.3, 0.95, 0.05),
) -> Tuple[float, float]:
    """Finds decision threshold that maximizes macro F_0.5 score on validation set."""
    best_thresh = 0.5
    best_score = -1.0

    s1_ids = val_df["source1_entity_id"].values
    cand_ids = val_df["candidate_entity_id"].values

    for thresh in threshold_range:
        # Build predictions dict for all validation S1 entities
        pred_map: Dict[str, list] = {s1: [] for s1 in val_ground_truth}
        above_thresh = val_probs >= thresh
        for s1, cand, matched in zip(s1_ids, cand_ids, above_thresh):
            if matched and s1 in pred_map:
                pred_map[s1].append(cand)

        score = f_half_score(pred_map, val_ground_truth)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh, best_score


def train_classifier(
    features_path: str = "data/interim/features.tsv",
    ground_truth_path: str = "data/raw/dataset/train/train_ground_truth.tsv",
    model_dir: str = "models",
) -> Dict[str, Any]:
    """Stub: Trains classifier and tunes F_0.5 decision threshold.

    Args:
        features_path: Path to computed features.tsv
        ground_truth_path: Path to train_ground_truth.tsv
        model_dir: Directory where model and threshold metadata will be saved

    Returns:
        Dict containing training metadata, best threshold, and best validation score
    """
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.isfile(features_path) or not os.path.isfile(ground_truth_path):
        print(f"Skipping training: features or ground truth file missing.")
        meta = {"model_type": "baseline_stub", "optimal_threshold": 0.65, "val_f_half": 0.0}
        with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

    # Read dataset
    feat_df = pd.read_csv(features_path, sep="\t")
    gt_map = load_ground_truth_sets(ground_truth_path)

    # Assign binary labels: is_match
    labels = []
    for _, row in feat_df.iterrows():
        s1 = str(row["source1_entity_id"])
        cand = str(row["candidate_entity_id"])
        labels.append(1 if cand in gt_map.get(s1, set()) else 0)
    feat_df["label"] = labels

    feature_cols = [c for c in feat_df.columns if c not in ("source1_entity_id", "candidate_entity_id", "label")]

    # Grouped split by source1_entity_id
    unique_s1 = feat_df["source1_entity_id"].unique()
    np.random.seed(42)
    val_s1 = set(np.random.choice(unique_s1, size=int(len(unique_s1) * 0.2), replace=False))

    train_mask = ~feat_df["source1_entity_id"].isin(val_s1)
    val_mask = feat_df["source1_entity_id"].isin(val_s1)

    X_train, y_train = feat_df.loc[train_mask, feature_cols], feat_df.loc[train_mask, "label"]
    X_val, y_val = feat_df.loc[val_mask, feature_cols], feat_df.loc[val_mask, "label"]

    # Fit classifier (LightGBM if available, fallback to scikit-learn)
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

    # Evaluate validation probabilities
    val_probs = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
    val_gt_subset = {s1: list(gt_map.get(s1, set())) for s1 in val_s1}

    best_thresh, best_f05 = optimize_threshold(feat_df.loc[val_mask], val_probs, val_gt_subset)
    print(f"Optimal F_0.5 Threshold: {best_thresh:.3f} (Val F_0.5: {best_f05:.4f})")

    # Save artifacts
    model_path = os.path.join(model_dir, "matching_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "model_path": model_path,
        "feature_cols": feature_cols,
        "optimal_threshold": best_thresh,
        "val_f_half": best_f05,
    }
    with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
