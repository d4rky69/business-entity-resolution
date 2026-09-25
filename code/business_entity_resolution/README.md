# Business Entity Resolution Pipeline

Record linkage and entity resolution across 3 noisy commercial data sources where **Source 1** serves as the deduplicated reference anchor. Evaluated on **macro-averaged $F_{0.5}$ score** per Source-1 entity.

---

## 1. Problem Overview

Commercial business identity records arrive with partial, inconsistent, and conflicting fragments (misspellings, legal suffix abbreviations, address variations, missing postal codes, and landmark references). The objective is to identify all corresponding records from **Source 2** and **Source 3** for each **Source 1** entity.

### Golden Rules & Constraints
- **File Format:** All input and output datasets are strictly tab-separated (`.tsv`). TSV reads and writes **must** explicitly specify `sep='\t'`.
- **Source Resolution:** Raw source files have **no source column**. The source is derived entirely from the entity ID prefix (`S1-` = Source 1, `S2-` = Source 2, `S3-` = Source 3).
- **Geographic Scope:** Training data covers `US` and `India`. Test data contains an additional unseen country: `France`. Country is an open string label; France must never be filtered out.
- **Model Constraints:** Open-weight models must adhere to MIT/Apache 2.0 licenses with parameter size $\le 8\text{B}$.
- **Fair Play:** External API lookups, web scraping, and external business registry queries are **strictly prohibited**.

---

## 2. Evaluation Metric: Macro-Averaged $F_{0.5}$

Submissions are evaluated on **macro-averaged $F_{0.5}$** across all Source 1 entities in the test set:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Precision-heavy:** Precision is weighted $2\times$ over recall to penalize false merges.
- **Singletons:** Source 1 entities with no true matches score $1.0$ when predicted as empty, and $0.0$ if any match is predicted.

---

## 3. Pipeline Stages

The pipeline is organized modularly under `src/`:

1. **Preprocessing & Normalization (`src/normalize.py`)**
   - Extracts source identifier from ID prefixes (`get_source`).
   - Normalizes unicode characters, case folding, and whitespace.
   - Cleans country-specific and generic legal suffixes (configured in `config/suffixes.json`).
   - Extracts structured postal codes / PINs.

2. **Candidate Generation / Blocking (`src/blocking.py`)**
   - Drastically reduces comparison space using multi-pass blocking keys:
     - Exact postal code + name token
     - Rare token inverted index
     - Phonetic blocking (Metaphone / Soundex)
   - Writes internal working format: `data/interim/candidates_long.tsv`

3. **Feature Engineering (`src/features.py`)**
   - Pairwise lexical, phonetic, and semantic similarities:
     - Levenshtein distance ratio, Token sort ratio, Jaccard token overlap
     - Address similarity and postal code alignment
     - Source and geographic consistency indicators
   - Writes `data/interim/features.tsv`

4. **Model Training & Threshold Tuning (`src/train.py`)**
   - Grouped train/validation split by Source 1 entity to prevent leakage.
   - Fits gradient boosted decision trees (LightGBM).
   - Optimizes probability decision threshold directly against macro $F_{0.5}$.

5. **High-Precision Verifier (`src/verifier.py`)**
   - Rule-based guardrails to reject false merges (country contradictions, conflicting postal codes).

6. **Inference & Aggregation (`src/infer.py`)**
   - Aggregates pairwise predictions into official competition format:
     - `output/matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`)
     - `output/candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`)

---

## 4. Setup & Installation

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. How to Run

### Step 1: Run Training & Threshold Tuning
```bash
python -m src.train
```

### Step 2: Run End-to-End Inference
Generate both `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python -m src.infer --test-dir data/raw/dataset/test --output-dir output
```

### Step 3: Validate Outputs with Organizer Tool
Before submitting to the portal, execute the official validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/raw/dataset/test
```

### Optional Diagnostic Check (All IDs Existence)
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/raw/dataset/test \
    --check-ids
```

### Step 4: Evaluate Validation Score
Calculate the macro $F_{0.5}$ score locally on validation splits:
```bash
python code/business_entity_resolution/utils/score_f_half.py \
    --pred output/matching_results.tsv \
    --truth data/raw/dataset/train/train_ground_truth.tsv
```

---

## 6. Team & Submission Deliverables

- **Team Name:** [Your Team Name]
- **Deliverables:**
  - `output/matching_results.tsv` (Leaderboard submission)
  - `output/candidate_pairs.tsv` (Candidate audit file)
  - `Documentation_template.md` (Methodology write-up)
  - Complete reproducible source code under `code/business_entity_resolution/`
