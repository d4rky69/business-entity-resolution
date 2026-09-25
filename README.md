# Business Entity Resolution — Hackathon Repository

An end-to-end machine learning pipeline for multi-source entity resolution (record linkage across 3 noisy business databases), evaluated on macro-averaged $F_{0.5}$ per Source 1 entity.

---

## Repository Structure

```
.
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── normalize.py       # Normalization, suffix stripping, ID prefix source extraction
│       │   ├── blocking.py        # Multi-pass candidate generation (candidates_long.tsv)
│       │   ├── features.py        # Pairwise similarity feature computation (features.tsv)
│       │   ├── train.py           # Classifier training & F_0.5 threshold optimization
│       │   ├── verifier.py        # High-precision guardrail verifier
│       │   └── infer.py           # End-to-end inference & official TSV formatting
│       ├── utils/
│       │   └── score_f_half.py    # Official Macro F_0.5 evaluation implementation
│       ├── SCHEMA.md              # Full dataset and submission TSV schema definitions
│       ├── README.md              # Pipeline documentation & reproduction guide
│       ├── requirements.txt       # Pinned dependencies
│       └── .gitignore             # Subdirectory gitignore
├── config/
│   └── suffixes.json              # Legal business suffixes (generic, US, IN, FR)
├── data/
│   ├── raw/                       # Linked / copied dataset/ (gitignored)
│   └── interim/                   # Working tables (candidates_long.tsv, features.tsv)
├── output/                        # Final TSV outputs for submission
│   ├── matching_results.tsv       # Leaderboard submission
│   └── candidate_pairs.tsv        # Audited candidate set
├── utils/
│   └── validate_submission.py     # Organizer-provided validation tool
├── Documentation_template.md      # Solution methodology template
└── .gitignore                     # Root gitignore (data, models, outputs, env)
```

---

## Quick Start

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run Inference & Generate Submission Outputs
```bash
python -m code.business_entity_resolution.src.infer \
    --test-dir data/raw/dataset/test \
    --output-dir output
```

### 3. Run Organizer Submission Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/raw/dataset/test
```
*(Optionally append `--check-ids` to verify every ID exists in test Source 2 and 3).*

### 4. Evaluate Validation Macro $F_{0.5}$
```bash
python code/business_entity_resolution/utils/score_f_half.py \
    --pred output/matching_results.tsv \
    --truth data/raw/dataset/train/train_ground_truth.tsv
```
