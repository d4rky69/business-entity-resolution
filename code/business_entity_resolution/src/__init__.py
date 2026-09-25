"""Business Entity Resolution Pipeline.

Modular pipeline for record linkage across multi-source noisy business records:
- normalize: Text cleaning, phonetic/suffix stripping, address parsing, source resolution
- blocking: Multi-pass candidate generation (PIN, tokens, phonetic, embeddings)
- features: Pairwise similarity computation (lexical, phonetic, geographic, embedding)
- train: Supervised classifier (LightGBM/XGBoost) with F_0.5 threshold optimization
- verifier: High-precision rule/second-stage verification to prevent false merges
- infer: End-to-end inference and formatting to competition TSV submission standards
"""

from .normalize import normalize_record, get_source
from .blocking import generate_candidates
from .infer import aggregate_to_submission_format, main

# Optional modules requiring third-party libraries (pandas, lightgbm, etc.)
try:
    from .features import compute_features
except ImportError:
    compute_features = None

try:
    from .train import train_classifier
except ImportError:
    train_classifier = None

try:
    from .verifier import verify_pair
except ImportError:
    verify_pair = None

__all__ = [
    "normalize_record",
    "get_source",
    "generate_candidates",
    "compute_features",
    "train_classifier",
    "verify_pair",
    "aggregate_to_submission_format",
    "main",
]
