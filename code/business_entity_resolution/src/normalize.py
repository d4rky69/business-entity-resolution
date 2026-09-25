"""Data Normalization and Preprocessing Module.

Simplest working version:
- Extract source identifier ('S1', 'S2', 'S3') from entity_id prefix
- Lowercase business_name
- Strip punctuation
- Strip common legal suffixes (inc, ltd, corp, pvt)
- Basic whitespace cleanup
"""

import re
from typing import Dict, Optional

# Regex compiled patterns for fast processing
SUFFIX_PATTERN = re.compile(r"\b(inc|ltd|corp|pvt)\b", re.IGNORECASE)
PUNCT_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def get_source(entity_id: str) -> str:
    """Extract source identifier ('S1', 'S2', or 'S3') from entity_id prefix.

    Note:
        Raw files contain NO separate source column. The source is derived
        strictly from the entity_id prefix ('S1-...', 'S2-...', 'S3-...').
    """
    if not entity_id or not isinstance(entity_id, str):
        return "UNKNOWN"
    prefix = entity_id.split("-", 1)[0].upper().strip()
    if prefix in ("S1", "S2", "S3"):
        return prefix
    return "UNKNOWN"


def normalize_name(name: Optional[str]) -> str:
    """Lowercase, strip punctuation, strip common legal suffixes, and collapse whitespace."""
    if not name or not isinstance(name, str):
        return ""
    # 1. Lowercase
    text = name.lower()
    # 2. Strip punctuation
    text = PUNCT_PATTERN.sub(" ", text)
    # 3. Strip common legal suffixes (inc, ltd, corp, pvt)
    text = SUFFIX_PATTERN.sub(" ", text)
    # 4. Collapse whitespace
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_record(record: dict, suffix_dict: Optional[dict] = None) -> dict:
    """Basic record normalization returning cleaned fields."""
    entity_id = record.get("entity_id", "")
    source = get_source(entity_id)
    raw_name = record.get("business_name") or ""
    clean_name = normalize_name(raw_name)
    country = (record.get("country") or "").strip().upper()

    return {
        "entity_id": entity_id,
        "source": source,
        "country": country,
        "raw_name": raw_name,
        "clean_name": clean_name,
        "base_name": clean_name,
        "raw_address": record.get("business_address") or "",
        "clean_address": (record.get("business_address") or "").strip().lower(),
    }
