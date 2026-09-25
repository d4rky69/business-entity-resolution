"""Data Normalization and Preprocessing Module.

Preprocesses noisy business records across diverse sources (Source 1, Source 2, Source 3)
and geographic jurisdictions (US, India, France). Handles:
- Source extraction from entity_id prefix (e.g., S1-00001 -> S1)
- Legal suffix normalization and stripping (e.g., "Pvt Ltd", "LLC", "SAS")
- Punctuation removal, case folding, and whitespace normalization
- Address component parsing (postal code, locality, street abbreviations)
"""

import re
import string
from typing import Any, Dict, Optional


def get_source(entity_id: str) -> str:
    """Extract source identifier ("S1", "S2", or "S3") from entity_id prefix.

    Note:
        Raw files contain NO separate source column. The source is derived
        strictly from the entity_id prefix ("S1-...", "S2-...", "S3-...").

    Args:
        entity_id: String ID of the entity (e.g., 'S1-00042', 'S2-00109')

    Returns:
        One of 'S1', 'S2', 'S3', or 'UNKNOWN'
    """
    if not entity_id or not isinstance(entity_id, str):
        return "UNKNOWN"
    prefix = entity_id.split("-", 1)[0].upper().strip()
    if prefix in ("S1", "S2", "S3"):
        return prefix
    return "UNKNOWN"


def clean_text(text: Optional[str]) -> str:
    """Basic lowercasing, punctuation normalization, and whitespace cleanup."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    # Normalize unicode quotes and dashes
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    # Replace punctuation with whitespace except alphanumeric
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", text).strip()


def strip_legal_suffixes(name: str, country: str, suffix_dict: Dict[str, list]) -> str:
    """Strips country-specific and generic legal suffixes from a business name."""
    if not name:
        return ""
    country_key = country.upper() if country else "GENERIC"
    suffixes = set(suffix_dict.get("generic", []))
    if country_key in suffix_dict:
        suffixes.update(suffix_dict[country_key])
    elif country_key in ("INDIA", "IND", "IN"):
        suffixes.update(suffix_dict.get("IN", []))
    elif country_key in ("UNITED STATES", "USA", "US"):
        suffixes.update(suffix_dict.get("US", []))
    elif country_key in ("FRANCE", "FRA", "FR"):
        suffixes.update(suffix_dict.get("FR", []))

    # Sort suffixes by length descending so longer compound phrases match first
    sorted_suffixes = sorted(suffixes, key=len, reverse=True)
    cleaned = name
    for sfx in sorted_suffixes:
        pattern = rf"\b{re.escape(sfx.lower())}\b"
        cleaned = re.sub(pattern, " ", cleaned)

    return re.sub(r"\s+", " ", cleaned).strip()


def extract_postal_code(address: str, country: str) -> Optional[str]:
    """Extracts country-specific postal code (US 5-digit, India 6-digit PIN, France 5-digit)."""
    if not address:
        return None
    # India PIN code: 6 digits, optionally starting with 1-9
    pin_match = re.search(r"\b([1-9][0-9]{5})\b", address)
    if pin_match:
        return pin_match.group(1)
    # US Zip / France Code Postal: 5 digits
    zip_match = re.search(r"\b([0-9]{5})\b", address)
    if zip_match:
        return zip_match.group(1)
    return None


def normalize_record(record: dict, suffix_dict: dict) -> dict:
    """Stub: Normalizes a raw business record dictionary into cleaned, structured format.

    Args:
        record: Raw dict containing 'entity_id', 'business_name', 'business_address', 'country'
        suffix_dict: Dictionary loaded from config/suffixes.json with keys 'generic', 'US', 'IN', 'FR'

    Returns:
        Cleaned dict with normalized fields and extracted attributes:
            - entity_id: original ID
            - source: 'S1', 'S2', or 'S3'
            - raw_name: original business_name
            - clean_name: lowercased, punctuation-cleaned name
            - base_name: clean_name with legal suffixes removed
            - raw_address: original business_address
            - clean_address: lowercased, cleaned address
            - postal_code: extracted postal code / PIN if available
            - country: normalized country string (US, India, France)
    """
    entity_id = record.get("entity_id", "")
    source = get_source(entity_id)
    country = (record.get("country") or "").strip()
    raw_name = record.get("business_name") or ""
    raw_address = record.get("business_address") or ""

    clean_nm = clean_text(raw_name)
    base_nm = strip_legal_suffixes(clean_nm, country, suffix_dict)
    clean_addr = clean_text(raw_address)
    postal_code = extract_postal_code(raw_address, country)

    return {
        "entity_id": entity_id,
        "source": source,
        "country": country,
        "raw_name": raw_name,
        "clean_name": clean_nm,
        "base_name": base_nm if base_nm else clean_nm,
        "raw_address": raw_address,
        "clean_address": clean_addr,
        "postal_code": postal_code,
    }
