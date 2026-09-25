#!/usr/bin/env python3
"""Convenience top-level runner for the Business Entity Resolution Pipeline."""

import sys
import os

# Ensure code/business_entity_resolution is in sys.path
sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

from src.infer import main

if __name__ == "__main__":
    main()
