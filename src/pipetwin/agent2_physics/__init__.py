"""Isolated workstream for E03 physics models under contract 1.0.0.

The shared registry is connected separately by its owner after integration review.
"""

CONTRACT_VERSION = "1.0.0"

from .adapter import make_model

__all__ = ["CONTRACT_VERSION", "make_model"]
