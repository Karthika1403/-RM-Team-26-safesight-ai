"""Claim extraction: split an LVLM description into claims c_k with type rho_k and tier (S3)."""
import re
from dataclasses import dataclass
from .config import *

@dataclass
class Claim:
    text: str
    ctype: str      # object | attribute | relation
    tier: int
    path_claim: bool

def classify_type(t: str) -> str:
    t = t.lower()
    if any(w in t for w in RELATION_WORDS): return "relation"
    if any(w in t.split() for w in COLOR_WORDS): return "attribute"
    return "object"

def lookup_tier(t: str) -> int:
    t = t.lower() + " "
    for tier in (4, 3, 2):
        if any(k in t for k in TIER_KEYWORDS[tier]): return tier
    return 1

def extract_claims(description: str):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|;\s+|\n+", description) if p.strip()]
    return [Claim(p, classify_type(p), lookup_tier(p),
                  any(w in p.lower() for w in PATH_WORDS) and lookup_tier(p) == 4) for p in parts]
