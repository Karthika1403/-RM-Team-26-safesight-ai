"""S1: pre-generation risk fusion  R_k = alpha*r_k + beta*u_k + gamma*i_k   (Eq. 1)

BASELINE NOTE: the full system uses a trained probe (HALP-style), a Dempster-Shafer conflict score and the
Instruction-Lens score. The baseline uses cheap, transparent PROXIES with the same interface, so each can be
swapped for the real signal later without touching the rest of the pipeline.
"""
import math
from .config import ABSOLUTE_WORDS, Config

TYPE_PRIOR = {"relation": 0.55, "object": 0.35, "attribute": 0.2}   # relation errors are least-evaluated [1]

def probe_risk(claim, token_logprobs=None):                     # r_k (probe proxy)
    r = TYPE_PRIOR[claim.ctype]
    if token_logprobs:
        r = 0.5 * r + 0.5 * (1 - math.exp(sum(token_logprobs) / len(token_logprobs)))
    return r

def conflict_score(claim):                                       # u_k (evidential-conflict proxy)
    low = claim.text.lower()
    return 0.8 if any(w in low for w in ABSOLUTE_WORDS) else 0.2

def lens_inconsistency(claim, token_logprobs=None):              # i_k = 1 - InsLen_k (proxy)
    if not token_logprobs: return 0.5
    return min(1.0, max(0.0, -min(token_logprobs) / 5.0))

def fused_risk(claim, cfg: Config, token_logprobs=None):
    r, u, i = probe_risk(claim, token_logprobs), conflict_score(claim), lens_inconsistency(claim, token_logprobs)
    return cfg.alpha * r + cfg.beta * u + cfg.gamma * i, dict(r=round(r, 3), u=round(u, 3), i=round(i, 3))
