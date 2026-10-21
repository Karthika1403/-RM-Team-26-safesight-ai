"""SafeSight AI baseline: S1..S6 combined (Algorithm 1)."""
from .config import Config, TIER_WEIGHT
from .claims import extract_claims
from .signals import fused_risk
from . import steering, policy

class SafeSightPipeline:
    def __init__(self, grounder, cfg: Config = None, ablate=()):
        """ablate: subset of {'S1_probe','S2','S3','S4','S5','S6'} -> ablation variants (V1,V3,V4,V5,V6,V8)."""
        self.g, self.cfg, self.ablate = grounder, cfg or Config(), set(ablate)

    def run(self, image, description, token_logprobs=None):
        c, traces = self.cfg, []
        for cl in extract_claims(description):
            R, parts = fused_risk(cl, c, token_logprobs)                                        # S1
            if "S1_probe" in self.ablate: R = (R - c.alpha * parts["r"]) / (1 - c.alpha)       # V1
            w = 1.0 if "S3" in self.ablate else TIER_WEIGHT[cl.tier]                           # S3 (V4)
            nu = c.nu_max if cl.path_claim else 1.0
            H = min(1.0, w * nu)
            ground = "S2" not in self.ablate and (R >= c.theta_low or cl.tier >= 3)            # S2 trigger
            g, Rt = None, R
            if ground:
                g = self.g.score(image, cl.text); Rt = 0.7 * (1 - g) + 0.3 * R                  # Eq. 2 variant (see README)
            gate = 1 if "S4" in self.ablate else int(Rt * H >= c.theta_b)                      # S4 (V5 = always on)
            steer = steering.simulate(cl, g if g is not None else 1.0, c) if gate and "S5" not in self.ablate else None
            ungr = bool(steer and steer["still_ungroundable"]) or (g is not None and g < c.theta_ground)
            mode = "confirm" if "S6" in self.ablate else policy.decide(Rt, cl.tier, ungr, c)    # S6 (V8)
            traces.append({"claim": cl.text, "type": cl.ctype, "tier": cl.tier, "weight": w, "urgency": nu,
                           "signals": parts, "R": round(R, 3), "grounded": ground,
                           "support": None if g is None else round(g, 3), "R_refined": round(Rt, 3),
                           "H": round(H, 3), "gate": gate, "steering": steer, "mode": mode,
                           "spoken": policy.speak(mode, cl.text), "why": policy.explain(mode, ungr)})
        return {"claims": traces, "spoken": " ".join(t["spoken"] for t in traces),
                "intervention_rate": sum(t["gate"] for t in traces) / max(1, len(traces))}
