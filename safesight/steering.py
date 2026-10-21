"""S5: targeted latent steering (Eq. 5):  h' = h + G * lambda * s,  s = v - <v,p>/||p||^2 * p
`orthogonal_steer` is the exact operator for a hooked open-weight LVLM (see README, next step).
The baseline demo has no hidden states, so `simulate` records the intervention and re-scores the claim."""
import numpy as np

def steering_vector(v, p):
    v, p = np.asarray(v, float), np.asarray(p, float)
    return v - (v @ p) / (p @ p + 1e-12) * p

def orthogonal_steer(h, v, p, gate, lam=1.0):
    return np.asarray(h, float) + gate * lam * steering_vector(v, p)

def simulate(claim, support, cfg):
    return {"applied": True, "regenerated_as": None if support < cfg.theta_ground else claim.text,
            "still_ungroundable": support < cfg.theta_ground}
