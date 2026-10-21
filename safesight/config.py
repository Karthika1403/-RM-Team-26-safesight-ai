"""Tunable parameters (Table VIII of the survey). All values are INITIAL and must be calibrated."""
from dataclasses import dataclass

@dataclass
class Config:
    alpha: float = 0.4       # weight of probe risk r_k
    beta: float = 0.3        # weight of evidential-conflict u_k
    gamma: float = 0.3       # weight of instruction-lens inconsistency i_k
    theta_low: float = 0.35  # invoke S2 grounding if R_k >= theta_low
    theta_b: float = 0.50    # gate threshold (per backbone in the full system)
    theta_abs: float = 0.75  # abstain threshold
    theta_ground: float = 0.30  # support below this = "ungroundable"
    nu_max: float = 1.5      # urgency cap for path claims
    steer_lambda: float = 1.0

TIER_WEIGHT = {4: 1.0, 3: 0.7, 2: 0.4, 1: 0.1}   # Table VI

TIER_KEYWORDS = {
    4: ["path is clear", "path ahead", "clear path", "way is clear", "stairs", "staircase", "step down",
        "steps", "curb", "drop-off", "drop off", "vehicle", "car ", "bus", "truck", "hole", "traffic"],
    3: ["obstacle", "bicycle", "bike", "pole", "wall", "door", "crossing", "signal", "exit", "medicine",
        "dosage", "tablet", "label", "braille", "allergen", "pothole"],
    2: ["sign", "person", "people", "pedestrian", "shop", "store", "bench", "tree", "landmark"],
}
RELATION_WORDS = ["ahead", "left", "right", "behind", "in front", "next to", "near", "on the", "beside", "above", "below"]
COLOR_WORDS = ["red", "green", "blue", "yellow", "black", "white", "grey", "gray", "brown", "orange", "pink", "purple"]
ABSOLUTE_WORDS = ["clear", "nothing", "no ", "nobody", "safe", "empty", "free of", "definitely", "always"]
PATH_WORDS = ["path", "ahead", "in front", "way is", "clear"]
