"""S6: accessible output policy (Table VII)."""
def decide(risk, tier, ungroundable, cfg):
    if risk >= cfg.theta_abs or (ungroundable and tier >= 3): return "abstain"
    if risk >= 0.45: return "hedge"
    return "confirm"

def speak(mode, text):
    t = text.rstrip(".!?")
    return {"confirm": text,
            "hedge": f"I think {t[0].lower() + t[1:]}, but I am not fully sure.",
            "abstain": "I cannot confirm that. Please check another way."}[mode]

def explain(mode, ungroundable):
    if mode == "abstain" and ungroundable:
        return "I could not match that to anything I can see clearly. Try turning slowly and asking again."
    if mode == "abstain": return "This could affect your safety and I am not sure enough, so I am not saying it."
    if mode == "hedge": return "I can see some evidence but not enough to be certain."
    return "What I said matches what I can see."
