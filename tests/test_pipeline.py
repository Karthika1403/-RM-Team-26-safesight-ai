import numpy as np
from safesight import SafeSightPipeline
from safesight.grounding import MockGrounder
from safesight.steering import orthogonal_steer, steering_vector
from safesight.metrics import swhr, critical_miss_rate

def run(desc, tags, **kw): return SafeSightPipeline(MockGrounder(tags), **kw).run(None, desc)["claims"]

def test_cs1_false_clearance_not_asserted():
    t = run("The path ahead is clear.", ["doorway"])[0]
    assert t["tier"] == 4 and t["gate"] == 1 and t["mode"] == "abstain"

def test_cs1_grounded_clearance_not_gated():
    t = run("The path ahead is clear.", ["clear"])[0]
    assert t["gate"] == 0 and t["mode"] in ("confirm", "hedge")

def test_cs2_hallucinated_bicycle_flagged():
    t = run("A bicycle is parked on your left.", ["street"])[0]
    assert t["gate"] == 1 and t["mode"] == "abstain"

def test_cs3_ungroundable_exit_sign_abstains():
    assert run("The exit sign is above the door.", ["wall"])[0]["mode"] == "abstain"

def test_low_tier_colour_not_gated():
    t = run("The jacket is red.", ["street"])[0]
    assert t["tier"] == 1 and t["gate"] == 0

def test_v8_no_abstention_asserts_falsely():
    assert run("The path ahead is clear.", [], ablate=["S6"])[0]["mode"] == "confirm"

def test_v5_always_on_gate_raises_intervention():
    full = SafeSightPipeline(MockGrounder(["jacket"])).run(None, "The jacket is red.")
    abl = SafeSightPipeline(MockGrounder(["jacket"]), ablate=["S4"]).run(None, "The jacket is red.")
    assert abl["intervention_rate"] > full["intervention_rate"]

def test_orthogonal_projection():
    v, p = np.array([1., 1., 0.]), np.array([1., 0., 0.])
    assert abs(steering_vector(v, p) @ p) < 1e-9
    assert np.allclose(orthogonal_steer([0, 0, 0], v, p, gate=0), 0)

def test_metrics():
    tr = run("The path ahead is clear. The jacket is red.", [])
    assert 0 <= swhr(tr, [True, False]) <= 1 and critical_miss_rate(tr, [True, False]) == 0
