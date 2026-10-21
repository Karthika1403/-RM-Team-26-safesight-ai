"""Direct metrics (Table IX). labels[i] = True if claim i is truly hallucinated."""
def swhr(traces, labels):
    return sum(t["weight"] for t, h in zip(traces, labels) if h) / max(1e-9, sum(t["weight"] for t in traces))

def critical_miss_rate(traces, labels):
    crit = [t for t, h in zip(traces, labels) if t["tier"] == 4 and h]
    return sum(t["mode"] == "confirm" for t in crit) / max(1, len(crit))

def hallucination_pass_rate(traces, labels):
    bad = [t for t, h in zip(traces, labels) if h]
    return sum(t["mode"] == "confirm" for t in bad) / max(1, len(bad))
