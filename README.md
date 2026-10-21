# SafeSight AI – Baseline (S1–S6 combined)

Risk-aware, event-driven hallucination detection and mitigation for BLV assistive vision.

## What "baseline" means here
A baseline is the simplest complete working version of the system. Every later improvement (real probe, VBackChecker grounding,
real latent steering) is measured against it, and the ablations V0–V12 switch parts of it off. Each stage has a fixed interface,
so a proxy can be replaced by the real method without changing the rest.

| Stage | File | Baseline implementation | Full-system upgrade |
|---|---|---|---|
| S1 risk fusion | signals.py | transparent proxies for r, u, i (claim type, absolute-claim cue, token log-probs) | HALP probe, evidential conflict, InsLen |
| S2 grounding | grounding.py | MockGrounder (scene tags) or CLIP support | VBackChecker pixel grounding |
| S3 hazard tier | claims.py, config.py | keyword tier table (Table VI) + urgency factor | participatory-validated table |
| S4 gate | pipeline.py | G = 1[R~·H >= theta_b] | per-backbone calibrated theta_b |
| S5 steering | steering.py | exact orthogonal-projection operator (Eq. 5, unit-tested); demo simulates the intervention | hook on an open-weight LVLM hidden layer |
| S6 output policy | policy.py | confirm / hedge / abstain / follow-up | user-studied wording |

Deviation from the paper: Eq. 2 uses `R~ = 0.7(1-g) + 0.3R` instead of `max(R, 1-g)` because the max rule kept grounded
claims above the gate. Report this and test it as an ablation.

## Run
    pip install -r requirements.txt
    pytest -q
    uvicorn app:app --reload          # open http://127.0.0.1:8000
Optional CLIP grounding: `pip install torch transformers` and `export SAFESIGHT_GROUNDER=clip`.

API: `POST /api/analyze` (form: description, scene_tags, ablate="S2,S4", image) · `GET /api/health`

## Deploy free (Hugging Face Spaces, Docker)
1. Create a Space → SDK: Docker. 2. Push this folder (Dockerfile exposes port 7860). 3. Copy the Space URL into the submission README.

## GitHub
    git init && git add . && git commit -m "SafeSight AI baseline"
    git branch -M main
    git remote add origin https://github.com/<user>/safesight-ai.git
    git push -u origin main
Suggested repo layout later: `/docs` (case study, PPT), `/data` (IEEE DataPort link + schema), `/experiments`, `/safesight`.

## Submission README checklist
Online server URL · GitHub URL · IEEE DataPort URL · member page with photos and emails.
