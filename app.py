"""FastAPI backend: POST /api/analyze, GET /api/health, static UI at /"""
import io, os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from safesight import SafeSightPipeline
from safesight.grounding import MockGrounder

app = FastAPI(title="SafeSight AI (baseline)")
_clip = None

def get_grounder(image_bytes, tags):
    global _clip
    if os.getenv("SAFESIGHT_GROUNDER", "mock") == "clip" and image_bytes:
        from PIL import Image
        from safesight.grounding import ClipGrounder
        _clip = _clip or ClipGrounder()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return _clip, img
    return MockGrounder(tags.split(",") if tags else []), None

@app.get("/api/health")
def health(): return {"status": "ok", "grounder": os.getenv("SAFESIGHT_GROUNDER", "mock")}

@app.post("/api/analyze")
async def analyze(description: str = Form(...), scene_tags: str = Form(""), ablate: str = Form(""),
                  image: UploadFile = File(None)):
    data = await image.read() if image else None
    grounder, img = get_grounder(data, scene_tags)
    pipe = SafeSightPipeline(grounder, ablate=[a for a in ablate.split(",") if a])
    return pipe.run(img, description)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
